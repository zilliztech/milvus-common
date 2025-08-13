// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once

#include <folly/executors/thread_factory/NamedThreadFactory.h>
#include <omp.h>

#ifdef __linux__

#if defined(__PPC64__) || defined(__ppc64__) || defined(__PPC64LE__) || defined(__ppc64le__) || defined(__powerpc64__)
#include <openblas/cblas.h>
#else
#include <cblas.h>
#endif

#include <sys/resource.h>
#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 30
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)
#endif
#endif

#include <cassert>
#include <cerrno>
#include <cstring>
#include <memory>
#include <thread>
#include <utility>

#include "folly/executors/CPUThreadPoolExecutor.h"
#include "folly/executors/task_queue/UnboundedBlockingQueue.h"
#include "folly/futures/Future.h"
#include "log/Log.h"

namespace knowhere {

class ThreadPool {
 public:
    enum class QueueType { LIFO, FIFO };
#ifdef __linux__
 private:
    class CustomPriorityThreadFactory : public folly::NamedThreadFactory {
     public:
        using folly::NamedThreadFactory::NamedThreadFactory;
        std::thread
        newThread(folly::Func&& func) override {
            return folly::NamedThreadFactory::newThread([&, func = std::move(func)]() mutable {
                if (setpriority(PRIO_PROCESS, gettid(), thread_priority_) != 0) {
                    // fallback to 19 priority due to SYS_NICE compatiblity
                    // it is designed that the thread pool shall have lower priority than normal
                    // in case of heartbeat thread starving
                    if (setpriority(PRIO_PROCESS, gettid(), 19) != 0) {
                        LOG_ERROR("Failed to set priority of knowhere thread. Error is: %s", std::strerror(errno));
                    } else {
                        LOG_WARN("Successfully set fallback priority of knowhere thread.");
                    }
                } else {
                    LOG_INFO("Successfully set priority of knowhere thread.");
                }
                func();
            });
        }

        explicit CustomPriorityThreadFactory(const std::string& thread_name_prefix, int thread_priority)
            : folly::NamedThreadFactory(thread_name_prefix), thread_priority_(thread_priority) {
            assert(thread_priority_ >= -20 && thread_priority_ < 20);
        }

     private:
        int thread_priority_;
    };

 public:
    explicit ThreadPool(uint32_t num_threads, const std::string& thread_name_prefix, QueueType queueT = QueueType::LIFO,
                        int thread_priority = 10)
        : pool_(queueT == QueueType::LIFO
                    ? folly::CPUThreadPoolExecutor(
                          num_threads,
                          std::make_unique<folly::LifoSemMPMCQueue<folly::CPUThreadPoolExecutor::CPUTask,
                                                                   folly::QueueBehaviorIfFull::BLOCK>>(
                              num_threads * kTaskQueueFactor),
                          std::make_shared<CustomPriorityThreadFactory>(thread_name_prefix, thread_priority))
                    : folly::CPUThreadPoolExecutor(
                          num_threads,
                          std::make_unique<folly::UnboundedBlockingQueue<folly::CPUThreadPoolExecutor::CPUTask>>(),
                          std::make_shared<CustomPriorityThreadFactory>(thread_name_prefix, thread_priority))) {
    }
#else
 public:
    // `thread_priority` is linux only, the param is kept here to make signature same between linux & mac one
    explicit ThreadPool(uint32_t num_threads, const std::string& thread_name_prefix, QueueType queueT = QueueType::LIFO,
                        int thread_priority = 10)
        : pool_(queueT == QueueType::LIFO
                    ? folly::CPUThreadPoolExecutor(
                          num_threads,
                          std::make_unique<folly::LifoSemMPMCQueue<folly::CPUThreadPoolExecutor::CPUTask,
                                                                   folly::QueueBehaviorIfFull::BLOCK>>(
                              num_threads * kTaskQueueFactor),
                          std::make_shared<folly::NamedThreadFactory>(thread_name_prefix))
                    : folly::CPUThreadPoolExecutor(
                          num_threads,
                          std::make_unique<folly::UnboundedBlockingQueue<folly::CPUThreadPoolExecutor::CPUTask>>(),
                          std::make_shared<folly::NamedThreadFactory>(thread_name_prefix))) {
    }
#endif

    ThreadPool(const ThreadPool&) = delete;

    ThreadPool&
    operator=(const ThreadPool&) = delete;

    ThreadPool(ThreadPool&&) noexcept = delete;

    ThreadPool&
    operator=(ThreadPool&&) noexcept = delete;

    template <typename Func, typename... Args>
    auto
    push(Func&& func, Args&&... args) {
        return folly::makeSemiFuture().via(&pool_).then(
            [func = std::forward<Func>(func), &args...](auto&&) mutable { return func(std::forward<Args>(args)...); });
    }

    [[nodiscard]] size_t
    size() const noexcept {
        return pool_.numThreads();
    }

    size_t
    GetPendingTaskCount() {
        return pool_.getPendingTaskCount();
    }

    folly::CPUThreadPoolExecutor&
    GetPool() {
        return pool_;
    }

    void
    SetNumThreads(uint32_t num_threads) {
        if (num_threads == 0) {
            LOG_ERROR("set number of threads can not be 0");
            return;
        } else {
            // setNumThreads() adjust the relevant variables instead of changing the number of threads directly;
            // If numThreads < active threads, reduce number of running threads.
            pool_.setNumThreads(num_threads);
            return;
        }
    }

    static ThreadPool
    CreateFIFO(uint32_t num_threads, const std::string& thread_name_prefix);

    static ThreadPool
    CreateLIFO(uint32_t num_threads, const std::string& thread_name_prefix);

    static void
    InitGlobalBuildThreadPool(uint32_t num_threads);

    static void
    InitGlobalSearchThreadPool(uint32_t num_threads);

    static void
    SetGlobalBuildThreadPoolSize(uint32_t num_threads);

    static size_t
    GetGlobalBuildThreadPoolSize();

    static void
    SetGlobalSearchThreadPoolSize(uint32_t num_threads);

    static size_t
    GetGlobalSearchThreadPoolSize();

    static size_t
    GetSearchThreadPoolPendingTaskCount();

    static size_t
    GetBuildThreadPoolPendingTaskCount();

    static std::shared_ptr<ThreadPool>
    GetGlobalBuildThreadPool();

    static std::shared_ptr<ThreadPool>
    GetGlobalSearchThreadPool();

    class ScopedBuildOmpSetter {
        int omp_before;
#ifdef OPENBLAS_OS_LINUX
        int blas_thread_before;
#endif
     public:
        explicit ScopedBuildOmpSetter(int num_threads = 0);

        ~ScopedBuildOmpSetter();
    };

    class ScopedSearchOmpSetter {
        int omp_before;

     public:
        explicit ScopedSearchOmpSetter(int num_threads = 1);

        ~ScopedSearchOmpSetter();
    };

 private:
    folly::CPUThreadPoolExecutor pool_;

    static std::mutex build_pool_mutex_;
    static std::shared_ptr<ThreadPool> build_pool_;

    static std::mutex search_pool_mutex_;
    static std::shared_ptr<ThreadPool> search_pool_;

    constexpr static size_t kTaskQueueFactor = 16;
};

// This class is used to wrap the thread pool and the inline executor
// If use_pool is true, the function will be pushed to the thread pool
// If use_pool is false, the function will be executed directly
class ThreadPoolWrapper {
 public:
    ThreadPoolWrapper(const std::shared_ptr<ThreadPool>& pool, bool use_pool = true)
        : pool_(pool), use_pool_(use_pool) {
    }

    template <typename Func, typename... Args>
    auto
    push(Func&& func, Args&&... args) {
        if (use_pool_) {
            return pool_->push(std::forward<Func>(func), std::forward<Args>(args)...);
        } else {
            // If the pool is not used, the function will be executed within the current thread directly
            return folly::makeSemiFuture()
                .via(&folly::InlineExecutor::instance())
                .then([func = std::forward<Func>(func), &args...](auto&&) mutable {
                    return func(std::forward<Args>(args)...);
                });
        }
    }

 private:
    std::shared_ptr<ThreadPool> pool_;
    bool use_pool_;
};

}  // namespace knowhere
