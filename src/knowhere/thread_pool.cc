#include "knowhere/thread_pool.h"

#include "log/Log.h"

namespace knowhere {
std::mutex ThreadPool::build_pool_mutex_;
std::shared_ptr<ThreadPool> ThreadPool::build_pool_ = nullptr;
std::mutex ThreadPool::search_pool_mutex_;
std::shared_ptr<ThreadPool> ThreadPool::search_pool_ = nullptr;

ThreadPool
ThreadPool::CreateFIFO(uint32_t num_threads, const std::string& thread_name_prefix) {
    return ThreadPool(num_threads, thread_name_prefix, QueueType::FIFO);
}

ThreadPool
ThreadPool::CreateLIFO(uint32_t num_threads, const std::string& thread_name_prefix) {
    return ThreadPool(num_threads, thread_name_prefix, QueueType::LIFO);
}

void
ThreadPool::InitGlobalBuildThreadPool(uint32_t num_threads) {
    if (num_threads <= 0) {
        LOG_ERROR("num_threads should be bigger than 0");
        return;
    }

    if (build_pool_ == nullptr) {
        std::lock_guard<std::mutex> lock(build_pool_mutex_);
        if (build_pool_ == nullptr) {
            build_pool_ = std::make_shared<ThreadPool>(num_threads, "knowhere_build");
            LOG_INFO(fmt::format("Init global build thread pool with size {}", num_threads));
            return;
        }
    } else {
        LOG_INFO(fmt::format("Global build thread pool size has already been initialized to {}", build_pool_->size()));
    }
}

void
ThreadPool::InitGlobalSearchThreadPool(uint32_t num_threads) {
    if (num_threads <= 0) {
        LOG_ERROR("num_threads should be bigger than 0");
        return;
    }

    if (search_pool_ == nullptr) {
        std::lock_guard<std::mutex> lock(search_pool_mutex_);
        if (search_pool_ == nullptr) {
            search_pool_ = std::make_shared<ThreadPool>(num_threads, "knowhere_search");
            LOG_INFO(fmt::format("Init global search thread pool with size {}", num_threads));
            return;
        }
    } else {
        LOG_INFO(
            fmt::format("Global search thread pool size has already been initialized to {}", search_pool_->size()));
    }
}

void
ThreadPool::SetGlobalBuildThreadPoolSize(uint32_t num_threads) {
    if (build_pool_ == nullptr) {
        InitGlobalBuildThreadPool(num_threads);
        return;
    } else {
        build_pool_->SetNumThreads(num_threads);
        LOG_INFO(fmt::format("Global build thread pool size has already been set to {}", build_pool_->size()));
        return;
    }
}

size_t
ThreadPool::GetGlobalBuildThreadPoolSize() {
    return (build_pool_ == nullptr ? 0 : build_pool_->size());
}

void
ThreadPool::SetGlobalSearchThreadPoolSize(uint32_t num_threads) {
    if (search_pool_ == nullptr) {
        InitGlobalSearchThreadPool(num_threads);
        return;
    } else {
        search_pool_->SetNumThreads(num_threads);
        LOG_INFO(fmt::format("Global search thread pool size has already been set to {}", search_pool_->size()));
        return;
    }
}

size_t
ThreadPool::GetGlobalSearchThreadPoolSize() {
    return (search_pool_ == nullptr ? 0 : search_pool_->size());
}

size_t
ThreadPool::GetSearchThreadPoolPendingTaskCount() {
    return ThreadPool::GetGlobalSearchThreadPool()->GetPendingTaskCount();
}

size_t
ThreadPool::GetBuildThreadPoolPendingTaskCount() {
    return ThreadPool::GetGlobalBuildThreadPool()->GetPendingTaskCount();
}

std::shared_ptr<ThreadPool>
ThreadPool::GetGlobalBuildThreadPool() {
    if (build_pool_ == nullptr) {
        InitGlobalBuildThreadPool(std::thread::hardware_concurrency());
    }
    return build_pool_;
}

std::shared_ptr<ThreadPool>
ThreadPool::GetGlobalSearchThreadPool() {
    if (search_pool_ == nullptr) {
        InitGlobalSearchThreadPool(std::thread::hardware_concurrency());
    }
    return search_pool_;
}

ThreadPool::ScopedBuildOmpSetter::ScopedBuildOmpSetter(int num_threads) {
    omp_before = (build_pool_ ? build_pool_->size() : omp_get_max_threads());
#ifdef OPENBLAS_OS_LINUX
    // to avoid thread spawn when IVF_PQ build
    blas_thread_before = openblas_get_num_threads();
    openblas_set_num_threads(1);
#endif
    omp_set_num_threads(num_threads <= 0 ? omp_before : num_threads);
}

ThreadPool::ScopedBuildOmpSetter::~ScopedBuildOmpSetter() {
#ifdef OPENBLAS_OS_LINUX
    openblas_set_num_threads(blas_thread_before);
#endif
    omp_set_num_threads(omp_before);
}

ThreadPool::ScopedSearchOmpSetter::ScopedSearchOmpSetter(int num_threads) {
    omp_before = (search_pool_ ? search_pool_->size() : omp_get_max_threads());
    omp_set_num_threads(num_threads <= 0 ? omp_before : num_threads);
}

ThreadPool::ScopedSearchOmpSetter::~ScopedSearchOmpSetter() {
    omp_set_num_threads(omp_before);
}

}  // namespace knowhere
