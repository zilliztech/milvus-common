#pragma once

#include <libaio.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <vector>

#include "log/Log.h"

constexpr size_t default_max_nr = 65536;
constexpr size_t default_max_events = 128;
constexpr size_t default_pool_size = default_max_nr / default_max_events;

class AioContextPool {
 public:
    AioContextPool(const AioContextPool&) = delete;

    AioContextPool&
    operator=(const AioContextPool&) = delete;

    AioContextPool(AioContextPool&&) noexcept = delete;

    AioContextPool&
    operator==(AioContextPool&&) noexcept = delete;

    size_t
    max_events_per_ctx() {
        return max_events_;
    }

    void
    push(io_context_t ctx) {
        {
            std::scoped_lock lk(ctx_mtx_);
            ctx_q_.push(ctx);
        }
        ctx_cv_.notify_one();
    }

    io_context_t
    pop() {
        std::unique_lock lk(ctx_mtx_);
        if (stop_) {
            return nullptr;
        }
        ctx_cv_.wait(lk, [this] { return stop_ || !ctx_q_.empty() || num_ctx_ == 0; });
        if (stop_) {
            return nullptr;
        }
        if (num_ctx_ == 0) {
            throw std::runtime_error("No usable AIO contexts remain");
        }
        auto ret = ctx_q_.front();
        ctx_q_.pop();
        return ret;
    }

    // Retire an exclusively borrowed context and try to replace it once.
    // The handle is consumed even if destruction fails; only successful
    // destruction guarantees that pending I/O has stopped. Failed setup or
    // enqueue reduces usable capacity. C++ exceptions may propagate.
    void
    DestroyAndRecreate(io_context_t& ctx);

    static bool
    InitGlobalAioPool(size_t num_ctx, size_t max_events);

    static std::shared_ptr<AioContextPool>
    GetGlobalAioPool();

    ~AioContextPool() {
        stop_ = true;
        for (auto ctx : ctx_bak_) {
            if (ctx != nullptr) {
                io_destroy(ctx);
            }
        }
        ctx_cv_.notify_all();
    }

 private:
    std::vector<io_context_t> ctx_bak_;
    std::queue<io_context_t> ctx_q_;
    std::mutex ctx_mtx_;
    std::condition_variable ctx_cv_;
    bool stop_ = false;
    // Usable contexts, including borrowed contexts and replacements in progress.
    size_t num_ctx_;
    size_t max_events_;
    static size_t global_aio_pool_size;
    static size_t global_aio_max_events;
    static std::mutex global_aio_pool_mut;

    AioContextPool(size_t num_ctx, size_t max_events) : num_ctx_(0), max_events_(max_events) {
        for (size_t i = 0; i < num_ctx; ++i) {
            io_context_t ctx = 0;
            int ret = -1;
            for (int retry = 0; (ret = io_setup(max_events, &ctx)) != 0 && retry < 5; ++retry) {
                if (-ret != EAGAIN) {
                    LOG_ERROR("Unknown error occur in io_setup, errno: %d, %s", -ret, ::strerror(-ret));
                }
            }
            if (ret != 0) {
                LOG_ERROR("io_setup() failed; returned %d, errno=%d: %s", ret, -ret, ::strerror(-ret));
            } else {
                LOG_DEBUG("allocating ctx: %p", (void*)ctx);
                ctx_q_.push(ctx);
                ctx_bak_.push_back(ctx);
                ++num_ctx_;
            }
        }
    }
};
