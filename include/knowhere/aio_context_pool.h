#pragma once

#include <libaio.h>

#include <condition_variable>
#include <mutex>
#include <queue>

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
        ctx_cv_.wait(lk, [this] { return ctx_q_.size(); });
        if (stop_) {
            return nullptr;
        }
        auto ret = ctx_q_.front();
        ctx_q_.pop();
        return ret;
    }

    static bool
    InitGlobalAioPool(size_t num_ctx, size_t max_events);

    static std::shared_ptr<AioContextPool>
    GetGlobalAioPool();

    ~AioContextPool() {
        stop_ = true;
        for (auto ctx : ctx_bak_) {
            io_destroy(ctx);
        }
        ctx_cv_.notify_all();
    }

 private:
    std::vector<io_context_t> ctx_bak_;
    std::queue<io_context_t> ctx_q_;
    std::mutex ctx_mtx_;
    std::condition_variable ctx_cv_;
    bool stop_ = false;
    size_t num_ctx_;
    size_t max_events_;
    static size_t global_aio_pool_size;
    static size_t global_aio_max_events;
    static std::mutex global_aio_pool_mut;

    AioContextPool(size_t num_ctx, size_t max_events) : num_ctx_(num_ctx), max_events_(max_events) {
        for (size_t i = 0; i < num_ctx_; ++i) {
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
            }
        }
    }
};
