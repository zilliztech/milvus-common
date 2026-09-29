#include "knowhere/aio_context_pool.h"

#include <algorithm>
#include <exception>

#include "log/Log.h"

size_t AioContextPool::global_aio_pool_size = 0;
size_t AioContextPool::global_aio_max_events = 0;
std::mutex AioContextPool::global_aio_pool_mut;

bool
AioContextPool::InitGlobalAioPool(size_t num_ctx, size_t max_events) {
    if (num_ctx <= 0) {
        LOG_ERROR("num_ctx should be bigger than 0");
        return false;
    }
    if (max_events > default_max_events) {
        LOG_ERROR("max_events %d should not be larger than %d", max_events, default_max_events);
        return false;
    }
    if (global_aio_pool_size == 0) {
        std::scoped_lock lk(global_aio_pool_mut);
        if (global_aio_pool_size == 0) {
            global_aio_pool_size = num_ctx;
            global_aio_max_events = max_events;
            return true;
        }
    }
    LOG_WARN("Global AioContextPool has already been inialized with context num: %d", global_aio_pool_size);
    return true;
}

std::shared_ptr<AioContextPool>
AioContextPool::GetGlobalAioPool() {
    if (global_aio_pool_size == 0) {
        std::scoped_lock lk(global_aio_pool_mut);
        if (global_aio_pool_size == 0) {
            global_aio_pool_size = default_pool_size;
            global_aio_max_events = default_max_events;
            LOG_WARN("Global AioContextPool has not been inialized yet, init it now with context num: %d",
                     global_aio_pool_size);
        }
    }
    static auto pool = std::shared_ptr<AioContextPool>(new AioContextPool(global_aio_pool_size, global_aio_max_events));
    return pool;
}

void
AioContextPool::DestroyAndRecreate(io_context_t& ctx) {
    size_t slot;
    {
        std::scoped_lock lock(ctx_mtx_);
        const auto it = std::find(ctx_bak_.begin(), ctx_bak_.end(), ctx);
        if (ctx == nullptr || it == ctx_bak_.end()) {
            LOG_ERROR("Cannot retire an unknown AIO context: {}", (void*)ctx);
            return;
        }
        slot = it - ctx_bak_.begin();
        // Reserve the registry entry before destruction. Kernel handle
        // addresses can be reused immediately by another concurrent setup.
        ctx_bak_[slot] = nullptr;
    }

    const auto retired = ctx;
    ctx = nullptr;
    const int destroy_ret = io_destroy(retired);
    io_context_t replacement = nullptr;
    const int setup_ret = io_setup(max_events_, &replacement);
    std::exception_ptr enqueue_error;
    bool exhausted;
    {
        std::scoped_lock lock(ctx_mtx_);
        if (setup_ret == 0) {
            try {
                ctx_q_.push(replacement);
                ctx_bak_[slot] = replacement;
            } catch (...) {
                enqueue_error = std::current_exception();
            }
        }
        if (setup_ret != 0 || enqueue_error) {
            --num_ctx_;
        }
        exhausted = num_ctx_ == 0;
    }
    if (exhausted) {
        ctx_cv_.notify_all();
    } else if (setup_ret == 0 && !enqueue_error) {
        ctx_cv_.notify_one();
    }
    if (enqueue_error) {
        const int cleanup_ret = io_destroy(replacement);
        if (cleanup_ret != 0) {
            LOG_ERROR("io_destroy failed for an unqueued AIO context {}, error: {}", (void*)replacement, -cleanup_ret);
        }
    }
    // Log after the pool state is consistent: formatting can also throw.
    if (destroy_ret != 0) {
        LOG_ERROR("io_destroy failed; abandoning AIO context {}, error: {}", (void*)retired, -destroy_ret);
    }
    if (setup_ret != 0) {
        LOG_ERROR("io_setup failed; AIO pool capacity reduced, error: {}", -setup_ret);
    }
    if (enqueue_error) {
        std::rethrow_exception(enqueue_error);
    }
}
