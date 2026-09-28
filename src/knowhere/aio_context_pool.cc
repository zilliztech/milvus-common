#include "knowhere/aio_context_pool.h"

#include <algorithm>
#include <system_error>

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

AioContextPool::AioContextPool(size_t num_ctx, size_t max_events)
    : ctx_bak_(num_ctx, nullptr), num_ctx_(num_ctx), max_events_(max_events) {
    ctx_q_.reserve(num_ctx);
    pending_recreation_.reserve(num_ctx);
    for (size_t i = 0; i < num_ctx; ++i) {
        io_context_t ctx = nullptr;
        int ret = -1;
        for (int retry = 0; (ret = io_setup(max_events, &ctx)) != 0 && retry < 5; ++retry) {
            if (ret != -EAGAIN) {
                LOG_ERROR("io_setup failed: {}", -ret);
            }
        }
        if (ret != 0) {
            LOG_ERROR("io_setup failed; retaining AIO slot for recreation: {}", -ret);
            pending_recreation_.push_back(i);
        } else {
            ctx_bak_[i] = ctx;
            ctx_q_.push_back(ctx);
        }
    }
}

void
AioContextPool::push(io_context_t ctx) {
    {
        std::scoped_lock lock(ctx_mtx_);
        ctx_q_.push_back(ctx);
    }
    ctx_cv_.notify_one();
}

io_context_t
AioContextPool::pop() {
    std::unique_lock lock(ctx_mtx_);
    ctx_cv_.wait(lock, [this] { return stop_ || !ctx_q_.empty() || !pending_recreation_.empty(); });
    if (stop_) {
        return nullptr;
    }
    if (!ctx_q_.empty()) {
        auto ctx = ctx_q_.back();
        ctx_q_.pop_back();
        return ctx;
    }

    const auto slot = pending_recreation_.back();
    pending_recreation_.pop_back();
    lock.unlock();
    io_context_t ctx = nullptr;
    const int ret = io_setup(max_events_, &ctx);
    lock.lock();
    if (ret != 0) {
        pending_recreation_.push_back(slot);
        lock.unlock();
        ctx_cv_.notify_one();
        throw std::system_error(-ret, std::generic_category(), "Could not recreate AIO context");
    }
    ctx_bak_[slot] = ctx;
    return ctx;
}

int
AioContextPool::DestroyAndRecreate(io_context_t& ctx) noexcept {
    size_t slot;
    {
        std::scoped_lock lock(ctx_mtx_);
        const auto it = std::find(ctx_bak_.begin(), ctx_bak_.end(), ctx);
        if (ctx == nullptr || it == ctx_bak_.end() || std::find(ctx_q_.begin(), ctx_q_.end(), ctx) != ctx_q_.end()) {
            return -EINVAL;
        }
        slot = it - ctx_bak_.begin();
        // Reserve the registry entry before destruction. Kernel handle
        // addresses can be reused immediately by another concurrent setup.
        ctx_bak_[slot] = nullptr;
    }

    const int destroy_ret = io_destroy(ctx);
    if (destroy_ret != 0) {
        std::scoped_lock lock(ctx_mtx_);
        ctx_bak_[slot] = ctx;
        return destroy_ret;
    }
    ctx = nullptr;

    io_context_t replacement = nullptr;
    const int setup_ret = io_setup(max_events_, &replacement);
    {
        std::scoped_lock lock(ctx_mtx_);
        if (setup_ret == 0) {
            ctx_bak_[slot] = replacement;
            ctx_q_.push_back(replacement);
        } else {
            pending_recreation_.push_back(slot);
        }
    }
    ctx_cv_.notify_one();
    return setup_ret;
}

AioContextPool::~AioContextPool() {
    {
        std::scoped_lock lock(ctx_mtx_);
        stop_ = true;
    }
    ctx_cv_.notify_all();
    for (auto ctx : ctx_bak_) {
        if (ctx != nullptr) {
            io_destroy(ctx);
        }
    }
}
