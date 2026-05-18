#pragma once

#include <libaio.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_set>
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

    size_t
    created_context_count() const {
        std::scoped_lock lk(ctx_mtx_);
        return ctx_bak_.size();
    }

    bool
    IsUsable() const {
        std::scoped_lock lk(ctx_mtx_);
        return !stop_ && ctx_bak_.size() == num_ctx_ && !ctx_bak_.empty();
    }

    void
    push(io_context_t ctx) {
        if (ctx == nullptr) {
            LOG_WARN("AioContextPool push gets null context");
            return;
        }

        {
            std::scoped_lock lk(ctx_mtx_);
            if (stop_) {
                checked_out_ctxs_.erase(ctx);
                return;
            }

            if (owned_ctxs_.find(ctx) == owned_ctxs_.end()) {
                LOG_WARN("AioContextPool rejects unknown context: {}", static_cast<void*>(ctx));
                return;
            }

            if (checked_out_ctxs_.erase(ctx) == 0) {
                LOG_WARN("AioContextPool rejects context not checked out: {}", static_cast<void*>(ctx));
                return;
            }
            ctx_q_.push(ctx);
        }
        ctx_cv_.notify_one();
    }

    io_context_t
    pop() {
        std::unique_lock lk(ctx_mtx_);
        ctx_cv_.wait(lk, [this] { return stop_ || !ctx_q_.empty(); });
        if (stop_) {
            return nullptr;
        }
        auto ret = ctx_q_.front();
        ctx_q_.pop();
        checked_out_ctxs_.insert(ret);
        return ret;
    }

    void
    Shutdown() {
        {
            std::scoped_lock lk(ctx_mtx_);
            if (stop_) {
                return;
            }
            stop_ = true;
        }
        ctx_cv_.notify_all();
    }

    static bool
    InitGlobalAioPool(size_t num_ctx, size_t max_events);

    static std::shared_ptr<AioContextPool>
    GetGlobalAioPool();

    static bool
    InitGlobalAioPoolWithValidation(size_t num_ctx, size_t max_events);

    static std::shared_ptr<AioContextPool>
    GetGlobalAioPoolDirect();

    static void
    ResetGlobalForTest();

    bool
    ResetCheckedOut(io_context_t ctx) {
        if (ctx == nullptr) {
            LOG_WARN("AioContextPool reset gets null context");
            return false;
        }

        {
            std::scoped_lock lk(ctx_mtx_);
            if (owned_ctxs_.find(ctx) == owned_ctxs_.end()) {
                LOG_WARN("AioContextPool rejects reset for unknown context: {}", static_cast<void*>(ctx));
                return false;
            }
            if (checked_out_ctxs_.find(ctx) == checked_out_ctxs_.end()) {
                LOG_WARN("AioContextPool rejects reset for context not checked out: {}", static_cast<void*>(ctx));
                return false;
            }
        }

        io_destroy(ctx);
        io_context_t new_ctx = 0;
        const auto ret = io_setup(max_events_, &new_ctx);
        if (ret == 0) {
            std::scoped_lock lk(ctx_mtx_);
            auto iter = std::find(ctx_bak_.begin(), ctx_bak_.end(), ctx);
            if (iter != ctx_bak_.end()) {
                *iter = new_ctx;
            }
            owned_ctxs_.erase(ctx);
            owned_ctxs_.insert(new_ctx);
            checked_out_ctxs_.erase(ctx);
            ctx_q_.push(new_ctx);
            ctx_cv_.notify_one();
            return true;
        }

        LOG_ERROR("io_setup failed while resetting AIO context with ret={}, errno={}: {}", ret, -ret, ::strerror(-ret));
        {
            std::scoped_lock lk(ctx_mtx_);
            checked_out_ctxs_.erase(ctx);
            owned_ctxs_.erase(ctx);
            auto iter = std::find(ctx_bak_.begin(), ctx_bak_.end(), ctx);
            if (iter != ctx_bak_.end()) {
                ctx_bak_.erase(iter);
            }
        }
        return false;
    }

    ~AioContextPool() {
        Shutdown();
        std::unordered_set<io_context_t> checked_out;
        {
            std::scoped_lock lk(ctx_mtx_);
            checked_out = checked_out_ctxs_;
        }
        if (!checked_out.empty()) {
            LOG_WARN("AioContextPool shutdown with {} checked-out contexts still not returned", checked_out.size());
        }
        for (auto ctx : ctx_bak_) {
            if (checked_out.find(ctx) == checked_out.end()) {
                io_destroy(ctx);
            }
        }
    }

 private:
    std::vector<io_context_t> ctx_bak_;
    std::queue<io_context_t> ctx_q_;
    std::unordered_set<io_context_t> owned_ctxs_;
    std::unordered_set<io_context_t> checked_out_ctxs_;
    mutable std::mutex ctx_mtx_;
    std::condition_variable ctx_cv_;
    bool stop_ = false;
    size_t num_ctx_;
    size_t max_events_;
    static std::atomic<size_t> global_aio_pool_size;
    static std::atomic<size_t> global_aio_max_events;
    static std::mutex global_aio_pool_mut;

    AioContextPool(size_t num_ctx, size_t max_events) : num_ctx_(num_ctx), max_events_(max_events) {
        for (size_t i = 0; i < num_ctx_; ++i) {
            io_context_t ctx = 0;
            int ret = -1;
            for (int retry = 0; (ret = io_setup(max_events, &ctx)) != 0 && retry < 5; ++retry) {
                if (-ret != EAGAIN) {
                    LOG_ERROR("Unknown error occur in io_setup, errno: {}, {}", -ret, ::strerror(-ret));
                }
            }
            if (ret != 0) {
                LOG_ERROR("io_setup() failed; returned {}, errno={}: {}", ret, -ret, ::strerror(-ret));
            } else {
                LOG_DEBUG("allocating ctx: {}", static_cast<void*>(ctx));
                ctx_q_.push(ctx);
                ctx_bak_.push_back(ctx);
                owned_ctxs_.insert(ctx);
            }
        }
        if (ctx_bak_.size() != num_ctx_) {
            stop_ = true;
            LOG_ERROR("AioContextPool initialization failed: created {} of {} requested contexts", ctx_bak_.size(),
                      num_ctx_);
        }
    }
};
