#pragma once

#include <libaio.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_set>
#include <vector>

#include "log/Log.h"
#include "syncpoint/sync_point.h"

constexpr size_t default_max_nr = 65536;
constexpr size_t default_max_events = 128;
constexpr size_t default_pool_size = default_max_nr / default_max_events;

class AioContextPool {
 public:
    enum class State {
        Healthy,
        Unusable,
        Stopped,
    };

    AioContextPool(const AioContextPool&) = delete;

    AioContextPool&
    operator=(const AioContextPool&) = delete;

    AioContextPool(AioContextPool&&) noexcept = delete;

    AioContextPool&
    operator=(AioContextPool&&) noexcept = delete;

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
        return state_ == State::Healthy && ctx_bak_.size() == num_ctx_ && !ctx_bak_.empty();
    }

    bool
    push(io_context_t ctx) {
        if (ctx == nullptr) {
            LOG_WARN("AioContextPool push gets null context");
            return false;
        }

        bool should_destroy = false;
        bool released = false;
        bool notify_all = false;
        {
            std::scoped_lock lk(ctx_mtx_);
            if (owned_ctxs_.find(ctx) == owned_ctxs_.end()) {
                LOG_WARN("AioContextPool rejects unknown context: {}", static_cast<void*>(ctx));
                return false;
            }

            if (checked_out_ctxs_.find(ctx) == checked_out_ctxs_.end()) {
                LOG_WARN("AioContextPool rejects context not checked out: {}", static_cast<void*>(ctx));
                return false;
            }
            if (state_ != State::Healthy) {
                RemoveTrackedContextLocked(ctx);
                should_destroy = true;
                notify_all = true;
            } else {
                try {
                    ctx_q_.push(ctx);
                    checked_out_ctxs_.erase(ctx);
                    released = true;
                } catch (const std::exception& e) {
                    LOG_ERROR("AioContextPool failed to requeue context {}: {}", static_cast<void*>(ctx), e.what());
                    RemoveTrackedContextLocked(ctx);
                    MarkUnusableLocked();
                    should_destroy = true;
                    notify_all = true;
                } catch (...) {
                    LOG_ERROR("AioContextPool failed to requeue context {}: unknown exception",
                              static_cast<void*>(ctx));
                    RemoveTrackedContextLocked(ctx);
                    MarkUnusableLocked();
                    should_destroy = true;
                    notify_all = true;
                }
            }
        }

        if (should_destroy) {
            DestroyContextNoThrow(ctx, "releasing");
            if (notify_all) {
                ctx_cv_.notify_all();
            }
            return false;
        }

        ctx_cv_.notify_one();
        return released;
    }

    io_context_t
    pop() {
        std::unique_lock lk(ctx_mtx_);
        ctx_cv_.wait(lk, [this] { return state_ != State::Healthy || !ctx_q_.empty(); });
        if (state_ != State::Healthy) {
            return nullptr;
        }
        auto ret = ctx_q_.front();
        try {
            const auto inserted = checked_out_ctxs_.insert(ret).second;
            if (!inserted) {
                LOG_ERROR("AioContextPool detected duplicate checked-out context: {}", static_cast<void*>(ret));
                MarkUnusableLocked();
                lk.unlock();
                ctx_cv_.notify_all();
                return nullptr;
            }
        } catch (const std::exception& e) {
            LOG_ERROR("AioContextPool failed to mark context checked out: {}", e.what());
            MarkUnusableLocked();
            lk.unlock();
            ctx_cv_.notify_all();
            return nullptr;
        } catch (...) {
            LOG_ERROR("AioContextPool failed to mark context checked out: unknown exception");
            MarkUnusableLocked();
            lk.unlock();
            ctx_cv_.notify_all();
            return nullptr;
        }
        ctx_q_.pop();
        return ret;
    }

    void
    Shutdown() {
        {
            std::scoped_lock lk(ctx_mtx_);
            if (state_ == State::Stopped) {
                return;
            }
            state_ = State::Stopped;
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

        if (!DestroyContextNoThrow(ctx, "resetting")) {
            {
                std::scoped_lock lk(ctx_mtx_);
                RemoveTrackedContextLocked(ctx);
                MarkUnusableLocked();
            }
            ctx_cv_.notify_all();
            return false;
        }

        io_context_t new_ctx = 0;
        int ret = 0;
#ifdef ENABLE_SYNCPOINT
        TEST_SYNC_POINT_CALLBACK("AioContextPool::ResetCheckedOut:BeforeSetup", &ret);
#endif
        if (ret == 0) {
            ret = io_setup(max_events_, &new_ctx);
        }
        if (ret == 0) {
            bool reusable = false;
            bool should_destroy_new_ctx = false;
            {
                std::scoped_lock lk(ctx_mtx_);
                auto iter = std::find(ctx_bak_.begin(), ctx_bak_.end(), ctx);
                if (state_ == State::Healthy && iter != ctx_bak_.end()) {
                    owned_ctxs_.erase(ctx);
                    checked_out_ctxs_.erase(ctx);
                    try {
                        const auto inserted = owned_ctxs_.insert(new_ctx).second;
                        if (!inserted) {
                            LOG_ERROR("AioContextPool replacement context already exists: {}",
                                      static_cast<void*>(new_ctx));
                            RemoveTrackedContextLocked(ctx);
                            MarkUnusableLocked();
                            should_destroy_new_ctx = true;
                        } else {
                            try {
                                ctx_q_.push(new_ctx);
                            } catch (...) {
                                owned_ctxs_.erase(new_ctx);
                                throw;
                            }
                            *iter = new_ctx;
                            reusable = true;
                        }
                    } catch (const std::exception& e) {
                        LOG_ERROR("AioContextPool failed to install replacement context: {}", e.what());
                        RemoveTrackedContextLocked(ctx);
                        MarkUnusableLocked();
                        should_destroy_new_ctx = true;
                    } catch (...) {
                        LOG_ERROR("AioContextPool failed to install replacement context: unknown exception");
                        RemoveTrackedContextLocked(ctx);
                        MarkUnusableLocked();
                        should_destroy_new_ctx = true;
                    }
                } else {
                    RemoveTrackedContextLocked(ctx);
                    should_destroy_new_ctx = true;
                }
            }
            if (reusable) {
                ctx_cv_.notify_one();
                return true;
            }
            if (should_destroy_new_ctx) {
                DestroyContextNoThrow(new_ctx, "discarding replacement");
            }
            ctx_cv_.notify_all();
            return false;
        }

        LOG_ERROR("io_setup failed while resetting AIO context with ret={}, errno={}: {}", ret, -ret, ::strerror(-ret));
        {
            std::scoped_lock lk(ctx_mtx_);
            RemoveTrackedContextLocked(ctx);
            MarkUnusableLocked();
        }
        ctx_cv_.notify_all();
        return false;
    }

    bool
    RetireCheckedOut(io_context_t ctx) {
        if (ctx == nullptr) {
            LOG_WARN("AioContextPool retire gets null context");
            return false;
        }

        {
            std::scoped_lock lk(ctx_mtx_);
            if (owned_ctxs_.find(ctx) == owned_ctxs_.end()) {
                LOG_WARN("AioContextPool rejects retire for unknown context: {}", static_cast<void*>(ctx));
                return false;
            }
            if (checked_out_ctxs_.find(ctx) == checked_out_ctxs_.end()) {
                LOG_WARN("AioContextPool rejects retire for context not checked out: {}", static_cast<void*>(ctx));
                return false;
            }

            RemoveTrackedContextLocked(ctx);
            MarkUnusableLocked();
        }

        const bool destroyed = DestroyContextNoThrow(ctx, "retiring");
        ctx_cv_.notify_all();
        return destroyed;
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
                DestroyContextNoThrow(ctx, "destroying during shutdown");
            }
        }
    }

 private:
    void
    MarkUnusableLocked() noexcept {
        if (state_ == State::Healthy) {
            state_ = State::Unusable;
        }
    }

    void
    RemoveTrackedContextLocked(io_context_t ctx) {
        checked_out_ctxs_.erase(ctx);
        owned_ctxs_.erase(ctx);
        auto iter = std::find(ctx_bak_.begin(), ctx_bak_.end(), ctx);
        if (iter != ctx_bak_.end()) {
            ctx_bak_.erase(iter);
        }
    }

    bool
    DestroyContextNoThrow(io_context_t ctx, const char* action) noexcept {
        int ret = io_destroy(ctx);
#ifdef ENABLE_SYNCPOINT
        TEST_SYNC_POINT_CALLBACK("AioContextPool::DestroyContext:AfterDestroy", &ret);
#endif
        if (ret != 0) {
            const int err = ret < 0 ? -ret : ret;
            LOG_ERROR("io_destroy failed while {} AIO context {} with ret={}, errno={}: {}", action,
                      static_cast<void*>(ctx), ret, err, ::strerror(err));
            return false;
        }
        return true;
    }

    std::vector<io_context_t> ctx_bak_;
    std::queue<io_context_t> ctx_q_;
    std::unordered_set<io_context_t> owned_ctxs_;
    std::unordered_set<io_context_t> checked_out_ctxs_;
    mutable std::mutex ctx_mtx_;
    std::condition_variable ctx_cv_;
    State state_ = State::Healthy;
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
            state_ = State::Unusable;
            LOG_ERROR("AioContextPool initialization failed: created {} of {} requested contexts", ctx_bak_.size(),
                      num_ctx_);
        }
    }
};
