#pragma once

#ifdef WITH_IO_URING

#include <liburing.h>

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_set>
#include <vector>

#include "log/Log.h"

constexpr size_t default_uring_max_entries = 128;

class UringContextPool {
 public:
    enum class State {
        Healthy,
        Unusable,
        Stopped,
    };

    UringContextPool(const UringContextPool&) = delete;

    UringContextPool&
    operator=(const UringContextPool&) = delete;

    UringContextPool(UringContextPool&&) noexcept = delete;

    UringContextPool&
    operator=(UringContextPool&&) noexcept = delete;

    size_t
    max_entries_per_ctx() {
        return max_entries_;
    }

    bool
    IsUsable() const {
        std::scoped_lock lk(ring_mtx_);
        return state_ == State::Healthy && ring_bak_.size() == num_ctx_ && !ring_bak_.empty();
    }

    bool
    push(struct io_uring* ring) {
        if (ring == nullptr) {
            LOG_WARN("UringContextPool push gets null ring");
            return false;
        }

        bool should_destroy = false;
        bool released = false;
        bool notify_all = false;
        {
            std::scoped_lock lk(ring_mtx_);
            if (owned_rings_.find(ring) == owned_rings_.end()) {
                LOG_WARN("UringContextPool rejects unknown ring: {}", static_cast<void*>(ring));
                return false;
            }

            if (checked_out_rings_.find(ring) == checked_out_rings_.end()) {
                LOG_WARN("UringContextPool rejects ring not checked out: {}", static_cast<void*>(ring));
                return false;
            }

            if (state_ != State::Healthy) {
                RemoveTrackedRingLocked(ring);
                should_destroy = true;
                notify_all = true;
            } else {
                try {
                    ring_q_.push(ring);
                    checked_out_rings_.erase(ring);
                    released = true;
                } catch (const std::exception& e) {
                    LOG_ERROR("UringContextPool failed to requeue ring {}: {}", static_cast<void*>(ring), e.what());
                    RemoveTrackedRingLocked(ring);
                    MarkUnusableLocked();
                    should_destroy = true;
                    notify_all = true;
                } catch (...) {
                    LOG_ERROR("UringContextPool failed to requeue ring {}: unknown exception",
                              static_cast<void*>(ring));
                    RemoveTrackedRingLocked(ring);
                    MarkUnusableLocked();
                    should_destroy = true;
                    notify_all = true;
                }
            }
        }

        if (should_destroy) {
            DestroyRing(ring);
            if (notify_all) {
                ring_cv_.notify_all();
            }
            return false;
        }

        ring_cv_.notify_one();
        return released;
    }

    struct io_uring*
    pop() {
        std::unique_lock lk(ring_mtx_);
        ring_cv_.wait(lk, [this] { return state_ != State::Healthy || !ring_q_.empty(); });
        if (state_ != State::Healthy) {
            return nullptr;
        }
        auto ret = ring_q_.front();
        try {
            const auto inserted = checked_out_rings_.insert(ret).second;
            if (!inserted) {
                LOG_ERROR("UringContextPool detected duplicate checked-out ring: {}", static_cast<void*>(ret));
                MarkUnusableLocked();
                lk.unlock();
                ring_cv_.notify_all();
                return nullptr;
            }
        } catch (const std::exception& e) {
            LOG_ERROR("UringContextPool failed to mark ring checked out: {}", e.what());
            MarkUnusableLocked();
            lk.unlock();
            ring_cv_.notify_all();
            return nullptr;
        } catch (...) {
            LOG_ERROR("UringContextPool failed to mark ring checked out: unknown exception");
            MarkUnusableLocked();
            lk.unlock();
            ring_cv_.notify_all();
            return nullptr;
        }
        ring_q_.pop();
        return ret;
    }

    bool
    ResetCheckedOut(struct io_uring* ring);

    bool
    RetireCheckedOut(struct io_uring* ring);

    static bool
    InitGlobalUringPool(size_t num_ctx, size_t max_entries);

    static std::shared_ptr<UringContextPool>
    GetGlobalUringPool();

    static bool
    InitGlobalUringPoolWithValidation(size_t num_ctx, size_t max_entries);

    static std::shared_ptr<UringContextPool>
    GetGlobalUringPoolDirect();

    static void
    ResetGlobalForTest();

    ~UringContextPool();

 private:
    void
    MarkUnusableLocked() noexcept {
        if (state_ == State::Healthy) {
            state_ = State::Unusable;
        }
    }

    void
    RemoveTrackedRingLocked(struct io_uring* ring) {
        checked_out_rings_.erase(ring);
        owned_rings_.erase(ring);
        auto iter = std::find(ring_bak_.begin(), ring_bak_.end(), ring);
        if (iter != ring_bak_.end()) {
            ring_bak_.erase(iter);
        }
    }

    static void
    DestroyRing(struct io_uring* ring) noexcept {
        io_uring_queue_exit(ring);
        delete ring;
    }

    std::vector<struct io_uring*> ring_bak_;
    std::queue<struct io_uring*> ring_q_;
    std::unordered_set<struct io_uring*> owned_rings_;
    std::unordered_set<struct io_uring*> checked_out_rings_;
    mutable std::mutex ring_mtx_;
    std::condition_variable ring_cv_;
    State state_ = State::Healthy;
    size_t num_ctx_;
    size_t max_entries_;

    static size_t global_uring_pool_size;
    static size_t global_uring_max_entries;
    static std::mutex global_uring_pool_mut;

    UringContextPool(size_t num_ctx, size_t max_entries);
};

#endif  // WITH_IO_URING
