#pragma once

#ifdef WITH_IO_URING

#include <liburing.h>

#include <algorithm>
#include <condition_variable>
#include <cstddef>
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
        {
            std::scoped_lock lk(ring_mtx_);
            if (owned_rings_.find(ring) == owned_rings_.end()) {
                LOG_WARN("UringContextPool rejects unknown ring: {}", static_cast<void*>(ring));
                return false;
            }

            if (checked_out_rings_.erase(ring) == 0) {
                LOG_WARN("UringContextPool rejects ring not checked out: {}", static_cast<void*>(ring));
                return false;
            }

            if (state_ != State::Healthy) {
                owned_rings_.erase(ring);
                auto iter = std::find(ring_bak_.begin(), ring_bak_.end(), ring);
                if (iter != ring_bak_.end()) {
                    ring_bak_.erase(iter);
                }
                should_destroy = true;
            } else {
                ring_q_.push(ring);
            }
        }

        if (should_destroy) {
            io_uring_queue_exit(ring);
            delete ring;
            return false;
        }

        ring_cv_.notify_one();
        return true;
    }

    struct io_uring*
    pop() {
        std::unique_lock lk(ring_mtx_);
        ring_cv_.wait(lk, [this] { return state_ != State::Healthy || !ring_q_.empty(); });
        if (state_ != State::Healthy) {
            return nullptr;
        }
        auto ret = ring_q_.front();
        ring_q_.pop();
        checked_out_rings_.insert(ret);
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
