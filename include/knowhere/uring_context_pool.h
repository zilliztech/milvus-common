#pragma once

#ifdef WITH_IO_URING

#include <liburing.h>

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
    UringContextPool(const UringContextPool&) = delete;

    UringContextPool&
    operator=(const UringContextPool&) = delete;

    UringContextPool(UringContextPool&&) noexcept = delete;

    UringContextPool&
    operator==(UringContextPool&&) noexcept = delete;

    size_t
    max_entries_per_ctx() {
        return max_entries_;
    }

    void
    push(struct io_uring* ring) {
        if (ring == nullptr) {
            LOG_WARN("UringContextPool push gets null ring");
            return;
        }

        {
            std::scoped_lock lk(ring_mtx_);
            if (stop_) {
                LOG_WARN("UringContextPool is stopping, reject returned ring: %p", static_cast<void*>(ring));
                return;
            }

            if (owned_rings_.find(ring) == owned_rings_.end()) {
                LOG_WARN("UringContextPool rejects unknown ring: %p", static_cast<void*>(ring));
                return;
            }

            if (checked_out_rings_.erase(ring) == 0) {
                LOG_WARN("UringContextPool rejects ring not checked out: %p", static_cast<void*>(ring));
                return;
            }

            ring_q_.push(ring);
        }

        ring_cv_.notify_one();
    }

    struct io_uring*
    pop() {
        std::unique_lock lk(ring_mtx_);
        ring_cv_.wait(lk, [this] { return stop_ || !ring_q_.empty(); });
        if (stop_) {
            return nullptr;
        }
        auto ret = ring_q_.front();
        ring_q_.pop();
        checked_out_rings_.insert(ret);
        return ret;
    }

    static bool
    InitGlobalUringPool(size_t num_ctx, size_t max_entries);

    static std::shared_ptr<UringContextPool>
    GetGlobalUringPool();

    ~UringContextPool();

 private:
    std::vector<struct io_uring*> ring_bak_;
    std::queue<struct io_uring*> ring_q_;
    std::unordered_set<struct io_uring*> owned_rings_;
    std::unordered_set<struct io_uring*> checked_out_rings_;
    std::mutex ring_mtx_;
    std::condition_variable ring_cv_;
    bool stop_ = false;
    size_t num_ctx_;
    size_t max_entries_;

    static size_t global_uring_pool_size;
    static size_t global_uring_max_entries;
    static std::mutex global_uring_pool_mut;

    UringContextPool(size_t num_ctx, size_t max_entries);
};

#endif  // WITH_IO_URING
