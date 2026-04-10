#ifdef WITH_IO_URING

#include "knowhere/uring_context_pool.h"

#include <cerrno>
#include <cstring>
#include <thread>

size_t UringContextPool::global_uring_pool_size = 0;
size_t UringContextPool::global_uring_max_entries = 0;
std::mutex UringContextPool::global_uring_pool_mut;

UringContextPool::UringContextPool(size_t num_ctx, size_t max_entries) : num_ctx_(num_ctx), max_entries_(max_entries) {
    ring_bak_.reserve(num_ctx_);

    for (size_t i = 0; i < num_ctx_; ++i) {
        auto* ring = new io_uring();
        std::memset(ring, 0, sizeof(io_uring));
        int ret = io_uring_queue_init(static_cast<unsigned>(max_entries_), ring, 0);
        if (ret < 0) {
            LOG_ERROR("io_uring_queue_init failed with ret=%d, errno=%d: %s", ret, -ret, ::strerror(-ret));
            delete ring;
            continue;
        }

        ring_q_.push(ring);
        ring_bak_.push_back(ring);
        owned_rings_.insert(ring);
    }

    if (ring_bak_.empty()) {
        stop_ = true;
        LOG_ERROR("UringContextPool initialization failed: no valid io_uring context created");
    }
}

bool
UringContextPool::InitGlobalUringPool(size_t num_ctx, size_t max_entries) {
    if (num_ctx == 0) {
        LOG_ERROR("num_ctx should be bigger than 0");
        return false;
    }

    if (max_entries == 0 || max_entries > default_uring_max_entries) {
        LOG_ERROR("max_entries %zu should be in range (0, %zu]", max_entries, default_uring_max_entries);
        return false;
    }

    std::scoped_lock lk(global_uring_pool_mut);
    if (global_uring_pool_size == 0) {
        global_uring_pool_size = num_ctx;
        global_uring_max_entries = max_entries;
        return true;
    }

    if (global_uring_pool_size != num_ctx || global_uring_max_entries != max_entries) {
        LOG_WARN("Global UringContextPool already initialized with context num: %zu, max_entries: %zu (requested %zu, %zu)",
                 global_uring_pool_size,
                 global_uring_max_entries,
                 num_ctx,
                 max_entries);
    } else {
        LOG_WARN("Global UringContextPool has already been initialized with context num: %zu", global_uring_pool_size);
    }

    return true;
}

std::shared_ptr<UringContextPool>
UringContextPool::GetGlobalUringPool() {
    std::scoped_lock lk(global_uring_pool_mut);
    if (global_uring_pool_size == 0) {
        global_uring_pool_size = 1;
        global_uring_max_entries = default_uring_max_entries;
        LOG_WARN("Global UringContextPool has not been initialized yet, init it now with context num: %zu", global_uring_pool_size);
    }

    static std::shared_ptr<UringContextPool> pool =
        std::shared_ptr<UringContextPool>(new UringContextPool(global_uring_pool_size, global_uring_max_entries));
    return pool;
}

UringContextPool::~UringContextPool() {
    {
        std::scoped_lock lk(ring_mtx_);
        stop_ = true;
    }
    ring_cv_.notify_all();

    for (size_t retry = 0; retry < 100; ++retry) {
        {
            std::scoped_lock lk(ring_mtx_);
            if (checked_out_rings_.empty()) {
                break;
            }

            if (retry == 99) {
                LOG_WARN("UringContextPool shutdown with %zu checked-out rings still not returned", checked_out_rings_.size());
            }
        }

        std::this_thread::yield();
    }

    for (auto* ring : ring_bak_) {
        io_uring_queue_exit(ring);
        delete ring;
    }
    ring_bak_.clear();
    owned_rings_.clear();
    checked_out_rings_.clear();
}

#endif  // WITH_IO_URING
