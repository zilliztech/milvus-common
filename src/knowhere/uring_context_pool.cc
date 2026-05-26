#ifdef WITH_IO_URING

#include "knowhere/uring_context_pool.h"

#include <algorithm>
#include <cerrno>
#include <cstring>

#include "knowhere/io_context_pool.h"
#include "syncpoint/sync_point.h"

namespace {
std::shared_ptr<UringContextPool> g_uring_pool;
}

size_t UringContextPool::global_uring_pool_size = 0;
size_t UringContextPool::global_uring_max_entries = 0;
std::mutex UringContextPool::global_uring_pool_mut;

UringContextPool::UringContextPool(size_t num_ctx, size_t max_entries) : num_ctx_(num_ctx), max_entries_(max_entries) {
    ring_bak_.reserve(num_ctx_);

    for (size_t i = 0; i < num_ctx_; ++i) {
        auto* ring = new io_uring();
        std::memset(ring, 0, sizeof(io_uring));
        int ret = 0;
#ifdef ENABLE_SYNCPOINT
        TEST_SYNC_POINT_CALLBACK("UringContextPool::Ctor:BeforeInit", &ret);
#endif
        if (ret == 0) {
            ret = io_uring_queue_init(static_cast<unsigned>(max_entries_), ring, 0);
        }
        if (ret < 0) {
            LOG_ERROR("io_uring_queue_init failed with ret={}, errno={}: {}", ret, -ret, ::strerror(-ret));
            delete ring;
            continue;
        }

        ring_q_.push(ring);
        ring_bak_.push_back(ring);
        owned_rings_.insert(ring);
    }

    if (ring_bak_.size() != num_ctx_) {
        state_ = State::Unusable;
        LOG_ERROR("UringContextPool initialization failed: created {} of {} requested contexts", ring_bak_.size(),
                  num_ctx_);
    }
}

bool
UringContextPool::InitGlobalUringPoolWithValidation(size_t num_ctx, size_t max_entries) {
    if (num_ctx == 0) {
        LOG_ERROR("num_ctx should be bigger than 0");
        return false;
    }

    if (max_entries == 0 || max_entries > default_uring_max_entries) {
        LOG_ERROR("max_entries {} should be in range (0, {}]", max_entries, default_uring_max_entries);
        return false;
    }

    std::scoped_lock lk(global_uring_pool_mut);
    if (global_uring_pool_size == 0) {
        global_uring_pool_size = num_ctx;
        global_uring_max_entries = max_entries;
        return true;
    }

    if (global_uring_pool_size != num_ctx || global_uring_max_entries != max_entries) {
        LOG_ERROR(
            "Global UringContextPool already initialized with context num: {}, max_entries: {} (requested {}, {})",
            global_uring_pool_size, global_uring_max_entries, num_ctx, max_entries);
        return false;
    }

    LOG_WARN("Global UringContextPool has already been initialized with context num: {}", global_uring_pool_size);
    return true;
}

bool
UringContextPool::ResetCheckedOut(struct io_uring* ring) {
    if (ring == nullptr) {
        LOG_WARN("UringContextPool reset gets null ring");
        return false;
    }

    {
        std::scoped_lock lk(ring_mtx_);
        if (owned_rings_.find(ring) == owned_rings_.end()) {
            LOG_WARN("UringContextPool rejects reset for unknown ring: {}", static_cast<void*>(ring));
            return false;
        }
        if (checked_out_rings_.find(ring) == checked_out_rings_.end()) {
            LOG_WARN("UringContextPool rejects reset for ring not checked out: {}", static_cast<void*>(ring));
            return false;
        }
    }

    io_uring_queue_exit(ring);
    std::memset(ring, 0, sizeof(io_uring));
    int ret = 0;
#ifdef ENABLE_SYNCPOINT
    TEST_SYNC_POINT_CALLBACK("UringContextPool::ResetCheckedOut:BeforeInit", &ret);
#endif
    if (ret == 0) {
        ret = io_uring_queue_init(static_cast<unsigned>(max_entries_), ring, 0);
    }
    if (ret == 0) {
        bool released = false;
        bool should_destroy = false;
        {
            std::scoped_lock lk(ring_mtx_);
            if (state_ == State::Healthy) {
                try {
                    ring_q_.push(ring);
                    checked_out_rings_.erase(ring);
                    released = true;
                } catch (const std::exception& e) {
                    LOG_ERROR("UringContextPool failed to requeue reset ring {}: {}", static_cast<void*>(ring),
                              e.what());
                    RemoveTrackedRingLocked(ring);
                    MarkUnusableLocked();
                    should_destroy = true;
                } catch (...) {
                    LOG_ERROR("UringContextPool failed to requeue reset ring {}: unknown exception",
                              static_cast<void*>(ring));
                    RemoveTrackedRingLocked(ring);
                    MarkUnusableLocked();
                    should_destroy = true;
                }
            } else {
                RemoveTrackedRingLocked(ring);
                should_destroy = true;
            }
        }
        if (released) {
            ring_cv_.notify_one();
            return true;
        }
        if (should_destroy) {
            DestroyRing(ring);
        }
        ring_cv_.notify_all();
        return false;
    }

    LOG_ERROR("io_uring_queue_init failed while resetting ring with ret={}, errno={}: {}", ret, -ret, ::strerror(-ret));
    {
        std::scoped_lock lk(ring_mtx_);
        RemoveTrackedRingLocked(ring);
        MarkUnusableLocked();
    }
    ring_cv_.notify_all();
    delete ring;
    return false;
}

bool
UringContextPool::RetireCheckedOut(struct io_uring* ring) {
    if (ring == nullptr) {
        LOG_WARN("UringContextPool retire gets null ring");
        return false;
    }

    {
        std::scoped_lock lk(ring_mtx_);
        if (owned_rings_.find(ring) == owned_rings_.end()) {
            LOG_WARN("UringContextPool rejects retire for unknown ring: {}", static_cast<void*>(ring));
            return false;
        }
        if (checked_out_rings_.find(ring) == checked_out_rings_.end()) {
            LOG_WARN("UringContextPool rejects retire for ring not checked out: {}", static_cast<void*>(ring));
            return false;
        }

        RemoveTrackedRingLocked(ring);
        MarkUnusableLocked();
    }

    DestroyRing(ring);
    ring_cv_.notify_all();
    return true;
}

void
UringContextPool::Shutdown() {
    {
        std::scoped_lock lk(ring_mtx_);
        if (state_ == State::Stopped) {
            return;
        }
        state_ = State::Stopped;
    }
    ring_cv_.notify_all();
}

std::shared_ptr<UringContextPool>
UringContextPool::GetGlobalUringPoolDirect() {
    std::scoped_lock lk(global_uring_pool_mut);
    if (global_uring_pool_size == 0) {
        IOContextPoolConfig cfg;
        global_uring_pool_size = cfg.num_ctx;
        global_uring_max_entries = cfg.max_events;
        LOG_WARN("Global UringContextPool has not been initialized yet, init it now with context num: {}",
                 global_uring_pool_size);
    }

    if (g_uring_pool == nullptr) {
        g_uring_pool =
            std::shared_ptr<UringContextPool>(new UringContextPool(global_uring_pool_size, global_uring_max_entries));
    }
    return g_uring_pool;
}

bool
UringContextPool::InitGlobalUringPool(size_t num_ctx, size_t max_entries) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = num_ctx;
    cfg.max_events = max_entries;

    if (!IOContextPool::InitGlobal(cfg)) {
        return false;
    }

    auto io_pool = IOContextPool::GetGlobal();
    if (io_pool == nullptr || !io_pool->IsInitialized()) {
        return false;
    }
    if (io_pool->Backend() != IOBackend::IO_URING) {
        LOG_ERROR("Global IOContextPool backend is {}, legacy io_uring API is unavailable", io_pool->BackendName());
        return false;
    }
    return true;
}

std::shared_ptr<UringContextPool>
UringContextPool::GetGlobalUringPool() {
    auto io_pool = IOContextPool::GetGlobal();
    if (io_pool == nullptr || !io_pool->IsInitialized()) {
        return nullptr;
    }
    if (io_pool->Backend() != IOBackend::IO_URING) {
        return nullptr;
    }
    return io_pool->GetUringPoolForLegacy();
}

void
UringContextPool::ResetGlobalForTest() {
    std::scoped_lock lk(global_uring_pool_mut);
    g_uring_pool.reset();
    global_uring_pool_size = 0;
    global_uring_max_entries = 0;
}

UringContextPool::~UringContextPool() {
    Shutdown();

    std::unordered_set<struct io_uring*> checked_out;
    {
        std::scoped_lock lk(ring_mtx_);
        checked_out = checked_out_rings_;
    }
    if (!checked_out.empty()) {
        LOG_WARN("UringContextPool shutdown with {} checked-out rings still not returned", checked_out.size());
    }

    for (auto* ring : ring_bak_) {
        if (checked_out.find(ring) == checked_out.end()) {
            DestroyRing(ring);
        }
    }
    ring_bak_.clear();
    owned_rings_.clear();
    checked_out_rings_.clear();
}

#endif  // WITH_IO_URING
