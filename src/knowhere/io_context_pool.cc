#include "knowhere/io_context_pool.h"

#include <mutex>

#include "log/Log.h"

namespace {
std::shared_ptr<IOContextPool> g_io_pool;
std::mutex g_io_pool_mutex;
}  // namespace

#ifdef WITH_IO_URING
bool
IOContextPool::TryInitUring(const IOContextPoolConfig& cfg, const std::shared_ptr<IOContextPool>& io_pool) {
    if (!UringContextPool::InitGlobalUringPoolWithValidation(cfg.num_ctx, cfg.max_events)) {
        return false;
    }

    auto pool = UringContextPool::GetGlobalUringPoolDirect();
    if (pool == nullptr || !pool->IsUsable()) {
        LOG_ERROR("Global UringContextPool is unavailable after initialization");
        return false;
    }

    io_pool->uring_pool_ = pool;
    io_pool->backend_ = IOBackend::IO_URING;
    io_pool->num_ctx_ = cfg.num_ctx;
    io_pool->max_events_per_ctx_ = pool->max_entries_per_ctx();
    return true;
}
#endif

#ifdef MILVUS_COMMON_WITH_LIBAIO
bool
IOContextPool::TryInitAio(const IOContextPoolConfig& cfg, const std::shared_ptr<IOContextPool>& io_pool) {
    if (!AioContextPool::InitGlobalAioPoolWithValidation(cfg.num_ctx, cfg.max_events)) {
        return false;
    }

    auto pool = AioContextPool::GetGlobalAioPoolDirect();
    if (pool == nullptr) {
        return false;
    }

    io_pool->aio_pool_ = pool;
    io_pool->backend_ = IOBackend::AIO;
    io_pool->num_ctx_ = cfg.num_ctx;
    io_pool->max_events_per_ctx_ = pool->max_events_per_ctx();
    return true;
}
#endif

bool
IOContextPool::InitGlobal(const IOContextPoolConfig& cfg) {
    if (cfg.num_ctx == 0) {
        LOG_ERROR("num_ctx should be bigger than 0");
        return false;
    }

    if (cfg.max_events == 0) {
        LOG_ERROR("max_events should be bigger than 0");
        return false;
    }

    std::scoped_lock lk(g_io_pool_mutex);
    if (g_io_pool != nullptr && g_io_pool->IsInitialized()) {
        if (cfg.max_events != g_io_pool->MaxEventsPerCtx()) {
            LOG_ERROR("Global IOContextPool already initialized with max_events=%zu, requested=%zu",
                      g_io_pool->MaxEventsPerCtx(), cfg.max_events);
            return false;
        }
        if (cfg.num_ctx != g_io_pool->num_ctx_) {
            LOG_ERROR("Global IOContextPool already initialized with num_ctx=%zu, requested=%zu", g_io_pool->num_ctx_,
                      cfg.num_ctx);
            return false;
        }
        LOG_WARN("Global IOContextPool has already been initialized with backend: %s",
                 g_io_pool->BackendName().c_str());
        return true;
    }

    auto io_pool = std::shared_ptr<IOContextPool>(new IOContextPool());

#ifdef WITH_IO_URING
    if (TryInitUring(cfg, io_pool)) {
        g_io_pool = io_pool;
        LOG_INFO("Global IOContextPool initialized with backend io_uring");
        return true;
    }
#ifdef MILVUS_COMMON_WITH_LIBAIO
    LOG_WARN("io_uring backend initialization failed, fallback to aio backend");
    if (TryInitAio(cfg, io_pool)) {
        g_io_pool = io_pool;
        LOG_WARN("Global IOContextPool fallback initialized with backend aio");
        return true;
    }
#endif
#elif defined(MILVUS_COMMON_WITH_LIBAIO)
    if (TryInitAio(cfg, io_pool)) {
        g_io_pool = io_pool;
        LOG_INFO("Global IOContextPool initialized with backend aio");
        return true;
    }
#endif

    LOG_ERROR("Failed to initialize IOContextPool with any backend");
    return false;
}

std::shared_ptr<IOContextPool>
IOContextPool::GetGlobal() {
    {
        std::scoped_lock lk(g_io_pool_mutex);
        if (g_io_pool != nullptr) {
            return g_io_pool;
        }
    }

    IOContextPoolConfig cfg;
    if (!InitGlobal(cfg)) {
        return nullptr;
    }

    std::scoped_lock lk(g_io_pool_mutex);
    return g_io_pool;
}

void
IOContextPool::ResetGlobalForTest() {
    std::scoped_lock lk(g_io_pool_mutex);
    g_io_pool.reset();
#ifdef MILVUS_COMMON_WITH_LIBAIO
    AioContextPool::ResetGlobalForTest();
#endif
#ifdef WITH_IO_URING
    UringContextPool::ResetGlobalForTest();
#endif
}

IOBackend
IOContextPool::Backend() const {
    return backend_;
}

std::string
IOContextPool::BackendName() const {
    switch (backend_) {
        case IOBackend::IO_URING:
            return "io_uring";
        case IOBackend::AIO:
            return "aio";
        default:
            return "unknown";
    }
}

bool
IOContextPool::IsInitialized() const {
    return backend_ != IOBackend::UNKNOWN;
}

size_t
IOContextPool::MaxEventsPerCtx() const {
    return max_events_per_ctx_;
}

IOContextHandle
IOContextPool::Pop() {
    IOContextHandle handle;
    handle.backend = backend_;
    switch (backend_) {
#ifdef WITH_IO_URING
        case IOBackend::IO_URING:
            handle.uring = PopUring();
            break;
#endif
#ifdef MILVUS_COMMON_WITH_LIBAIO
        case IOBackend::AIO:
            handle.aio = PopAio();
            break;
#endif
        default:
            break;
    }
    return handle;
}

void
IOContextPool::Push(IOContextHandle handle) {
    if (handle.backend != backend_) {
        LOG_WARN("IOContextPool rejects handle for backend %d while active backend is %d",
                 static_cast<int>(handle.backend), static_cast<int>(backend_));
        return;
    }

    switch (handle.backend) {
#ifdef WITH_IO_URING
        case IOBackend::IO_URING:
            PushUring(handle.uring);
            break;
#endif
#ifdef MILVUS_COMMON_WITH_LIBAIO
        case IOBackend::AIO:
            PushAio(handle.aio);
            break;
#endif
        default:
            break;
    }
}

#ifdef WITH_IO_URING
struct io_uring*
IOContextPool::PopUring() {
    if (uring_pool_ == nullptr) {
        return nullptr;
    }
    return uring_pool_->pop();
}

void
IOContextPool::PushUring(struct io_uring* ring) {
    if (uring_pool_ != nullptr) {
        uring_pool_->push(ring);
    }
}

bool
IOContextPool::ResetUring(struct io_uring* ring) {
    return uring_pool_ != nullptr && uring_pool_->ResetCheckedOut(ring);
}

std::shared_ptr<UringContextPool>
IOContextPool::GetUringPoolForLegacy() const {
    return uring_pool_;
}
#endif

#ifdef MILVUS_COMMON_WITH_LIBAIO
io_context_t
IOContextPool::PopAio() {
    if (aio_pool_ == nullptr) {
        return nullptr;
    }
    return aio_pool_->pop();
}

void
IOContextPool::PushAio(io_context_t ctx) {
    if (aio_pool_ != nullptr) {
        aio_pool_->push(ctx);
    }
}

std::shared_ptr<AioContextPool>
IOContextPool::GetAioPoolForLegacy() const {
    return aio_pool_;
}
#endif
