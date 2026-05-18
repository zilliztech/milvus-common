#include "knowhere/io_context_pool.h"

#include <exception>
#include <mutex>
#include <utility>

#include "log/Log.h"

namespace {
std::shared_ptr<IOContextPool> g_io_pool;
std::mutex g_io_pool_mutex;
}  // namespace

std::atomic<uint64_t> IOContextPool::next_generation_{1};

IOContextHandle::~IOContextHandle() {
    ReleaseNoThrow();
}

IOContextHandle::IOContextHandle(IOContextHandle&& other) noexcept {
    *this = std::move(other);
}

IOContextHandle&
IOContextHandle::operator=(IOContextHandle&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    ReleaseNoThrow();
    backend = other.backend;
#ifdef WITH_IO_URING
    uring = other.uring;
    other.uring = nullptr;
#endif
#ifdef MILVUS_COMMON_WITH_LIBAIO
    aio = other.aio;
    other.aio = nullptr;
#endif
    owner_ = std::move(other.owner_);
    owner_generation_ = other.owner_generation_;
    other.backend = IOBackend::UNKNOWN;
    other.owner_generation_ = 0;
    return *this;
}

bool
IOContextHandle::HasContext() const noexcept {
    switch (backend) {
#ifdef WITH_IO_URING
        case IOBackend::IO_URING:
            return uring != nullptr;
#endif
#ifdef MILVUS_COMMON_WITH_LIBAIO
        case IOBackend::AIO:
            return aio != nullptr;
#endif
        default:
            return false;
    }
}

void
IOContextHandle::ClearNoRelease() noexcept {
    backend = IOBackend::UNKNOWN;
#ifdef WITH_IO_URING
    uring = nullptr;
#endif
#ifdef MILVUS_COMMON_WITH_LIBAIO
    aio = nullptr;
#endif
    owner_.reset();
    owner_generation_ = 0;
}

void
IOContextHandle::ReleaseNoThrow() noexcept {
    if (!HasContext()) {
        ClearNoRelease();
        return;
    }

    auto owner = owner_;
    if (owner == nullptr) {
        LOG_WARN("IOContextHandle drops context without owner for backend {}", static_cast<int>(backend));
        ClearNoRelease();
        return;
    }

    try {
        owner->Release(*this, IOContextReleaseDisposition::Clean);
    } catch (const std::exception& e) {
        LOG_ERROR("IOContextHandle failed to release context: {}", e.what());
        ClearNoRelease();
    } catch (...) {
        LOG_ERROR("IOContextHandle failed to release context: unknown exception");
        ClearNoRelease();
    }
}

#ifdef WITH_IO_URING
bool
IOContextPool::TryInitUring(const IOContextPoolConfig& cfg, const std::shared_ptr<IOContextPool>& io_pool) {
    if (!UringContextPool::InitGlobalUringPoolWithValidation(cfg.num_ctx, cfg.max_events)) {
        return false;
    }

    auto pool = UringContextPool::GetGlobalUringPoolDirect();
    if (pool == nullptr || !pool->IsUsable()) {
        LOG_ERROR("Global UringContextPool is unavailable after initialization");
        UringContextPool::ResetGlobalForTest();
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
    if (pool == nullptr || !pool->IsUsable()) {
        LOG_ERROR("Global AioContextPool is unavailable after initialization");
        AioContextPool::ResetGlobalForTest();
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
            LOG_ERROR("Global IOContextPool already initialized with max_events={}, requested={}",
                      g_io_pool->MaxEventsPerCtx(), cfg.max_events);
            return false;
        }
        if (cfg.num_ctx != g_io_pool->num_ctx_) {
            LOG_ERROR("Global IOContextPool already initialized with num_ctx={}, requested={}", g_io_pool->num_ctx_,
                      cfg.num_ctx);
            return false;
        }
        LOG_WARN("Global IOContextPool has already been initialized with backend: {}", g_io_pool->BackendName());
        return true;
    }

    auto io_pool = std::shared_ptr<IOContextPool>(new IOContextPool());
    io_pool->generation_ = next_generation_.fetch_add(1, std::memory_order_relaxed);

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
    std::scoped_lock lk(g_io_pool_mutex);
    return g_io_pool;
}

std::shared_ptr<IOContextPool>
IOContextPool::GetGlobalOrInit(const IOContextPoolConfig& cfg) {
    {
        std::scoped_lock lk(g_io_pool_mutex);
        if (g_io_pool != nullptr) {
            return g_io_pool;
        }
    }

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
    if (handle.HasContext()) {
        handle.owner_ = shared_from_this();
        handle.owner_generation_ = generation_;
    } else {
        handle.backend = IOBackend::UNKNOWN;
    }
    return handle;
}

bool
IOContextPool::Push(IOContextHandle&& handle) {
    return Release(std::move(handle), IOContextReleaseDisposition::Clean);
}

bool
IOContextPool::Push(IOContextHandle& handle) {
    return Push(std::move(handle));
}

bool
IOContextPool::Reset(IOContextHandle&& handle) {
    return Release(std::move(handle), IOContextReleaseDisposition::Dirty);
}

bool
IOContextPool::Reset(IOContextHandle& handle) {
    return Reset(std::move(handle));
}

bool
IOContextPool::Release(IOContextHandle&& handle, IOContextReleaseDisposition disposition) {
    if (!handle.HasContext()) {
        handle.ClearNoRelease();
        return true;
    }
    if (handle.owner_.get() != this) {
        LOG_WARN("IOContextPool rejects release for handle owned by a different pool");
        return false;
    }
    if (handle.owner_generation_ != generation_) {
        LOG_WARN("IOContextPool rejects stale handle for generation {} while active generation is {}",
                 handle.owner_generation_, generation_);
        handle.ClearNoRelease();
        return false;
    }
    if (handle.backend != backend_) {
        LOG_WARN("IOContextPool rejects release for backend {} while active backend is {}",
                 static_cast<int>(handle.backend), static_cast<int>(backend_));
        handle.ClearNoRelease();
        return false;
    }

    bool released = false;
    switch (handle.backend) {
#ifdef WITH_IO_URING
        case IOBackend::IO_URING:
            if (disposition == IOContextReleaseDisposition::Clean) {
                released = PushUring(handle.uring);
            } else if (disposition == IOContextReleaseDisposition::Dirty) {
                released = ResetUring(handle.uring);
                if (released) {
                    released = PushUring(handle.uring);
                }
            } else {
                released = RetireUring(handle.uring);
            }
            break;
#endif
#ifdef MILVUS_COMMON_WITH_LIBAIO
        case IOBackend::AIO:
            if (disposition == IOContextReleaseDisposition::Clean) {
                released = PushAio(handle.aio);
            } else if (disposition == IOContextReleaseDisposition::Dirty) {
                released = ResetAio(handle.aio);
            } else {
                released = RetireAio(handle.aio);
            }
            break;
#endif
        default:
            break;
    }
    handle.ClearNoRelease();
    return released;
}

bool
IOContextPool::Release(IOContextHandle& handle, IOContextReleaseDisposition disposition) {
    return Release(std::move(handle), disposition);
}

#ifdef WITH_IO_URING
struct io_uring*
IOContextPool::PopUring() {
    if (uring_pool_ == nullptr) {
        return nullptr;
    }
    return uring_pool_->pop();
}

bool
IOContextPool::PushUring(struct io_uring* ring) {
    if (uring_pool_ != nullptr) {
        return uring_pool_->push(ring);
    }
    return false;
}

bool
IOContextPool::ResetUring(struct io_uring* ring) {
    return uring_pool_ != nullptr && uring_pool_->ResetCheckedOut(ring);
}

bool
IOContextPool::RetireUring(struct io_uring* ring) {
    return uring_pool_ != nullptr && uring_pool_->RetireCheckedOut(ring);
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

bool
IOContextPool::PushAio(io_context_t ctx) {
    if (aio_pool_ != nullptr) {
        return aio_pool_->push(ctx);
    }
    return false;
}

bool
IOContextPool::ResetAio(io_context_t ctx) {
    return aio_pool_ != nullptr && aio_pool_->ResetCheckedOut(ctx);
}

bool
IOContextPool::RetireAio(io_context_t ctx) {
    return aio_pool_ != nullptr && aio_pool_->RetireCheckedOut(ctx);
}

std::shared_ptr<AioContextPool>
IOContextPool::GetAioPoolForLegacy() const {
    return aio_pool_;
}
#endif
