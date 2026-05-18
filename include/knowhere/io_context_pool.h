#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#ifdef MILVUS_COMMON_WITH_LIBAIO
#include <libaio.h>
#endif

#ifdef WITH_IO_URING
#include <liburing.h>
#endif

#ifdef MILVUS_COMMON_WITH_LIBAIO
#include "knowhere/aio_context_pool.h"
#endif

#ifdef WITH_IO_URING
#include "knowhere/uring_context_pool.h"
#endif

class IOContextPool;

enum class IOBackend {
    UNKNOWN,
    IO_URING,
    AIO,
};

enum class IOContextReleaseDisposition {
    Clean,
    Dirty,
    Retire,
};

constexpr size_t default_io_ctx_pool_size = 65536 / 128;

struct IOContextPoolConfig {
#ifdef MILVUS_COMMON_WITH_LIBAIO
    size_t num_ctx = default_pool_size;
#else
    size_t num_ctx = default_io_ctx_pool_size;
#endif
    size_t max_events = 128;
};

// Lifecycle invariants:
// - Healthy backend pools account every context as either available or leased.
// - Dirty/retired contexts never return to the available queue without a successful reset.
// - If reset/replacement cannot restore capacity, backend pools become fail-fast.
// - IOContextHandle is the single lease token; moving or destroying it releases the old lease.
struct IOContextHandle {
    IOContextHandle() = default;
    ~IOContextHandle();

    IOContextHandle(const IOContextHandle&) = delete;
    IOContextHandle&
    operator=(const IOContextHandle&) = delete;

    IOContextHandle(IOContextHandle&& other) noexcept;

    IOContextHandle&
    operator=(IOContextHandle&& other) noexcept;

    bool
    HasContext() const noexcept;

    IOBackend backend = IOBackend::UNKNOWN;
#ifdef WITH_IO_URING
    struct io_uring* uring = nullptr;
#endif
#ifdef MILVUS_COMMON_WITH_LIBAIO
    io_context_t aio = nullptr;
#endif

 private:
    friend class IOContextPool;

    void
    ClearNoRelease() noexcept;

    void
    ReleaseNoThrow() noexcept;

    std::shared_ptr<IOContextPool> owner_;
    uint64_t owner_generation_ = 0;
};

class IOContextPool : public std::enable_shared_from_this<IOContextPool> {
 public:
    IOContextPool(const IOContextPool&) = delete;
    IOContextPool&
    operator=(const IOContextPool&) = delete;

    static bool
    InitGlobal(const IOContextPoolConfig& cfg);

    static std::shared_ptr<IOContextPool>
    GetGlobal();

    static std::shared_ptr<IOContextPool>
    GetGlobalOrInit(const IOContextPoolConfig& cfg = IOContextPoolConfig{});

    static void
    ResetGlobalForTest();

    IOBackend
    Backend() const;

    std::string
    BackendName() const;

    bool
    IsInitialized() const;

    size_t
    MaxEventsPerCtx() const;

    IOContextHandle
    Pop();

    bool
    Push(IOContextHandle&& handle);

    bool
    Push(IOContextHandle& handle);

    bool
    Reset(IOContextHandle&& handle);

    bool
    Reset(IOContextHandle& handle);

    bool
    Release(IOContextHandle&& handle, IOContextReleaseDisposition disposition);

    bool
    Release(IOContextHandle& handle, IOContextReleaseDisposition disposition);

#ifdef WITH_IO_URING
    struct io_uring*
    PopUring();

    bool
    PushUring(struct io_uring* ring);

    bool
    ResetUring(struct io_uring* ring);

    bool
    RetireUring(struct io_uring* ring);

    std::shared_ptr<UringContextPool>
    GetUringPoolForLegacy() const;
#endif

#ifdef MILVUS_COMMON_WITH_LIBAIO
    io_context_t
    PopAio();

    bool
    PushAio(io_context_t ctx);

    bool
    ResetAio(io_context_t ctx);

    bool
    RetireAio(io_context_t ctx);

    std::shared_ptr<AioContextPool>
    GetAioPoolForLegacy() const;
#endif

 private:
    IOContextPool() = default;

#ifdef WITH_IO_URING
    static bool
    TryInitUring(const IOContextPoolConfig& cfg, const std::shared_ptr<IOContextPool>& io_pool);
#endif

#ifdef MILVUS_COMMON_WITH_LIBAIO
    static bool
    TryInitAio(const IOContextPoolConfig& cfg, const std::shared_ptr<IOContextPool>& io_pool);
#endif

    IOBackend backend_ = IOBackend::UNKNOWN;
    size_t num_ctx_ = 0;
    size_t max_events_per_ctx_ = 0;
    uint64_t generation_ = 0;
    static std::atomic<uint64_t> next_generation_;

#ifdef WITH_IO_URING
    std::shared_ptr<UringContextPool> uring_pool_;
#endif

#ifdef MILVUS_COMMON_WITH_LIBAIO
    std::shared_ptr<AioContextPool> aio_pool_;
#endif
};
