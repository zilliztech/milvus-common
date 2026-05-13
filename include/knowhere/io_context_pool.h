#pragma once

#include <cstddef>
#include <memory>
#include <string>

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

enum class IOBackend {
    UNKNOWN,
    IO_URING,
    AIO,
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

struct IOContextHandle {
    IOBackend backend = IOBackend::UNKNOWN;
#ifdef WITH_IO_URING
    struct io_uring* uring = nullptr;
#endif
#ifdef MILVUS_COMMON_WITH_LIBAIO
    io_context_t aio = nullptr;
#endif
};

class IOContextPool {
 public:
    IOContextPool(const IOContextPool&) = delete;
    IOContextPool&
    operator=(const IOContextPool&) = delete;

    static bool
    InitGlobal(const IOContextPoolConfig& cfg);

    static std::shared_ptr<IOContextPool>
    GetGlobal();

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

    void
    Push(IOContextHandle handle);

#ifdef WITH_IO_URING
    struct io_uring*
    PopUring();

    void
    PushUring(struct io_uring* ring);

    bool
    ResetUring(struct io_uring* ring);

    std::shared_ptr<UringContextPool>
    GetUringPoolForLegacy() const;
#endif

#ifdef MILVUS_COMMON_WITH_LIBAIO
    io_context_t
    PopAio();

    void
    PushAio(io_context_t ctx);

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

#ifdef WITH_IO_URING
    std::shared_ptr<UringContextPool> uring_pool_;
#endif

#ifdef MILVUS_COMMON_WITH_LIBAIO
    std::shared_ptr<AioContextPool> aio_pool_;
#endif
};
