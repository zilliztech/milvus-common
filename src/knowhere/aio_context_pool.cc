#include "knowhere/aio_context_pool.h"

#include "knowhere/io_context_pool.h"
#include "log/Log.h"

namespace {
std::shared_ptr<AioContextPool> g_aio_pool;
}

std::atomic<size_t> AioContextPool::global_aio_pool_size{0};
std::atomic<size_t> AioContextPool::global_aio_max_events{0};
std::mutex AioContextPool::global_aio_pool_mut;

bool
AioContextPool::InitGlobalAioPoolWithValidation(size_t num_ctx, size_t max_events) {
    if (num_ctx <= 0) {
        LOG_ERROR("num_ctx should be bigger than 0");
        return false;
    }
    if (max_events == 0) {
        LOG_ERROR("max_events should be bigger than 0");
        return false;
    }
    if (max_events > default_max_events) {
        LOG_ERROR("max_events {} should not be larger than {}", max_events, default_max_events);
        return false;
    }
    if (global_aio_pool_size == 0) {
        std::scoped_lock lk(global_aio_pool_mut);
        if (global_aio_pool_size == 0) {
            global_aio_pool_size = num_ctx;
            global_aio_max_events = max_events;
            return true;
        }
    }
    if (global_aio_pool_size != num_ctx || global_aio_max_events != max_events) {
        LOG_ERROR(
            "Global AioContextPool already initialized with context num: {}, max_events: {} (requested {}, {})",
            global_aio_pool_size.load(), global_aio_max_events.load(), num_ctx, max_events);
        return false;
    }
    LOG_WARN("Global AioContextPool has already been initialized with context num: {}", global_aio_pool_size.load());
    return true;
}

std::shared_ptr<AioContextPool>
AioContextPool::GetGlobalAioPoolDirect() {
    std::scoped_lock lk(global_aio_pool_mut);
    if (global_aio_pool_size == 0) {
        global_aio_pool_size = default_pool_size;
        global_aio_max_events = default_max_events;
        LOG_WARN("Global AioContextPool has not been inialized yet, init it now with context num: {}",
                 global_aio_pool_size.load());
    }
    if (g_aio_pool == nullptr) {
        g_aio_pool = std::shared_ptr<AioContextPool>(new AioContextPool(global_aio_pool_size, global_aio_max_events));
    }
    return g_aio_pool;
}

bool
AioContextPool::InitGlobalAioPool(size_t num_ctx, size_t max_events) {
    return InitGlobalAioPoolWithValidation(num_ctx, max_events) && GetGlobalAioPoolDirect() != nullptr;
}

std::shared_ptr<AioContextPool>
AioContextPool::GetGlobalAioPool() {
    auto io_pool = IOContextPool::GetGlobal();
    if (io_pool != nullptr && io_pool->IsInitialized() && io_pool->Backend() == IOBackend::AIO) {
        return io_pool->GetAioPoolForLegacy();
    }

    if (io_pool != nullptr && io_pool->IsInitialized()) {
        LOG_WARN("Returning independent legacy AIO pool while unified IOContextPool backend is {}",
                 io_pool->BackendName());
    }
    return GetGlobalAioPoolDirect();
}

void
AioContextPool::ResetGlobalForTest() {
    std::scoped_lock lk(global_aio_pool_mut);
    g_aio_pool.reset();
    global_aio_pool_size = 0;
    global_aio_max_events = 0;
}
