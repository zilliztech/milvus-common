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
        LOG_ERROR("max_events %zu should not be larger than %zu", max_events, default_max_events);
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
            "Global AioContextPool already initialized with context num: %zu, max_events: %zu (requested %zu, %zu)",
            global_aio_pool_size.load(), global_aio_max_events.load(), num_ctx, max_events);
        return false;
    }
    LOG_WARN("Global AioContextPool has already been initialized with context num: %zu", global_aio_pool_size.load());
    return true;
}

std::shared_ptr<AioContextPool>
AioContextPool::GetGlobalAioPoolDirect() {
    std::scoped_lock lk(global_aio_pool_mut);
    if (global_aio_pool_size == 0) {
        global_aio_pool_size = default_pool_size;
        global_aio_max_events = default_max_events;
        LOG_WARN("Global AioContextPool has not been inialized yet, init it now with context num: %zu",
                 global_aio_pool_size.load());
    }
    if (g_aio_pool == nullptr) {
        g_aio_pool = std::shared_ptr<AioContextPool>(new AioContextPool(global_aio_pool_size, global_aio_max_events));
    }
    return g_aio_pool;
}

bool
AioContextPool::InitGlobalAioPool(size_t num_ctx, size_t max_events) {
    IOContextPoolConfig cfg;
    cfg.num_ctx = num_ctx;
    cfg.max_events = max_events;

    if (!IOContextPool::InitGlobal(cfg)) {
        return false;
    }

    auto io_pool = IOContextPool::GetGlobal();
    if (io_pool == nullptr || !io_pool->IsInitialized()) {
        return false;
    }
    if (io_pool->Backend() != IOBackend::AIO) {
        LOG_ERROR("Global IOContextPool backend is %s, legacy AIO API is unavailable", io_pool->BackendName().c_str());
        return false;
    }
    return true;
}

std::shared_ptr<AioContextPool>
AioContextPool::GetGlobalAioPool() {
    auto io_pool = IOContextPool::GetGlobal();
    if (io_pool == nullptr || !io_pool->IsInitialized()) {
        return nullptr;
    }
    if (io_pool->Backend() != IOBackend::AIO) {
        return nullptr;
    }
    return io_pool->GetAioPoolForLegacy();
}

void
AioContextPool::ResetGlobalForTest() {
    std::scoped_lock lk(global_aio_pool_mut);
    g_aio_pool.reset();
    global_aio_pool_size = 0;
    global_aio_max_events = 0;
}
