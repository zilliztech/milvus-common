#pragma once

#include <libaio.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

#include "log/Log.h"

constexpr size_t default_max_nr = 65536;
constexpr size_t default_max_events = 128;
constexpr size_t default_pool_size = default_max_nr / default_max_events;

class AioContextPool {
 public:
    AioContextPool(const AioContextPool&) = delete;

    AioContextPool&
    operator=(const AioContextPool&) = delete;

    AioContextPool(AioContextPool&&) noexcept = delete;

    AioContextPool&
    operator==(AioContextPool&&) noexcept = delete;

    size_t
    max_events_per_ctx() {
        return max_events_;
    }

    void
    push(io_context_t ctx);

    // Throws std::runtime_error if no usable contexts remain.
    io_context_t
    pop();

    // Retire an exclusively borrowed context and try to replace it once.
    // The handle is consumed even if destruction fails: log the error and
    // abandon that context without returning it to the pool. Failed setup
    // reduces pool capacity; pop() does not retry it. This best-effort cleanup
    // cannot guarantee that pending I/O has stopped if destruction fails.
    void
    DestroyAndRecreate(io_context_t& ctx) noexcept;

    static bool
    InitGlobalAioPool(size_t num_ctx, size_t max_events);

    static std::shared_ptr<AioContextPool>
    GetGlobalAioPool();

    ~AioContextPool();

 private:
    std::vector<io_context_t> ctx_bak_;
    // Reserve the configured capacity up front, so returning
    // or retiring a borrowed context does not allocate while unwinding.
    std::vector<io_context_t> ctx_q_;
    std::mutex ctx_mtx_;
    std::condition_variable ctx_cv_;
    bool stop_ = false;
    // Usable contexts, including borrowed contexts and replacements in progress.
    size_t num_ctx_ = 0;
    size_t max_events_;
    static size_t global_aio_pool_size;
    static size_t global_aio_max_events;
    static std::mutex global_aio_pool_mut;

    AioContextPool(size_t num_ctx, size_t max_events);
};
