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

    // May throw std::system_error if a retired slot cannot be recreated.
    io_context_t
    pop();

    // Consume an exclusively borrowed context after synchronously destroying
    // its pending I/O, then return a replacement to the pool. The caller must
    // keep its I/O buffers alive until this operation completes.
    //
    // Returns 0 on success, or a negative libaio error. A null ctx means the
    // old I/O is safely destroyed; if setup failed, the pool retains the slot
    // and pop() will try to recreate it. A non-null ctx on error means destroy
    // did not succeed and the caller still owns the context and its buffers.
    [[nodiscard]] int
    DestroyAndRecreate(io_context_t& ctx) noexcept;

    static bool
    InitGlobalAioPool(size_t num_ctx, size_t max_events);

    static std::shared_ptr<AioContextPool>
    GetGlobalAioPool();

    ~AioContextPool();

 private:
    std::vector<io_context_t> ctx_bak_;
    // These vectors reserve the configured capacity up front, so returning
    // or retiring a borrowed context does not allocate while unwinding.
    std::vector<io_context_t> ctx_q_;
    std::vector<size_t> pending_recreation_;
    std::mutex ctx_mtx_;
    std::condition_variable ctx_cv_;
    bool stop_ = false;
    size_t num_ctx_;
    size_t max_events_;
    static size_t global_aio_pool_size;
    static size_t global_aio_max_events;
    static std::mutex global_aio_pool_mut;

    AioContextPool(size_t num_ctx, size_t max_events);
};
