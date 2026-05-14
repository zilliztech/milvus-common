#include "knowhere/io_reader.h"

#include "io_reader_internal.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "log/Log.h"

namespace {
std::future<bool>
MakeReadyFuture(bool value) {
    return std::async(std::launch::deferred, [value] { return value; });
}

class IOContextHandleGuard {
 public:
    IOContextHandleGuard(std::shared_ptr<IOContextPool> pool, IOContextHandle handle)
        : pool_(std::move(pool)), handle_(handle) {
    }

    IOContextHandleGuard(const IOContextHandleGuard&) = delete;
    IOContextHandleGuard&
    operator=(const IOContextHandleGuard&) = delete;

    IOContextHandleGuard(IOContextHandleGuard&& other) noexcept
        : pool_(std::move(other.pool_)), handle_(other.handle_), active_(other.active_) {
        other.active_ = false;
    }

    ~IOContextHandleGuard() {
        Reset();
    }

    IOContextHandle&
    Handle() {
        return handle_;
    }

    void
    Reset() {
        if (active_ && pool_ != nullptr) {
            pool_->Push(handle_);
            active_ = false;
        }
    }

 private:
    std::shared_ptr<IOContextPool> pool_;
    IOContextHandle handle_;
    bool active_ = true;
};

constexpr size_t kNumRetries = 10;

struct BatchWaitResult {
    size_t completed = 0;
    bool complete = false;
    bool ok = false;
};

#ifdef MILVUS_COMMON_WITH_LIBAIO
size_t
SubmitAioBatch(io_context_t ctx, int fd, const std::vector<std::byte*>& buffers, size_t size,
               const std::vector<size_t>& offsets, size_t start, size_t batch, std::vector<struct iocb>& cbs) {
    cbs.resize(batch);
    std::vector<struct iocb*> cb_ptrs(batch);
    for (size_t i = 0; i < batch; ++i) {
        const auto idx = start + i;
        io_prep_pread(&cbs[i], fd, reinterpret_cast<void*>(buffers[idx]), size, offsets[idx]);
        cb_ptrs[i] = &cbs[i];
    }

    size_t submitted_total = 0;
    size_t retry = 0;
    while (submitted_total < batch) {
        const auto submitted =
            io_submit(ctx, static_cast<long>(batch - submitted_total), cb_ptrs.data() + submitted_total);
        if (submitted < 0) {
            if (-submitted == EINTR) {
                if (!knowhere_internal::ShouldRetryInterruptedSyscall(retry, kNumRetries)) {
                    break;
                }
                continue;
            }
            break;
        }
        if (submitted == 0) {
            if (++retry > kNumRetries) {
                break;
            }
            continue;
        }
        submitted_total += static_cast<size_t>(submitted);
        if (submitted_total < batch && ++retry > kNumRetries) {
            break;
        }
    }
    return submitted_total;
}

BatchWaitResult
WaitAioBatch(io_context_t ctx, size_t size, size_t submitted_total) {
    if (submitted_total == 0) {
        return {0, true, true};
    }

    std::vector<struct io_event> events(submitted_total);
    BatchWaitResult result;
    result.ok = true;
    size_t retry = 0;
    while (result.completed < submitted_total) {
        const auto completed = io_getevents(ctx, 1, static_cast<long>(submitted_total - result.completed),
                                            events.data() + result.completed, nullptr);
        if (completed < 0) {
            if (-completed == EINTR) {
                if (!knowhere_internal::ShouldRetryInterruptedSyscall(retry, kNumRetries)) {
                    break;
                }
                continue;
            }
            break;
        }
        if (completed == 0) {
            if (++retry > kNumRetries) {
                break;
            }
            continue;
        }
        for (size_t i = result.completed; i < result.completed + static_cast<size_t>(completed); ++i) {
            if (events[i].res < 0 || static_cast<size_t>(events[i].res) != size) {
                result.ok = false;
            }
        }
        result.completed += static_cast<size_t>(completed);
        if (result.completed < submitted_total && ++retry > kNumRetries) {
            break;
        }
    }
    result.complete = result.completed == submitted_total;
    result.ok = result.ok && result.complete;
    return result;
}

class AioReadState {
 public:
    AioReadState(IOContextHandleGuard guard, size_t size, size_t first_submitted, std::vector<struct iocb>&& first_cbs)
        : guard_(std::move(guard)), size_(size), first_remaining_(first_submitted), first_cbs_(std::move(first_cbs)) {
    }

    AioReadState(const AioReadState&) = delete;
    AioReadState&
    operator=(const AioReadState&) = delete;

    AioReadState(AioReadState&& other) noexcept
        : guard_(std::move(other.guard_)),
          size_(other.size_),
          first_remaining_(other.first_remaining_),
          first_cbs_(std::move(other.first_cbs_)) {
        other.first_remaining_ = 0;
    }

    ~AioReadState() {
        CollectFirst();
    }

    io_context_t
    Context() {
        return guard_.Handle().aio;
    }

    BatchWaitResult
    CollectFirst() {
        if (first_remaining_ == 0) {
            return {0, true, true};
        }
        auto result = WaitAioBatch(Context(), size_, first_remaining_);
        first_remaining_ -= result.completed;
        result.complete = first_remaining_ == 0;
        result.ok = result.ok && result.complete;
        return result;
    }

 private:
    IOContextHandleGuard guard_;
    size_t size_ = 0;
    size_t first_remaining_ = 0;
    std::vector<struct iocb> first_cbs_;
};

std::future<bool>
ReadAioAsync(int fd, size_t size, std::vector<std::byte*>&& buffers, std::vector<size_t>&& offsets,
             std::shared_ptr<IOContextPool> pool) {
    auto handle = pool->Pop();
    if (handle.aio == nullptr) {
        return MakeReadyFuture(false);
    }

    IOContextHandleGuard guard(pool, handle);
    const size_t max_batch = pool->MaxEventsPerCtx();
    if (max_batch == 0) {
        return MakeReadyFuture(false);
    }

    const size_t first_batch = std::min(max_batch, buffers.size());
    std::vector<struct iocb> first_cbs;
    const size_t first_submitted = SubmitAioBatch(handle.aio, fd, buffers, size, offsets, 0, first_batch, first_cbs);
    if (first_submitted == 0) {
        return MakeReadyFuture(false);
    }

    return std::async(
        std::launch::deferred,
        [fd, size, buffers = std::move(buffers), offsets = std::move(offsets), max_batch, first_batch, first_submitted,
         state = AioReadState(std::move(guard), size, first_submitted, std::move(first_cbs))]() mutable -> bool {
            auto ctx = state.Context();
            const auto first_result = state.CollectFirst();
            if (!first_result.complete || !first_result.ok || first_submitted != first_batch) {
                return false;
            }

            size_t processed = first_batch;
            while (processed < buffers.size()) {
                const size_t batch = std::min(max_batch, buffers.size() - processed);
                std::vector<struct iocb> cbs;
                const size_t submitted = SubmitAioBatch(ctx, fd, buffers, size, offsets, processed, batch, cbs);
                if (submitted != batch) {
                    WaitAioBatch(ctx, size, submitted);
                    return false;
                }
                const auto result = WaitAioBatch(ctx, size, submitted);
                if (!result.complete || !result.ok) {
                    return false;
                }
                processed += batch;
            }
            return true;
        });
}
#endif

#ifdef WITH_IO_URING
size_t
PrepareUringBatch(io_uring* ring, int fd, const std::vector<std::byte*>& buffers, size_t size,
                  const std::vector<size_t>& offsets, size_t start) {
    size_t batch = 0;
    for (; start + batch < buffers.size(); ++batch) {
        auto* sqe = io_uring_get_sqe(ring);
        if (sqe == nullptr) {
            break;
        }
        const auto idx = start + batch;
        io_uring_prep_read(sqe, fd, reinterpret_cast<void*>(buffers[idx]), size, offsets[idx]);
        sqe->user_data = idx;
    }
    return batch;
}

BatchWaitResult
WaitUringBatch(io_uring* ring, size_t size, size_t submitted_total) {
    BatchWaitResult result;
    result.ok = true;
    size_t retry = 0;
    while (result.completed < submitted_total) {
        io_uring_cqe* cqe = nullptr;
        const auto wait_result = io_uring_wait_cqe(ring, &cqe);
        if (wait_result < 0) {
            if (-wait_result == EINTR) {
                if (!knowhere_internal::ShouldRetryInterruptedSyscall(retry, kNumRetries)) {
                    break;
                }
                continue;
            }
            break;
        }
        if (cqe == nullptr) {
            break;
        }
        if (cqe->res < 0 || static_cast<size_t>(cqe->res) != size) {
            result.ok = false;
        }
        io_uring_cqe_seen(ring, cqe);
        ++result.completed;
    }
    result.complete = result.completed == submitted_total;
    result.ok = result.ok && result.complete;
    return result;
}

class UringReadState {
 public:
    UringReadState(IOContextHandleGuard guard, size_t size, size_t first_submitted)
        : guard_(std::move(guard)), size_(size), first_remaining_(first_submitted) {
    }

    UringReadState(const UringReadState&) = delete;
    UringReadState&
    operator=(const UringReadState&) = delete;

    UringReadState(UringReadState&& other) noexcept
        : guard_(std::move(other.guard_)), size_(other.size_), first_remaining_(other.first_remaining_) {
        other.first_remaining_ = 0;
    }

    ~UringReadState() {
        CollectFirst();
    }

    io_uring*
    Ring() {
        return guard_.Handle().uring;
    }

    BatchWaitResult
    CollectFirst() {
        if (first_remaining_ == 0) {
            return {0, true, true};
        }
        auto result = WaitUringBatch(Ring(), size_, first_remaining_);
        first_remaining_ -= result.completed;
        result.complete = first_remaining_ == 0;
        result.ok = result.ok && result.complete;
        return result;
    }

 private:
    IOContextHandleGuard guard_;
    size_t size_ = 0;
    size_t first_remaining_ = 0;
};

std::future<bool>
ReadUringAsync(int fd, size_t size, std::vector<std::byte*>&& buffers, std::vector<size_t>&& offsets,
               std::shared_ptr<IOContextPool> pool) {
    auto handle = pool->Pop();
    if (handle.uring == nullptr) {
        return MakeReadyFuture(false);
    }

    IOContextHandleGuard guard(pool, handle);
    auto* ring = handle.uring;
    const size_t first_batch = PrepareUringBatch(ring, fd, buffers, size, offsets, 0);
    if (first_batch == 0) {
        return MakeReadyFuture(false);
    }

    const auto submitted = io_uring_submit(ring);
    if (submitted <= 0) {
        return MakeReadyFuture(false);
    }
    const auto first_submitted = static_cast<size_t>(submitted);

    return std::async(
        std::launch::deferred,
        [fd, size, buffers = std::move(buffers), offsets = std::move(offsets), first_batch, first_submitted,
         state = UringReadState(std::move(guard), size, first_submitted)]() mutable -> bool {
            auto* ring = state.Ring();
            const auto first_result = state.CollectFirst();
            if (!first_result.complete || !first_result.ok || first_submitted != first_batch) {
                return false;
            }

            size_t processed = first_batch;
            while (processed < buffers.size()) {
                const size_t batch = PrepareUringBatch(ring, fd, buffers, size, offsets, processed);
                if (batch == 0) {
                    return false;
                }
                const auto submitted = io_uring_submit(ring);
                if (submitted <= 0) {
                    return false;
                }
                const auto submitted_count = static_cast<size_t>(submitted);
                const auto result = WaitUringBatch(ring, size, submitted_count);
                if (!result.complete || !result.ok || submitted_count != batch) {
                    return false;
                }
                processed += batch;
            }
            return true;
        });
}
#endif
}  // namespace

IOReader::IOReader() : io_pool_(IOContextPool::GetGlobal()) {
}

IOReader::IOReader(int fd) : fd_(fd), io_pool_(IOContextPool::GetGlobal()) {
}

IOReader::IOReader(int fd, std::shared_ptr<IOContextPool> io_pool) : fd_(fd), io_pool_(std::move(io_pool)) {
}

IOReader::IOReader(std::shared_ptr<IOContextPool> io_pool) : io_pool_(std::move(io_pool)) {
}

bool
IOReader::Read(IOReaderSpan<std::byte*> buf, size_t size, IOReaderSpan<off_t> offsets) const {
    if (buf.size() != offsets.size()) {
        throw std::invalid_argument("buffers and offsets must have same size");
    }

    std::vector<std::byte*> buffers(buf.size());
    std::vector<size_t> read_offsets(offsets.size());

    for (size_t i = 0; i < buf.size(); ++i) {
        if (buf[i] == nullptr) {
            throw std::invalid_argument("buffer pointer should not be null");
        }
        if (offsets[i] < 0) {
            throw std::invalid_argument("offset should be non-negative");
        }
        buffers[i] = buf[i];
        read_offsets[i] = static_cast<size_t>(offsets[i]);
    }

    return ReadAsync(std::move(buffers), size, std::move(read_offsets)).get();
}

std::future<bool>
IOReader::ReadAsync(std::vector<std::byte*>&& buffers, size_t size, std::vector<size_t>&& offsets) const {
    if (size == 0) {
        throw std::invalid_argument("size should be greater than 0");
    }
    if (buffers.size() != offsets.size()) {
        throw std::invalid_argument("buffers and offsets must have same size");
    }
    if (buffers.empty()) {
        return MakeReadyFuture(true);
    }
    if (fd_ < 0) {
        throw std::invalid_argument("invalid file descriptor");
    }

    for (const auto* buffer : buffers) {
        if (buffer == nullptr) {
            throw std::invalid_argument("buffer pointer should not be null");
        }
    }

    auto pool = io_pool_ ? io_pool_ : IOContextPool::GetGlobal();
    if (pool == nullptr || !pool->IsInitialized()) {
        throw std::runtime_error("IOContextPool is not initialized");
    }

    switch (pool->Backend()) {
#ifdef MILVUS_COMMON_WITH_LIBAIO
        case IOBackend::AIO:
            return ReadAioAsync(fd_, size, std::move(buffers), std::move(offsets), std::move(pool));
#endif
#ifdef WITH_IO_URING
        case IOBackend::IO_URING:
            return ReadUringAsync(fd_, size, std::move(buffers), std::move(offsets), std::move(pool));
#endif
        default:
            return MakeReadyFuture(false);
    }
}

IOBackend
IOReader::Backend() const {
    return io_pool_ ? io_pool_->Backend() : IOBackend::UNKNOWN;
}

std::string
IOReader::BackendName() const {
    return io_pool_ ? io_pool_->BackendName() : "unknown";
}

bool
IOReader::IsReady() const {
    return fd_ >= 0 && io_pool_ != nullptr && io_pool_->IsInitialized();
}
