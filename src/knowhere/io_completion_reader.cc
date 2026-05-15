#include "knowhere/io_completion_reader.h"

#include "io_reader_internal.h"

#include "syncpoint/sync_point.h"

#include <algorithm>
#include <cerrno>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
constexpr size_t kNumRetries = 10;

#ifdef WITH_IO_URING
int
SubmitRing(struct io_uring* ring) {
#ifdef ENABLE_SYNCPOINT
    int forced_ret = 0;
    TEST_SYNC_POINT_CALLBACK("IOCompletionReader::SubmitRing:Before", &forced_ret);
    if (forced_ret != 0) {
        return forced_ret;
    }
#endif
    return io_uring_submit(ring);
}
#endif
}  // namespace

IOCompletionReader::IOCompletionReader(int fd, std::shared_ptr<IOContextPool> io_pool)
    : fd_(fd), io_pool_(std::move(io_pool)) {
    if (fd_ < 0) {
        throw std::invalid_argument("invalid file descriptor");
    }

    if (io_pool_ == nullptr || !io_pool_->IsInitialized()) {
        throw std::runtime_error("IOContextPool is not initialized");
    }

#ifndef WITH_IO_URING
    throw std::runtime_error("IOCompletionReader requires io_uring support");
#else
    if (io_pool_->Backend() != IOBackend::IO_URING) {
        throw std::runtime_error("IOCompletionReader requires io_uring backend");
    }

    handle_ = io_pool_->Pop();
    if (handle_.backend != IOBackend::IO_URING) {
        throw std::runtime_error("failed to acquire io_uring context handle");
    }
    if (handle_.uring == nullptr) {
        throw std::runtime_error("failed to acquire io_uring context handle");
    }
#endif
}

IOCompletionReader::IOCompletionReader(IOCompletionReader&& other) noexcept
    : fd_(other.fd_),
      io_pool_(std::move(other.io_pool_)),
      handle_(other.handle_),
      next_request_id_(other.next_request_id_),
      pending_requests_(std::move(other.pending_requests_)),
      ready_completions_(std::move(other.ready_completions_)) {
    other.fd_ = -1;
    other.handle_ = IOContextHandle{};
    other.next_request_id_ = 1;
}

IOCompletionReader&
IOCompletionReader::operator=(IOCompletionReader&& other) {
    if (this == &other) {
        return *this;
    }

    DrainOutstandingNoThrow();
    ReleaseHandle();

    fd_ = other.fd_;
    io_pool_ = std::move(other.io_pool_);
    handle_ = other.handle_;
    next_request_id_ = other.next_request_id_;
    pending_requests_ = std::move(other.pending_requests_);
    ready_completions_ = std::move(other.ready_completions_);

    other.fd_ = -1;
    other.handle_ = IOContextHandle{};
    other.next_request_id_ = 1;
    return *this;
}

IOCompletionReader::~IOCompletionReader() {
    DrainOutstandingNoThrow();
    ReleaseHandle();
}

IOCompletionReader::RequestId
IOCompletionReader::Submit(std::span<std::byte* const> buffers, size_t size, std::span<const size_t> offsets) {
#ifndef WITH_IO_URING
    (void)buffers;
    (void)size;
    (void)offsets;
    throw std::runtime_error("IOCompletionReader requires io_uring support");
#else
    if (!IsReady()) {
        throw std::runtime_error("IOCompletionReader is not ready");
    }
    if (size == 0) {
        throw std::invalid_argument("size should be greater than 0");
    }
    if (buffers.size() != offsets.size()) {
        throw std::invalid_argument("buffers and offsets must have same size");
    }
    if (buffers.empty()) {
        throw std::invalid_argument("buffers should not be empty");
    }
    for (const auto* buffer : buffers) {
        if (buffer == nullptr) {
            throw std::invalid_argument("buffer pointer should not be null");
        }
    }

    const auto max_events = io_pool_->MaxEventsPerCtx();
    if (max_events == 0) {
        throw std::runtime_error("IOCompletionReader has no io_uring event capacity");
    }
    if (buffers.size() > max_events) {
        throw std::invalid_argument("buffers should not exceed io_uring event capacity");
    }

    ProcessAvailableCompletions();
    if (PendingOperationCount() + buffers.size() > max_events) {
        throw std::runtime_error("too many outstanding io_uring requests");
    }

    auto request_id = next_request_id_++;
    auto [iter, inserted] = pending_requests_.emplace(request_id, RequestState{});
    if (!inserted) {
        throw std::runtime_error("duplicate io_uring request id");
    }
    auto& state = iter->second;
    state.expected_size = size;
    state.ok = true;

    size_t prepared = 0;
    while (prepared < buffers.size()) {
        auto* sqe = io_uring_get_sqe(handle_.uring);
        if (sqe == nullptr) {
            const auto flushed = SubmitRing(handle_.uring);
            if (flushed < 0) {
                CleanupFailedSubmit(request_id, prepared, state.remaining);
                throw std::runtime_error("io_uring_submit failed while preparing request");
            }
            if (flushed == 0) {
                CleanupFailedSubmit(request_id, prepared, state.remaining);
                throw std::runtime_error("io_uring_submit made no progress while preparing request");
            }
            state.remaining += static_cast<size_t>(flushed);
            continue;
        }

        io_uring_prep_read(sqe, fd_, reinterpret_cast<void*>(buffers[prepared]), size, offsets[prepared]);
        sqe->user_data = request_id;
        ++prepared;
    }

    while (state.remaining < prepared) {
        const auto flushed = SubmitRing(handle_.uring);
        if (flushed < 0) {
            CleanupFailedSubmit(request_id, prepared, state.remaining);
            throw std::runtime_error("io_uring_submit failed");
        }
        if (flushed == 0) {
            CleanupFailedSubmit(request_id, prepared, state.remaining);
            throw std::runtime_error("io_uring_submit made no progress");
        }
        state.remaining += static_cast<size_t>(flushed);
    }

    return request_id;
#endif
}

IOCompletionReader::Completion
IOCompletionReader::WaitCompleted() {
#ifndef WITH_IO_URING
    throw std::runtime_error("IOCompletionReader requires io_uring support");
#else
    if (!ready_completions_.empty()) {
        auto completion = ready_completions_.front();
        ready_completions_.pop_front();
        return completion;
    }
    if (!IsReady()) {
        throw std::runtime_error("IOCompletionReader is not ready");
    }
    if (pending_requests_.empty()) {
        throw std::runtime_error("IOCompletionReader has no pending requests");
    }

    while (ready_completions_.empty()) {
        WaitOneCompletion();
    }

    auto completion = ready_completions_.front();
    ready_completions_.pop_front();
    return completion;
#endif
}

std::vector<IOCompletionReader::Completion>
IOCompletionReader::PollCompleted() {
#ifndef WITH_IO_URING
    throw std::runtime_error("IOCompletionReader requires io_uring support");
#else
    if (IsReady()) {
        ProcessAvailableCompletions();
    } else if (ready_completions_.empty()) {
        throw std::runtime_error("IOCompletionReader is not ready");
    }

    std::vector<Completion> completed;
    while (!ready_completions_.empty()) {
        completed.push_back(ready_completions_.front());
        ready_completions_.pop_front();
    }
    return completed;
#endif
}

bool
IOCompletionReader::IsReady() const {
#ifdef WITH_IO_URING
    return fd_ >= 0 && io_pool_ != nullptr && io_pool_->Backend() == IOBackend::IO_URING &&
           handle_.backend == IOBackend::IO_URING && handle_.uring != nullptr;
#else
    return false;
#endif
}

std::optional<std::string>
IOCompletionReader::ProcessCqe(struct io_uring_cqe* cqe) {
#ifdef WITH_IO_URING
    const auto request_id = static_cast<RequestId>(cqe->user_data);
    auto iter = pending_requests_.find(request_id);
    if (iter == pending_requests_.end()) {
        return std::string("unknown io_uring completion request id");
    }

    auto& state = iter->second;
    if (cqe->res < 0 || static_cast<size_t>(cqe->res) != state.expected_size) {
        state.ok = false;
    }

    if (state.remaining > 0) {
        --state.remaining;
    }

    if (state.remaining == 0) {
        ready_completions_.push_back({request_id, state.ok});
        pending_requests_.erase(iter);
    }
    return std::nullopt;
#else
    (void)cqe;
    return std::nullopt;
#endif
}

void
IOCompletionReader::WaitOneCompletion() {
#ifdef WITH_IO_URING
    size_t retry = 0;
    while (true) {
        io_uring_cqe* cqe = nullptr;
        const auto ret = io_uring_wait_cqe(handle_.uring, &cqe);
        if (ret < 0) {
            if (-ret == EINTR) {
                if (!knowhere_internal::ShouldRetryInterruptedSyscall(retry, kNumRetries)) {
                    throw std::runtime_error("io_uring_wait_cqe interrupted too many times");
                }
                continue;
            }
            throw std::runtime_error("io_uring_wait_cqe failed");
        }
        if (cqe == nullptr) {
            continue;
        }

        auto error = ProcessCqe(cqe);
        io_uring_cqe_seen(handle_.uring, cqe);
        if (error) {
            throw std::runtime_error(*error);
        }
        return;
    }
#else
    throw std::runtime_error("IOCompletionReader requires io_uring support");
#endif
}

void
IOCompletionReader::ProcessAvailableCompletions() {
#ifdef WITH_IO_URING
#ifdef ENABLE_SYNCPOINT
    bool skip = false;
    TEST_SYNC_POINT_CALLBACK("IOCompletionReader::ProcessAvailableCompletions:Skip", &skip);
    if (skip) {
        return;
    }
#endif
    size_t retry = 0;
    while (true) {
        io_uring_cqe* cqe = nullptr;
        const auto ret = io_uring_peek_cqe(handle_.uring, &cqe);
        if (ret == -EAGAIN) {
            break;
        }
        if (ret < 0) {
            if (-ret == EINTR) {
                if (!knowhere_internal::ShouldRetryInterruptedSyscall(retry, kNumRetries)) {
                    throw std::runtime_error("io_uring_peek_cqe interrupted too many times");
                }
                continue;
            }
            throw std::runtime_error("io_uring_peek_cqe failed");
        }
        if (cqe == nullptr) {
            break;
        }

        auto error = ProcessCqe(cqe);
        io_uring_cqe_seen(handle_.uring, cqe);
        if (error) {
            throw std::runtime_error(*error);
        }
        retry = 0;
    }
#endif
}

size_t
IOCompletionReader::PendingOperationCount() const {
    size_t count = 0;
    for (const auto& pending : pending_requests_) {
        count += pending.second.remaining;
    }
    return count;
}

void
IOCompletionReader::DrainOutstandingNoThrow() noexcept {
#ifdef WITH_IO_URING
    if (!IsReady()) {
        ready_completions_.clear();
        pending_requests_.clear();
        return;
    }

    bool drained = true;
    size_t retry = 0;
    while (!pending_requests_.empty()) {
        try {
            io_uring_cqe* cqe = nullptr;
            const auto ret = io_uring_wait_cqe(handle_.uring, &cqe);
            if (ret < 0) {
                if (-ret == EINTR) {
                    if (!knowhere_internal::ShouldRetryInterruptedSyscall(retry, kNumRetries)) {
                        drained = false;
                        break;
                    }
                    continue;
                }
                drained = false;
                break;
            }
            if (cqe == nullptr) {
                continue;
            }

            auto error = ProcessCqe(cqe);
            io_uring_cqe_seen(handle_.uring, cqe);
            if (error) {
                drained = false;
                break;
            }
        } catch (...) {
            drained = false;
            break;
        }
    }
    ready_completions_.clear();
    pending_requests_.clear();
    if (!drained) {
        ResetHandleUring();
    }
#endif
}

void
IOCompletionReader::DrainOutstanding() {
#ifdef WITH_IO_URING
    while (!pending_requests_.empty()) {
        WaitCompleted();
    }
    ready_completions_.clear();
#endif
}

void
IOCompletionReader::DrainOutstanding(RequestId request_id) {
#ifdef WITH_IO_URING
    while (pending_requests_.find(request_id) != pending_requests_.end()) {
        WaitOneCompletion();
    }
#endif
}

void
IOCompletionReader::CleanupFailedSubmit(RequestId request_id, size_t prepared, size_t submitted) {
#ifdef WITH_IO_URING
    if (prepared > submitted) {
        FailPendingRequests(request_id);
        ResetHandleUring();
        return;
    }

    if (submitted > 0) {
        try {
            DrainOutstanding(request_id);
            RemoveReadyCompletion(request_id);
        } catch (...) {
            FailPendingRequests(request_id);
            ResetHandleUring();
        }
    } else {
        pending_requests_.erase(request_id);
    }
#else
    (void)request_id;
    (void)prepared;
    (void)submitted;
#endif
}

void
IOCompletionReader::FailPendingRequests(RequestId excluded_request_id) {
    RemoveReadyCompletion(excluded_request_id);
    for (const auto& pending : pending_requests_) {
        if (pending.first != excluded_request_id) {
            ready_completions_.push_back({pending.first, false});
        }
    }
    pending_requests_.clear();
}

void
IOCompletionReader::RemoveReadyCompletion(RequestId request_id) {
    auto new_end = std::remove_if(ready_completions_.begin(), ready_completions_.end(),
                                  [request_id](const auto& completion) { return completion.request_id == request_id; });
    ready_completions_.erase(new_end, ready_completions_.end());
}

bool
IOCompletionReader::ResetHandleUring() {
#ifdef WITH_IO_URING
    if (io_pool_ == nullptr || handle_.backend != IOBackend::IO_URING || handle_.uring == nullptr) {
        handle_ = IOContextHandle{};
        return false;
    }

    auto handle = handle_;
    handle_ = IOContextHandle{};
    if (io_pool_->ResetUring(handle.uring)) {
        io_pool_->Push(handle);
        return true;
    }
#endif
    return false;
}

void
IOCompletionReader::ReleaseHandle() {
    if (io_pool_ != nullptr && handle_.backend != IOBackend::UNKNOWN) {
        io_pool_->Push(handle_);
    }
    handle_ = IOContextHandle{};
}
