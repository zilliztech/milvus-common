#include "knowhere/io_completion_reader.h"

#include <cerrno>
#include <stdexcept>
#include <utility>

IOCompletionReader::IOCompletionReader(int fd, std::shared_ptr<IOContextPool> io_pool)
    : fd_(fd), io_pool_(std::move(io_pool)) {
    if (fd_ < 0) {
        throw std::invalid_argument("invalid file descriptor");
    }

    if (io_pool_ == nullptr || !io_pool_->IsInitialized()) {
        throw std::runtime_error("IOContextPool is not initialized");
    }

    if (io_pool_->Backend() != IOBackend::IO_URING) {
        throw std::runtime_error("IOCompletionReader requires io_uring backend");
    }

    handle_ = io_pool_->Pop();
    if (handle_.backend != IOBackend::IO_URING || handle_.uring == nullptr) {
        throw std::runtime_error("failed to acquire io_uring context handle");
    }
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
IOCompletionReader::operator=(IOCompletionReader&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    DrainOutstanding();
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
    DrainOutstanding();
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

    auto request_id = next_request_id_++;
    size_t submitted = 0;
    while (submitted < buffers.size()) {
        auto* sqe = io_uring_get_sqe(handle_.uring);
        if (sqe == nullptr) {
            const auto flushed = io_uring_submit(handle_.uring);
            if (flushed <= 0) {
                throw std::runtime_error("io_uring_submit failed while preparing request");
            }
            continue;
        }

        io_uring_prep_read(sqe, fd_, reinterpret_cast<void*>(buffers[submitted]), size, offsets[submitted]);
        sqe->user_data = request_id;
        ++submitted;
    }

    const auto final_submitted = io_uring_submit(handle_.uring);
    if (final_submitted <= 0) {
        throw std::runtime_error("io_uring_submit failed");
    }

    auto& state = pending_requests_[request_id];
    state.remaining = buffers.size();
    state.expected_size = size;
    state.ok = true;
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

    while (true) {
        io_uring_cqe* cqe = nullptr;
        const auto ret = io_uring_wait_cqe(handle_.uring, &cqe);
        if (ret < 0) {
            if (-ret == EINTR) {
                continue;
            }
            throw std::runtime_error("io_uring_wait_cqe failed");
        }
        if (cqe == nullptr) {
            continue;
        }

        ProcessCqe(cqe);
        io_uring_cqe_seen(handle_.uring, cqe);

        if (!ready_completions_.empty()) {
            auto completion = ready_completions_.front();
            ready_completions_.pop_front();
            return completion;
        }
    }
#endif
}

std::vector<IOCompletionReader::Completion>
IOCompletionReader::PollCompleted() {
#ifndef WITH_IO_URING
    return {};
#else
    while (true) {
        io_uring_cqe* cqe = nullptr;
        const auto ret = io_uring_peek_cqe(handle_.uring, &cqe);
        if (ret == -EAGAIN || cqe == nullptr) {
            break;
        }
        if (ret < 0) {
            if (-ret == EINTR) {
                continue;
            }
            throw std::runtime_error("io_uring_peek_cqe failed");
        }

        ProcessCqe(cqe);
        io_uring_cqe_seen(handle_.uring, cqe);
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

void
IOCompletionReader::ProcessCqe(struct io_uring_cqe* cqe) {
#ifdef WITH_IO_URING
    const auto request_id = static_cast<RequestId>(cqe->user_data);
    auto iter = pending_requests_.find(request_id);
    if (iter == pending_requests_.end()) {
        return;
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
#else
    (void)cqe;
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
IOCompletionReader::ReleaseHandle() {
    if (io_pool_ != nullptr && handle_.backend != IOBackend::UNKNOWN) {
        io_pool_->Push(handle_);
    }
    handle_ = IOContextHandle{};
}
