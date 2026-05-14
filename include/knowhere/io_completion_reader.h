#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "knowhere/io_context_pool.h"

// Worker-local, single-threaded completion reader. Not thread-safe.
class IOCompletionReader {
 public:
    using RequestId = uint64_t;

    struct Completion {
        RequestId request_id = 0;
        bool ok = false;
    };

    IOCompletionReader(int fd, std::shared_ptr<IOContextPool> io_pool);

    IOCompletionReader(const IOCompletionReader&) = delete;
    IOCompletionReader&
    operator=(const IOCompletionReader&) = delete;

    IOCompletionReader(IOCompletionReader&& other) noexcept;
    IOCompletionReader&
    operator=(IOCompletionReader&& other);

    ~IOCompletionReader();

    RequestId
    Submit(std::span<std::byte* const> buffers, size_t size, std::span<const size_t> offsets);

    Completion
    WaitCompleted();

    std::vector<Completion>
    PollCompleted();

    bool
    IsReady() const;

 private:
    struct RequestState {
        size_t remaining = 0;
        size_t expected_size = 0;
        bool ok = true;
    };

    std::optional<std::string>
    ProcessCqe(struct io_uring_cqe* cqe);

    void
    DrainOutstandingNoThrow() noexcept;

    void
    DrainOutstanding();

    void
    DrainOutstanding(RequestId request_id);

    void
    CleanupFailedSubmit(RequestId request_id, size_t prepared, size_t submitted);

    void
    RemoveReadyCompletion(RequestId request_id);

    bool
    ResetHandleUring();

    void
    ReleaseHandle();

    int fd_ = -1;
    std::shared_ptr<IOContextPool> io_pool_;
    IOContextHandle handle_;
    RequestId next_request_id_ = 1;
    std::unordered_map<RequestId, RequestState> pending_requests_;
    std::deque<Completion> ready_completions_;
};
