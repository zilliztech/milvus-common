#include "knowhere/io_reader.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "log/Log.h"

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
        return std::async(std::launch::deferred, [] { return true; });
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

    return std::async(
        std::launch::deferred,
        [fd = fd_, size, buffers = std::move(buffers), offsets = std::move(offsets), pool]() mutable -> bool {
            switch (pool->Backend()) {
#ifdef MILVUS_COMMON_WITH_LIBAIO
                case IOBackend::AIO: {
                    auto handle = pool->Pop();
                    auto ctx = handle.aio;
                    if (ctx == nullptr) {
                        return false;
                    }

                    const size_t max_batch = pool->MaxEventsPerCtx();
                    if (max_batch == 0) {
                        pool->Push(handle);
                        return false;
                    }

                    size_t pending = 0;
                    auto drain_pending = [&]() {
                        while (pending > 0) {
                            std::vector<struct io_event> drain_events(std::min(pending, max_batch));
                            const auto drained = io_getevents(ctx, 1, static_cast<long>(drain_events.size()),
                                                              drain_events.data(), nullptr);
                            if (drained <= 0) {
                                break;
                            }
                            pending -= static_cast<size_t>(drained);
                        }
                    };

                    bool ok = true;
                    size_t processed = 0;
                    while (ok && processed < buffers.size()) {
                        const size_t batch = std::min(max_batch, buffers.size() - processed);
                        std::vector<struct iocb> cbs(batch);
                        std::vector<struct iocb*> cb_ptrs(batch);
                        for (size_t i = 0; i < batch; ++i) {
                            const auto idx = processed + i;
                            io_prep_pread(&cbs[i], fd, reinterpret_cast<void*>(buffers[idx]), size, offsets[idx]);
                            cb_ptrs[i] = &cbs[i];
                        }

                        size_t submitted_total = 0;
                        while (submitted_total < batch) {
                            const auto submitted = io_submit(ctx, static_cast<long>(batch - submitted_total),
                                                             cb_ptrs.data() + submitted_total);
                            if (submitted <= 0) {
                                ok = false;
                                break;
                            }
                            submitted_total += static_cast<size_t>(submitted);
                            pending += static_cast<size_t>(submitted);
                        }

                        std::vector<struct io_event> events(batch);
                        size_t completed_total = 0;
                        while (ok && completed_total < submitted_total) {
                            const auto completed =
                                io_getevents(ctx, 1, static_cast<long>(submitted_total - completed_total),
                                             events.data() + completed_total, nullptr);
                            if (completed <= 0) {
                                ok = false;
                                break;
                            }
                            completed_total += static_cast<size_t>(completed);
                            pending -= static_cast<size_t>(completed);
                        }

                        for (size_t i = 0; ok && i < completed_total; ++i) {
                            if (events[i].res < 0 || static_cast<size_t>(events[i].res) != size) {
                                ok = false;
                            }
                        }

                        processed += batch;
                    }

                    if (!ok) {
                        drain_pending();
                        pool->Push(handle);
                        return false;
                    }

                    pool->Push(handle);
                    return true;
                }
#endif
#ifdef WITH_IO_URING
                case IOBackend::IO_URING: {
                    auto handle = pool->Pop();
                    auto* ring = handle.uring;
                    if (ring == nullptr) {
                        return false;
                    }

                    bool ok = true;
                    size_t processed = 0;
                    while (processed < buffers.size()) {
                        size_t batch = 0;
                        for (; processed + batch < buffers.size(); ++batch) {
                            auto* sqe = io_uring_get_sqe(ring);
                            if (sqe == nullptr) {
                                break;
                            }
                            const auto idx = processed + batch;
                            io_uring_prep_read(sqe, fd, reinterpret_cast<void*>(buffers[idx]), size, offsets[idx]);
                            sqe->user_data = idx;
                        }

                        if (batch == 0) {
                            ok = false;
                            break;
                        }

                        const auto submitted = io_uring_submit(ring);
                        if (submitted < 0 || static_cast<size_t>(submitted) != batch) {
                            ok = false;
                            break;
                        }

                        size_t completed = 0;
                        while (completed < batch) {
                            io_uring_cqe* cqe = nullptr;
                            if (io_uring_wait_cqe(ring, &cqe) < 0 || cqe == nullptr) {
                                ok = false;
                                break;
                            }
                            if (cqe->res < 0 || static_cast<size_t>(cqe->res) != size) {
                                io_uring_cqe_seen(ring, cqe);
                                ok = false;
                                break;
                            }
                            io_uring_cqe_seen(ring, cqe);
                            ++completed;
                        }

                        if (!ok) {
                            break;
                        }

                        processed += batch;
                    }

                    pool->Push(handle);
                    return ok;
                }
#endif
                default:
                    return false;
            }
        });
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
