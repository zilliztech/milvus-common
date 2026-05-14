#pragma once

#include <sys/types.h>

#include <cstddef>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace knowhere_compat {
template <typename T>
class span {
 public:
    span(T* data, size_t size) : data_(data), size_(size) {
    }

    T&
    operator[](size_t idx) const {
        return data_[idx];
    }

    size_t
    size() const {
        return size_;
    }

    T*
    data() const {
        return data_;
    }

 private:
    T* data_;
    size_t size_;
};
}  // namespace knowhere_compat

#include "knowhere/io_context_pool.h"

template <typename T>
using IOReaderSpan = knowhere_compat::span<T>;

class IOReader {
 public:
    IOReader();

    explicit IOReader(int fd);

    IOReader(int fd, std::shared_ptr<IOContextPool> io_pool);

    explicit IOReader(std::shared_ptr<IOContextPool> io_pool);

    bool
    Read(IOReaderSpan<std::byte*> buf, size_t size, IOReaderSpan<off_t> offsets) const;

    std::future<bool>
    ReadAsync(std::vector<std::byte*>&& buffers, size_t size, std::vector<size_t>&& offsets) const;

    IOBackend
    Backend() const;

    std::string
    BackendName() const;

    bool
    IsReady() const;

 private:
    int fd_ = -1;
    std::shared_ptr<IOContextPool> io_pool_;
};
