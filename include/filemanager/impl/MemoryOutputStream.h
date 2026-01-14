#pragma once

#include <unistd.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

#include "filemanager/OutputStream.h"

namespace milvus {

class MemoryOutputStream : public OutputStream {
 public:
    MemoryOutputStream() : capacity_(kInitCapacity), data_(std::make_unique<uint8_t[]>(kInitCapacity)) {
    }

    [[nodiscard]] size_t
    Tell() const override {
        return offset_;
    }

    size_t
    Write(const void* buffer, size_t size) override {
        TryExpand(size);
        size_t write_size = std::min(size, capacity_ - offset_);
        memcpy(data_.get() + offset_, buffer, write_size);
        offset_ += write_size;
        return write_size;
    }

    std::pair<uint8_t*, int64_t>
    Release() {
        return {data_.release(), offset_};
    }

    size_t
    Write(int fd, size_t size) override {
        TryExpand(size);
        size_t write_size = std::min(size, capacity_ - offset_);
        read(fd, data_.get() + offset_, write_size);
        assert(bytes_read >= 0 && static_cast<size_t>(bytes_read) == write_size);
        offset_ += write_size;
        return write_size;
    }

    void
    Close() override {
        // do nothing
    }

 private:
    void
    TryExpand(size_t size) {
        if (offset_ + size > capacity_) {
            capacity_ = static_cast<size_t>(kExpandRatio * std::max(static_cast<size_t>(capacity_), offset_ + size));
            std::unique_ptr<uint8_t[]> new_data = std::make_unique<uint8_t[]>(capacity_);
            std::memcpy(new_data.get(), data_.get(), offset_);
            data_.swap(new_data);
        }
    }

    size_t offset_{};
    size_t capacity_;
    std::unique_ptr<uint8_t[]> data_;
    static constexpr float kExpandRatio = 1.5;
    static constexpr size_t kInitCapacity = 2 * 1024 * 1024;
};

}  // namespace milvus
