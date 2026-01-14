#pragma once

#include <unistd.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "filemanager/InputStream.h"

namespace milvus {

class MemoryInputStream : public InputStream {
 public:
    MemoryInputStream(const uint8_t* data, size_t size) : size_(size), data_(data) {
    }

    [[nodiscard]] size_t
    Size() const override {
        return size_;
    }

    bool
    Seek(int64_t offset) override {
        if (offset < 0 || offset >= static_cast<int64_t>(size_)) {
            return false;
        }
        offset_ = offset;
        return true;
    }

    [[nodiscard]] size_t
    Tell() const override {
        return offset_;
    }

    [[nodiscard]] bool
    Eof() const override {
        return offset_ >= size_;
    }

    size_t
    Read(void* buffer, size_t size) override {
        if (size_ == 0) {
            return 0;
        }
        assert(offset_ + size <= size_);
        size_t read_size = std::min(size, size_ - offset_);
        memcpy(buffer, data_ + offset_, read_size);
        offset_ += read_size;
        return read_size;
    }

    size_t
    ReadAt(void* buffer, size_t offset, size_t size) override {
        if (size_ == 0) {
            return 0;
        }
        assert(offset + size <= size_);
        size_t read_size = std::min(size, size_ - offset);
        memcpy(buffer, data_ + offset, read_size);
        return read_size;
    }

    size_t
    Read(int fd, size_t size) override {
        if (size_ == 0) {
            return 0;
        }
        assert(offset_ + size <= size_);
        size_t read_size = std::min(size, size_ - offset_);
        write(fd, data_ + offset_, read_size);
        assert(written >= 0 && static_cast<size_t>(written) == read_size);
        ::fsync(fd);
        offset_ += read_size;
        return read_size;
    }

 private:
    size_t size_;
    size_t offset_ = 0;
    const uint8_t* data_ = nullptr;
};

}  // namespace milvus
