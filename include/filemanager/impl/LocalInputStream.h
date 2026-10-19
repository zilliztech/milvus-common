#pragma once

#include <cstddef>
#include <fstream>
#include <mutex>

#include "filemanager/InputStream.h"

namespace milvus {

class LocalInputStream : public InputStream {
 public:
    LocalInputStream(const std::string& filename);
    ~LocalInputStream() override;

    size_t
    Size() const override;

    bool
    Seek(int64_t offset) override;

    size_t
    Tell() const override;

    size_t
    Read(void* ptr, size_t size) override;

    size_t
    ReadAt(void* ptr, size_t offset, size_t size) override;

    // For local validation only; this is not a performance-critical path.
    // Performs blocking IO inline and returns a ready future containing the result or exception.
    [[nodiscard]] folly::SemiFuture<size_t>
    ReadAtAsync(void* ptr, size_t offset, size_t size) override {
        return folly::makeSemiFutureWith([this, ptr, offset, size] { return ReadAt(ptr, offset, size); });
    }

    template <typename T>
    size_t
    Read(T& value) {
        return Read(&value, sizeof(T));
    }

    size_t
    Read(int fd, size_t size) override;

    bool
    Eof() const override;

 private:
    mutable std::ifstream stream_;
    size_t size_;
    std::string filename_;
    std::mutex mutex_;
};

}  // namespace milvus
