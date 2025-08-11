#include "filemanager/impl/LocalOutputStream.h"
#include <unistd.h>
#include <algorithm>
#include <vector>

namespace milvus {

LocalOutputStream::LocalOutputStream(const std::string& filename) : filename_(filename) {
    stream_.open(filename_, std::ios::binary | std::ios::out);
    if (!stream_.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename_);
    }
}

LocalOutputStream::~LocalOutputStream() {
    stream_.close();
}

size_t
LocalOutputStream::Tell() const {
    return stream_.tellp();
}

size_t
LocalOutputStream::Write(const void* ptr, size_t size) {
    stream_.write(static_cast<const char*>(ptr), size);
    return size;
}

size_t
LocalOutputStream::Write(int fd, size_t size) {
    size_t buffer_size = std::min(size, static_cast<size_t>(4096));
    std::vector<char> buffer(buffer_size);

    size_t remain_size = size;
    size_t total_write_size = 0;

    while (remain_size > 0) {
        size_t write_size = std::min(remain_size, buffer_size);
        size_t read_size = ::read(fd, buffer.data(), write_size);
        if (read_size != write_size) {
            throw std::runtime_error("Read from fd " + std::to_string(fd) + " failed");
        }
        stream_.write(buffer.data(), write_size);
        total_write_size += write_size;
        remain_size -= write_size;
    }

    return total_write_size;
}

}  // namespace milvus
