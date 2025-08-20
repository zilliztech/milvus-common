#include "filemanager/impl/LocalInputStream.h"

#include <unistd.h>

#include <algorithm>
#include <vector>

namespace milvus {

LocalInputStream::LocalInputStream(const std::string& filename) : filename_(filename) {
    stream_.open(filename_, std::ios::binary | std::ios::in);
    if (!stream_.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename_);
    }

    stream_.seekg(0, std::ios::end);
    size_ = static_cast<size_t>(stream_.tellg());
    stream_.seekg(0, std::ios::beg);
}

LocalInputStream::~LocalInputStream() {
    stream_.close();
}

bool
LocalInputStream::Seek(int64_t offset) {
    stream_.seekg(offset);
    return true;
}

size_t
LocalInputStream::Size() const {
    return size_;
}

size_t
LocalInputStream::Tell() const {
    return stream_.tellg();
}

bool
LocalInputStream::Eof() const {
    return stream_.eof();
}

size_t
LocalInputStream::Read(void* ptr, size_t size) {
    size_t cur = stream_.tellg();
    if (cur + size > size_) {
        throw std::runtime_error("Read out of range");
    }
    stream_.read(static_cast<char*>(ptr), size);
    return stream_.gcount();
}

size_t
LocalInputStream::ReadAt(void* ptr, size_t offset, size_t size) {
    size_t cur = stream_.tellg();
    if (cur + size > size_) {
        throw std::runtime_error("Read out of range");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    stream_.seekg(offset);
    stream_.read(static_cast<char*>(ptr), size);
    return stream_.gcount();
}

size_t
LocalInputStream::Read(int fd, size_t size) {
    size_t cur = stream_.tellg();
    if (cur + size > size_) {
        throw std::runtime_error("Read out of range");
    }

    size_t buffer_size = std::min(size, static_cast<size_t>(4096));
    std::vector<char> buffer(buffer_size);

    size_t remain_size = size;
    size_t total_read_size = 0;

    while (remain_size > 0) {
        size_t write_size = std::min(remain_size, buffer_size);
        size_t read_size = stream_.read(buffer.data(), write_size).gcount();
        if (read_size != write_size) {
            throw std::runtime_error("Read from stream " + filename_ + " failed");
        }
        if (static_cast<size_t>(::write(fd, buffer.data(), read_size)) != read_size) {
            throw std::runtime_error("Write to fd " + std::to_string(fd) + " failed");
        }
        total_read_size += read_size;
        remain_size -= write_size;
    }
    ::fsync(fd);

    return total_read_size;
}

}  // namespace milvus
