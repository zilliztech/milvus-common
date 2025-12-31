#pragma once

#include <cstddef>

namespace milvus {

class SpanBytes {
public:
    SpanBytes(void* data, size_t size) : data_(data), size_(size) {}
    
    void* data() const { return data_; }
    size_t size() const { return size_; }

private:
    void* data_;
    size_t size_;
};

} // namespace milvus