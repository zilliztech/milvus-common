#pragma once

#include <cstddef>

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

    bool
    empty() const {
        return size_ == 0;
    }

    T*
    data() const {
        return data_;
    }

    T*
    begin() const {
        return data_;
    }

    T*
    end() const {
        return data_ + size_;
    }

 private:
    T* data_;
    size_t size_;
};
}  // namespace knowhere_compat

