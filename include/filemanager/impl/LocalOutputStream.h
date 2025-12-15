#pragma once

#include <fstream>

#include "filemanager/OutputStream.h"

namespace milvus {

class LocalOutputStream : public OutputStream {
 public:
    LocalOutputStream(const std::string& filename);
    ~LocalOutputStream() override;

    size_t
    Tell() const override;

    size_t
    Write(const void* ptr, size_t size) override;

    template <typename T>
    size_t
    Write(T& value) {
        return Write(&value, sizeof(T));
    }

    size_t
    Write(int fd, size_t size) override;

    void
    Close() override;

 private:
    mutable std::ofstream stream_;
    std::string filename_;
};

}  // namespace milvus
