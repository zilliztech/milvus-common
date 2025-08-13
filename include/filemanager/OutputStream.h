// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#pragma once

#include <string>

namespace milvus {
class OutputStream {
 public:
    virtual ~OutputStream() = default;

    /**
     * @brief get the current position in the stream
     *
     * @return
     */
    virtual size_t
    Tell() const = 0;

    /**
     * @brief writes a specified number of bytes from the stream into ptr
     *
     * @param ptr
     * @param size
     * @return
     */

    virtual size_t
    Write(const void* ptr, size_t size) = 0;

    /**
     * @brief read data from the stream to a object with given type
     *
     * @param
     * @return
     */
    template <typename T>
    size_t
    Write(const T& value) {
        return Write(&value, sizeof(T));
    }

    /**
     * @brief write data from a file to the stream
     *
     * @param file
     * @param size
     * @return the number of bytes written
     */

    virtual size_t
    Write(int fd, size_t size) = 0;
};
}  // namespace milvus
