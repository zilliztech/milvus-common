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

#include <folly/futures/Future.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace milvus {
class InputStream {
 public:
    virtual ~InputStream() = default;

    /**
     * @brief get the total size of the stream
     *
     * @return
     */
    virtual size_t
    Size() const = 0;

    /**
     * @brief get the current position in the stream
     *
     * @return true if the seek is successful, false otherwise
     */
    virtual bool
    Seek(int64_t offset) = 0;

    /**
     * @brief get the current position in the stream
     *
     * @return the current position in the stream
     */
    virtual size_t
    Tell() const = 0;

    /**
     * @brief check if the end of the stream has been reached
     *
     * @return true if the end of the stream has been reached, false otherwise
     */
    virtual bool
    Eof() const = 0;

    /**
     * @brief reads a specified number of bytes from the stream into ptr
     *
     * @param ptr
     * @param size
     * @return the number of bytes read
     */

    virtual size_t
    Read(void* ptr, size_t size) = 0;

    /**
     * @brief reads a specified number of bytes from the stream into ptr at the given offset
     *
     * @note this function is thread safe
     * @param ptr
     * @param offset
     * @param size
     * @return the number of bytes read
     */

    virtual size_t
    ReadAt(void* ptr, size_t offset, size_t size) = 0;

    /**
     * @brief asynchronously reads bytes into ptr at the given offset
     *
     * The default implementation defers the synchronous ReadAt call until the
     * returned future is driven. Callers can attach an executor with via() to
     * keep the synchronous fallback off the current thread. Implementations
     * backed by native asynchronous IO should override this method.
     *
     * The stream and ptr must remain valid until the future completes.
     *
     * @param ptr
     * @param offset
     * @param size
     * @return a future containing the number of bytes read
     */
    virtual folly::SemiFuture<size_t>
    ReadAtAsync(void* ptr, size_t offset, size_t size) {
        return folly::makeSemiFuture().deferValue(
            [this, ptr, offset, size](folly::Unit) { return ReadAt(ptr, offset, size); });
    }

    /**
     * @brief read data from the stream to a object with given type
     *
     * @param value
     * @return the number of bytes read
     */
    template <typename T>
    size_t
    Read(T& value) {
        return Read(&value, sizeof(T));
    }

    /**
     * @brief read data from the stream to a file
     *
     * @param file
     * @param size
     * @return the number of bytes read
     */
    virtual size_t
    Read(int fd, size_t size) = 0;
};
}  // namespace milvus
