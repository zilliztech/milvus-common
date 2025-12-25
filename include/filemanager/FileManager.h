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

#include <memory>
#include <optional>
#include <string>

#include "filemanager/InputStream.h"
#include "filemanager/OutputStream.h"

namespace milvus {

struct FileMeta {
    std::string file_path;
    size_t file_size;
};

/**
 * @brief This FileManager is used to manage file, including its replication, backup, ect.
 * It will act as a cloud-like client, and Knowhere need to call load/add to better support
 * distribution of the whole service.
 */
class FileManager {
 public:
    virtual ~FileManager() = default;
    /**
     * @brief Load a file to the local disk, so we can use stl lib to operate it.
     *
     * @param filename
     * @return false if any error, or return true.
     */
    virtual bool
    LoadFile(const std::string& filename) = 0;

    /**
     * @brief Add file to FileManager to manipulate it.
     *
     * @param filename
     * @return false if any error, or return true.
     */
    virtual bool
    AddFile(const std::string& filename) = 0;

    /**
     * @brief Check if a file exists.
     *
     * @param filename
     * @return std::nullopt if any error, or return if the file exists.
     */
    virtual std::optional<bool>
    IsExisted(const std::string& filename) = 0;

    /**
     * @brief Delete a file from FileManager.
     *
     * @param filename
     * @return false if any error, or return true.
     */
    virtual bool
    RemoveFile(const std::string& filename) = 0;

    /**
     * @brief Open a file as an input stream.
     *
     * @param filename
     * @return a shared pointer to the input stream.
     */
    virtual std::shared_ptr<InputStream>
    OpenInputStream(const std::string& filename) = 0;

    /**
     * @brief Open a file as an output stream.
     *
     * @param filename
     * @return a shared pointer to the output stream.
     */
    virtual std::shared_ptr<OutputStream>
    OpenOutputStream(const std::string& filename) = 0;
};

}  // namespace milvus
