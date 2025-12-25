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

#include <unordered_set>

#include "filemanager/FileManager.h"
#include "filemanager/InputStream.h"
#include "filemanager/OutputStream.h"

namespace milvus {
/**
 * @brief LocalFileManager is used for placeholder purpose. It will not do anything to the file on disk.
 *
 * This class is not thread-safe.
 */
class LocalFileManager : public FileManager {
 public:
    auto
    LoadFile(const std::string& filename) -> bool override;

    auto
    AddFile(const std::string& filename) -> bool override;

    auto
    IsExisted(const std::string& filename) -> std::optional<bool> override;

    auto
    RemoveFile(const std::string& filename) -> bool override;

    auto
    OpenInputStream(const std::string& filename) -> std::shared_ptr<InputStream> override;

    auto
    OpenOutputStream(const std::string& filename) -> std::shared_ptr<OutputStream> override;

    ~LocalFileManager() override = default;

 private:
    std::unordered_set<std::string> files;
};

}  // namespace milvus
