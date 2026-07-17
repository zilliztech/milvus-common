// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "cachinglayer/Utils.h"

namespace milvus::cachinglayer {

struct LoadingOverheadDimensionConfig {
    int64_t upper_bound;
    std::string group;
};

// A missing dimension is not group-capped. Its loading overhead passes through
// unchanged and is still checked against the DList resource limit.
struct LoadingOverheadConfig {
    LoadingOverheadConfig() = default;

    LoadingOverheadConfig(std::optional<LoadingOverheadDimensionConfig> memory,
                          std::optional<LoadingOverheadDimensionConfig> file)
        : memory(std::move(memory)), file(std::move(file)) {
    }

    // Compatibility entry point for callers using the original shared-group config.
    LoadingOverheadConfig(const ResourceUsage& upper_bound, std::string group)
        : memory(LoadingOverheadDimensionConfig{upper_bound.memory_bytes, group}),
          file(LoadingOverheadDimensionConfig{upper_bound.file_bytes, std::move(group)}) {
    }

    std::optional<LoadingOverheadDimensionConfig> memory;
    std::optional<LoadingOverheadDimensionConfig> file;
};

}  // namespace milvus::cachinglayer
