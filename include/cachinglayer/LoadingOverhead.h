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

namespace milvus::cachinglayer {

struct LoadingOverheadDimensionConfig {
    int64_t upper_bound;
    std::string group;
};

// A missing dimension is not group-capped. Its loading overhead passes through
// unchanged and is still checked against the DList resource limit.
struct LoadingOverheadConfig {
    std::optional<LoadingOverheadDimensionConfig> memory;
    std::optional<LoadingOverheadDimensionConfig> file;
};

}  // namespace milvus::cachinglayer
