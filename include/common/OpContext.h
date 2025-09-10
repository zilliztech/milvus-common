// Copyright (C) 2019-2020 Zilliz. All rights reserved.
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
#include <atomic>

namespace milvus {

/**
 * @brief Operation context for tracking operation-specific metadata and resources.
 *
 * OpContext provides a unified container for operation-related information including
 * resource usage tracking, performance metrics, and extensible metadata. It is designed
 * to be passed through operation call chains to maintain context and enable observability.
 */
struct OpContext {
    // Storage Usage tracking
    struct {
        std::atomic<int64_t> cold_bytes{0};
        std::atomic<int64_t> used_bytes{0};
    } storage_usage;

    // TODO: OpenTelemetry Tracing integration

    OpContext() = default;
    ~OpContext() = default;
    OpContext(const OpContext&) = delete;
    OpContext&
    operator=(const OpContext&) = delete;
    OpContext(OpContext&&) = delete;
    OpContext&
    operator=(OpContext&&) = delete;
};

}  // namespace milvus
