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
#include <folly/CancellationToken.h>

#include <atomic>
#include <memory>

namespace milvus {

/**
 * @brief Storage usage tracking for operations.
 *
 * StorageUsage tracks the bytes scanned during an operation, separated into
 * cold (not cached) and total bytes. This struct can be shared across multiple
 * OpContext objects to aggregate usage across related operations.
 */
struct StorageUsage {
    std::atomic<int64_t> scanned_cold_bytes{0};
    std::atomic<int64_t> scanned_total_bytes{0};
};

/**
 * @brief Operation context for tracking operation-specific metadata and resources.
 *
 * OpContext provides a unified container for operation-related information including
 * resource usage tracking, performance metrics, and extensible metadata. It is designed
 * to be passed through operation call chains to maintain context and enable observability.
 */
struct OpContext {
    // Storage Usage tracking (shared_ptr allows sharing across multiple OpContext objects)
    std::shared_ptr<StorageUsage> storage_usage;

    folly::CancellationToken cancellation_token;

    OpContext() : storage_usage(std::make_shared<StorageUsage>()) {
    }
    OpContext(const folly::CancellationToken& cancellation_token)
        : storage_usage(std::make_shared<StorageUsage>()), cancellation_token(cancellation_token) {
    }
    OpContext(std::shared_ptr<StorageUsage> storage_usage) : storage_usage(std::move(storage_usage)) {
    }
    OpContext(std::shared_ptr<StorageUsage> storage_usage, const folly::CancellationToken& cancellation_token)
        : storage_usage(std::move(storage_usage)), cancellation_token(cancellation_token) {
    }
    ~OpContext() = default;
    OpContext(const OpContext&) = delete;
    OpContext&
    operator=(const OpContext&) = delete;
    OpContext(OpContext&&) = delete;
    OpContext&
    operator=(OpContext&&) = delete;
};

}  // namespace milvus
