// Copyright (C) 2019-2025 Zilliz. All rights reserved.
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

#include <folly/futures/Future.h>
#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>

#include <atomic>
#include <cstdint>

#include "common/CommonMonitor.h"
#include "common/EasyAssert.h"
#include "folly/executors/InlineExecutor.h"

namespace milvus::cachinglayer {

using uid_t = int64_t;
using cid_t = int64_t;

enum class StorageType {
    MEMORY,
    DISK,
    MIXED,
};

enum class CellDataType {
    VECTOR_FIELD,
    VECTOR_INDEX,
    SCALAR_FIELD,
    SCALAR_INDEX,
    OTHER,  // e.g. InsertRecord and DeleteRecord
};

enum class CellIdMappingMode : uint8_t {
    CUSTOMIZED = 0,   // the cell id should be parsed from the uid by the translator
    IDENTICAL = 1,    // the cell id is identical to the uid
    ALWAYS_ZERO = 2,  // the cell id is always 0
};

// TODO(tiered storage 4): this is a temporary function to get the result of a future
// by running it on the inline executor. We don't need this once we are fully async.
template <typename T>
T
SemiInlineGet(folly::SemiFuture<T>&& future) {
    return std::move(future).via(&folly::InlineExecutor::instance()).get();
}

inline std::string
FormatBytes(int64_t bytes) {
    constexpr int64_t kUnlimitedThreshold = std::numeric_limits<int64_t>::max() / 100;
    if (bytes >= kUnlimitedThreshold) {
        return "UNSET";
    }
    if (bytes < 1024) {
        return fmt::format("{} B", bytes);
    } else if (bytes < 1024 * 1024) {
        return fmt::format("{:.2f} KB ({} B)", bytes / 1024.0, bytes);
    } else if (bytes < 1024 * 1024 * 1024) {
        return fmt::format("{:.2f} MB ({} B)", bytes / (1024.0 * 1024.0), bytes);
    } else {
        return fmt::format("{:.2f} GB ({} B)", bytes / (1024.0 * 1024.0 * 1024.0), bytes);
    }
}

struct ResourceUsage {
    int64_t memory_bytes{0};
    int64_t file_bytes{0};

    ResourceUsage() noexcept = default;
    ResourceUsage(int64_t mem, int64_t file) noexcept : memory_bytes(mem), file_bytes(file) {
    }

    ResourceUsage
    operator+(const ResourceUsage& rhs) const {
        return {memory_bytes + rhs.memory_bytes, file_bytes + rhs.file_bytes};
    }

    void
    operator+=(const ResourceUsage& rhs) {
        memory_bytes += rhs.memory_bytes;
        file_bytes += rhs.file_bytes;
    }

    ResourceUsage
    operator-(const ResourceUsage& rhs) const {
        return {memory_bytes - rhs.memory_bytes, file_bytes - rhs.file_bytes};
    }

    ResourceUsage
    operator*(double factor) const {
        return {static_cast<int64_t>(std::round(memory_bytes * factor)),
                static_cast<int64_t>(std::round(file_bytes * factor))};
    }

    friend ResourceUsage
    operator*(double factor, const ResourceUsage& usage) {
        return usage * factor;
    }

    void
    operator-=(const ResourceUsage& rhs) {
        memory_bytes -= rhs.memory_bytes;
        file_bytes -= rhs.file_bytes;
    }

    bool
    operator==(const ResourceUsage& rhs) const {
        return memory_bytes == rhs.memory_bytes && file_bytes == rhs.file_bytes;
    }

    bool
    operator!=(const ResourceUsage& rhs) const {
        return !(*this == rhs);
    }

    [[nodiscard]] bool
    AnyGTZero() const {
        return memory_bytes > 0 || file_bytes > 0;
    }

    [[nodiscard]] bool
    AllGEZero() const {
        return memory_bytes >= 0 && file_bytes >= 0;
    }

    [[nodiscard]] bool
    CanHold(const ResourceUsage& rhs) const {
        return memory_bytes >= rhs.memory_bytes && file_bytes >= rhs.file_bytes;
    }

    [[nodiscard]] StorageType
    storage_type() const {
        if (memory_bytes > 0 && file_bytes > 0) {
            return StorageType::MIXED;
        }
        return memory_bytes > 0 ? StorageType::MEMORY : StorageType::DISK;
    }

    [[nodiscard]] std::string
    ToString() const {
        if (memory_bytes == 0 && file_bytes == 0) {
            return "EMPTY";
        }

        std::string result;
        if (memory_bytes > 0) {
            result += fmt::format("memory {}", FormatBytes(memory_bytes));
        } else if (memory_bytes < 0) {
            // it should never happen
            result += fmt::format("memory -{}", FormatBytes(-memory_bytes));
        }
        if (file_bytes > 0) {
            if (!result.empty()) {
                result += ", ";
            }
            result += fmt::format("disk {}", FormatBytes(file_bytes));
        } else if (file_bytes < 0) {
            // it should never happen
            if (!result.empty()) {
                result += ", ";
            }
            result += fmt::format("disk -{}", FormatBytes(-file_bytes));
        }
        return result;
    }
};

inline std::ostream&
operator<<(std::ostream& os, const ResourceUsage& usage) {
    os << usage.ToString();
    return os;
}

inline void
operator+=(std::atomic<ResourceUsage>& atomic_lhs, const ResourceUsage& rhs) {
    ResourceUsage current = atomic_lhs.load();
    ResourceUsage new_value;
    do {
        new_value = current;
        new_value += rhs;
    } while (!atomic_lhs.compare_exchange_weak(current, new_value));
}

inline void
operator-=(std::atomic<ResourceUsage>& atomic_lhs, const ResourceUsage& rhs) {
    ResourceUsage current = atomic_lhs.load();
    ResourceUsage new_value;
    do {
        new_value = current;
        new_value -= rhs;
    } while (!atomic_lhs.compare_exchange_weak(current, new_value));
}

// helper struct for ConfigureTieredStorage, so the list of arguments is not too long.
struct CacheWarmupPolicies {
    CacheWarmupPolicy scalarFieldCacheWarmupPolicy;
    CacheWarmupPolicy vectorFieldCacheWarmupPolicy;
    CacheWarmupPolicy scalarIndexCacheWarmupPolicy;
    CacheWarmupPolicy vectorIndexCacheWarmupPolicy;

    CacheWarmupPolicies()
        : scalarFieldCacheWarmupPolicy(CacheWarmupPolicy::CacheWarmupPolicy_Sync),
          vectorFieldCacheWarmupPolicy(CacheWarmupPolicy::CacheWarmupPolicy_Disable),
          scalarIndexCacheWarmupPolicy(CacheWarmupPolicy::CacheWarmupPolicy_Sync),
          vectorIndexCacheWarmupPolicy(CacheWarmupPolicy::CacheWarmupPolicy_Sync) {
    }

    CacheWarmupPolicies(CacheWarmupPolicy scalarFieldCacheWarmupPolicy, CacheWarmupPolicy vectorFieldCacheWarmupPolicy,
                        CacheWarmupPolicy scalarIndexCacheWarmupPolicy, CacheWarmupPolicy vectorIndexCacheWarmupPolicy)
        : scalarFieldCacheWarmupPolicy(scalarFieldCacheWarmupPolicy),
          vectorFieldCacheWarmupPolicy(vectorFieldCacheWarmupPolicy),
          scalarIndexCacheWarmupPolicy(scalarIndexCacheWarmupPolicy),
          vectorIndexCacheWarmupPolicy(vectorIndexCacheWarmupPolicy) {
    }

    [[nodiscard]] std::string
    ToString() const {
        auto policyToString = [](CacheWarmupPolicy policy) -> std::string {
            switch (policy) {
                case CacheWarmupPolicy::CacheWarmupPolicy_Sync:
                    return "Sync";
                case CacheWarmupPolicy::CacheWarmupPolicy_Async:
                    return "Async";
                case CacheWarmupPolicy::CacheWarmupPolicy_Disable:
                    return "Disable";
                default:
                    return "Unknown";
            }
        };
        return fmt::format(
            "warmup policies: scalarField: {}, vectorField: {}, "
            "scalarIndex: {}, vectorIndex: {}",
            policyToString(scalarFieldCacheWarmupPolicy), policyToString(vectorFieldCacheWarmupPolicy),
            policyToString(scalarIndexCacheWarmupPolicy), policyToString(vectorIndexCacheWarmupPolicy));
    }
};

struct CacheLimit {
    int64_t memory_low_watermark_bytes;
    int64_t memory_high_watermark_bytes;
    int64_t memory_max_bytes;
    int64_t disk_low_watermark_bytes;
    int64_t disk_high_watermark_bytes;
    int64_t disk_max_bytes;
    CacheLimit()
        : memory_low_watermark_bytes(0),
          memory_high_watermark_bytes(0),
          memory_max_bytes(0),
          disk_low_watermark_bytes(0),
          disk_high_watermark_bytes(0),
          disk_max_bytes(0) {
    }

    CacheLimit(int64_t memory_low_watermark_bytes, int64_t memory_high_watermark_bytes, int64_t memory_max_bytes,
               int64_t disk_low_watermark_bytes, int64_t disk_high_watermark_bytes, int64_t disk_max_bytes)
        : memory_low_watermark_bytes(memory_low_watermark_bytes),
          memory_high_watermark_bytes(memory_high_watermark_bytes),
          memory_max_bytes(memory_max_bytes),
          disk_low_watermark_bytes(disk_low_watermark_bytes),
          disk_high_watermark_bytes(disk_high_watermark_bytes),
          disk_max_bytes(disk_max_bytes) {
    }
};

struct EvictionConfig {
    // Touch a node means to move it to the head of the list, which requires locking the entire list.
    // Use cache_touch_window_ms to reduce the frequency of touching and reduce contention.
    std::chrono::milliseconds cache_touch_window;
    bool background_eviction_enabled;
    std::chrono::milliseconds eviction_interval;
    // Time after which an unaccessed cache cell will be evicted
    std::chrono::seconds cache_cell_unaccessed_survival_time;
    // Overloaded memory threshold percentage - limits cache memory usage to this percentage of total physical memory
    float overloaded_memory_threshold_percentage;
    // Max disk usage percentage - limits disk cache usage to this percentage of total disk space (not used yet)
    float max_disk_usage_percentage;
    std::string disk_path;
    // Loading resource factor to reserve more resources, preventing poor resource estimation during the loading
    // process.
    float loading_resource_factor;

    EvictionConfig()
        : cache_touch_window(std::chrono::milliseconds(0)),
          background_eviction_enabled(false),
          eviction_interval(std::chrono::milliseconds(0)),
          cache_cell_unaccessed_survival_time(std::chrono::seconds(0)),
          overloaded_memory_threshold_percentage(0.9),
          max_disk_usage_percentage(0.95),
          disk_path(""),
          loading_resource_factor(1.0f) {
    }

    EvictionConfig(int64_t cache_touch_window_ms, bool background_eviction_enabled, int64_t eviction_interval_ms)
        : cache_touch_window(std::chrono::milliseconds(cache_touch_window_ms)),
          background_eviction_enabled(background_eviction_enabled),
          eviction_interval(std::chrono::milliseconds(eviction_interval_ms)),
          cache_cell_unaccessed_survival_time(std::chrono::seconds(0)),
          overloaded_memory_threshold_percentage(0.9),
          max_disk_usage_percentage(0.95),
          disk_path(""),
          loading_resource_factor(1.0f) {
    }

    EvictionConfig(int64_t cache_touch_window_ms, bool background_eviction_enabled, int64_t eviction_interval_ms,
                   int64_t cache_cell_unaccessed_survival_time, float overloaded_memory_threshold_percentage,
                   float max_disk_usage_percentage, const std::string& disk_path, float loading_resource_factor)
        : cache_touch_window(std::chrono::milliseconds(cache_touch_window_ms)),
          background_eviction_enabled(background_eviction_enabled),
          eviction_interval(std::chrono::milliseconds(eviction_interval_ms)),
          cache_cell_unaccessed_survival_time(std::chrono::seconds(cache_cell_unaccessed_survival_time)),
          overloaded_memory_threshold_percentage(overloaded_memory_threshold_percentage),
          max_disk_usage_percentage(max_disk_usage_percentage),
          disk_path(disk_path),
          loading_resource_factor(loading_resource_factor) {
    }
};

namespace internal {

struct SystemResourceInfo {
    int64_t total_bytes{0};
    int64_t used_bytes{0};
};

int64_t
getHostTotalMemory();
int64_t
getContainerMemLimit();

// Returns unlimited if failed to get memory info, or if the platform is not supported.
SystemResourceInfo
getSystemMemoryInfo();

// Returns unlimited if failed to get disk info, or if the platform is not supported.
SystemResourceInfo
getSystemDiskInfo(const std::string& disk_path);

// Returns 0 if failed to get memory usage, or if the platform is not supported.
int64_t
getCurrentProcessMemoryUsage();

}  // namespace internal

}  // namespace milvus::cachinglayer
