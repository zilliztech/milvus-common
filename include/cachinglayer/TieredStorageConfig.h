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

#include <chrono>
#include <shared_mutex>

#include "cachinglayer/Utils.h"

namespace milvus::cachinglayer {

// Runtime-updatable configuration for the caching layer.
// Fields here can be changed at runtime via UpdateAll() (called from CGO).
// Fields that are init-only (eviction_enabled, prefetch_pool, DList) live in Manager.
class TieredStorageConfig {
 public:
    // Consistent snapshot of all config fields, read under a single lock hold.
    struct Snapshot {
        bool storage_usage_tracking_enabled;
        std::chrono::milliseconds loading_timeout;
        std::chrono::milliseconds warmup_loading_timeout;
        CacheWarmupPolicies warmup_policies;
    };

    static TieredStorageConfig&
    GetInstance() {
        static TieredStorageConfig instance;
        return instance;
    }

    TieredStorageConfig(const TieredStorageConfig&) = delete;
    TieredStorageConfig&
    operator=(const TieredStorageConfig&) = delete;
    TieredStorageConfig(TieredStorageConfig&&) = delete;
    TieredStorageConfig&
    operator=(TieredStorageConfig&&) = delete;

    // --- Snapshot read (single lock hold) ---

    [[nodiscard]] Snapshot
    GetSnapshot() const {
        std::shared_lock lock(mtx_);
        return {storage_usage_tracking_enabled_, loading_timeout_, warmup_loading_timeout_, warmup_policies_};
    }

    // --- Individual readers (shared lock) ---

    [[nodiscard]] bool
    storage_usage_tracking_enabled() const {
        std::shared_lock lock(mtx_);
        return storage_usage_tracking_enabled_;
    }

    [[nodiscard]] std::chrono::milliseconds
    loading_timeout() const {
        std::shared_lock lock(mtx_);
        return loading_timeout_;
    }

    [[nodiscard]] std::chrono::milliseconds
    warmup_loading_timeout() const {
        std::shared_lock lock(mtx_);
        return warmup_loading_timeout_;
    }

    [[nodiscard]] CacheWarmupPolicies
    warmup_policies() const {
        std::shared_lock lock(mtx_);
        return warmup_policies_;
    }

    // --- Atomic bulk update (single exclusive lock hold) ---

    void
    UpdateAll(bool storage_usage_tracking_enabled, std::chrono::milliseconds loading_timeout,
              std::chrono::milliseconds warmup_loading_timeout, CacheWarmupPolicies warmup_policies) {
        std::unique_lock lock(mtx_);
        storage_usage_tracking_enabled_ = storage_usage_tracking_enabled;
        loading_timeout_ = loading_timeout;
        warmup_loading_timeout_ = warmup_loading_timeout;
        warmup_policies_ = warmup_policies;
    }

    // --- Individual writers (exclusive lock) ---

    void
    SetStorageUsageTrackingEnabled(bool enabled) {
        std::unique_lock lock(mtx_);
        storage_usage_tracking_enabled_ = enabled;
    }

    void
    SetLoadingTimeout(std::chrono::milliseconds timeout) {
        std::unique_lock lock(mtx_);
        loading_timeout_ = timeout;
    }

    void
    SetWarmupLoadingTimeout(std::chrono::milliseconds timeout) {
        std::unique_lock lock(mtx_);
        warmup_loading_timeout_ = timeout;
    }

    void
    SetWarmupPolicies(CacheWarmupPolicies policies) {
        std::unique_lock lock(mtx_);
        warmup_policies_ = policies;
    }

 private:
    TieredStorageConfig() = default;

    mutable std::shared_mutex mtx_;
    bool storage_usage_tracking_enabled_{false};
    std::chrono::milliseconds loading_timeout_{100000};
    std::chrono::milliseconds warmup_loading_timeout_{0};
    CacheWarmupPolicies warmup_policies_{};
};

}  // namespace milvus::cachinglayer
