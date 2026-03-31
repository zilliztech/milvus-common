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

#include <folly/executors/CPUThreadPoolExecutor.h>

#include <memory>

#include "cachinglayer/CacheSlot.h"
#include "cachinglayer/LoadingOverheadTracker.h"
#include "cachinglayer/TieredStorageConfig.h"
#include "cachinglayer/Translator.h"
#include "cachinglayer/Utils.h"
#include "cachinglayer/lrucache/DList.h"
#include "common/common_type_c.h"

namespace milvus::cachinglayer {

class Manager {
 public:
    static Manager&
    GetInstance();

    // Must be called exactly once before any CacheSlot is created.
    static void
    ConfigureTieredStorage(CacheWarmupPolicies warmup_policies, CacheLimit cache_limit,
                           bool storage_usage_tracking_enabled, bool eviction_enabled, EvictionConfig eviction_config,
                           std::chrono::milliseconds loading_timeout,
                           std::chrono::milliseconds warmup_loading_timeout = std::chrono::milliseconds(0),
                           uint32_t prefetch_pool_threads = 0);

    // Update runtime-configurable fields. Can be called from CGO at runtime.
    static void
    UpdateConfig(std::chrono::milliseconds loading_timeout, std::chrono::milliseconds warmup_loading_timeout,
                 bool storage_usage_tracking_enabled, CacheWarmupPolicies warmup_policies);

    ~Manager();

    Manager(const Manager&) = delete;
    Manager&
    operator=(const Manager&) = delete;
    Manager(Manager&&) = delete;
    Manager&
    operator=(Manager&&) = delete;

    template <typename CellT>
    std::shared_ptr<CacheSlot<CellT>>
    CreateCacheSlot(std::unique_ptr<Translator<CellT>> translator, OpContext* ctx = nullptr) {
        AssertInfo(dlist_ != nullptr,
                   "dlist_ must be initialized by ConfigureTieredStorage before any CacheSlot is created");
        if (ctx && ctx->cancellation_token.isCancellationRequested()) {
            throw std::runtime_error("Operation cancelled, stop creating cache slot");
        }
        auto config = TieredStorageConfig::GetInstance().GetSnapshot();
        auto evictable = translator->meta()->support_eviction && eviction_enabled_;
        auto self_reserve = eviction_enabled_;

        // Register loading overhead upper bound for this CellDataType.
        // If the translator specifies a finite UB, use it; otherwise register
        // with unlimited UB to ensure every type goes through the tracker.
        if (translator->meta()->loading_overhead_upper_bound.has_value()) {
            loading_overhead_tracker_.RegisterUpperBound(translator->meta()->cell_data_type,
                                                         translator->meta()->loading_overhead_upper_bound.value());
        } else {
            loading_overhead_tracker_.EnsureRegistered(translator->meta()->cell_data_type);
        }

        auto cache_slot = std::make_shared<CacheSlot<CellT>>(
            std::move(translator), dlist_.get(), evictable, self_reserve, config.storage_usage_tracking_enabled,
            config.loading_timeout, config.warmup_loading_timeout, &loading_overhead_tracker_);
        cache_slot->Warmup(ctx, prefetch_pool_);
        return cache_slot;
    }

    bool
    ReserveLoadingResourceWithTimeout(const ResourceUsage& size, std::chrono::milliseconds timeout,
                                      const std::string& ctx_info = "") {
        auto result = SemiInlineGet(dlist_->ReserveLoadingResourceWithTimeout(size, timeout, nullptr));
        if (result) {
            monitor::cache_loading_bytes(CellDataType::OTHER, StorageType::MEMORY).Increment(size.memory_bytes);
            monitor::cache_loading_bytes(CellDataType::OTHER, StorageType::DISK).Increment(size.file_bytes);
            LOG_TRACE("[MCL] ReserveLoadingResourceWithTimeout for [{}] with size={}", ctx_info, size.ToString());
        }
        return result;
    }

    void
    ReleaseLoadingResource(const ResourceUsage& size, const std::string& ctx_info = "") {
        dlist_->ReleaseLoadingResource(size);
        monitor::cache_loading_bytes(CellDataType::OTHER, StorageType::MEMORY).Decrement(size.memory_bytes);
        monitor::cache_loading_bytes(CellDataType::OTHER, StorageType::DISK).Decrement(size.file_bytes);
        LOG_TRACE("[MCL] ReleaseLoadingResource for [{}] with size={}", ctx_info, size.ToString());
    }

    void
    ChargeLoadedResource(const ResourceUsage& size, const std::string& ctx_info = "") {
        monitor::cache_loaded_bytes(CellDataType::OTHER, StorageType::MEMORY).Increment(size.memory_bytes);
        monitor::cache_loaded_bytes(CellDataType::OTHER, StorageType::DISK).Increment(size.file_bytes);
        dlist_->ChargeLoadedResource(size);
        LOG_TRACE("[MCL] ChargeLoadedResource for [{}] with size={}", ctx_info, size.ToString());
    }

    void
    RefundLoadedResource(const ResourceUsage& size, const std::string& ctx_info = "") {
        dlist_->RefundLoadedResource(size);
        monitor::cache_loaded_bytes(CellDataType::OTHER, StorageType::MEMORY).Decrement(size.memory_bytes);
        monitor::cache_loaded_bytes(CellDataType::OTHER, StorageType::DISK).Decrement(size.file_bytes);
        LOG_TRACE("[MCL] RefundLoadedResource for [{}] with size={}", ctx_info, size.ToString());
    }

    // memory overhead for managing all cache slots/cells/translators/policies.
    [[nodiscard]] size_t
    memory_overhead() const;

    [[nodiscard]] CacheWarmupPolicy
    getScalarFieldCacheWarmupPolicy() const {
        return TieredStorageConfig::GetInstance().warmup_policies().scalarFieldCacheWarmupPolicy;
    }

    [[nodiscard]] CacheWarmupPolicy
    getVectorFieldCacheWarmupPolicy() const {
        return TieredStorageConfig::GetInstance().warmup_policies().vectorFieldCacheWarmupPolicy;
    }

    [[nodiscard]] CacheWarmupPolicy
    getScalarIndexCacheWarmupPolicy() const {
        return TieredStorageConfig::GetInstance().warmup_policies().scalarIndexCacheWarmupPolicy;
    }

    [[nodiscard]] CacheWarmupPolicy
    getVectorIndexCacheWarmupPolicy() const {
        return TieredStorageConfig::GetInstance().warmup_policies().vectorIndexCacheWarmupPolicy;
    }

    [[nodiscard]] bool
    isEvictionEnabled() const {
        return eviction_enabled_;
    }

    [[nodiscard]] std::shared_ptr<folly::CPUThreadPoolExecutor>
    GetPrefetchPool() const {
        return prefetch_pool_;
    }

 private:
    Manager() = default;

    std::shared_ptr<internal::DList> dlist_{nullptr};
    std::shared_ptr<folly::CPUThreadPoolExecutor> prefetch_pool_{nullptr};
    LoadingOverheadTracker loading_overhead_tracker_;
    bool eviction_enabled_{false};
};  // class Manager

}  // namespace milvus::cachinglayer
