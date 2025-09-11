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

#include <memory>

#include "cachinglayer/CacheSlot.h"
#include "cachinglayer/Translator.h"
#include "cachinglayer/Utils.h"
#include "cachinglayer/lrucache/DList.h"
#include "common/common_type_c.h"

namespace milvus::cachinglayer {

class Manager {
 public:
    static Manager&
    GetInstance();

    // This function is not thread safe, must be called exactly once before any CacheSlot is created,
    // and before any Manager instance method is called.
    // TODO(tiered storage 4): support dynamic update.
    static void
    ConfigureTieredStorage(CacheWarmupPolicies warmup_policies, CacheLimit cache_limit, bool evictionEnabled,
                           EvictionConfig eviction_config);

    Manager(const Manager&) = delete;
    Manager&
    operator=(const Manager&) = delete;
    Manager(Manager&&) = delete;
    Manager&
    operator=(Manager&&) = delete;

    template <typename CellT>
    std::shared_ptr<CacheSlot<CellT>>
    CreateCacheSlot(std::unique_ptr<Translator<CellT>> translator) {
        auto evictable = translator->meta()->support_eviction && evictionEnabled_;
        auto self_reserve = evictionEnabled_;
        auto cache_slot =
            std::make_shared<CacheSlot<CellT>>(std::move(translator), dlist_.get(), evictable, self_reserve);
        cache_slot->Warmup();
        return cache_slot;
    }

    bool
    ReserveLoadingResourceWithTimeout(const ResourceUsage& size, std::chrono::milliseconds timeout) {
        auto result = SemiInlineGet(dlist_->ReserveLoadingResourceWithTimeout(size, timeout));
        if (result) {
            monitor::cache_loading_bytes(CellDataType::OTHER, StorageType::MEMORY).Increment(size.memory_bytes);
            monitor::cache_loading_bytes(CellDataType::OTHER, StorageType::DISK).Increment(size.file_bytes);
        }
        return result;
    }

    void
    ReleaseLoadingResource(const ResourceUsage& size) {
        dlist_->ReleaseLoadingResource(size);
        monitor::cache_loading_bytes(CellDataType::OTHER, StorageType::MEMORY).Decrement(size.memory_bytes);
        monitor::cache_loading_bytes(CellDataType::OTHER, StorageType::DISK).Decrement(size.file_bytes);
    }

    void
    ChargeLoadedResource(const ResourceUsage& size) {
        monitor::cache_loaded_bytes(CellDataType::OTHER, StorageType::MEMORY).Increment(size.memory_bytes);
        monitor::cache_loaded_bytes(CellDataType::OTHER, StorageType::DISK).Increment(size.file_bytes);
        dlist_->ChargeLoadedResource(size);
    }

    void
    RefundLoadedResource(const ResourceUsage& size) {
        dlist_->RefundLoadedResource(size);
        monitor::cache_loaded_bytes(CellDataType::OTHER, StorageType::MEMORY).Decrement(size.memory_bytes);
        monitor::cache_loaded_bytes(CellDataType::OTHER, StorageType::DISK).Decrement(size.file_bytes);
    }

    // memory overhead for managing all cache slots/cells/translators/policies.
    [[nodiscard]] size_t
    memory_overhead() const;

    [[nodiscard]] CacheWarmupPolicy
    getScalarFieldCacheWarmupPolicy() const {
        return warmup_policies_.scalarFieldCacheWarmupPolicy;
    }

    [[nodiscard]] CacheWarmupPolicy
    getVectorFieldCacheWarmupPolicy() const {
        return warmup_policies_.vectorFieldCacheWarmupPolicy;
    }

    [[nodiscard]] CacheWarmupPolicy
    getScalarIndexCacheWarmupPolicy() const {
        return warmup_policies_.scalarIndexCacheWarmupPolicy;
    }

    [[nodiscard]] CacheWarmupPolicy
    getVectorIndexCacheWarmupPolicy() const {
        return warmup_policies_.vectorIndexCacheWarmupPolicy;
    }

    [[nodiscard]] bool
    isEvictionEnabled() const {
        return evictionEnabled_;
    }

 private:
    friend void
    ConfigureTieredStorage(CacheWarmupPolicies warmup_policies, CacheLimit cache_limit, bool evictionEnabled,
                           EvictionConfig eviction_config);

    Manager() = default;

    std::unique_ptr<internal::DList> dlist_{nullptr};
    CacheWarmupPolicies warmup_policies_{};
    bool evictionEnabled_{false};
};  // class Manager

}  // namespace milvus::cachinglayer
