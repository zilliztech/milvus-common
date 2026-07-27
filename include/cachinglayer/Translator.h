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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "cachinglayer/LoadingOverhead.h"
#include "cachinglayer/Utils.h"
#include "common/OpContext.h"
#include "common/common_type_c.h"

namespace milvus::cachinglayer {

struct MetricAttribution {
    // Optional stable shard/channel label for attributed cache-slot disk usage metrics.
    // Keep this value bounded in cardinality, such as loaded shards/channels, not request/user IDs.
    // The shard disk usage collector removes the time series after the last slot for this label is gone.
    // Empty means this translator is unattributed and no shard metric is emitted.
    std::string shard;
};

struct Meta {
    // This storage type is currently used only by metrics to distinguish the slot type.
    // In actual resource reservation, we use the actual size of the cell to determine the type.
    StorageType storage_type;
    CellIdMappingMode cell_id_mapping_mode;
    CellDataType cell_data_type;
    CacheWarmupPolicy cache_warmup_policy;
    // Whether the translator supports strategy based eviction.
    // Does not affect manual eviction.
    bool support_eviction;
    // Loading-overhead configuration for this translator.
    // Each configured resource dimension is capped across CacheSlots sharing its group.
    // An omitted dimension passes through unchanged and remains subject to DList admission.
    // If the config is not set, no capping is applied (existing behavior).
    std::optional<LoadingOverheadConfig> loading_overhead_config;
    std::optional<MetricAttribution> metric_attribution;
    explicit Meta(StorageType storage_type, CellIdMappingMode cell_id_mapping_mode, CellDataType cell_data_type,
                  CacheWarmupPolicy cache_warmup_policy, bool support_eviction,
                  std::optional<LoadingOverheadConfig> loading_overhead_config = std::nullopt,
                  std::optional<MetricAttribution> metric_attribution = std::nullopt)
        : storage_type(storage_type),
          cell_id_mapping_mode(cell_id_mapping_mode),
          cell_data_type(cell_data_type),
          cache_warmup_policy(cache_warmup_policy),
          support_eviction(support_eviction),
          loading_overhead_config(std::move(loading_overhead_config)),
          metric_attribution(std::move(metric_attribution)) {
    }
};

template <typename CellT>
class Translator {
 public:
    using value_type = CellT;

    virtual size_t
    num_cells() const = 0;
    virtual cid_t
    cell_id_of(uid_t uid) const = 0;
    // For resource reservation when a cell is about to be loaded.
    // Returns {loaded_usage, loading_overhead}:
    //   - loaded_usage (first): the final resource usage after the cell is fully loaded and in cache.
    //   - loading_overhead (second): a conservative upper bound for the *temporary* resource usage during loading
    //     (e.g., preprocessing buffers), excluding the final loaded usage. For grouped dimensions it must cover the
    //     request's actual transient usage from successful DList reservation until the paired Release.
    // When a loading_overhead dimension is configured in Meta, the total reservation across all CacheSlots sharing
    // that dimension's group is governed by the Group policy. Omitted dimensions pass through unchanged.
    // If a cell is about to be pinned and loaded, and there are not enough resource for it, EvictionManager
    // will try to evict some other cells to make space. Both estimates must be greater than or equal to the actual
    // usage. Underestimation can break admission safety and may make the load fail.
    virtual std::pair<ResourceUsage, ResourceUsage>
    estimated_byte_size_of_cell(cid_t cid) const = 0;
    // must be unique to identify a CacheSlot.
    virtual const std::string&
    key() const = 0;

    virtual Meta*
    meta() = 0;

    // Translator may choose to fetch more than requested cells. The default behavior is to not include extra cells.
    virtual std::vector<cid_t>
    bonus_cells_to_be_loaded(const std::vector<cid_t>& cids) const {
        return {};
    }

    // this method is used to get the byte size of a specific cell in persistent storage
    virtual int64_t
    cells_storage_bytes(const std::vector<cid_t>& cids) const = 0;

    // extra cells strategy should be added in cell_ids_to_be_loaded(), get_cells() should just a load executor.
    virtual std::vector<std::pair<cid_t, std::unique_ptr<CellT>>>
    get_cells(OpContext* ctx, const std::vector<cid_t>& cids) = 0;
    virtual ~Translator() = default;
};

}  // namespace milvus::cachinglayer
