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

#include <folly/coro/Task.h>

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
    // Optional stable shard/channel label for attributed cache-slot usage metrics.
    // Keep this value bounded in cardinality, such as loaded shards/channels, not request/user IDs.
    // The shard usage collector removes the time series after the last slot for this label is gone.
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
    // Shared-overhead configuration for this translator. Private overhead is never capped.
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
    // Estimate one complete get_cells request, accounting for its loading concurrency.
    // Resident is the sum of final cell sizes; each temporary category bounds its batch peak,
    // excluding resident. Do not include private allocations in shared overhead: only scratch
    // whose entire lifetime is covered by the bound group's runtime admission may be capped.
    // Unbound shared dimensions pass through unchanged. All values must be non-negative and
    // conservative from successful reservation until get_cells completes (including failure).
    // Empty input returns zero. Loaded cells are charged individually using CellByteSize().
    virtual LoadingResourceUsage
    estimated_loading_usage(const std::vector<cid_t>& cids) const = 0;
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

    /// Synchronously load requested cells; extra cells belong in bonus_cells_to_be_loaded().
    virtual std::vector<std::pair<cid_t, std::unique_ptr<CellT>>>
    get_cells(OpContext* ctx, const std::vector<cid_t>& cids) = 0;

    /**
     * @brief Load requested cells on the awaiting caller's executor.
     * @param ctx Borrowed operation context, valid until completion; may be null.
     * @param cids Borrowed cell IDs, valid until completion.
     * @note The default invokes synchronous get_cells() on that executor and may block.
     * Override to await asynchronous work or offload blocking I/O. Drain issued work
     * before returning or throwing. Extra cells belong in bonus_cells_to_be_loaded().
     * Keep the translator alive until completion. Overrides should share a private
     * load implementation with get_cells(), rather than delegate to each other.
     */
    virtual folly::coro::Task<std::vector<std::pair<cid_t, std::unique_ptr<CellT>>>>
    get_cells_async(OpContext* ctx, const std::vector<cid_t>& cids) {
        co_return get_cells(ctx, cids);
    }
    virtual ~Translator() = default;
};

}  // namespace milvus::cachinglayer
