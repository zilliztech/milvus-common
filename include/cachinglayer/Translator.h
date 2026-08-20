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
#include <utility>
#include <vector>

#include "cachinglayer/Utils.h"
#include "common/OpContext.h"
#include "common/common_type_c.h"

namespace milvus::cachinglayer {

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
    explicit Meta(StorageType storage_type, CellIdMappingMode cell_id_mapping_mode, CellDataType cell_data_type,
                  CacheWarmupPolicy cache_warmup_policy, bool support_eviction)
        : storage_type(storage_type),
          cell_id_mapping_mode(cell_id_mapping_mode),
          cell_data_type(cell_data_type),
          cache_warmup_policy(cache_warmup_policy),
          support_eviction(support_eviction) {
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
    // Estimate resources for loading a batch of cells. The first value estimates the total resource retained after
    // the cells are loaded, and the second estimates the peak resource used while loading the batch. The Translator
    // owns these estimates because it knows how many cells can be loaded concurrently. Actual loaded resource usage
    // is reported by CellT::CellByteSize().
    //
    // The loading estimate should generally be greater than or equal to the loaded estimate. Underestimating either
    // value may cause the load to fail after insufficient resources are reserved. An empty batch returns two zero
    // ResourceUsage values.
    virtual std::pair<ResourceUsage, ResourceUsage>
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

    // extra cells strategy should be added in cell_ids_to_be_loaded(), get_cells() should just a load executor.
    virtual std::vector<std::pair<cid_t, std::unique_ptr<CellT>>>
    get_cells(OpContext* ctx, const std::vector<cid_t>& cids) = 0;
    virtual ~Translator() = default;
};

}  // namespace milvus::cachinglayer
