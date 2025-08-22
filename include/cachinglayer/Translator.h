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
#include "common/common_type_c.h"

namespace milvus::cachinglayer {

enum class CellIdMappingMode : uint8_t {
    CUSTOMIZED = 0,   // the cell id should be parsed from the uid by the translator
    IDENTICAL = 1,    // the cell id is identical to the uid
    ALWAYS_ZERO = 2,  // the cell id is always 0
};

struct Meta {
    // This storage type is currently used only by metrics to distinguish the slot type.
    // In actual resource reservation, we use the actual size of the cell to determine the type.
    StorageType storage_type;
    CellIdMappingMode cell_id_mapping_mode;
    CacheWarmupPolicy cache_warmup_policy;
    // Whether the translator supports strategy based eviction.
    // Does not affect manual eviction.
    bool support_eviction;
    explicit Meta(StorageType storage_type, CellIdMappingMode cell_id_mapping_mode,
                  CacheWarmupPolicy cache_warmup_policy, bool support_eviction)
        : storage_type(storage_type),
          cell_id_mapping_mode(cell_id_mapping_mode),
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
    // For resource reservation when a cell is about to be loaded.
    // There are two types of resource usage for a cell: the first is the usage after it has been loaded,
    // and the second is the usage during loading. Typically, the loading usage is greater than the loaded usage
    // due to the preprocessing stage.
    // If a cell is about to be pinned and loaded, and there are not enough resource for it, EvictionManager
    // will try to evict some other cells to make space. Thus this estimation should generally be greater
    // than or equal to the actual size. If the estimation is smaller than the actual size, with insufficient
    // resource reserved, the load may fail.
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

    // extra cells strategy should be added in cell_ids_to_be_loaded(), get_cells() should just a load executor.
    virtual std::vector<std::pair<cid_t, std::unique_ptr<CellT>>>
    get_cells(const std::vector<cid_t>& cids) = 0;
    virtual ~Translator() = default;
};

}  // namespace milvus::cachinglayer
