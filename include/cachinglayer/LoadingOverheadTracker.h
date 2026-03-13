// Copyright (C) 2019-2026 Zilliz. All rights reserved.
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

#include <algorithm>
#include <mutex>
#include <unordered_map>

#include "cachinglayer/Utils.h"
#include "log/Log.h"

namespace milvus::cachinglayer {

// Manages per-CellDataType loading overhead reservation with an overhead upper bound (UB).
//
// The total loading overhead reserved from DList for a given type is capped at UB:
//   DList loading overhead reservation = min(sum_of_overhead, UB)
//
// Each Reserve/Release call returns the incremental delta to apply to DList.
// The tracker directly tracks `overhead_reserved` (actual amount of overhead currently reserved in DList) to ensure
// correctness even when UB changes at runtime.
//
// The total DList resource reservation across all requests of a type = sum(loaded_i) + min(sum(overhead_i), UB),
// which does not inflate with the number of queuing requests.
//
// By default, types are registered with kUnlimited UB, preserving original behavior (no capping).
class LoadingOverheadTracker {
 public:
    static inline const ResourceUsage kUnlimited{std::numeric_limits<int64_t>::max(),
                                                 std::numeric_limits<int64_t>::max()};

    // Register the upper bound for a given CellDataType.
    // If already registered with a finite UB, uses the larger of the two UBs per dimension.
    // If previously registered with kUnlimited (default), replaces with the given UB.
    void
    RegisterUpperBound(CellDataType type, const ResourceUsage& upper_bound) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto [it, inserted] = type_state_.try_emplace(type, TypeState{upper_bound, {}, {}});
        if (!inserted) {
            auto& existing = it->second.upper_bound;
            if (existing == kUnlimited) {
                existing = upper_bound;
            } else {
                existing.memory_bytes = std::max(existing.memory_bytes, upper_bound.memory_bytes);
                existing.file_bytes = std::max(existing.file_bytes, upper_bound.file_bytes);
            }
        }
        LOG_INFO("[MCL] LoadingOverheadTracker registered UB for type {}: {}", static_cast<int>(type),
                 upper_bound.ToString());
    }

    // Ensure a type is registered. If not yet registered, registers with kUnlimited.
    void
    EnsureRegistered(CellDataType type) {
        std::lock_guard<std::mutex> lock(mtx_);
        type_state_.try_emplace(type, TypeState{kUnlimited, {}, {}});
    }

    // Called before loading. Returns the delta to reserve from DList for loading overhead.
    // Caller should: DList.Reserve(loaded_resource + delta)
    ResourceUsage
    Reserve(CellDataType type, const ResourceUsage& loading_overhead) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto& state = getOrCreateState(type);
        state.sum_of_overhead += loading_overhead;
        auto target = cappedAmount(state.sum_of_overhead, state.upper_bound);
        auto delta = target - state.overhead_reserved;
        delta.memory_bytes = std::max(delta.memory_bytes, int64_t{0});
        delta.file_bytes = std::max(delta.file_bytes, int64_t{0});
        state.overhead_reserved += delta;
        LOG_TRACE(
            "[MCL] LoadingOverheadTracker Reserve type={}: loading={}, "
            "sum={}, UB={}, overhead_reserved={}, delta={}",
            static_cast<int>(type), loading_overhead.ToString(), state.sum_of_overhead.ToString(),
            state.upper_bound.ToString(), state.overhead_reserved.ToString(), delta.ToString());
        return delta;
    }

    // Called to release a previous Reserve (either after loading completes, or to undo
    // a Reserve when DList reservation fails / bonus cells retry).
    // Returns the delta to release from DList for loading overhead.
    ResourceUsage
    Release(CellDataType type, const ResourceUsage& loading_overhead) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto& state = getOrCreateState(type);
        state.sum_of_overhead -= loading_overhead;
        if (state.sum_of_overhead.memory_bytes < 0) {
            LOG_ERROR("[MCL] LoadingOverheadTracker ReleaseInternal type={}: sum_of_overhead.memory_bytes < 0",
                      static_cast<int>(type));
            state.sum_of_overhead.memory_bytes = 0;
        }
        if (state.sum_of_overhead.file_bytes < 0) {
            LOG_ERROR("[MCL] LoadingOverheadTracker ReleaseInternal type={}: sum_of_overhead.file_bytes < 0",
                      static_cast<int>(type));
            state.sum_of_overhead.file_bytes = 0;
        }
        auto target = cappedAmount(state.sum_of_overhead, state.upper_bound);
        auto delta = state.overhead_reserved - target;
        delta.memory_bytes = std::max(delta.memory_bytes, int64_t{0});
        delta.file_bytes = std::max(delta.file_bytes, int64_t{0});
        state.overhead_reserved -= delta;
        LOG_TRACE(
            "[MCL] LoadingOverheadTracker Release type={}: loading={}, "
            "sum={}, UB={}, overhead_reserved={}, delta={}",
            static_cast<int>(type), loading_overhead.ToString(), state.sum_of_overhead.ToString(),
            state.upper_bound.ToString(), state.overhead_reserved.ToString(), delta.ToString());
        return delta;
    }

    // Check if a type has a finite (non-unlimited) upper bound registered.
    bool
    HasFiniteUpperBound(CellDataType type) const {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = type_state_.find(type);
        return it != type_state_.end() && !(it->second.upper_bound == kUnlimited);
    }

    // Get the upper bound for a type.
    ResourceUsage
    GetUpperBound(CellDataType type) const {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = type_state_.find(type);
        if (it == type_state_.end()) {
            return kUnlimited;
        }
        return it->second.upper_bound;
    }

 private:
    struct TypeState {
        ResourceUsage upper_bound;
        ResourceUsage sum_of_overhead;    // total loading overhead requested (uncapped)
        ResourceUsage overhead_reserved;  // actual amount currently reserved in DList
    };

    static ResourceUsage
    cappedAmount(const ResourceUsage& sum, const ResourceUsage& ub) {
        return {std::min(std::max(sum.memory_bytes, int64_t{0}), ub.memory_bytes),
                std::min(std::max(sum.file_bytes, int64_t{0}), ub.file_bytes)};
    }

    TypeState&
    getOrCreateState(CellDataType type) {
        auto [it, _] = type_state_.try_emplace(type, TypeState{kUnlimited, {}, {}});
        return it->second;
    }

    mutable std::mutex mtx_;
    std::unordered_map<CellDataType, TypeState> type_state_;
};

}  // namespace milvus::cachinglayer
