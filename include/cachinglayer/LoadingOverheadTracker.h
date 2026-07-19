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
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>

#include "cachinglayer/LoadingOverhead.h"
#include "cachinglayer/Utils.h"
#include "log/Log.h"

namespace milvus::cachinglayer {

namespace internal {
class DList;
}

// Manages per-dimension, per-group loading overhead reservation with an upper bound (UB).
//
// Memory and file groups are independent even when they use the same string key.
// Registration returns a composite uint64_t handle for O(1) lookup on the hot path.
//
// The total loading overhead reserved from DList for each configured dimension is capped:
//   DList dimension reservation = min(sum_of_overhead, dimension_UB)
// An unconfigured dimension passes through unchanged. INT64_MAX is an explicit
// unlimited upper bound. A group's effective upper bound is the maximum reported
// by its live registrations, so refreshing or unregistering one registration can
// safely recompute the group bound without affecting other contributors.
//
// Each Reserve/Release call returns the incremental delta to apply to DList.
// The tracker directly tracks `overhead_reserved` (actual amount of overhead currently
// reserved in DList) to ensure correctness.
//
// The compatibility Register overload configures both dimensions. A missing
// dimension in LoadingOverheadConfig preserves pass-through behavior.
class LoadingOverheadTracker {
 public:
    static inline const ResourceUsage kUnlimited{std::numeric_limits<int64_t>::max(),
                                                 std::numeric_limits<int64_t>::max()};

    static constexpr uint64_t kInvalidHandle = 0;

    uint64_t
    Register(const LoadingOverheadConfig& config) {
        std::lock_guard<std::mutex> lock(mtx_);
        RegistrationState registration;
        if (config.memory.has_value()) {
            registration.memory.group_handle =
                registerDimensionGroup(memory_name_to_group_handle_, config.memory.value(), "memory");
            registration.memory.upper_bound = config.memory->upper_bound;
        }
        if (config.file.has_value()) {
            registration.file.group_handle =
                registerDimensionGroup(file_name_to_group_handle_, config.file.value(), "file");
            registration.file.upper_bound = config.file->upper_bound;
        }

        auto handle = next_registration_handle_++;
        registration_state_[handle] = registration;
        return handle;
    }

    // Compatibility entry point. The two dimensions use independent group state.
    uint64_t
    Register(const std::string& group, const ResourceUsage& upper_bound) {
        return Register(LoadingOverheadConfig{LoadingOverheadDimensionConfig{upper_bound.memory_bytes, group},
                                              LoadingOverheadDimensionConfig{upper_bound.file_bytes, group}});
    }

    // Called before loading. Returns the delta to reserve from DList for loading overhead.
    ResourceUsage
    Reserve(uint64_t handle, const ResourceUsage& loading_overhead) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = registration_state_.find(handle);
        if (it == registration_state_.end()) {
            return loading_overhead;
        }
        auto delta = loading_overhead;
        if (it->second.memory.group_handle != kInvalidHandle) {
            delta.memory_bytes = reserveDimension(it->second.memory.group_handle, loading_overhead.memory_bytes);
        }
        if (it->second.file.group_handle != kInvalidHandle) {
            delta.file_bytes = reserveDimension(it->second.file.group_handle, loading_overhead.file_bytes);
        }
        return delta;
    }

    // Called to release a previous Reserve.
    // Returns the delta to release from DList for loading overhead.
    ResourceUsage
    Release(uint64_t handle, const ResourceUsage& loading_overhead) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = registration_state_.find(handle);
        if (it == registration_state_.end()) {
            return loading_overhead;
        }
        auto delta = loading_overhead;
        if (it->second.memory.group_handle != kInvalidHandle) {
            delta.memory_bytes = releaseDimension(it->second.memory.group_handle, loading_overhead.memory_bytes);
        }
        if (it->second.file.group_handle != kInvalidHandle) {
            delta.file_bytes = releaseDimension(it->second.file.group_handle, loading_overhead.file_bytes);
        }
        return delta;
    }

    bool
    HasFiniteUpperBound(uint64_t handle) const {
        std::lock_guard<std::mutex> lock(mtx_);
        return !(getUpperBoundLocked(handle) == kUnlimited);
    }

    ResourceUsage
    GetUpperBound(uint64_t handle) const {
        std::lock_guard<std::mutex> lock(mtx_);
        return getUpperBoundLocked(handle);
    }

    // Refresh a registration without changing its ref count. Dimensions and
    // group names are fixed at registration; only each registration's upper-bound
    // contribution is updated. The group's effective bound remains the maximum
    // contribution from all live registrations.
    void
    RefreshUpperBound(uint64_t handle, const LoadingOverheadConfig& config) {
        if (handle == kInvalidHandle) {
            return;
        }
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = registration_state_.find(handle);
        if (it == registration_state_.end()) {
            return;
        }
        if (config.memory.has_value()) {
            refreshDimensionUpperBound(it->second.memory.group_handle, it->second.memory.upper_bound,
                                       config.memory.value(), "memory");
        }
        if (config.file.has_value()) {
            refreshDimensionUpperBound(it->second.file.group_handle, it->second.file.upper_bound, config.file.value(),
                                       "file");
        }
    }

    // Decrement ref count for a group. When ref count reaches 0, the group is
    // unconditionally removed. Safe to call from CacheSlot destructor.
    void
    Unregister(uint64_t handle) {
        if (handle == kInvalidHandle) {
            return;
        }
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = registration_state_.find(handle);
        if (it == registration_state_.end()) {
            return;
        }
        unregisterDimensionGroup(memory_name_to_group_handle_, it->second.memory.group_handle,
                                 it->second.memory.upper_bound, "memory");
        unregisterDimensionGroup(file_name_to_group_handle_, it->second.file.group_handle, it->second.file.upper_bound,
                                 "file");
        registration_state_.erase(it);
    }

 private:
    friend class internal::DList;

    struct DimensionGroupState {
        int64_t upper_bound{0};
        int64_t sum_of_overhead{0};
        int64_t overhead_reserved{0};
        uint64_t ref_count{0};
        std::string group_name;
        std::map<int64_t, uint64_t> upper_bound_ref_counts;
    };

    struct DimensionRegistrationState {
        uint64_t group_handle{kInvalidHandle};
        int64_t upper_bound{0};
    };

    struct RegistrationState {
        DimensionRegistrationState memory;
        DimensionRegistrationState file;
    };

    static void
    addUpperBoundContribution(DimensionGroupState& state, int64_t upper_bound) {
        state.upper_bound_ref_counts[upper_bound]++;
        state.upper_bound = state.upper_bound_ref_counts.rbegin()->first;
    }

    static bool
    removeUpperBoundContribution(DimensionGroupState& state, int64_t upper_bound) {
        auto it = state.upper_bound_ref_counts.find(upper_bound);
        if (it == state.upper_bound_ref_counts.end()) {
            LOG_ERROR("[MCL] LoadingOverheadTracker group '{}' is missing UB contribution {}", state.group_name,
                      upper_bound);
            return false;
        }
        if (--it->second == 0) {
            state.upper_bound_ref_counts.erase(it);
        }
        state.upper_bound = state.upper_bound_ref_counts.empty() ? 0 : state.upper_bound_ref_counts.rbegin()->first;
        return true;
    }

    uint64_t
    registerDimensionGroup(std::unordered_map<std::string, uint64_t>& name_to_handle,
                           const LoadingOverheadDimensionConfig& config, const char* dimension) {
        auto it = name_to_handle.find(config.group);
        if (it != name_to_handle.end()) {
            auto& state = dimension_group_state_[it->second];
            auto previous_upper_bound = state.upper_bound;
            addUpperBoundContribution(state, config.upper_bound);
            state.ref_count++;
            if (previous_upper_bound != config.upper_bound) {
                LOG_DEBUG(
                    "[MCL] LoadingOverheadTracker added {} UB contribution for group '{}' (handle {}): "
                    "existing={}, new={}, effective={}",
                    dimension, config.group, it->second, previous_upper_bound, config.upper_bound, state.upper_bound);
            } else {
                LOG_DEBUG("[MCL] LoadingOverheadTracker re-registered {} group '{}' (handle {}, refs={}), UB unchanged",
                          dimension, config.group, it->second, state.ref_count);
            }
            return it->second;
        }

        auto handle = next_group_handle_++;
        name_to_handle[config.group] = handle;
        auto& state = dimension_group_state_[handle];
        state.ref_count = 1;
        state.group_name = config.group;
        addUpperBoundContribution(state, config.upper_bound);
        LOG_INFO("[MCL] LoadingOverheadTracker registered {} group '{}' (handle {}, refs=1): UB={}", dimension,
                 config.group, handle, config.upper_bound);
        return handle;
    }

    int64_t
    reserveDimension(uint64_t group_handle, int64_t overhead) {
        auto& state = dimension_group_state_.at(group_handle);
        state.sum_of_overhead += overhead;
        auto target = std::min(std::max(state.sum_of_overhead, int64_t{0}), state.upper_bound);
        auto delta = std::max(target - state.overhead_reserved, int64_t{0});
        state.overhead_reserved += delta;
        return delta;
    }

    void
    refreshDimensionUpperBound(uint64_t group_handle, int64_t& registered_upper_bound,
                               const LoadingOverheadDimensionConfig& config, const char* dimension) {
        if (group_handle == kInvalidHandle) {
            return;
        }
        auto it = dimension_group_state_.find(group_handle);
        if (it == dimension_group_state_.end()) {
            return;
        }
        auto& state = it->second;
        if (state.group_name != config.group) {
            LOG_WARN("[MCL] LoadingOverheadTracker cannot refresh {} group '{}' through registration for group '{}'",
                     dimension, config.group, state.group_name);
            return;
        }
        if (registered_upper_bound == config.upper_bound) {
            return;
        }
        auto previous_group_upper_bound = state.upper_bound;
        addUpperBoundContribution(state, config.upper_bound);
        if (!removeUpperBoundContribution(state, registered_upper_bound)) {
            removeUpperBoundContribution(state, config.upper_bound);
            return;
        }
        auto previous_registration_upper_bound = registered_upper_bound;
        registered_upper_bound = config.upper_bound;
        LOG_INFO(
            "[MCL] LoadingOverheadTracker refreshed {} group '{}' (handle {}): registration UB {} -> {}, "
            "group UB {} -> {}",
            dimension, config.group, group_handle, previous_registration_upper_bound, config.upper_bound,
            previous_group_upper_bound, state.upper_bound);
    }

    // Undo a failed Reserve using the exact delta returned by that attempt.
    // This differs from Release when an upper-bound refresh makes Reserve catch
    // up previously capped overhead in addition to the current request.
    void
    rollbackReserve(uint64_t handle, const ResourceUsage& loading_overhead, const ResourceUsage& reserved_delta) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = registration_state_.find(handle);
        if (it == registration_state_.end()) {
            return;
        }
        if (it->second.memory.group_handle != kInvalidHandle) {
            rollbackReserveDimension(it->second.memory.group_handle, loading_overhead.memory_bytes,
                                     reserved_delta.memory_bytes);
        }
        if (it->second.file.group_handle != kInvalidHandle) {
            rollbackReserveDimension(it->second.file.group_handle, loading_overhead.file_bytes,
                                     reserved_delta.file_bytes);
        }
    }

    void
    rollbackReserveDimension(uint64_t group_handle, int64_t overhead, int64_t reserved_delta) {
        auto& state = dimension_group_state_.at(group_handle);
        state.sum_of_overhead -= overhead;
        state.overhead_reserved -= reserved_delta;
        if (state.sum_of_overhead < 0 || state.overhead_reserved < 0) {
            LOG_ERROR(
                "[MCL] LoadingOverheadTracker rollback group handle {} produced negative state: "
                "sum_of_overhead={}, overhead_reserved={}",
                group_handle, state.sum_of_overhead, state.overhead_reserved);
            state.sum_of_overhead = std::max(state.sum_of_overhead, int64_t{0});
            state.overhead_reserved = std::max(state.overhead_reserved, int64_t{0});
        }
    }

    int64_t
    releaseDimension(uint64_t group_handle, int64_t overhead) {
        auto& state = dimension_group_state_.at(group_handle);
        state.sum_of_overhead -= overhead;
        if (state.sum_of_overhead < 0) {
            LOG_ERROR("[MCL] LoadingOverheadTracker Release group handle {}: sum_of_overhead < 0", group_handle);
            state.sum_of_overhead = 0;
        }
        auto target = std::min(state.sum_of_overhead, state.upper_bound);
        auto delta = std::max(state.overhead_reserved - target, int64_t{0});
        state.overhead_reserved -= delta;
        return delta;
    }

    ResourceUsage
    getUpperBoundLocked(uint64_t registration_handle) const {
        auto it = registration_state_.find(registration_handle);
        if (it == registration_state_.end()) {
            return kUnlimited;
        }
        ResourceUsage result = kUnlimited;
        if (it->second.memory.group_handle != kInvalidHandle) {
            result.memory_bytes = dimension_group_state_.at(it->second.memory.group_handle).upper_bound;
        }
        if (it->second.file.group_handle != kInvalidHandle) {
            result.file_bytes = dimension_group_state_.at(it->second.file.group_handle).upper_bound;
        }
        return result;
    }

    void
    unregisterDimensionGroup(std::unordered_map<std::string, uint64_t>& name_to_handle, uint64_t group_handle,
                             int64_t registered_upper_bound, const char* dimension) {
        if (group_handle == kInvalidHandle) {
            return;
        }
        auto it = dimension_group_state_.find(group_handle);
        if (it == dimension_group_state_.end()) {
            return;
        }
        auto& state = it->second;
        removeUpperBoundContribution(state, registered_upper_bound);
        if (state.ref_count > 0) {
            state.ref_count--;
        }
        if (state.ref_count > 0) {
            LOG_DEBUG("[MCL] LoadingOverheadTracker {} group handle {} ref_count decremented to {}", dimension,
                      group_handle, state.ref_count);
            return;
        }
        if (state.sum_of_overhead > 0 || state.overhead_reserved > 0) {
            LOG_ERROR(
                "[MCL] LoadingOverheadTracker {} group handle {} ref_count=0 with residual reservations: "
                "sum_of_overhead={}, overhead_reserved={}. Cleaning up anyway to avoid leak.",
                dimension, group_handle, state.sum_of_overhead, state.overhead_reserved);
        }
        LOG_INFO("[MCL] LoadingOverheadTracker unregistered {} group '{}' (handle {})", dimension, state.group_name,
                 group_handle);
        name_to_handle.erase(state.group_name);
        dimension_group_state_.erase(it);
    }

    mutable std::mutex mtx_;
    std::unordered_map<std::string, uint64_t> memory_name_to_group_handle_;
    std::unordered_map<std::string, uint64_t> file_name_to_group_handle_;
    std::unordered_map<uint64_t, DimensionGroupState> dimension_group_state_;
    std::unordered_map<uint64_t, RegistrationState> registration_state_;
    uint64_t next_group_handle_{1};
    uint64_t next_registration_handle_{1};
};

}  // namespace milvus::cachinglayer
