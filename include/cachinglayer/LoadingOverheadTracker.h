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
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>

#include "cachinglayer/Utils.h"
#include "log/Log.h"

namespace milvus::cachinglayer {

// Manages per-group loading overhead reservation with an overhead upper bound (UB).
//
// Groups are identified by string keys at registration time. Registration returns
// a uint64_t handle for O(1) lookup on the hot path (Reserve/Release).
//
// The total loading overhead reserved from DList for a given group is capped at UB:
//   DList loading overhead reservation = min(sum_of_overhead, UB)
//
// Each Reserve/Release call returns the incremental delta to apply to DList.
// The tracker directly tracks `overhead_reserved` (actual amount of overhead currently
// reserved in DList) to ensure correctness.
//
// By default, groups are registered with kUnlimited UB, preserving original behavior.
class LoadingOverheadTracker {
 public:
    static inline const ResourceUsage kUnlimited{std::numeric_limits<int64_t>::max(),
                                                 std::numeric_limits<int64_t>::max()};

    static constexpr uint64_t kInvalidHandle = 0;

    // Register a group with an upper bound. Returns a handle for hot-path use.
    // If the group already exists with a finite UB, takes the larger of the two per dimension.
    // If previously registered with kUnlimited, replaces with the given UB.
    // Each Register increments a ref count; call Unregister to decrement.
    uint64_t
    Register(const std::string& group, const ResourceUsage& upper_bound) {
        enum class LogEvent {
            kNone,
            kSetUpperBound,
            kMismatch,
            kReregister,
            kNewGroup,
        };

        uint64_t handle = kInvalidHandle;
        uint64_t ref_count = 0;
        ResourceUsage old_upper_bound{};
        ResourceUsage effective_upper_bound{};
        LogEvent log_event = LogEvent::kNone;

        auto update_existing_state = [&](uint64_t existing_handle, const std::shared_ptr<GroupState>& state) {
            handle = existing_handle;
            std::lock_guard<std::mutex> state_lock(state->mtx);

            state->ref_count++;
            ref_count = state->ref_count;
            old_upper_bound = state->upper_bound;
            if (state->upper_bound == kUnlimited) {
                state->upper_bound = upper_bound;
                effective_upper_bound = state->upper_bound;
                log_event = LogEvent::kSetUpperBound;
            } else if (state->upper_bound.memory_bytes < upper_bound.memory_bytes ||
                       state->upper_bound.file_bytes < upper_bound.file_bytes) {
                state->upper_bound.memory_bytes = std::max(state->upper_bound.memory_bytes, upper_bound.memory_bytes);
                state->upper_bound.file_bytes = std::max(state->upper_bound.file_bytes, upper_bound.file_bytes);
                effective_upper_bound = state->upper_bound;
                log_event = LogEvent::kMismatch;
            } else {
                effective_upper_bound = state->upper_bound;
                log_event = LogEvent::kReregister;
            }
        };

        {
            std::shared_lock<std::shared_mutex> map_lock(map_mtx_);
            auto it = name_to_handle_.find(group);
            if (it != name_to_handle_.end()) {
                auto state_it = handle_state_.find(it->second);
                AssertInfo(state_it != handle_state_.end(),
                           "[MCL] LoadingOverheadTracker group '{}' has no state for handle {}", group, it->second);
                update_existing_state(it->second, state_it->second);
            }
        }

        if (handle == kInvalidHandle) {
            std::unique_lock<std::shared_mutex> map_lock(map_mtx_);
            auto it = name_to_handle_.find(group);
            if (it != name_to_handle_.end()) {
                auto state_it = handle_state_.find(it->second);
                AssertInfo(state_it != handle_state_.end(),
                           "[MCL] LoadingOverheadTracker group '{}' has no state for handle {}", group, it->second);
                update_existing_state(it->second, state_it->second);
            } else {
                handle = next_handle_++;
                auto state = std::make_shared<GroupState>(upper_bound, group);
                name_to_handle_[group] = handle;
                handle_state_[handle] = std::move(state);
                ref_count = 1;
                effective_upper_bound = upper_bound;
                log_event = LogEvent::kNewGroup;
            }
        }

        switch (log_event) {
            case LogEvent::kSetUpperBound:
                LOG_INFO("[MCL] LoadingOverheadTracker set UB for group '{}' (handle {}, refs={}): {}", group, handle,
                         ref_count, effective_upper_bound.ToString());
                break;
            case LogEvent::kMismatch:
                LOG_WARN(
                    "[MCL] LoadingOverheadTracker UB mismatch for group '{}' (handle {}): existing={}, new={}. "
                    "Taking max per dimension, effective={}.",
                    group, handle, old_upper_bound.ToString(), upper_bound.ToString(),
                    effective_upper_bound.ToString());
                break;
            case LogEvent::kReregister:
                LOG_DEBUG("[MCL] LoadingOverheadTracker re-registered group '{}' (handle {}, refs={}), UB unchanged",
                          group, handle, ref_count);
                break;
            case LogEvent::kNewGroup:
                LOG_INFO("[MCL] LoadingOverheadTracker registered group '{}' (handle {}, refs=1): UB={}", group, handle,
                         effective_upper_bound.ToString());
                break;
            case LogEvent::kNone:
                break;
        }
        return handle;
    }

    // Called before loading. Returns the delta to reserve from DList for loading overhead.
    ResourceUsage
    Reserve(uint64_t handle, const ResourceUsage& loading_overhead) {
        std::shared_lock<std::shared_mutex> map_lock(map_mtx_);
        auto it = handle_state_.find(handle);
        if (it == handle_state_.end()) {
            return loading_overhead;
        }
        auto state = it->second;
        std::lock_guard<std::mutex> state_lock(state->mtx);
        state->sum_of_overhead += loading_overhead;
        auto target = cappedAmount(state->sum_of_overhead, state->upper_bound);
        auto delta = target - state->overhead_reserved;
        delta.memory_bytes = std::max(delta.memory_bytes, int64_t{0});
        delta.file_bytes = std::max(delta.file_bytes, int64_t{0});
        state->overhead_reserved += delta;
        return delta;
    }

    // Called to release a previous Reserve.
    // Returns the delta to release from DList for loading overhead.
    ResourceUsage
    Release(uint64_t handle, const ResourceUsage& loading_overhead) {
        std::shared_lock<std::shared_mutex> map_lock(map_mtx_);
        auto it = handle_state_.find(handle);
        if (it == handle_state_.end()) {
            return loading_overhead;
        }
        auto state = it->second;
        std::lock_guard<std::mutex> state_lock(state->mtx);
        state->sum_of_overhead -= loading_overhead;
        if (state->sum_of_overhead.memory_bytes < 0) {
            LOG_ERROR("[MCL] LoadingOverheadTracker Release handle {}: sum_of_overhead.memory_bytes < 0", handle);
            state->sum_of_overhead.memory_bytes = 0;
        }
        if (state->sum_of_overhead.file_bytes < 0) {
            LOG_ERROR("[MCL] LoadingOverheadTracker Release handle {}: sum_of_overhead.file_bytes < 0", handle);
            state->sum_of_overhead.file_bytes = 0;
        }
        auto target = cappedAmount(state->sum_of_overhead, state->upper_bound);
        auto delta = state->overhead_reserved - target;
        delta.memory_bytes = std::max(delta.memory_bytes, int64_t{0});
        delta.file_bytes = std::max(delta.file_bytes, int64_t{0});
        state->overhead_reserved -= delta;
        return delta;
    }

    bool
    HasFiniteUpperBound(uint64_t handle) const {
        std::shared_lock<std::shared_mutex> map_lock(map_mtx_);
        auto it = handle_state_.find(handle);
        if (it == handle_state_.end()) {
            return false;
        }
        auto state = it->second;
        std::lock_guard<std::mutex> state_lock(state->mtx);
        return !(state->upper_bound == kUnlimited);
    }

    ResourceUsage
    GetUpperBound(uint64_t handle) const {
        std::shared_lock<std::shared_mutex> map_lock(map_mtx_);
        auto it = handle_state_.find(handle);
        if (it == handle_state_.end()) {
            return kUnlimited;
        }
        auto state = it->second;
        std::lock_guard<std::mutex> state_lock(state->mtx);
        return state->upper_bound;
    }

    // Decrement ref count for a group. When ref count reaches 0, the group is
    // unconditionally removed. Safe to call from CacheSlot destructor.
    void
    Unregister(uint64_t handle) {
        if (handle == kInvalidHandle) {
            return;
        }

        uint64_t ref_count = 0;
        bool should_log_unregister = false;
        bool should_log_decrement = false;
        bool has_residual = false;
        ResourceUsage residual_sum{};
        ResourceUsage residual_reserved{};
        std::string group_name;

        {
            std::unique_lock<std::shared_mutex> map_lock(map_mtx_);
            auto it = handle_state_.find(handle);
            if (it == handle_state_.end()) {
                return;
            }
            auto state = it->second;
            std::lock_guard<std::mutex> state_lock(state->mtx);

            if (state->ref_count > 0) {
                state->ref_count--;
            }
            ref_count = state->ref_count;
            if (ref_count > 0) {
                group_name = state->group_name;
                should_log_decrement = true;
            } else {
                group_name = state->group_name;
                residual_sum = state->sum_of_overhead;
                residual_reserved = state->overhead_reserved;
                has_residual = residual_sum.AnyGTZero() || residual_reserved.AnyGTZero();
                name_to_handle_.erase(group_name);
                handle_state_.erase(it);
                should_log_unregister = true;
            }
        }

        if (should_log_decrement) {
            LOG_DEBUG("[MCL] LoadingOverheadTracker handle {} ref_count decremented to {}", handle, ref_count);
            return;
        }

        if (has_residual) {
            LOG_ERROR(
                "[MCL] LoadingOverheadTracker handle {} ref_count=0 with residual reservations: "
                "sum_of_overhead={}, overhead_reserved={}. Cleaning up anyway to avoid leak.",
                handle, residual_sum.ToString(), residual_reserved.ToString());
        }
        if (should_log_unregister) {
            LOG_INFO("[MCL] LoadingOverheadTracker unregistered group '{}' (handle {})", group_name, handle);
        }
    }

 private:
    struct GroupState {
        GroupState(ResourceUsage ub, std::string group) : upper_bound(ub), ref_count(1), group_name(std::move(group)) {
        }

        mutable std::mutex mtx;
        ResourceUsage upper_bound;
        ResourceUsage sum_of_overhead;
        ResourceUsage overhead_reserved;
        uint64_t ref_count{0};
        std::string group_name;
    };

    static ResourceUsage
    cappedAmount(const ResourceUsage& sum, const ResourceUsage& ub) {
        return {std::min(std::max(sum.memory_bytes, int64_t{0}), ub.memory_bytes),
                std::min(std::max(sum.file_bytes, int64_t{0}), ub.file_bytes)};
    }

    mutable std::shared_mutex map_mtx_;
    std::unordered_map<std::string, uint64_t> name_to_handle_;
    std::unordered_map<uint64_t, std::shared_ptr<GroupState>> handle_state_;
    uint64_t next_handle_{1};
};

}  // namespace milvus::cachinglayer
