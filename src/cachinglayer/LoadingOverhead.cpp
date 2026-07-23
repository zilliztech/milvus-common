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

#include "cachinglayer/LoadingOverhead.h"

#include "log/Log.h"

namespace milvus::cachinglayer {

namespace {

const char*
DimensionName(LoadingOverheadDimension dimension) noexcept {
    return dimension == LoadingOverheadDimension::kMemory ? "memory" : "file";
}

}  // namespace

void
LoadingOverheadGroup::validateBinding(const internal::DList* owner, LoadingOverheadDimension dimension,
                                      const std::optional<int64_t>& max_runtime_unit) const {
    if (owner_ != owner || dimension_ != dimension) {
        throw std::invalid_argument("loading-overhead binding requires a Group from the matching dimension");
    }
    if (max_runtime_unit.has_value() && max_runtime_unit.value() < 0) {
        throw std::invalid_argument("loading-overhead binding runtime-unit bound must be non-negative");
    }
    if (!max_runtime_unit.has_value() && policy_.RequiresRuntimeUnitBound()) {
        throw std::invalid_argument("bounded loading-overhead Group binding requires max_runtime_unit");
    }
}

void
LoadingOverheadGroup::bind(const std::optional<int64_t>& max_runtime_unit) {
    if (max_runtime_unit.has_value()) {
        runtime_unit_bounds_.insert(max_runtime_unit.value());
        max_runtime_unit_ = std::max(max_runtime_unit_, max_runtime_unit.value());
    }
    ++binding_count_;
    LOG_DEBUG("[MCL] LoadingOverheadGroup bound {} Group (bindings={})", DimensionName(dimension_), binding_count_);
}

void
LoadingOverheadGroup::unbind(const internal::DList* owner, LoadingOverheadDimension dimension,
                             const std::optional<int64_t>& max_runtime_unit) noexcept {
    if (owner_ != owner || dimension_ != dimension) {
        LOG_ERROR("[MCL] LoadingOverheadGroup cannot unbind invalid {} Group", DimensionName(dimension));
        return;
    }

    if (max_runtime_unit.has_value()) {
        const auto unit_it = runtime_unit_bounds_.find(max_runtime_unit.value());
        if (unit_it == runtime_unit_bounds_.end()) {
            LOG_ERROR("[MCL] LoadingOverheadGroup runtime-unit binding {} not found", max_runtime_unit.value());
        } else {
            runtime_unit_bounds_.erase(unit_it);
            max_runtime_unit_ = runtime_unit_bounds_.empty() ? int64_t{0} : *runtime_unit_bounds_.rbegin();
        }
    }

    if (binding_count_ > 0) {
        --binding_count_;
    }
    LOG_DEBUG("[MCL] LoadingOverheadGroup unbound {} Group (bindings={})", DimensionName(dimension_), binding_count_);
    if (binding_count_ == 0 && (sum_of_overhead_ > 0 || overhead_reserved_ > 0)) {
        LOG_ERROR(
            "[MCL] LoadingOverheadGroup {} Group binding_count=0 with residual reservations: "
            "sum_of_overhead={}, overhead_reserved={}",
            DimensionName(dimension_), sum_of_overhead_, overhead_reserved_);
    }
}

LoadingOverheadUpdateResult
LoadingOverheadGroup::updatePolicy(const internal::DList* owner, LoadingOverheadPolicy policy) {
    if (owner_ != owner) {
        return LoadingOverheadUpdateResult::kInvalidArgument;
    }
    if (policy.RequiresRuntimeUnitBound() && hasMissingRuntimeUnit()) {
        return LoadingOverheadUpdateResult::kIncompatiblePolicy;
    }
    policy_ = std::move(policy);
    return LoadingOverheadUpdateResult::kApplied;
}

int64_t
LoadingOverheadGroup::reserve(int64_t overhead) {
    sum_of_overhead_ += overhead;
    const auto delta = std::max(computeTarget() - overhead_reserved_, int64_t{0});
    overhead_reserved_ += delta;
    return delta;
}

void
LoadingOverheadGroup::rollbackReserve(int64_t overhead, int64_t reserved) noexcept {
    sum_of_overhead_ -= overhead;
    if (sum_of_overhead_ < 0) {
        LOG_ERROR("[MCL] LoadingOverheadGroup Reserve rollback: sum_of_overhead < 0");
        sum_of_overhead_ = 0;
    }
    overhead_reserved_ -= reserved;
    if (overhead_reserved_ < 0) {
        LOG_ERROR("[MCL] LoadingOverheadGroup Reserve rollback: overhead_reserved < 0");
        overhead_reserved_ = 0;
    }
}

int64_t
LoadingOverheadGroup::release(int64_t overhead) noexcept {
    sum_of_overhead_ -= overhead;
    if (sum_of_overhead_ < 0) {
        LOG_ERROR("[MCL] LoadingOverheadGroup Release: sum_of_overhead < 0");
        sum_of_overhead_ = 0;
    }

    const auto delta = std::max(overhead_reserved_ - computeTarget(), int64_t{0});
    overhead_reserved_ -= delta;
    return delta;
}

int64_t
LoadingOverheadGroup::computeTarget() const noexcept {
    const auto bound = policy_.ResolveBound(max_runtime_unit_);
    return std::min(std::max(sum_of_overhead_, int64_t{0}), bound);
}

bool
LoadingOverheadGroup::hasMissingRuntimeUnit() const noexcept {
    return runtime_unit_bounds_.size() != binding_count_;
}

}  // namespace milvus::cachinglayer
