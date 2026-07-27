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
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "cachinglayer/Utils.h"

namespace milvus::cachinglayer {

/** @brief Immutable Group policy used to derive a reservation bound. */
class LoadingOverheadPolicy {
 public:
    enum class Kind {
        kFixed,
        kPassthrough,
        kBudget,
        kExecutor,
    };

    static LoadingOverheadPolicy
    Fixed(int64_t upper_bound) {
        return {Kind::kFixed, ValidateNonNegative(upper_bound, "fixed upper bound")};
    }

    static LoadingOverheadPolicy
    Passthrough() {
        return {Kind::kPassthrough, 0};
    }

    static LoadingOverheadPolicy
    Budget(int64_t capacity_bytes) {
        return {Kind::kBudget, ValidateNonNegative(capacity_bytes, "Budget capacity")};
    }

    static LoadingOverheadPolicy
    Executor(int64_t configured_workers) {
        return {Kind::kExecutor, ValidateNonNegative(configured_workers, "executor worker count")};
    }

    [[nodiscard]] int64_t
    ResolveBound(int64_t max_runtime_unit_bytes) const noexcept {
        switch (kind_) {
            case Kind::kFixed:
                return value_;
            case Kind::kPassthrough:
                return std::numeric_limits<int64_t>::max();
            case Kind::kBudget:
                if (max_runtime_unit_bytes < 0 || value_ == 0) {
                    return std::numeric_limits<int64_t>::max();
                }
                return std::max(value_, max_runtime_unit_bytes);
            case Kind::kExecutor:
                return SaturatingMultiply(value_, max_runtime_unit_bytes);
        }
        return std::numeric_limits<int64_t>::max();
    }

    [[nodiscard]] bool
    RequiresRuntimeUnitBound() const noexcept {
        return kind_ == Kind::kBudget || kind_ == Kind::kExecutor;
    }

 private:
    LoadingOverheadPolicy(Kind kind, int64_t value) : kind_(kind), value_(value) {
    }

    static int64_t
    ValidateNonNegative(int64_t value, const char* name) {
        if (value < 0) {
            throw std::invalid_argument(std::string("LoadingOverheadPolicy ") + name + " must be non-negative");
        }
        return value;
    }

    static int64_t
    SaturatingMultiply(int64_t lhs, int64_t rhs) noexcept {
        if (lhs < 0 || rhs < 0) {
            return std::numeric_limits<int64_t>::max();
        }
        if (lhs == 0 || rhs == 0) {
            return 0;
        }
        if (lhs > std::numeric_limits<int64_t>::max() / rhs) {
            return std::numeric_limits<int64_t>::max();
        }
        return lhs * rhs;
    }

    Kind kind_;
    int64_t value_;
};

/** @brief Resource dimension managed by a loading-overhead Group. */
enum class LoadingOverheadDimension {
    kMemory,
    kFile,
};

enum class LoadingOverheadUpdateResult;

namespace internal {
class DList;
}

// Opaque shared state for one independently created loading-overhead Group.
// DList serializes mutations with admission accounting under its list mutex.
class LoadingOverheadGroup {
 public:
    LoadingOverheadGroup(const LoadingOverheadGroup&) = delete;
    LoadingOverheadGroup&
    operator=(const LoadingOverheadGroup&) = delete;
    LoadingOverheadGroup(LoadingOverheadGroup&&) = delete;
    LoadingOverheadGroup&
    operator=(LoadingOverheadGroup&&) = delete;
    ~LoadingOverheadGroup() = default;

 private:
    friend class internal::DList;

    LoadingOverheadGroup(const internal::DList* owner, LoadingOverheadDimension dimension, LoadingOverheadPolicy policy)
        : owner_(owner), dimension_(dimension), policy_(std::move(policy)) {
    }

    void
    validateBinding(const internal::DList* owner, LoadingOverheadDimension dimension,
                    const std::optional<int64_t>& max_runtime_unit) const;

    void
    bind(const std::optional<int64_t>& max_runtime_unit);

    void
    unbind(const internal::DList* owner, LoadingOverheadDimension dimension,
           const std::optional<int64_t>& max_runtime_unit) noexcept;

    LoadingOverheadUpdateResult
    updatePolicy(const internal::DList* owner, LoadingOverheadPolicy policy);

    int64_t
    reserve(int64_t overhead);

    void
    rollbackReserve(int64_t overhead, int64_t reserved) noexcept;

    int64_t
    release(int64_t overhead) noexcept;

    [[nodiscard]] int64_t
    computeTarget() const noexcept;

    [[nodiscard]] bool
    hasMissingRuntimeUnit() const noexcept;

    const internal::DList* owner_;
    LoadingOverheadDimension dimension_;
    LoadingOverheadPolicy policy_;
    int64_t sum_of_overhead_{0};
    std::multiset<int64_t> runtime_unit_bounds_;
    int64_t max_runtime_unit_{0};
    int64_t overhead_reserved_{0};
    uint64_t binding_count_{0};
};

/**
 * @brief Translator binding to one loading-overhead Group.
 */
struct LoadingOverheadGroupBinding {
    /**
     * @brief Group created independently before this binding is attached.
     */
    std::shared_ptr<LoadingOverheadGroup> group;

    /**
     * @brief Optional conservative bound for one runtime unit from this binding.
     *
     * The Group caches the maximum value across attached bindings. A
     * policy that bounds Budget acquisitions or executor tasks requires
     * this value on every binding; binding or policy replacement is
     * rejected otherwise. Fixed and Passthrough Groups do not require it. It
     * must describe one runtime unit, not the sum of a multi-cell load request.
     *
     * @pre When present, the value is non-negative.
     */
    std::optional<int64_t> max_runtime_unit;
};

/**
 * @brief Loading-overhead configuration for a Translator.
 *
 * Each resource dimension binds independently to a Group. An absent dimension is
 * not assigned to a Group; its request-local loading overhead passes through
 * unchanged and remains subject to DList admission limits.
 */
struct LoadingOverheadConfig {
    /** @brief Memory-dimension binding, or std::nullopt for request-local passthrough. */
    std::optional<LoadingOverheadGroupBinding> memory;

    /** @brief File-dimension binding, or std::nullopt for request-local passthrough. */
    std::optional<LoadingOverheadGroupBinding> file;
};

/** @brief Result of updating a loading-overhead Group. */
enum class LoadingOverheadUpdateResult {
    /** The desired policy was committed. */
    kApplied,

    /** The requested policy is incompatible with the Group's bindings. */
    kIncompatiblePolicy,

    /** The Group handle was invalid. */
    kInvalidArgument,
};

}  // namespace milvus::cachinglayer
