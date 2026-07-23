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

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "cachinglayer/Utils.h"

namespace milvus::cachinglayer {

class LoadingOverheadGroup;
using LoadingOverheadGroupHandle = std::shared_ptr<LoadingOverheadGroup>;

/** @brief Immutable Group policy configuration. */
class LoadingOverheadPolicy {
 private:
    enum class Kind {
        kFixed,
        kPassthrough,
        kBudget,
        kExecutor,
    };

 public:
    static LoadingOverheadPolicy
    Fixed(int64_t upper_bound) {
        LoadingOverheadPolicy policy{Kind::kFixed};
        policy.fixed_upper_bound_ = ValidateNonNegative(upper_bound, "fixed upper bound");
        return policy;
    }

    static LoadingOverheadPolicy
    Passthrough() {
        return LoadingOverheadPolicy{Kind::kPassthrough};
    }

    static LoadingOverheadPolicy
    Budget(int64_t capacity_bytes) {
        LoadingOverheadPolicy policy{Kind::kBudget};
        policy.budget_capacity_bytes_ = ValidateNonNegative(capacity_bytes, "Budget capacity");
        return policy;
    }

    static LoadingOverheadPolicy
    Executor(int64_t configured_workers) {
        LoadingOverheadPolicy policy{Kind::kExecutor};
        policy.configured_workers_ = ValidateNonNegative(configured_workers, "executor worker count");
        return policy;
    }

 private:
    friend class LoadingOverheadGroup;

    explicit LoadingOverheadPolicy(Kind kind) : kind_(kind), unused_(0) {
    }

    static int64_t
    ValidateNonNegative(int64_t value, const char* name) {
        if (value < 0) {
            throw std::invalid_argument(std::string("LoadingOverheadPolicy ") + name + " must be non-negative");
        }
        return value;
    }

    Kind kind_;
    union {
        int64_t fixed_upper_bound_;
        int64_t budget_capacity_bytes_;
        int64_t configured_workers_;
        int64_t unused_;
    };
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

    LoadingOverheadGroup(LoadingOverheadDimension dimension, LoadingOverheadPolicy policy)
        : dimension_(dimension), policy_(policy) {
    }

    void
    validateBinding(LoadingOverheadDimension dimension, const std::optional<int64_t>& max_runtime_unit) const;

    void
    bind(const std::optional<int64_t>& max_runtime_unit);

    void
    unbind(LoadingOverheadDimension dimension, const std::optional<int64_t>& max_runtime_unit) noexcept;

    LoadingOverheadUpdateResult
    updatePolicy(LoadingOverheadPolicy policy);

    void
    validateReserve(int64_t overhead) const;

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

    [[nodiscard]] static bool
    requiresRuntimeUnitBound(const LoadingOverheadPolicy& policy) noexcept;

    [[nodiscard]] static int64_t
    resolveBound(const LoadingOverheadPolicy& policy, int64_t max_runtime_unit_bytes) noexcept;

    [[nodiscard]] static int64_t
    saturatingMultiply(int64_t lhs, int64_t rhs) noexcept;

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
    LoadingOverheadGroupHandle group;

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
