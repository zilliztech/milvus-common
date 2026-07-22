#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>

#include "cachinglayer/LoadingOverhead.h"

namespace milvus::cachinglayer {
namespace {

template <typename Policy>
concept EvaluatesLoadingOverhead = requires(const Policy& policy) {
    policy.ResolveBound(0);
    policy.RequiresRuntimeUnitBound();
};

static_assert(!EvaluatesLoadingOverhead<LoadingOverheadPolicy>);

TEST(LoadingOverheadPolicyTest, FixedRejectsNegativeBound) {
    EXPECT_THROW(LoadingOverheadPolicy::Fixed(-1), std::invalid_argument);
}

TEST(LoadingOverheadPolicyTest, BudgetRejectsNegativeCapacity) {
    EXPECT_THROW(LoadingOverheadPolicy::Budget(-1), std::invalid_argument);
}

TEST(LoadingOverheadPolicyTest, ExecutorRejectsNegativeWorkerCount) {
    EXPECT_THROW(LoadingOverheadPolicy::Executor(-1), std::invalid_argument);
}

}  // namespace
}  // namespace milvus::cachinglayer
