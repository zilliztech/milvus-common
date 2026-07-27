#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "cachinglayer/LoadingOverhead.h"

namespace milvus::cachinglayer {
namespace {

TEST(LoadingOverheadPolicyTest, FixedUsesConfiguredBound) {
    const auto policy = LoadingOverheadPolicy::Fixed(400);

    EXPECT_EQ(policy.ResolveBound(/*max_runtime_unit_bytes=*/100), 400);
    EXPECT_FALSE(policy.RequiresRuntimeUnitBound());
}

TEST(LoadingOverheadPolicyTest, FixedRejectsNegativeBound) {
    EXPECT_THROW(LoadingOverheadPolicy::Fixed(-1), std::invalid_argument);
}

TEST(LoadingOverheadPolicyTest, PassthroughUsesUnlimitedBound) {
    const auto policy = LoadingOverheadPolicy::Passthrough();

    EXPECT_EQ(policy.ResolveBound(/*max_runtime_unit_bytes=*/100), std::numeric_limits<int64_t>::max());
    EXPECT_FALSE(policy.RequiresRuntimeUnitBound());
}

TEST(LoadingOverheadPolicyTest, BudgetUsesLargerOfCapacityAndRuntimeUnit) {
    const auto policy = LoadingOverheadPolicy::Budget(400);

    EXPECT_EQ(policy.ResolveBound(/*max_runtime_unit_bytes=*/100), 400);
    EXPECT_EQ(policy.ResolveBound(/*max_runtime_unit_bytes=*/500), 500);
    EXPECT_TRUE(policy.RequiresRuntimeUnitBound());
}

TEST(LoadingOverheadPolicyTest, ZeroBudgetUsesUnlimitedBound) {
    const auto policy = LoadingOverheadPolicy::Budget(0);

    EXPECT_EQ(policy.ResolveBound(/*max_runtime_unit_bytes=*/50), std::numeric_limits<int64_t>::max());
}

TEST(LoadingOverheadPolicyTest, BudgetRejectsNegativeCapacity) {
    EXPECT_THROW(LoadingOverheadPolicy::Budget(-1), std::invalid_argument);
}

TEST(LoadingOverheadPolicyTest, ExecutorScalesRuntimeUnitByWorkerCount) {
    const auto policy = LoadingOverheadPolicy::Executor(6);

    EXPECT_EQ(policy.ResolveBound(/*max_runtime_unit_bytes=*/100), 600);
    EXPECT_TRUE(policy.RequiresRuntimeUnitBound());
}

TEST(LoadingOverheadPolicyTest, ExecutorSaturatesMultiplication) {
    const auto policy = LoadingOverheadPolicy::Executor(std::numeric_limits<int64_t>::max());

    EXPECT_EQ(policy.ResolveBound(/*max_runtime_unit_bytes=*/2), std::numeric_limits<int64_t>::max());
}

TEST(LoadingOverheadPolicyTest, ExecutorRejectsNegativeWorkerCount) {
    EXPECT_THROW(LoadingOverheadPolicy::Executor(-1), std::invalid_argument);
}

}  // namespace
}  // namespace milvus::cachinglayer
