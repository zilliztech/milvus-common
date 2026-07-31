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

template <typename Fn>
void
ExpectInvalidParameter(Fn&& fn) {
    try {
        fn();
        ADD_FAILURE() << "Expected InvalidParameter";
    } catch (const milvus::SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), milvus::ErrorCode::InvalidParameter);
    } catch (const std::exception& error) {
        ADD_FAILURE() << "Expected SegcoreError, got: " << error.what();
    }
}

TEST(LoadingOverheadPolicyTest, BudgetRejectsNegativeCapacity) {
    ExpectInvalidParameter([] { (void)LoadingOverheadPolicy::Budget(-1); });
}

TEST(LoadingOverheadPolicyTest, ExecutorRejectsNegativeWorkerCount) {
    ExpectInvalidParameter([] { (void)LoadingOverheadPolicy::Executor(-1); });
}

}  // namespace
}  // namespace milvus::cachinglayer
