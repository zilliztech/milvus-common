#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "cachinglayer/LoadingOverhead.h"
#include "cachinglayer/Utils.h"
#include "cachinglayer/lrucache/DList.h"

using namespace milvus::cachinglayer;
using milvus::cachinglayer::internal::DList;

class LoadingOverheadGroupTest : public ::testing::Test {
 protected:
    std::shared_ptr<LoadingOverheadGroup>
    CreateMemoryGroup(const LoadingOverheadPolicy& policy) {
        return dlist_->CreateLoadingOverheadGroup(LoadingOverheadDimension::kMemory, policy);
    }

    LoadingOverheadConfig
    BindMemory(const std::shared_ptr<LoadingOverheadGroup>& group,
               std::optional<int64_t> max_runtime_unit = std::nullopt) {
        LoadingOverheadConfig binding{
            LoadingOverheadGroupBinding{group, max_runtime_unit},
            std::nullopt,
        };
        dlist_->BindLoadingOverheadGroups(binding);
        return binding;
    }

    ResourceUsage
    Reserve(const LoadingOverheadConfig& binding, ResourceUsage overhead) {
        auto result = std::move(dlist_->ReserveLoadingResourceWithTimeout(
                                    /*loaded=*/{}, overhead, &binding, std::chrono::milliseconds(0)))
                          .get();
        EXPECT_TRUE(result.success);
        return result.reserved;
    }

    ResourceUsage
    Release(const LoadingOverheadConfig& binding, ResourceUsage overhead) {
        return dlist_->ReleaseLoadingResource(/*loaded=*/{}, overhead, &binding);
    }

    void
    ReserveMemory(const LoadingOverheadConfig& binding, int count, int64_t overhead) {
        for (int i = 0; i < count; ++i) {
            Reserve(binding, {overhead, 0});
        }
    }

    std::shared_ptr<DList> dlist_ = std::make_shared<DList>(false, ResourceUsage{1'000'000, 1'000'000}, ResourceUsage{},
                                                            ResourceUsage{}, EvictionConfig{});
};

TEST_F(LoadingOverheadGroupTest, BindingRequiresGroup) {
    EXPECT_THROW(dlist_->BindLoadingOverheadGroups(LoadingOverheadConfig{
                     LoadingOverheadGroupBinding{},
                     std::nullopt,
                 }),
                 std::invalid_argument);
}

TEST_F(LoadingOverheadGroupTest, FailedBindingDoesNotAttachConfiguredDimensions) {
    auto memory = CreateMemoryGroup(LoadingOverheadPolicy::Passthrough());

    EXPECT_THROW(dlist_->BindLoadingOverheadGroups(LoadingOverheadConfig{
                     LoadingOverheadGroupBinding{memory},
                     LoadingOverheadGroupBinding{memory},
                 }),
                 std::invalid_argument);

    EXPECT_EQ(dlist_->UpdateLoadingOverheadGroup(memory, LoadingOverheadPolicy::Executor(1)),
              LoadingOverheadUpdateResult::kApplied);
}

TEST_F(LoadingOverheadGroupTest, FixedGroupCapsAndReleasesReservation) {
    auto group = CreateMemoryGroup(LoadingOverheadPolicy::Fixed(200));
    auto binding = BindMemory(group);

    EXPECT_EQ(Reserve(binding, {150, 0}), (ResourceUsage{150, 0}));
    EXPECT_EQ(Reserve(binding, {100, 0}), (ResourceUsage{50, 0}));
    EXPECT_EQ(Reserve(binding, {100, 0}), ResourceUsage{});

    EXPECT_EQ(Release(binding, {100, 0}), ResourceUsage{});
    EXPECT_EQ(Release(binding, {100, 0}), (ResourceUsage{50, 0}));
    EXPECT_EQ(Release(binding, {150, 0}), (ResourceUsage{150, 0}));
}

TEST_F(LoadingOverheadGroupTest, GroupsReserveIndependently) {
    auto vector_group = CreateMemoryGroup(LoadingOverheadPolicy::Fixed(200));
    auto scalar_group = CreateMemoryGroup(LoadingOverheadPolicy::Fixed(100));

    auto vector = BindMemory(vector_group);
    auto scalar = BindMemory(scalar_group);

    EXPECT_EQ(Reserve(vector, {300, 0}), (ResourceUsage{200, 0}));
    EXPECT_EQ(Reserve(scalar, {300, 0}), (ResourceUsage{100, 0}));
}

TEST_F(LoadingOverheadGroupTest, ConcurrentReserveReleasePreservesAccounting) {
    auto group = CreateMemoryGroup(LoadingOverheadPolicy::Fixed(200));
    auto binding = BindMemory(group);

    constexpr int kThreadCount = 10;
    constexpr int kOperationsPerThread = 100;
    std::vector<std::thread> threads;
    std::atomic<int64_t> total_reserved{0};
    std::atomic<int64_t> total_released{0};

    threads.reserve(kThreadCount);
    for (int i = 0; i < kThreadCount; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < kOperationsPerThread; ++j) {
                total_reserved += Reserve(binding, {10, 0}).memory_bytes;
            }
            for (int j = 0; j < kOperationsPerThread; ++j) {
                total_released += Release(binding, {10, 0}).memory_bytes;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_EQ(total_reserved.load(), total_released.load());
}

TEST_F(LoadingOverheadGroupTest, BindingRejectsInvalidMetadata) {
    auto group = CreateMemoryGroup(LoadingOverheadPolicy::Passthrough());

    EXPECT_THROW(dlist_->BindLoadingOverheadGroups(LoadingOverheadConfig{
                     LoadingOverheadGroupBinding{group, -1},
                     std::nullopt,
                 }),
                 std::invalid_argument);
}

TEST_F(LoadingOverheadGroupTest, GroupUsesBoundRuntimeUnit) {
    auto group = CreateMemoryGroup(LoadingOverheadPolicy::Executor(1));
    auto binding = BindMemory(group, 100);

    EXPECT_EQ(Reserve(binding, {300, 0}), (ResourceUsage{100, 0}));
    EXPECT_EQ(Release(binding, {300, 0}), (ResourceUsage{100, 0}));
}

TEST_F(LoadingOverheadGroupTest, BudgetAndExecutorPoliciesCanReplaceEachOther) {
    auto group = CreateMemoryGroup(LoadingOverheadPolicy::Budget(200));
    auto config = BindMemory(group, 100);

    EXPECT_EQ(Reserve(config, {500, 0}), (ResourceUsage{200, 0}));

    EXPECT_EQ(dlist_->UpdateLoadingOverheadGroup(group, LoadingOverheadPolicy::Executor(1)),
              LoadingOverheadUpdateResult::kApplied);
    EXPECT_EQ(Release(config, {100, 0}), (ResourceUsage{100, 0}));

    EXPECT_EQ(dlist_->UpdateLoadingOverheadGroup(group, LoadingOverheadPolicy::Budget(300)),
              LoadingOverheadUpdateResult::kApplied);
    EXPECT_EQ(Reserve(config, {100, 0}), (ResourceUsage{200, 0}));
    EXPECT_EQ(Release(config, {500, 0}), (ResourceUsage{300, 0}));
}

TEST_F(LoadingOverheadGroupTest, BoundedGroupRequiresBoundRuntimeUnit) {
    auto executor = CreateMemoryGroup(LoadingOverheadPolicy::Executor(1));

    EXPECT_THROW(dlist_->BindLoadingOverheadGroups(LoadingOverheadConfig{
                     LoadingOverheadGroupBinding{executor},
                     std::nullopt,
                 }),
                 std::invalid_argument);

    auto budget = CreateMemoryGroup(LoadingOverheadPolicy::Budget(100));
    EXPECT_THROW(dlist_->BindLoadingOverheadGroups(LoadingOverheadConfig{
                     LoadingOverheadGroupBinding{budget},
                     std::nullopt,
                 }),
                 std::invalid_argument);
}

TEST_F(LoadingOverheadGroupTest, PassthroughAllowsMissingRuntimeUnitButBoundedReconfigurationDoesNot) {
    auto group = CreateMemoryGroup(LoadingOverheadPolicy::Passthrough());
    auto binding = BindMemory(group);

    EXPECT_EQ(Reserve(binding, {300, 0}), (ResourceUsage{300, 0}));
    EXPECT_EQ(Release(binding, {300, 0}), (ResourceUsage{300, 0}));

    EXPECT_EQ(dlist_->UpdateLoadingOverheadGroup(group, LoadingOverheadPolicy::Executor(1)),
              LoadingOverheadUpdateResult::kIncompatiblePolicy);

    dlist_->UnbindLoadingOverheadGroups(binding);
    EXPECT_EQ(dlist_->UpdateLoadingOverheadGroup(group, LoadingOverheadPolicy::Executor(1)),
              LoadingOverheadUpdateResult::kApplied);
}

TEST_F(LoadingOverheadGroupTest, GroupCachesMaximumAcrossRuntimeUnitBounds) {
    auto group = CreateMemoryGroup(LoadingOverheadPolicy::Executor(1));

    auto large = BindMemory(group, 300);
    auto small = BindMemory(group, 100);

    EXPECT_EQ(Reserve(small, {500, 0}), (ResourceUsage{300, 0}));
    EXPECT_EQ(Release(small, {500, 0}), (ResourceUsage{300, 0}));

    dlist_->UnbindLoadingOverheadGroups(large);
    EXPECT_EQ(Reserve(small, {500, 0}), (ResourceUsage{100, 0}));
    EXPECT_EQ(Release(small, {500, 0}), (ResourceUsage{100, 0}));
}

TEST_F(LoadingOverheadGroupTest, GroupSurvivesWithoutBindings) {
    auto group = CreateMemoryGroup(LoadingOverheadPolicy::Executor(1));
    auto first = BindMemory(group, 100);
    dlist_->UnbindLoadingOverheadGroups(first);

    auto second = BindMemory(group, 50);
    EXPECT_EQ(Reserve(second, {300, 0}), (ResourceUsage{50, 0}));
    EXPECT_EQ(Release(second, {300, 0}), (ResourceUsage{50, 0}));
}

TEST_F(LoadingOverheadGroupTest, PolicyTighteningAppliesImmediately) {
    auto group = CreateMemoryGroup(LoadingOverheadPolicy::Executor(4));
    auto binding = BindMemory(group, 100);
    ReserveMemory(binding, 5, 100);

    EXPECT_EQ(dlist_->UpdateLoadingOverheadGroup(group, LoadingOverheadPolicy::Executor(1)),
              LoadingOverheadUpdateResult::kApplied);

    EXPECT_EQ(Release(binding, {100, 0}), (ResourceUsage{300, 0}));
    EXPECT_EQ(Release(binding, {100, 0}), ResourceUsage{});
    EXPECT_EQ(Release(binding, {100, 0}), ResourceUsage{});
    EXPECT_EQ(Release(binding, {100, 0}), ResourceUsage{});
    EXPECT_EQ(Release(binding, {100, 0}), (ResourceUsage{100, 0}));
}

TEST_F(LoadingOverheadGroupTest, PolicyExpansionReconcilesOnNextReserve) {
    auto group = CreateMemoryGroup(LoadingOverheadPolicy::Executor(1));
    auto binding = BindMemory(group, 100);
    ReserveMemory(binding, 5, 100);

    EXPECT_EQ(dlist_->UpdateLoadingOverheadGroup(group, LoadingOverheadPolicy::Executor(2)),
              LoadingOverheadUpdateResult::kApplied);

    EXPECT_EQ(Reserve(binding, {100, 0}), (ResourceUsage{100, 0}));
    EXPECT_EQ(Release(binding, {100, 0}), ResourceUsage{});
    EXPECT_EQ(Release(binding, {500, 0}), (ResourceUsage{200, 0}));
}

TEST_F(LoadingOverheadGroupTest, FailedReserveRevertsDeltaAfterPolicyExpansion) {
    auto group = CreateMemoryGroup(LoadingOverheadPolicy::Executor(1));
    auto binding = BindMemory(group, 100);
    ASSERT_EQ(Reserve(binding, {500, 0}), (ResourceUsage{100, 0}));
    ASSERT_EQ(dlist_->UpdateLoadingOverheadGroup(group, LoadingOverheadPolicy::Executor(4)),
              LoadingOverheadUpdateResult::kApplied);

    ASSERT_TRUE(std::move(dlist_->ReserveLoadingResourceWithTimeout({999'700, 0}, std::chrono::milliseconds(0))).get());
    const auto failed = std::move(dlist_->ReserveLoadingResourceWithTimeout(
                                      /*loaded=*/{}, /*overhead=*/{100, 0}, &binding, std::chrono::milliseconds(0)))
                            .get();
    EXPECT_FALSE(failed.success);
    dlist_->ReleaseLoadingResource({999'700, 0});

    EXPECT_EQ(Release(binding, {500, 0}), (ResourceUsage{100, 0}));
}

TEST_F(LoadingOverheadGroupTest, DimensionsUseIndependentGroupsWhileAbsentFilePassesThrough) {
    auto memory_group = CreateMemoryGroup(LoadingOverheadPolicy::Fixed(200));
    auto file_group =
        dlist_->CreateLoadingOverheadGroup(LoadingOverheadDimension::kFile, LoadingOverheadPolicy::Fixed(50));

    auto scalar_binding = BindMemory(memory_group);
    LoadingOverheadConfig field_binding{
        LoadingOverheadGroupBinding{memory_group},
        LoadingOverheadGroupBinding{file_group},
    };
    dlist_->BindLoadingOverheadGroups(field_binding);

    auto scalar_first = Reserve(scalar_binding, {150, 100});
    EXPECT_EQ(scalar_first, (ResourceUsage{150, 100}));

    auto field_first = Reserve(field_binding, {100, 40});
    EXPECT_EQ(field_first, (ResourceUsage{50, 40}));

    auto scalar_second = Reserve(scalar_binding, {100, 200});
    EXPECT_EQ(scalar_second, (ResourceUsage{0, 200}));

    auto field_second = Reserve(field_binding, {0, 20});
    EXPECT_EQ(field_second, (ResourceUsage{0, 10}));

    auto scalar_first_release = Release(scalar_binding, {150, 100});
    EXPECT_EQ(scalar_first_release, (ResourceUsage{0, 100}));

    auto field_first_release = Release(field_binding, {100, 40});
    EXPECT_EQ(field_first_release, (ResourceUsage{100, 30}));

    auto scalar_second_release = Release(scalar_binding, {100, 200});
    EXPECT_EQ(scalar_second_release, (ResourceUsage{100, 200}));

    auto field_second_release = Release(field_binding, {0, 20});
    EXPECT_EQ(field_second_release, (ResourceUsage{0, 20}));
}
