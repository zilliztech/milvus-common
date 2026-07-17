#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "cachinglayer/LoadingOverhead.h"
#include "cachinglayer/LoadingOverheadTracker.h"
#include "cachinglayer/Utils.h"

using namespace milvus::cachinglayer;

class LoadingOverheadTrackerTest : public ::testing::Test {
 protected:
    LoadingOverheadTracker tracker_;
};

TEST_F(LoadingOverheadTrackerTest, NoUpperBoundPassThrough) {
    // Without registering a UB, all amounts pass through unchanged.
    auto handle = tracker_.Register("vector_index", LoadingOverheadTracker::kUnlimited);
    auto delta = tracker_.Reserve(handle, {100, 0});
    EXPECT_EQ(delta.memory_bytes, 100);
    EXPECT_EQ(delta.file_bytes, 0);

    auto release = tracker_.Release(handle, {100, 0});
    EXPECT_EQ(release.memory_bytes, 100);
    EXPECT_EQ(release.file_bytes, 0);
}

TEST_F(LoadingOverheadTrackerTest, BasicCapping) {
    auto handle = tracker_.Register("vector_index", {200, 0});

    // First reserve: 100, sum=100 <= UB=200, full amount passes through
    auto d1 = tracker_.Reserve(handle, {100, 0});
    EXPECT_EQ(d1.memory_bytes, 100);

    // Second reserve: 100, sum=200 <= UB=200, full amount passes through
    auto d2 = tracker_.Reserve(handle, {100, 0});
    EXPECT_EQ(d2.memory_bytes, 100);

    // Third reserve: 100, sum=300 > UB=200, capped: delta = 0
    auto d3 = tracker_.Reserve(handle, {100, 0});
    EXPECT_EQ(d3.memory_bytes, 0);

    // Fourth reserve: 100, sum=400 > UB=200, capped: delta = 0
    auto d4 = tracker_.Reserve(handle, {100, 0});
    EXPECT_EQ(d4.memory_bytes, 0);
}

TEST_F(LoadingOverheadTrackerTest, BasicRelease) {
    auto handle = tracker_.Register("vector_index", {200, 0});

    // Reserve 4x100, total sum=400, actual DList reserve = 200
    tracker_.Reserve(handle, {100, 0});
    tracker_.Reserve(handle, {100, 0});
    tracker_.Reserve(handle, {100, 0});
    tracker_.Reserve(handle, {100, 0});

    // Release first 100: sum 400->300, both >= UB, release 0
    auto r1 = tracker_.Release(handle, {100, 0});
    EXPECT_EQ(r1.memory_bytes, 0);

    // Release second 100: sum 300->200, release 0
    auto r2 = tracker_.Release(handle, {100, 0});
    EXPECT_EQ(r2.memory_bytes, 0);

    // Release third 100: sum 200->100, release 100
    auto r3 = tracker_.Release(handle, {100, 0});
    EXPECT_EQ(r3.memory_bytes, 100);

    // Release fourth 100: sum 100->0, release 100
    auto r4 = tracker_.Release(handle, {100, 0});
    EXPECT_EQ(r4.memory_bytes, 100);
}

TEST_F(LoadingOverheadTrackerTest, TotalReservedEqualsReleased) {
    auto handle = tracker_.Register("vector_index", {200, 0});

    int64_t total_reserved = 0;
    int64_t total_released = 0;

    // Reserve 10 x 100
    for (int i = 0; i < 10; i++) {
        total_reserved += tracker_.Reserve(handle, {100, 0}).memory_bytes;
    }
    // Should have reserved exactly UB = 200
    EXPECT_EQ(total_reserved, 200);

    // Release all 10
    for (int i = 0; i < 10; i++) {
        total_released += tracker_.Release(handle, {100, 0}).memory_bytes;
    }
    // Total released should equal total reserved
    EXPECT_EQ(total_released, 200);
}

TEST_F(LoadingOverheadTrackerTest, PartialCapping) {
    auto handle = tracker_.Register("vector_index", {200, 0});

    // Reserve 150: sum=150 <= UB=200, full amount
    auto d1 = tracker_.Reserve(handle, {150, 0});
    EXPECT_EQ(d1.memory_bytes, 150);

    // Reserve 100: sum=250 > UB=200, delta = 200-150 = 50
    auto d2 = tracker_.Reserve(handle, {100, 0});
    EXPECT_EQ(d2.memory_bytes, 50);
}

TEST_F(LoadingOverheadTrackerTest, ReleaseUndoesReserve) {
    auto handle = tracker_.Register("vector_index", {200, 0});

    // Reserve 150: actual = 150
    auto d1 = tracker_.Reserve(handle, {150, 0});
    EXPECT_EQ(d1.memory_bytes, 150);

    // Release undoes the reserve: sum 150->0, release = 150
    auto undo = tracker_.Release(handle, {150, 0});
    EXPECT_EQ(undo.memory_bytes, 150);
}

TEST_F(LoadingOverheadTrackerTest, MultipleTypes) {
    auto vec_handle = tracker_.Register("vector_index", {200, 0});
    auto scalar_handle = tracker_.Register("scalar_field", {100, 0});

    // Types are tracked independently
    auto d1 = tracker_.Reserve(vec_handle, {200, 0});
    EXPECT_EQ(d1.memory_bytes, 200);

    auto d2 = tracker_.Reserve(scalar_handle, {100, 0});
    EXPECT_EQ(d2.memory_bytes, 100);

    // Both at UB, further reserves return 0
    auto d3 = tracker_.Reserve(vec_handle, {100, 0});
    EXPECT_EQ(d3.memory_bytes, 0);

    auto d4 = tracker_.Reserve(scalar_handle, {50, 0});
    EXPECT_EQ(d4.memory_bytes, 0);
}

TEST_F(LoadingOverheadTrackerTest, RegisterUpperBoundTakesMax) {
    auto handle = tracker_.Register("vector_index", {100, 50});
    handle = tracker_.Register("vector_index", {200, 30});

    // UB should be {200, 50} (max per dimension)
    auto ub = tracker_.GetUpperBound(handle);
    EXPECT_EQ(ub.memory_bytes, 200);
    EXPECT_EQ(ub.file_bytes, 50);

    auto d1 = tracker_.Reserve(handle, {200, 50});
    EXPECT_EQ(d1.memory_bytes, 200);
    EXPECT_EQ(d1.file_bytes, 50);

    // Next reserve should be fully capped
    auto d2 = tracker_.Reserve(handle, {100, 100});
    EXPECT_EQ(d2.memory_bytes, 0);
    EXPECT_EQ(d2.file_bytes, 0);
}

TEST_F(LoadingOverheadTrackerTest, HasFiniteUpperBound) {
    auto unlimited_handle = tracker_.Register("vector_index", LoadingOverheadTracker::kUnlimited);
    EXPECT_FALSE(tracker_.HasFiniteUpperBound(unlimited_handle));

    auto finite_handle = tracker_.Register("finite_vector_index", {200, 0});
    EXPECT_TRUE(tracker_.HasFiniteUpperBound(finite_handle));

    auto scalar_handle = tracker_.Register("scalar_field", LoadingOverheadTracker::kUnlimited);
    EXPECT_FALSE(tracker_.HasFiniteUpperBound(scalar_handle));
}

TEST_F(LoadingOverheadTrackerTest, ConcurrentReserveRelease) {
    auto handle = tracker_.Register("vector_index", {200, 0});

    const int num_threads = 10;
    const int ops_per_thread = 100;
    std::vector<std::thread> threads;
    std::atomic<int64_t> total_reserved{0};
    std::atomic<int64_t> total_released{0};

    threads.reserve(num_threads);
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back([&]() {
            for (int j = 0; j < ops_per_thread; j++) {
                auto reserved = tracker_.Reserve(handle, {10, 0});
                total_reserved += reserved.memory_bytes;
            }
            for (int j = 0; j < ops_per_thread; j++) {
                auto released = tracker_.Release(handle, {10, 0});
                total_released += released.memory_bytes;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Total reserved must equal total released
    EXPECT_EQ(total_reserved.load(), total_released.load());
}

TEST_F(LoadingOverheadTrackerTest, DefaultUnlimitedUBFallback) {
    // Register with kUnlimited -> unlimited, behaves like no capping.
    auto handle = tracker_.Register("vector_index", LoadingOverheadTracker::kUnlimited);

    EXPECT_FALSE(tracker_.HasFiniteUpperBound(handle));

    auto d1 = tracker_.Reserve(handle, {1000000000, 0});
    EXPECT_EQ(d1.memory_bytes, 1000000000);

    auto d2 = tracker_.Reserve(handle, {2000000000, 0});
    EXPECT_EQ(d2.memory_bytes, 2000000000);

    auto r1 = tracker_.Release(handle, {1000000000, 0});
    EXPECT_EQ(r1.memory_bytes, 1000000000);

    auto r2 = tracker_.Release(handle, {2000000000, 0});
    EXPECT_EQ(r2.memory_bytes, 2000000000);
}

TEST_F(LoadingOverheadTrackerTest, RegisterUnlimitedThenFiniteKeepsUnlimited) {
    auto handle = tracker_.Register("vector_index", LoadingOverheadTracker::kUnlimited);
    EXPECT_FALSE(tracker_.HasFiniteUpperBound(handle));

    handle = tracker_.Register("vector_index", {200, 0});
    EXPECT_FALSE(tracker_.HasFiniteUpperBound(handle));

    // INT64_MAX is an explicit unlimited upper bound. Use a missing dimension
    // when loading overhead should pass through without joining a capped group.
    auto d1 = tracker_.Reserve(handle, {300, 0});
    EXPECT_EQ(d1.memory_bytes, 300);
}

TEST_F(LoadingOverheadTrackerTest, UnregisteredTypeAutoCreatesUnlimited) {
    auto handle = tracker_.Register("scalar_field", LoadingOverheadTracker::kUnlimited);
    auto d1 = tracker_.Reserve(handle, {500, 0});
    EXPECT_EQ(d1.memory_bytes, 500);

    auto r1 = tracker_.Release(handle, {500, 0});
    EXPECT_EQ(r1.memory_bytes, 500);

    EXPECT_FALSE(tracker_.HasFiniteUpperBound(handle));
}

TEST_F(LoadingOverheadTrackerTest, UBChangesMidFlight) {
    auto handle = tracker_.Register("vector_index", {200, 0});

    // Reserve 3x100 under UB=200: dlist gets 100+100+0 = 200
    auto d1 = tracker_.Reserve(handle, {100, 0});
    EXPECT_EQ(d1.memory_bytes, 100);
    auto d2 = tracker_.Reserve(handle, {100, 0});
    EXPECT_EQ(d2.memory_bytes, 100);
    auto d3 = tracker_.Reserve(handle, {100, 0});
    EXPECT_EQ(d3.memory_bytes, 0);

    int64_t total_reserved = d1.memory_bytes + d2.memory_bytes + d3.memory_bytes;
    EXPECT_EQ(total_reserved, 200);

    // UB changes to 400
    handle = tracker_.Register("vector_index", {400, 0});

    // Release all 3: should release exactly 200 total
    auto r1 = tracker_.Release(handle, {100, 0});
    EXPECT_EQ(r1.memory_bytes, 0);
    auto r2 = tracker_.Release(handle, {100, 0});
    EXPECT_EQ(r2.memory_bytes, 100);
    auto r3 = tracker_.Release(handle, {100, 0});
    EXPECT_EQ(r3.memory_bytes, 100);

    int64_t total_released = r1.memory_bytes + r2.memory_bytes + r3.memory_bytes;
    EXPECT_EQ(total_released, 200);
    EXPECT_EQ(total_reserved, total_released);
}

TEST_F(LoadingOverheadTrackerTest, UnlimitedUBDoesNotDecreaseMidFlight) {
    auto handle = tracker_.Register("vector_index", LoadingOverheadTracker::kUnlimited);
    auto d1 = tracker_.Reserve(handle, {1000, 0});
    EXPECT_EQ(d1.memory_bytes, 1000);
    auto d2 = tracker_.Reserve(handle, {1000, 0});
    EXPECT_EQ(d2.memory_bytes, 1000);

    // A later finite registration must not lower an explicit unlimited upper bound.
    handle = tracker_.Register("vector_index", {200, 0});

    auto d3 = tracker_.Reserve(handle, {100, 0});
    EXPECT_EQ(d3.memory_bytes, 100);

    auto r1 = tracker_.Release(handle, {1000, 0});
    EXPECT_EQ(r1.memory_bytes, 1000);
    auto r2 = tracker_.Release(handle, {1000, 0});
    EXPECT_EQ(r2.memory_bytes, 1000);
    auto r3 = tracker_.Release(handle, {100, 0});
    EXPECT_EQ(r3.memory_bytes, 100);

    int64_t total_reserved = d1.memory_bytes + d2.memory_bytes + d3.memory_bytes;
    int64_t total_released = r1.memory_bytes + r2.memory_bytes + r3.memory_bytes;
    EXPECT_EQ(total_reserved, 2100);
    EXPECT_EQ(total_released, 2100);
}

TEST_F(LoadingOverheadTrackerTest, GetUpperBound) {
    auto unlimited_handle = tracker_.Register("vector_index", LoadingOverheadTracker::kUnlimited);
    EXPECT_EQ(tracker_.GetUpperBound(unlimited_handle), LoadingOverheadTracker::kUnlimited);

    auto finite_handle = tracker_.Register("finite_vector_index", {200, 100});
    auto ub = tracker_.GetUpperBound(finite_handle);
    EXPECT_EQ(ub.memory_bytes, 200);
    EXPECT_EQ(ub.file_bytes, 100);
}

TEST_F(LoadingOverheadTrackerTest, PartialUnlimitedRemainsUnlimitedRegardlessOfRegistrationOrder) {
    constexpr auto kMax = std::numeric_limits<int64_t>::max();

    auto max_first = tracker_.Register("max_first", {kMax, 0});
    tracker_.Register("max_first", {200, 0});
    EXPECT_EQ(tracker_.GetUpperBound(max_first), (ResourceUsage{kMax, 0}));

    auto finite_first = tracker_.Register("finite_first", {200, 0});
    tracker_.Register("finite_first", {kMax, 0});
    EXPECT_EQ(tracker_.GetUpperBound(finite_first), (ResourceUsage{kMax, 0}));
}

TEST_F(LoadingOverheadTrackerTest, LegacyConfigConstructorConfiguresBothDimensions) {
    ResourceUsage upper_bound{200, 50};
    LoadingOverheadConfig config(upper_bound, "legacy_group");

    ASSERT_TRUE(config.memory.has_value());
    EXPECT_EQ(config.memory->upper_bound, 200);
    EXPECT_EQ(config.memory->group, "legacy_group");
    ASSERT_TRUE(config.file.has_value());
    EXPECT_EQ(config.file->upper_bound, 50);
    EXPECT_EQ(config.file->group, "legacy_group");

    auto handle = tracker_.Register(config);
    EXPECT_EQ(tracker_.GetUpperBound(handle), upper_bound);
}

TEST_F(LoadingOverheadTrackerTest, DimensionsShareMemoryWhileScalarFilePassesThrough) {
    auto scalar_handle =
        tracker_.Register(LoadingOverheadConfig{LoadingOverheadDimensionConfig{200, "load_transient"}, std::nullopt});
    auto field_handle = tracker_.Register(LoadingOverheadConfig{LoadingOverheadDimensionConfig{200, "load_transient"},
                                                                LoadingOverheadDimensionConfig{50, "load_transient"}});

    auto scalar_first = tracker_.Reserve(scalar_handle, {150, 100});
    EXPECT_EQ(scalar_first, (ResourceUsage{150, 100}));

    auto field_first = tracker_.Reserve(field_handle, {100, 40});
    EXPECT_EQ(field_first, (ResourceUsage{50, 40}));

    auto scalar_second = tracker_.Reserve(scalar_handle, {100, 200});
    EXPECT_EQ(scalar_second, (ResourceUsage{0, 200}));

    auto field_second = tracker_.Reserve(field_handle, {0, 20});
    EXPECT_EQ(field_second, (ResourceUsage{0, 10}));

    auto scalar_first_release = tracker_.Release(scalar_handle, {150, 100});
    EXPECT_EQ(scalar_first_release, (ResourceUsage{0, 100}));

    auto field_first_release = tracker_.Release(field_handle, {100, 40});
    EXPECT_EQ(field_first_release, (ResourceUsage{100, 30}));

    auto scalar_second_release = tracker_.Release(scalar_handle, {100, 200});
    EXPECT_EQ(scalar_second_release, (ResourceUsage{100, 200}));

    auto field_second_release = tracker_.Release(field_handle, {0, 20});
    EXPECT_EQ(field_second_release, (ResourceUsage{0, 20}));
}

TEST_F(LoadingOverheadTrackerTest, PassthroughRegistrationDoesNotPolluteFiniteFileGroup) {
    auto field_handle = tracker_.Register(LoadingOverheadConfig{LoadingOverheadDimensionConfig{200, "load_transient"},
                                                                LoadingOverheadDimensionConfig{50, "load_transient"}});
    auto scalar_handle =
        tracker_.Register(LoadingOverheadConfig{LoadingOverheadDimensionConfig{200, "load_transient"}, std::nullopt});

    EXPECT_EQ(tracker_.GetUpperBound(field_handle), (ResourceUsage{200, 50}));
    EXPECT_EQ(tracker_.GetUpperBound(scalar_handle), (ResourceUsage{200, std::numeric_limits<int64_t>::max()}));

    auto scalar = tracker_.Reserve(scalar_handle, {0, 100});
    auto field = tracker_.Reserve(field_handle, {0, 100});
    EXPECT_EQ(scalar.file_bytes, 100);
    EXPECT_EQ(field.file_bytes, 50);
}
