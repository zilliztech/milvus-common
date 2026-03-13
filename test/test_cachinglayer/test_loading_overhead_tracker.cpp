#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "cachinglayer/LoadingOverheadTracker.h"
#include "cachinglayer/Utils.h"

using namespace milvus::cachinglayer;

class LoadingOverheadTrackerTest : public ::testing::Test {
 protected:
    LoadingOverheadTracker tracker_;
};

TEST_F(LoadingOverheadTrackerTest, NoUpperBoundPassThrough) {
    // Without registering a UB, all amounts pass through unchanged.
    auto delta = tracker_.Reserve(CellDataType::VECTOR_INDEX, {100, 0});
    EXPECT_EQ(delta.memory_bytes, 100);
    EXPECT_EQ(delta.file_bytes, 0);

    auto release = tracker_.Release(CellDataType::VECTOR_INDEX, {100, 0});
    EXPECT_EQ(release.memory_bytes, 100);
    EXPECT_EQ(release.file_bytes, 0);
}

TEST_F(LoadingOverheadTrackerTest, BasicCapping) {
    tracker_.RegisterUpperBound(CellDataType::VECTOR_INDEX, {200, 0});

    // First reserve: 100, sum=100 <= UB=200, full amount passes through
    auto d1 = tracker_.Reserve(CellDataType::VECTOR_INDEX, {100, 0});
    EXPECT_EQ(d1.memory_bytes, 100);

    // Second reserve: 100, sum=200 <= UB=200, full amount passes through
    auto d2 = tracker_.Reserve(CellDataType::VECTOR_INDEX, {100, 0});
    EXPECT_EQ(d2.memory_bytes, 100);

    // Third reserve: 100, sum=300 > UB=200, capped: delta = 0
    auto d3 = tracker_.Reserve(CellDataType::VECTOR_INDEX, {100, 0});
    EXPECT_EQ(d3.memory_bytes, 0);

    // Fourth reserve: 100, sum=400 > UB=200, capped: delta = 0
    auto d4 = tracker_.Reserve(CellDataType::VECTOR_INDEX, {100, 0});
    EXPECT_EQ(d4.memory_bytes, 0);
}

TEST_F(LoadingOverheadTrackerTest, BasicRelease) {
    tracker_.RegisterUpperBound(CellDataType::VECTOR_INDEX, {200, 0});

    // Reserve 4x100, total sum=400, actual DList reserve = 200
    tracker_.Reserve(CellDataType::VECTOR_INDEX, {100, 0});
    tracker_.Reserve(CellDataType::VECTOR_INDEX, {100, 0});
    tracker_.Reserve(CellDataType::VECTOR_INDEX, {100, 0});
    tracker_.Reserve(CellDataType::VECTOR_INDEX, {100, 0});

    // Release first 100: sum 400->300, both >= UB, release 0
    auto r1 = tracker_.Release(CellDataType::VECTOR_INDEX, {100, 0});
    EXPECT_EQ(r1.memory_bytes, 0);

    // Release second 100: sum 300->200, release 0
    auto r2 = tracker_.Release(CellDataType::VECTOR_INDEX, {100, 0});
    EXPECT_EQ(r2.memory_bytes, 0);

    // Release third 100: sum 200->100, release 100
    auto r3 = tracker_.Release(CellDataType::VECTOR_INDEX, {100, 0});
    EXPECT_EQ(r3.memory_bytes, 100);

    // Release fourth 100: sum 100->0, release 100
    auto r4 = tracker_.Release(CellDataType::VECTOR_INDEX, {100, 0});
    EXPECT_EQ(r4.memory_bytes, 100);
}

TEST_F(LoadingOverheadTrackerTest, TotalReservedEqualsReleased) {
    tracker_.RegisterUpperBound(CellDataType::VECTOR_INDEX, {200, 0});

    int64_t total_reserved = 0;
    int64_t total_released = 0;

    // Reserve 10 x 100
    for (int i = 0; i < 10; i++) {
        total_reserved += tracker_.Reserve(CellDataType::VECTOR_INDEX, {100, 0}).memory_bytes;
    }
    // Should have reserved exactly UB = 200
    EXPECT_EQ(total_reserved, 200);

    // Release all 10
    for (int i = 0; i < 10; i++) {
        total_released += tracker_.Release(CellDataType::VECTOR_INDEX, {100, 0}).memory_bytes;
    }
    // Total released should equal total reserved
    EXPECT_EQ(total_released, 200);
}

TEST_F(LoadingOverheadTrackerTest, PartialCapping) {
    tracker_.RegisterUpperBound(CellDataType::VECTOR_INDEX, {200, 0});

    // Reserve 150: sum=150 <= UB=200, full amount
    auto d1 = tracker_.Reserve(CellDataType::VECTOR_INDEX, {150, 0});
    EXPECT_EQ(d1.memory_bytes, 150);

    // Reserve 100: sum=250 > UB=200, delta = 200-150 = 50
    auto d2 = tracker_.Reserve(CellDataType::VECTOR_INDEX, {100, 0});
    EXPECT_EQ(d2.memory_bytes, 50);
}

TEST_F(LoadingOverheadTrackerTest, ReleaseUndoesReserve) {
    tracker_.RegisterUpperBound(CellDataType::VECTOR_INDEX, {200, 0});

    // Reserve 150: actual = 150
    auto d1 = tracker_.Reserve(CellDataType::VECTOR_INDEX, {150, 0});
    EXPECT_EQ(d1.memory_bytes, 150);

    // Release undoes the reserve: sum 150->0, release = 150
    auto undo = tracker_.Release(CellDataType::VECTOR_INDEX, {150, 0});
    EXPECT_EQ(undo.memory_bytes, 150);
}

TEST_F(LoadingOverheadTrackerTest, MultipleTypes) {
    tracker_.RegisterUpperBound(CellDataType::VECTOR_INDEX, {200, 0});
    tracker_.RegisterUpperBound(CellDataType::SCALAR_INDEX, {100, 0});

    // Types are tracked independently
    auto d1 = tracker_.Reserve(CellDataType::VECTOR_INDEX, {200, 0});
    EXPECT_EQ(d1.memory_bytes, 200);

    auto d2 = tracker_.Reserve(CellDataType::SCALAR_INDEX, {100, 0});
    EXPECT_EQ(d2.memory_bytes, 100);

    // Both at UB, further reserves return 0
    auto d3 = tracker_.Reserve(CellDataType::VECTOR_INDEX, {100, 0});
    EXPECT_EQ(d3.memory_bytes, 0);

    auto d4 = tracker_.Reserve(CellDataType::SCALAR_INDEX, {50, 0});
    EXPECT_EQ(d4.memory_bytes, 0);
}

TEST_F(LoadingOverheadTrackerTest, RegisterUpperBoundTakesMax) {
    tracker_.RegisterUpperBound(CellDataType::VECTOR_INDEX, {100, 50});
    tracker_.RegisterUpperBound(CellDataType::VECTOR_INDEX, {200, 30});

    // UB should be {200, 50} (max per dimension)
    auto ub = tracker_.GetUpperBound(CellDataType::VECTOR_INDEX);
    EXPECT_EQ(ub.memory_bytes, 200);
    EXPECT_EQ(ub.file_bytes, 50);

    auto d1 = tracker_.Reserve(CellDataType::VECTOR_INDEX, {200, 50});
    EXPECT_EQ(d1.memory_bytes, 200);
    EXPECT_EQ(d1.file_bytes, 50);

    // Next reserve should be fully capped
    auto d2 = tracker_.Reserve(CellDataType::VECTOR_INDEX, {100, 100});
    EXPECT_EQ(d2.memory_bytes, 0);
    EXPECT_EQ(d2.file_bytes, 0);
}

TEST_F(LoadingOverheadTrackerTest, HasFiniteUpperBound) {
    EXPECT_FALSE(tracker_.HasFiniteUpperBound(CellDataType::VECTOR_INDEX));
    tracker_.RegisterUpperBound(CellDataType::VECTOR_INDEX, {200, 0});
    EXPECT_TRUE(tracker_.HasFiniteUpperBound(CellDataType::VECTOR_INDEX));
    EXPECT_FALSE(tracker_.HasFiniteUpperBound(CellDataType::SCALAR_INDEX));
}

TEST_F(LoadingOverheadTrackerTest, ConcurrentReserveRelease) {
    tracker_.RegisterUpperBound(CellDataType::VECTOR_INDEX, {200, 0});

    const int num_threads = 10;
    const int ops_per_thread = 100;
    std::vector<std::thread> threads;
    std::atomic<int64_t> total_reserved{0};
    std::atomic<int64_t> total_released{0};

    threads.reserve(num_threads);
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back([&]() {
            for (int j = 0; j < ops_per_thread; j++) {
                auto reserved = tracker_.Reserve(CellDataType::VECTOR_INDEX, {10, 0});
                total_reserved += reserved.memory_bytes;
            }
            for (int j = 0; j < ops_per_thread; j++) {
                auto released = tracker_.Release(CellDataType::VECTOR_INDEX, {10, 0});
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
    // EnsureRegistered without explicit UB -> unlimited, behaves like no capping.
    tracker_.EnsureRegistered(CellDataType::VECTOR_INDEX);

    EXPECT_FALSE(tracker_.HasFiniteUpperBound(CellDataType::VECTOR_INDEX));

    auto d1 = tracker_.Reserve(CellDataType::VECTOR_INDEX, {1000000000, 0});
    EXPECT_EQ(d1.memory_bytes, 1000000000);

    auto d2 = tracker_.Reserve(CellDataType::VECTOR_INDEX, {2000000000, 0});
    EXPECT_EQ(d2.memory_bytes, 2000000000);

    auto r1 = tracker_.Release(CellDataType::VECTOR_INDEX, {1000000000, 0});
    EXPECT_EQ(r1.memory_bytes, 1000000000);

    auto r2 = tracker_.Release(CellDataType::VECTOR_INDEX, {2000000000, 0});
    EXPECT_EQ(r2.memory_bytes, 2000000000);
}

TEST_F(LoadingOverheadTrackerTest, EnsureRegisteredThenRegisterFiniteUB) {
    tracker_.EnsureRegistered(CellDataType::VECTOR_INDEX);
    EXPECT_FALSE(tracker_.HasFiniteUpperBound(CellDataType::VECTOR_INDEX));

    tracker_.RegisterUpperBound(CellDataType::VECTOR_INDEX, {200, 0});
    EXPECT_TRUE(tracker_.HasFiniteUpperBound(CellDataType::VECTOR_INDEX));

    // Now capping should apply.
    auto d1 = tracker_.Reserve(CellDataType::VECTOR_INDEX, {300, 0});
    EXPECT_EQ(d1.memory_bytes, 200);
}

TEST_F(LoadingOverheadTrackerTest, UnregisteredTypeAutoCreatesUnlimited) {
    auto d1 = tracker_.Reserve(CellDataType::SCALAR_FIELD, {500, 0});
    EXPECT_EQ(d1.memory_bytes, 500);

    auto r1 = tracker_.Release(CellDataType::SCALAR_FIELD, {500, 0});
    EXPECT_EQ(r1.memory_bytes, 500);

    EXPECT_FALSE(tracker_.HasFiniteUpperBound(CellDataType::SCALAR_FIELD));
}

TEST_F(LoadingOverheadTrackerTest, UBChangesMidFlight) {
    tracker_.RegisterUpperBound(CellDataType::VECTOR_INDEX, {200, 0});

    // Reserve 3x100 under UB=200: dlist gets 100+100+0 = 200
    auto d1 = tracker_.Reserve(CellDataType::VECTOR_INDEX, {100, 0});
    EXPECT_EQ(d1.memory_bytes, 100);
    auto d2 = tracker_.Reserve(CellDataType::VECTOR_INDEX, {100, 0});
    EXPECT_EQ(d2.memory_bytes, 100);
    auto d3 = tracker_.Reserve(CellDataType::VECTOR_INDEX, {100, 0});
    EXPECT_EQ(d3.memory_bytes, 0);

    int64_t total_reserved = d1.memory_bytes + d2.memory_bytes + d3.memory_bytes;
    EXPECT_EQ(total_reserved, 200);

    // UB changes to 400
    tracker_.RegisterUpperBound(CellDataType::VECTOR_INDEX, {400, 0});

    // Release all 3: should release exactly 200 total
    auto r1 = tracker_.Release(CellDataType::VECTOR_INDEX, {100, 0});
    EXPECT_EQ(r1.memory_bytes, 0);
    auto r2 = tracker_.Release(CellDataType::VECTOR_INDEX, {100, 0});
    EXPECT_EQ(r2.memory_bytes, 100);
    auto r3 = tracker_.Release(CellDataType::VECTOR_INDEX, {100, 0});
    EXPECT_EQ(r3.memory_bytes, 100);

    int64_t total_released = r1.memory_bytes + r2.memory_bytes + r3.memory_bytes;
    EXPECT_EQ(total_released, 200);
    EXPECT_EQ(total_reserved, total_released);
}

TEST_F(LoadingOverheadTrackerTest, UBDecreasesFromUnlimitedMidFlight) {
    auto d1 = tracker_.Reserve(CellDataType::VECTOR_INDEX, {1000, 0});
    EXPECT_EQ(d1.memory_bytes, 1000);
    auto d2 = tracker_.Reserve(CellDataType::VECTOR_INDEX, {1000, 0});
    EXPECT_EQ(d2.memory_bytes, 1000);

    // Now register finite UB=200
    tracker_.RegisterUpperBound(CellDataType::VECTOR_INDEX, {200, 0});

    // Further reserves should be capped
    auto d3 = tracker_.Reserve(CellDataType::VECTOR_INDEX, {100, 0});
    EXPECT_EQ(d3.memory_bytes, 0);

    // Release all 3: should release exactly 2000
    auto r1 = tracker_.Release(CellDataType::VECTOR_INDEX, {1000, 0});
    EXPECT_EQ(r1.memory_bytes, 1800);
    auto r2 = tracker_.Release(CellDataType::VECTOR_INDEX, {1000, 0});
    EXPECT_EQ(r2.memory_bytes, 100);
    auto r3 = tracker_.Release(CellDataType::VECTOR_INDEX, {100, 0});
    EXPECT_EQ(r3.memory_bytes, 100);

    int64_t total_reserved = d1.memory_bytes + d2.memory_bytes + d3.memory_bytes;
    int64_t total_released = r1.memory_bytes + r2.memory_bytes + r3.memory_bytes;
    EXPECT_EQ(total_reserved, 2000);
    EXPECT_EQ(total_released, 2000);
}

TEST_F(LoadingOverheadTrackerTest, GetUpperBound) {
    EXPECT_EQ(tracker_.GetUpperBound(CellDataType::VECTOR_INDEX), LoadingOverheadTracker::kUnlimited);

    tracker_.RegisterUpperBound(CellDataType::VECTOR_INDEX, {200, 100});
    auto ub = tracker_.GetUpperBound(CellDataType::VECTOR_INDEX);
    EXPECT_EQ(ub.memory_bytes, 200);
    EXPECT_EQ(ub.file_bytes, 100);
}
