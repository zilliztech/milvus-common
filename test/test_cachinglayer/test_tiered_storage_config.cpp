// Copyright (C) 2019-2025 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License
#include <gtest/gtest.h>

#include <chrono>
#include <thread>
#include <vector>

#include "cachinglayer/TieredStorageConfig.h"

using namespace milvus::cachinglayer;

class TieredStorageConfigTest : public ::testing::Test {
 protected:
    TieredStorageConfig& config = TieredStorageConfig::GetInstance();

    void
    TearDown() override {
        // Reset to defaults after each test
        config.UpdateAll(false, std::chrono::milliseconds(100000), std::chrono::milliseconds(0), CacheWarmupPolicies{});
    }
};

TEST_F(TieredStorageConfigTest, DefaultValues) {
    EXPECT_FALSE(config.storage_usage_tracking_enabled());
    EXPECT_EQ(config.loading_timeout(), std::chrono::milliseconds(100000));
    EXPECT_EQ(config.warmup_loading_timeout(), std::chrono::milliseconds(0));
}

TEST_F(TieredStorageConfigTest, SetAndGetStorageUsageTracking) {
    config.SetStorageUsageTrackingEnabled(true);
    EXPECT_TRUE(config.storage_usage_tracking_enabled());

    config.SetStorageUsageTrackingEnabled(false);
    EXPECT_FALSE(config.storage_usage_tracking_enabled());
}

TEST_F(TieredStorageConfigTest, SetAndGetLoadingTimeout) {
    config.SetLoadingTimeout(std::chrono::milliseconds(5000));
    EXPECT_EQ(config.loading_timeout(), std::chrono::milliseconds(5000));
}

TEST_F(TieredStorageConfigTest, SetAndGetWarmupLoadingTimeout) {
    config.SetWarmupLoadingTimeout(std::chrono::milliseconds(3000));
    EXPECT_EQ(config.warmup_loading_timeout(), std::chrono::milliseconds(3000));

    // Zero means best-effort (fail immediately, no waiting queue)
    config.SetWarmupLoadingTimeout(std::chrono::milliseconds(0));
    EXPECT_EQ(config.warmup_loading_timeout(), std::chrono::milliseconds(0));
}

TEST_F(TieredStorageConfigTest, SetAndGetWarmupPolicies) {
    CacheWarmupPolicies policies(CacheWarmupPolicy::CacheWarmupPolicy_Async,
                                 CacheWarmupPolicy::CacheWarmupPolicy_Disable,
                                 CacheWarmupPolicy::CacheWarmupPolicy_Sync, CacheWarmupPolicy::CacheWarmupPolicy_Async);
    config.SetWarmupPolicies(policies);

    auto result = config.warmup_policies();
    EXPECT_EQ(result.scalarFieldCacheWarmupPolicy, CacheWarmupPolicy::CacheWarmupPolicy_Async);
    EXPECT_EQ(result.vectorFieldCacheWarmupPolicy, CacheWarmupPolicy::CacheWarmupPolicy_Disable);
    EXPECT_EQ(result.scalarIndexCacheWarmupPolicy, CacheWarmupPolicy::CacheWarmupPolicy_Sync);
    EXPECT_EQ(result.vectorIndexCacheWarmupPolicy, CacheWarmupPolicy::CacheWarmupPolicy_Async);
}

TEST_F(TieredStorageConfigTest, SnapshotReturnsConsistentView) {
    config.UpdateAll(
        true, std::chrono::milliseconds(5000), std::chrono::milliseconds(200),
        CacheWarmupPolicies(CacheWarmupPolicy::CacheWarmupPolicy_Async, CacheWarmupPolicy::CacheWarmupPolicy_Async,
                            CacheWarmupPolicy::CacheWarmupPolicy_Async, CacheWarmupPolicy::CacheWarmupPolicy_Async));

    auto snapshot = config.GetSnapshot();
    EXPECT_TRUE(snapshot.storage_usage_tracking_enabled);
    EXPECT_EQ(snapshot.loading_timeout, std::chrono::milliseconds(5000));
    EXPECT_EQ(snapshot.warmup_loading_timeout, std::chrono::milliseconds(200));
    EXPECT_EQ(snapshot.warmup_policies.scalarFieldCacheWarmupPolicy, CacheWarmupPolicy::CacheWarmupPolicy_Async);
}

TEST_F(TieredStorageConfigTest, UpdateAllIsAtomic) {
    // Set initial values
    config.UpdateAll(false, std::chrono::milliseconds(100), std::chrono::milliseconds(0), CacheWarmupPolicies{});

    // Update all fields atomically
    CacheWarmupPolicies new_policies(
        CacheWarmupPolicy::CacheWarmupPolicy_Disable, CacheWarmupPolicy::CacheWarmupPolicy_Disable,
        CacheWarmupPolicy::CacheWarmupPolicy_Disable, CacheWarmupPolicy::CacheWarmupPolicy_Disable);
    config.UpdateAll(true, std::chrono::milliseconds(9999), std::chrono::milliseconds(42), new_policies);

    // Snapshot should see all new values together
    auto snapshot = config.GetSnapshot();
    EXPECT_TRUE(snapshot.storage_usage_tracking_enabled);
    EXPECT_EQ(snapshot.loading_timeout, std::chrono::milliseconds(9999));
    EXPECT_EQ(snapshot.warmup_loading_timeout, std::chrono::milliseconds(42));
    EXPECT_EQ(snapshot.warmup_policies.scalarFieldCacheWarmupPolicy, CacheWarmupPolicy::CacheWarmupPolicy_Disable);
}

TEST_F(TieredStorageConfigTest, ConcurrentReadsAndWrites) {
    // Smoke test: concurrent snapshot readers and UpdateAll writer don't crash
    std::atomic<bool> stop{false};

    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i) {
        readers.emplace_back([&]() {
            while (!stop.load()) {
                (void)config.GetSnapshot();
                (void)config.loading_timeout();
                (void)config.warmup_loading_timeout();
                (void)config.storage_usage_tracking_enabled();
                (void)config.warmup_policies();
            }
        });
    }

    std::thread writer([&]() {
        for (int i = 0; i < 100; ++i) {
            config.UpdateAll(i % 3 == 0, std::chrono::milliseconds(i * 100), std::chrono::milliseconds(i),
                             CacheWarmupPolicies{});
        }
        stop.store(true);
    });

    writer.join();
    for (auto& r : readers) {
        r.join();
    }
}
