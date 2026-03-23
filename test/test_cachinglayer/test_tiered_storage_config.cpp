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
        config.SetEvictionEnabled(false);
        config.SetStorageUsageTrackingEnabled(false);
        config.SetLoadingTimeout(std::chrono::milliseconds(100000));
        config.SetWarmupLoadingTimeout(std::chrono::milliseconds(0));
        config.SetWarmupPolicies(CacheWarmupPolicies{});
    }
};

TEST_F(TieredStorageConfigTest, DefaultValues) {
    EXPECT_FALSE(config.eviction_enabled());
    EXPECT_FALSE(config.storage_usage_tracking_enabled());
    EXPECT_EQ(config.loading_timeout(), std::chrono::milliseconds(100000));
    EXPECT_EQ(config.warmup_loading_timeout(), std::chrono::milliseconds(0));
}

TEST_F(TieredStorageConfigTest, SetAndGetEvictionEnabled) {
    config.SetEvictionEnabled(true);
    EXPECT_TRUE(config.eviction_enabled());

    config.SetEvictionEnabled(false);
    EXPECT_FALSE(config.eviction_enabled());
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

TEST_F(TieredStorageConfigTest, ConcurrentReadsAndWrites) {
    // Smoke test: concurrent readers and a writer don't crash
    std::atomic<bool> stop{false};

    // Readers
    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i) {
        readers.emplace_back([&]() {
            while (!stop.load()) {
                (void)config.eviction_enabled();
                (void)config.loading_timeout();
                (void)config.warmup_loading_timeout();
                (void)config.storage_usage_tracking_enabled();
                (void)config.warmup_policies();
            }
        });
    }

    // Writer
    std::thread writer([&]() {
        for (int i = 0; i < 100; ++i) {
            config.SetEvictionEnabled(i % 2 == 0);
            config.SetLoadingTimeout(std::chrono::milliseconds(i * 100));
            config.SetWarmupLoadingTimeout(std::chrono::milliseconds(i));
            config.SetStorageUsageTrackingEnabled(i % 3 == 0);
        }
        stop.store(true);
    });

    writer.join();
    for (auto& r : readers) {
        r.join();
    }
}
