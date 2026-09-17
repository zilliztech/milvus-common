// Copyright (C) 2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cachinglayer/Metrics.h"

namespace milvus::cachinglayer::monitor {
namespace {

using Samples = std::map<std::pair<std::string, std::string>, double>;

Samples
ExportedSamples() {
    Samples samples;
    for (const auto& family : milvus::monitor::getPrometheusClient().GetRegistry().Collect()) {
        if (family.name != "internal_cache_shard_disk_usage_bytes") {
            continue;
        }
        for (const auto& metric : family.metric) {
            std::string type;
            std::string shard;
            for (const auto& label : metric.label) {
                if (label.name == "data_type") {
                    type = label.value;
                } else if (label.name == "shard") {
                    shard = label.value;
                }
            }
            EXPECT_TRUE(samples.emplace(std::make_pair(type, shard), metric.gauge.value).second);
        }
    }
    return samples;
}

double
ExportedValue(const std::string& type, const std::string& shard) {
    const auto samples = ExportedSamples();
    const auto it = samples.find({type, cache_shard_disk_usage_metrics_aggregate() ? "all" : shard});
    return it == samples.end() ? 0 : it->second;
}

class CacheShardDiskUsageMetricTest : public testing::Test {
 protected:
    void
    SetUp() override {
        EXPECT_TRUE(collect_cache_shard_disk_usage_stats().empty());
    }

    void
    TearDown() override {
        EXPECT_TRUE(collect_cache_shard_disk_usage_stats().empty());
        for (const auto& [labels, bytes] : ExportedSamples()) {
            EXPECT_TRUE(cache_shard_disk_usage_metrics_aggregate());
            EXPECT_EQ(labels.second, "all");
            EXPECT_EQ(bytes, 0);
        }
    }
};

TEST_F(CacheShardDiskUsageMetricTest, KeepsRealShardStatisticsWithSharedGauge) {
    auto a = create_cache_shard_disk_usage_metric_handle(CellDataType::VECTOR_FIELD, "shard-a");
    auto a2 = create_cache_shard_disk_usage_metric_handle(CellDataType::VECTOR_FIELD, "shard-a");
    auto b = create_cache_shard_disk_usage_metric_handle(CellDataType::VECTOR_FIELD, "shard-b");
    auto scalar = create_cache_shard_disk_usage_metric_handle(CellDataType::SCALAR_INDEX, "shard-a");
    a->Increment(10);
    a2->Increment(3);
    b->Increment(20);
    scalar->Increment(7);

    EXPECT_EQ(a->Value(), 13);
    EXPECT_EQ(a2->Value(), 13);
    EXPECT_EQ(b->Value(), 20);
    EXPECT_EQ(cache_shard_disk_usage_bytes_value(CellDataType::SCALAR_INDEX, "shard-a"), 7);
    const auto stats = collect_cache_shard_disk_usage_stats();
    ASSERT_EQ(stats.size(), 3);
    for (const auto& stat : stats) {
        EXPECT_NE(stat.shard, "all");
        EXPECT_EQ(stat.disk_bytes, stat.cell_data_type == CellDataType::SCALAR_INDEX ? 7
                                   : stat.shard == "shard-a"                         ? 13
                                                                                     : 20);
    }
    EXPECT_EQ(ExportedValue("vector_field", "shard-a"), cache_shard_disk_usage_metrics_aggregate() ? 33 : 13);
    EXPECT_EQ(ExportedValue("scalar_index", "shard-a"), 7);

    a->Decrement(10);
    a.reset();
    EXPECT_EQ(a2->Value(), 3);
    a2->Decrement(3);
    a2.reset();
    EXPECT_EQ(cache_shard_disk_usage_bytes_value(CellDataType::VECTOR_FIELD, "shard-a"), std::nullopt);
    EXPECT_EQ(b->Value(), 20);
    EXPECT_EQ(ExportedValue("vector_field", "shard-b"), 20);
    b->Decrement(20);
    scalar->Decrement(7);
}

TEST_F(CacheShardDiskUsageMetricTest, RetiresOnlyTheLastHandlesContribution) {
    auto a = create_cache_shard_disk_usage_metric_handle(CellDataType::VECTOR_INDEX, "retired-a");
    auto b = create_cache_shard_disk_usage_metric_handle(CellDataType::VECTOR_INDEX, "retired-b");
    a->Increment(10);
    b->Increment(20);
    a.reset();
    // Aggregate retirement takes effect without a scrape or stats sweep.
    if (cache_shard_disk_usage_metrics_aggregate()) {
        EXPECT_EQ(ExportedValue("vector_index", "retired-b"), 20);
    }
    auto replacement = create_cache_shard_disk_usage_metric_handle(CellDataType::VECTOR_INDEX, "retired-a");
    EXPECT_EQ(replacement->Value(), 0);
    replacement->Increment(5);
    EXPECT_EQ(b->Value(), 20);
    EXPECT_EQ(ExportedValue("vector_index", "retired-a"), cache_shard_disk_usage_metrics_aggregate() ? 25 : 5);
    replacement->Decrement(5);
    b->Decrement(20);
}

TEST_F(CacheShardDiskUsageMetricTest, LeavesEmptyShardsUnattributed) {
    const auto before = ExportedSamples();
    EXPECT_EQ(create_cache_shard_disk_usage_metric_handle(CellDataType::OTHER, ""), nullptr);
    EXPECT_EQ(cache_shard_disk_usage_bytes_value(CellDataType::OTHER, ""), std::nullopt);
    EXPECT_EQ(ExportedSamples(), before);
}

TEST_F(CacheShardDiskUsageMetricTest, FreezesModeBeforeAndAfterRetirement) {
    const bool aggregate = cache_shard_disk_usage_metrics_aggregate();
    auto handle = create_cache_shard_disk_usage_metric_handle(CellDataType::OTHER, "mode-test");
    EXPECT_TRUE(set_cache_shard_disk_usage_metrics_mode(aggregate));
    EXPECT_FALSE(set_cache_shard_disk_usage_metrics_mode(!aggregate));
    handle.reset();
    EXPECT_TRUE(collect_cache_shard_disk_usage_stats().empty());
    EXPECT_FALSE(set_cache_shard_disk_usage_metrics_mode(!aggregate));
    EXPECT_EQ(cache_shard_disk_usage_metrics_aggregate(), aggregate);
}

TEST_F(CacheShardDiskUsageMetricTest, CardinalityDoesNotGrowInAggregateMode) {
    constexpr size_t kShards = 2000;
    for (int cycle = 0; cycle < 3; ++cycle) {
        std::vector<std::unique_ptr<CacheShardDiskUsageMetricHandle>> handles;
        for (size_t i = 0; i < kShards; ++i) {
            auto handle = create_cache_shard_disk_usage_metric_handle(CellDataType::OTHER, std::to_string(i));
            handle->Increment(2);
            handles.push_back(std::move(handle));
        }
        EXPECT_EQ(collect_cache_shard_disk_usage_stats().size(), kShards);
        if (cache_shard_disk_usage_metrics_aggregate()) {
            EXPECT_LE(ExportedSamples().size(), 5);
            EXPECT_EQ(ExportedValue("other", "all"), 2 * kShards);
        } else {
            EXPECT_EQ(ExportedSamples().size(), kShards);
        }
        for (auto& handle : handles) {
            handle->Decrement(2);
        }
        handles.clear();
        EXPECT_TRUE(collect_cache_shard_disk_usage_stats().empty());
        EXPECT_EQ(ExportedValue("other", "all"), 0);
        if (cache_shard_disk_usage_metrics_aggregate()) {
            const auto samples = ExportedSamples();
            ASSERT_TRUE(samples.contains({"other", "all"}));
            EXPECT_EQ(samples.at({"other", "all"}), 0);
        }
    }
}

TEST_F(CacheShardDiskUsageMetricTest, ConcurrentUpdatesScrapesAndSameShardRecreation) {
    constexpr int kThreads = 8;
    constexpr int kIterations = 1000;
    std::atomic<bool> finished{false};
    std::thread reader([&] {
        while (!finished.load()) {
            for (const auto& stat : collect_cache_shard_disk_usage_stats()) {
                EXPECT_NE(stat.shard, "all");
                EXPECT_GE(stat.disk_bytes, 0);
            }
            ExportedSamples();
            std::this_thread::yield();
        }
    });
    std::vector<std::thread> writers;
    for (int thread = 0; thread < kThreads; ++thread) {
        writers.emplace_back([&] {
            for (int i = 0; i < kIterations; ++i) {
                const auto shard = "concurrent-" + std::to_string(i % 16);
                auto first = create_cache_shard_disk_usage_metric_handle(CellDataType::VECTOR_FIELD, shard);
                auto second = create_cache_shard_disk_usage_metric_handle(CellDataType::VECTOR_FIELD, shard);
                first->Increment(11);
                second->Increment(7);
                first->Decrement(11);
                first.reset();
                second->Decrement(7);
            }
        });
    }
    for (auto& writer : writers) {
        writer.join();
    }
    finished.store(true);
    reader.join();
}

}  // namespace
}  // namespace milvus::cachinglayer::monitor
