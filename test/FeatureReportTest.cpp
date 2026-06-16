// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "common/FeatureReport.h"
#include "common/PrometheusClient.h"

namespace milvus::monitor::feature_report {
namespace {

bool
MetricsContain(const std::string& metrics, const std::string& feature, const std::string& value) {
    const auto expected = "milvus_feature_report_total{feature=\"" + feature + "\",source=\"cpp\"} " + value + "\n";
    return metrics.find(expected) != std::string::npos;
}

TEST(FeatureReportTest, Throttle) {
    auto& reporter = HybridSearch();
    reporter.ResetForTest();
    const auto now = std::chrono::steady_clock::time_point(std::chrono::seconds(100));

    ASSERT_TRUE(reporter.RecordAtForTest(now));
    ASSERT_FALSE(reporter.RecordAtForTest(now + std::chrono::minutes(1)));
    ASSERT_TRUE(reporter.RecordAtForTest(now + std::chrono::hours(1)));

    ASSERT_TRUE(MetricsContain(getPrometheusClient().GetMetrics(), std::string(reporter.Name()), "2"));
}

TEST(FeatureReportTest, ConcurrentCalls) {
    auto& reporter = PartitionKey();
    reporter.ResetForTest();
    const auto now = std::chrono::steady_clock::time_point(std::chrono::seconds(200));
    constexpr int kThreads = 32;

    std::atomic<int> reported{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&] {
            if (reporter.RecordAtForTest(now)) {
                reported.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    ASSERT_EQ(reported.load(std::memory_order_relaxed), 1);
    ASSERT_TRUE(MetricsContain(getPrometheusClient().GetMetrics(), std::string(reporter.Name()), "1"));
}

TEST(FeatureReportTest, PredeclaredReporter) {
    DynamicField().ResetForTest();
    ASSERT_EQ(DynamicField().Name(), kDynamicField);
    ASSERT_TRUE(DynamicField().RecordAtForTest(std::chrono::steady_clock::time_point(std::chrono::seconds(300))));
}

}  // namespace
}  // namespace milvus::monitor::feature_report
