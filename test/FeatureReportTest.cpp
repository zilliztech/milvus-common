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
#include <string_view>
#include <thread>
#include <vector>

#include "common/FeatureReport.h"
#include "common/PrometheusClient.h"

namespace milvus::monitor::feature_report {

struct FeatureReporterTestPeer {
    static bool
    RecordAt(FeatureReporter& reporter, std::chrono::steady_clock::time_point now) {
        return reporter.recordAt(now);
    }

    static void
    Reset(FeatureReporter& reporter) {
        reporter.reset();
    }
};

namespace {

double
FeatureReportValue(std::string_view feature) {
    for (const auto& family : getPrometheusClient().GetRegistry().Collect()) {
        if (family.name != "milvus_feature_report_total" || family.type != prometheus::MetricType::Counter) {
            continue;
        }

        for (const auto& metric : family.metric) {
            bool feature_match = false;
            bool source_match = false;
            for (const auto& label : metric.label) {
                if (label.name == "feature" && std::string_view(label.value) == feature) {
                    feature_match = true;
                } else if (label.name == "source" && label.value == "cpp") {
                    source_match = true;
                }
            }
            if (feature_match && source_match) {
                return metric.counter.value;
            }
        }
    }
    return 0.0;
}

TEST(FeatureReportTest, Throttle) {
    auto& reporter = HybridSearch();
    FeatureReporterTestPeer::Reset(reporter);
    const auto before = FeatureReportValue(reporter.Name());
    const auto now = std::chrono::steady_clock::time_point(std::chrono::seconds(100));

    ASSERT_TRUE(FeatureReporterTestPeer::RecordAt(reporter, now));
    ASSERT_FALSE(FeatureReporterTestPeer::RecordAt(reporter, now + std::chrono::minutes(1)));
    ASSERT_TRUE(FeatureReporterTestPeer::RecordAt(reporter, now + std::chrono::hours(1)));

    ASSERT_DOUBLE_EQ(FeatureReportValue(reporter.Name()) - before, 2.0);
}

TEST(FeatureReportTest, ConcurrentCalls) {
    auto& reporter = PartitionKey();
    FeatureReporterTestPeer::Reset(reporter);
    const auto before = FeatureReportValue(reporter.Name());
    const auto now = std::chrono::steady_clock::time_point(std::chrono::seconds(200));
    constexpr int kThreads = 32;

    std::atomic<int> reported{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&] {
            if (FeatureReporterTestPeer::RecordAt(reporter, now)) {
                reported.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    ASSERT_EQ(reported.load(std::memory_order_relaxed), 1);
    ASSERT_DOUBLE_EQ(FeatureReportValue(reporter.Name()) - before, 1.0);
}

TEST(FeatureReportTest, PredeclaredReporter) {
    FeatureReporterTestPeer::Reset(DynamicField());
    ASSERT_EQ(DynamicField().Name(), kDynamicField);
    ASSERT_TRUE(FeatureReporterTestPeer::RecordAt(DynamicField(),
                                                  std::chrono::steady_clock::time_point(std::chrono::seconds(300))));
}

}  // namespace
}  // namespace milvus::monitor::feature_report
