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

#include "common/FeatureReport.h"

#include <prometheus/family.h>

#include <string>

#include "common/PrometheusClient.h"

namespace milvus::monitor::feature_report {
namespace {

constexpr std::string_view kSourceCpp = "cpp";
constexpr auto kReportInterval = std::chrono::hours(1);

prometheus::Family<prometheus::Counter>&
FeatureReportFamily() {
    static auto& family = prometheus::BuildCounter()
                              .Name("milvus_feature_report_total")
                              .Help("Count of throttled feature reports.")
                              .Register(milvus::monitor::getPrometheusClient().GetRegistry());
    return family;
}

prometheus::Counter&
CounterForFeature(std::string_view feature) {
    return FeatureReportFamily().Add({{"feature", std::string(feature)}, {"source", std::string(kSourceCpp)}});
}

int64_t
ToNanos(std::chrono::steady_clock::time_point time_point) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(time_point.time_since_epoch()).count();
}

}  // namespace

FeatureReporter::FeatureReporter(std::string_view name) : name_(name), counter_(CounterForFeature(name_)) {
}

std::string_view
FeatureReporter::Name() const noexcept {
    return std::string_view(name_);
}

bool
FeatureReporter::Record() {
    return recordAt(std::chrono::steady_clock::now());
}

bool
FeatureReporter::RecordAtForTest(std::chrono::steady_clock::time_point now) {
    return recordAt(now);
}

void
FeatureReporter::ResetForTest() {
    next_allowed_nanos_.store(0, std::memory_order_release);
}

bool
FeatureReporter::recordAt(std::chrono::steady_clock::time_point now) {
    if (name_.empty()) {
        return false;
    }

    const auto now_nanos = ToNanos(now);
    const auto next_nanos = ToNanos(now + kReportInterval);

    auto old = next_allowed_nanos_.load(std::memory_order_acquire);
    while (now_nanos >= old) {
        if (next_allowed_nanos_.compare_exchange_weak(old, next_nanos, std::memory_order_acq_rel,
                                                      std::memory_order_acquire)) {
            counter_.Increment();
            return true;
        }
    }
    return false;
}

#define DEFINE_FEATURE_REPORTER(name, label)      \
    FeatureReporter& name() {                     \
        static FeatureReporter reporter{k##name}; \
        return reporter;                          \
    }

MILVUS_FEATURE_REPORTERS(DEFINE_FEATURE_REPORTER)

#undef DEFINE_FEATURE_REPORTER

}  // namespace milvus::monitor::feature_report
