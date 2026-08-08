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

#pragma once

#include <prometheus/counter.h>

#include <atomic>
#include <chrono>
#include <string>
#include <string_view>

namespace milvus::monitor::feature_report {

#define MILVUS_FEATURE_REPORTERS(FEATURE)    \
    FEATURE(HybridSearch, "hybrid_search")   \
    FEATURE(PartitionKey, "partition_key")   \
    FEATURE(DynamicField, "dynamic_field")   \
    FEATURE(BM25Function, "bm25_function")   \
    FEATURE(ResourceGroup, "resource_group") \
    FEATURE(BulkImport, "bulk_import")

class alignas(64) FeatureReporter {
 public:
    FeatureReporter(const FeatureReporter&) = delete;
    FeatureReporter&
    operator=(const FeatureReporter&) = delete;

    std::string_view
    Name() const noexcept;

    bool
    Record();

 private:
    explicit FeatureReporter(std::string_view name);

    bool
    recordAt(std::chrono::steady_clock::time_point now);

    void
    reset();

    friend struct FeatureReporterTestPeer;

#define FRIEND_FEATURE_REPORTER(name, label) friend FeatureReporter& name();

    MILVUS_FEATURE_REPORTERS(FRIEND_FEATURE_REPORTER)

#undef FRIEND_FEATURE_REPORTER

 private:
    std::atomic<int64_t> next_allowed_nanos_{0};
    std::string name_;
    prometheus::Counter& counter_;
};

#define DECLARE_FEATURE_REPORTER(name, label)   \
    constexpr std::string_view k##name = label; \
    FeatureReporter& name();

MILVUS_FEATURE_REPORTERS(DECLARE_FEATURE_REPORTER)

#undef DECLARE_FEATURE_REPORTER

}  // namespace milvus::monitor::feature_report
