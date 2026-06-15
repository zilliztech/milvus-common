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

constexpr std::string_view kHybridSearch = "hybrid_search";
constexpr std::string_view kPartitionKey = "partition_key";
constexpr std::string_view kDynamicField = "dynamic_field";
constexpr std::string_view kBM25Function = "bm25_function";
constexpr std::string_view kResourceGroup = "resource_group";
constexpr std::string_view kBulkImport = "bulk_import";

class alignas(64) FeatureReporter {
 public:
    // Prefer the predeclared static reporters below on hot paths.
    explicit FeatureReporter(std::string_view name);

    FeatureReporter(const FeatureReporter&) = delete;
    FeatureReporter&
    operator=(const FeatureReporter&) = delete;

    std::string_view
    Name() const noexcept;

    bool
    Record();

    bool
    RecordAtForTest(std::chrono::steady_clock::time_point now);

    void
    ResetForTest();

 private:
    bool
    recordAt(std::chrono::steady_clock::time_point now);

 private:
    std::atomic<int64_t> next_allowed_nanos_{0};
    std::string name_;
    prometheus::Counter& counter_;
};

FeatureReporter&
HybridSearch();

FeatureReporter&
PartitionKey();

FeatureReporter&
DynamicField();

FeatureReporter&
BM25Function();

FeatureReporter&
ResourceGroup();

FeatureReporter&
BulkImport();

}  // namespace milvus::monitor::feature_report
