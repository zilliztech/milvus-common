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
#include "cachinglayer/Manager.h"

#include <folly/executors/thread_factory/NamedThreadFactory.h>

#include <memory>
#include <mutex>

#include "cachinglayer/Utils.h"
#include "log/Log.h"

namespace milvus::cachinglayer {

Manager&
Manager::GetInstance() {
    static Manager instance;
    return instance;
}

Manager::~Manager() {
    if (prefetch_pool_) {
        prefetch_pool_->stop();
        prefetch_pool_->join();
    }
}

void
Manager::ConfigureTieredStorage(const TieredStorageOptions& options) {
    static std::once_flag init_once;
    std::call_once(init_once, [&]() {
        auto& config = TieredStorageConfig::GetInstance();
        config.UpdateAll(options.storage_usage_tracking_enabled, options.loading_timeout,
                         options.warmup_loading_timeout, options.warmup_policies, options.max_loading_mem_ratio);
        const auto max_loading_mem_size = internal::getMaxLoadingMemSize(options.max_loading_mem_ratio);

        Manager& manager = GetInstance();
        manager.eviction_enabled_ = options.eviction_enabled;

        if (options.prefetch_pool_threads > 0) {
            manager.prefetch_pool_ = std::make_shared<folly::CPUThreadPoolExecutor>(
                options.prefetch_pool_threads, std::make_shared<folly::NamedThreadFactory>("milvus_prefetch"));
            LOG_INFO("[MCL] Prefetch pool initialized with {} threads", options.prefetch_pool_threads);
        }

        ResourceUsage max{options.cache_limit.memory_max_bytes, options.cache_limit.disk_max_bytes};
        ResourceUsage low_watermark{options.cache_limit.memory_low_watermark_bytes,
                                    options.cache_limit.disk_low_watermark_bytes};
        ResourceUsage high_watermark{options.cache_limit.memory_high_watermark_bytes,
                                     options.cache_limit.disk_high_watermark_bytes};

        manager.dlist_ = std::make_shared<internal::DList>(options.eviction_enabled, max, low_watermark, high_watermark,
                                                           options.eviction_config, max_loading_mem_size);

        LOG_INFO(
            "[MCL] Configured Tiered Storage manager with "
            "memory watermark: low {}, high {}, max {}, "
            "disk watermark: low {}, high {}, max {}, "
            "cache touch window: {} ms, "
            "background eviction enabled: {}, eviction interval: {} ms, "
            "physical memory max ratio: {}, max disk usage percentage: {}, "
            "loading resource factor: {}, cache cell unaccessed survival time: "
            "{} s, max loading memory ratio: {}, max loading memory size: {}, warmup policies: {}",
            FormatBytes(low_watermark.memory_bytes), FormatBytes(high_watermark.memory_bytes),
            FormatBytes(max.memory_bytes), FormatBytes(low_watermark.file_bytes),
            FormatBytes(high_watermark.file_bytes), FormatBytes(max.file_bytes),
            options.eviction_config.cache_touch_window.count(), options.eviction_config.background_eviction_enabled,
            options.eviction_config.eviction_interval.count(),
            options.eviction_config.overloaded_memory_threshold_percentage,
            options.eviction_config.max_disk_usage_percentage, options.eviction_config.loading_resource_factor,
            options.eviction_config.cache_cell_unaccessed_survival_time.count(),
            options.max_loading_mem_ratio, FormatBytes(max_loading_mem_size), options.warmup_policies.ToString());
    });
}

void
Manager::ConfigureTieredStorage(CacheWarmupPolicies warmup_policies, CacheLimit cache_limit,
                                bool storage_usage_tracking_enabled, bool eviction_enabled,
                                EvictionConfig eviction_config, std::chrono::milliseconds loading_timeout,
                                std::chrono::milliseconds warmup_loading_timeout, uint32_t prefetch_pool_threads) {
    TieredStorageOptions options;
    options.warmup_policies = warmup_policies;
    options.cache_limit = cache_limit;
    options.storage_usage_tracking_enabled = storage_usage_tracking_enabled;
    options.eviction_enabled = eviction_enabled;
    options.eviction_config = eviction_config;
    options.loading_timeout = loading_timeout;
    options.warmup_loading_timeout = warmup_loading_timeout;
    options.prefetch_pool_threads = prefetch_pool_threads;
    ConfigureTieredStorage(options);
}

void
Manager::UpdateConfig(std::chrono::milliseconds loading_timeout, std::chrono::milliseconds warmup_loading_timeout,
                      bool storage_usage_tracking_enabled, CacheWarmupPolicies warmup_policies) {
    TieredStorageConfig::GetInstance().UpdateAll(storage_usage_tracking_enabled, loading_timeout,
                                                 warmup_loading_timeout, warmup_policies);
    LOG_INFO(
        "[MCL] Config updated: loading_timeout={}ms, warmup_loading_timeout={}ms, "
        "storage_usage_tracking={}, warmup_policies={}",
        loading_timeout.count(), warmup_loading_timeout.count(), storage_usage_tracking_enabled,
        warmup_policies.ToString());
}

void
Manager::UpdateConfig(std::chrono::milliseconds loading_timeout, std::chrono::milliseconds warmup_loading_timeout,
                      bool storage_usage_tracking_enabled, CacheWarmupPolicies warmup_policies,
                      double max_loading_mem_ratio) {
    TieredStorageConfig::GetInstance().UpdateAll(storage_usage_tracking_enabled, loading_timeout,
                                                 warmup_loading_timeout, warmup_policies, max_loading_mem_ratio);
    const auto max_loading_mem_size = internal::getMaxLoadingMemSize(max_loading_mem_ratio);
    Manager& manager = GetInstance();
    if (manager.dlist_) {
        manager.dlist_->UpdateMaxLoadingMemSize(max_loading_mem_size);
    }
    LOG_INFO(
        "[MCL] Config updated: loading_timeout={}ms, warmup_loading_timeout={}ms, "
        "storage_usage_tracking={}, max_loading_mem_ratio={}, max_loading_mem_size={}, warmup_policies={}",
        loading_timeout.count(), warmup_loading_timeout.count(), storage_usage_tracking_enabled,
        max_loading_mem_ratio, FormatBytes(max_loading_mem_size), warmup_policies.ToString());
}

void
Manager::UpdateMaxLoadingMemRatio(double max_loading_mem_ratio) {
    TieredStorageConfig::GetInstance().SetMaxLoadingMemRatio(max_loading_mem_ratio);
    const auto max_loading_mem_size = internal::getMaxLoadingMemSize(max_loading_mem_ratio);
    Manager& manager = GetInstance();
    if (manager.dlist_) {
        manager.dlist_->UpdateMaxLoadingMemSize(max_loading_mem_size);
    }
}

size_t
Manager::memory_overhead() const {
    // TODO(tiered storage 2): calculate memory overhead
    return 0;
}

}  // namespace milvus::cachinglayer
