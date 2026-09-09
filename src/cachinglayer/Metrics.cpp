#include "cachinglayer/Metrics.h"

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace milvus::cachinglayer::monitor {

DEFINE_LABEL_MAP_WITH_DATA_TYPE_AND_LOCATION(vector_field, memory);
DEFINE_LABEL_MAP_WITH_DATA_TYPE_AND_LOCATION(vector_index, memory);
DEFINE_LABEL_MAP_WITH_DATA_TYPE_AND_LOCATION(scalar_field, memory);
DEFINE_LABEL_MAP_WITH_DATA_TYPE_AND_LOCATION(scalar_index, memory);
DEFINE_LABEL_MAP_WITH_DATA_TYPE_AND_LOCATION(other, memory);
DEFINE_LABEL_MAP_WITH_DATA_TYPE_AND_LOCATION(vector_field, disk);
DEFINE_LABEL_MAP_WITH_DATA_TYPE_AND_LOCATION(vector_index, disk);
DEFINE_LABEL_MAP_WITH_DATA_TYPE_AND_LOCATION(scalar_field, disk);
DEFINE_LABEL_MAP_WITH_DATA_TYPE_AND_LOCATION(scalar_index, disk);
DEFINE_LABEL_MAP_WITH_DATA_TYPE_AND_LOCATION(other, disk);
DEFINE_LABEL_MAP_WITH_DATA_TYPE_AND_LOCATION(vector_field, mixed);
DEFINE_LABEL_MAP_WITH_DATA_TYPE_AND_LOCATION(vector_index, mixed);
DEFINE_LABEL_MAP_WITH_DATA_TYPE_AND_LOCATION(scalar_field, mixed);
DEFINE_LABEL_MAP_WITH_DATA_TYPE_AND_LOCATION(scalar_index, mixed);
DEFINE_LABEL_MAP_WITH_DATA_TYPE_AND_LOCATION(other, mixed);

DEFINE_LABEL_MAP_WITH_LOCATION(memory);
DEFINE_LABEL_MAP_WITH_LOCATION(disk);
DEFINE_LABEL_MAP_WITH_LOCATION(mixed);

/* Metrics for Cache Resource Usage */
DEFINE_PROMETHEUS_GAUGE_METRIC_WITH_LOCATION(internal_cache_capacity_bytes, "[cpp]cache capacity bytes");
DEFINE_PROMETHEUS_GAUGE_METRIC_WITH_LOCATION(internal_cache_high_watermark_bytes, "[cpp]cache high watermark bytes");
DEFINE_PROMETHEUS_GAUGE_METRIC_WITH_LOCATION(internal_cache_low_watermark_bytes, "[cpp]cache low watermark bytes");
DEFINE_PROMETHEUS_GAUGE_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_slot_count, "[cpp]cache slot count");
DEFINE_PROMETHEUS_GAUGE_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_cell_count, "[cpp]cache cell count");
DEFINE_PROMETHEUS_GAUGE_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_loaded_bytes, "[cpp]cache loaded bytes");
DEFINE_PROMETHEUS_GAUGE_METRIC_WITH_DATA_TYPE_AND_LOCATION(
    internal_cache_loading_bytes, "[cpp]estimated resource bytes held by active cache load requests");
DEFINE_PROMETHEUS_GAUGE_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_cell_loading_count,
                                                           "[cpp]cache cell loading count");
DEFINE_PROMETHEUS_GAUGE_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_cell_loaded_count,
                                                           "[cpp]cache cell loaded count");
DEFINE_PROMETHEUS_GAUGE_FAMILY(internal_cache_shard_disk_usage_bytes,
                               "[cpp]attributed cache-slot loaded disk usage bytes by shard");
DEFINE_PROMETHEUS_GAUGE_FAMILY(internal_cache_shard_memory_usage_bytes,
                               "[cpp]attributed cache loaded memory usage bytes by shard");

/* Metrics for Cache Cell Access */
DEFINE_PROMETHEUS_COUNTER_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_access_event_total,
                                                             "[cpp]cache access event total");
DEFINE_PROMETHEUS_COUNTER_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_cell_access_hit_bytes_total,
                                                             "[cpp]cache cell access hit bytes total");
DEFINE_PROMETHEUS_COUNTER_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_cell_access_miss_bytes_total,
                                                             "[cpp]cache cell access miss bytes total");

/* Metrics for Cache Cell Loading */
DEFINE_PROMETHEUS_COUNTER_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_load_event_fail_total,
                                                             "[cpp]cache load event fail total");
DEFINE_PROMETHEUS_HISTOGRAM_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_load_latency_microseconds,
                                                               milvus::monitor::secondsBuckets,
                                                               "[cpp]cache load latency microseconds");

/* Metrics for Cache Cell Eviction */
DEFINE_PROMETHEUS_COUNTER_METRIC_WITH_LOCATION(internal_cache_eviction_event_total, "[cpp]cache eviction event total");
DEFINE_PROMETHEUS_COUNTER_METRIC_WITH_LOCATION(internal_cache_evicted_bytes_total, "[cpp]cache evicted bytes total");
DEFINE_PROMETHEUS_HISTOGRAM_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_cell_lifetime_seconds,
                                                               milvus::monitor::secondsBuckets,
                                                               "[cpp]cache cell lifetime seconds");

namespace {

struct ShardUsageMetricKey {
    CellDataType cell_data_type;
    std::string shard;

    bool
    operator<(const ShardUsageMetricKey& other) const {
        if (cell_data_type != other.cell_data_type) {
            return static_cast<int>(cell_data_type) < static_cast<int>(other.cell_data_type);
        }
        return shard < other.shard;
    }
};

struct ShardUsageMetricValue {
    std::weak_ptr<CacheShardUsageMetricEntry> entry;
    prometheus::Gauge* gauge{nullptr};
};

std::mutex shard_usage_mutex;
std::map<StorageType, std::map<ShardUsageMetricKey, ShardUsageMetricValue>> shard_usage_metrics;

const char*
CellDataTypeLabel(CellDataType type) {
    switch (type) {
        case CellDataType::VECTOR_FIELD:
            return "vector_field";
        case CellDataType::VECTOR_INDEX:
            return "vector_index";
        case CellDataType::SCALAR_FIELD:
            return "scalar_field";
        case CellDataType::SCALAR_INDEX:
            return "scalar_index";
        case CellDataType::OTHER:
            return "other";
    }
    ThrowInfo(ErrorCode::UnexpectedError, "Unknown CellDataType");
}

ShardUsageMetricKey
MakeShardUsageMetricKey(CellDataType type, const std::string& shard) {
    static_cast<void>(CellDataTypeLabel(type));
    return {type, shard};
}

std::map<std::string, std::string>
MakeShardUsageLabels(const ShardUsageMetricKey& key) {
    return {{"data_type", CellDataTypeLabel(key.cell_data_type)}, {"shard", key.shard}};
}

prometheus::Family<prometheus::Gauge>&
GetShardUsageMetricFamily(StorageType storage_type) {
    if (storage_type == StorageType::MEMORY) {
        return internal_cache_shard_memory_usage_bytes_family;
    }
    if (storage_type == StorageType::DISK) {
        return internal_cache_shard_disk_usage_bytes_family;
    }
    ThrowInfo(ErrorCode::UnexpectedError, "Shard usage metrics require MEMORY or DISK storage type");
}

}  // namespace

struct CacheShardUsageMetricEntry {
    ShardUsageMetricKey key;
    prometheus::Gauge* gauge{nullptr};
};

CacheShardUsageMetricHandle::CacheShardUsageMetricHandle(std::shared_ptr<CacheShardUsageMetricEntry> entry)
    : entry_(std::move(entry)) {
}

CacheShardUsageMetricHandle::~CacheShardUsageMetricHandle() = default;

void
CacheShardUsageMetricHandle::Increment(double value) {
    std::lock_guard<std::mutex> lock(shard_usage_mutex);
    if (entry_->gauge != nullptr) {
        entry_->gauge->Increment(value);
    }
}

void
CacheShardUsageMetricHandle::Decrement(double value) {
    std::lock_guard<std::mutex> lock(shard_usage_mutex);
    if (entry_->gauge != nullptr) {
        entry_->gauge->Decrement(value);
    }
}

double
CacheShardUsageMetricHandle::Value() const {
    std::lock_guard<std::mutex> lock(shard_usage_mutex);
    if (entry_->gauge == nullptr) {
        return 0;
    }
    return entry_->gauge->Value();
}

std::unique_ptr<CacheShardUsageMetricHandle>
create_cache_shard_usage_metric_handle(CellDataType type, const std::string& shard, StorageType storage_type) {
    if (shard.empty()) {
        return nullptr;
    }
    auto key = MakeShardUsageMetricKey(type, shard);
    auto& family = GetShardUsageMetricFamily(storage_type);

    std::lock_guard<std::mutex> lock(shard_usage_mutex);
    auto& metrics = shard_usage_metrics[storage_type];
    auto it = metrics.find(key);
    if (it != metrics.end()) {
        if (auto entry = it->second.entry.lock()) {
            return std::unique_ptr<CacheShardUsageMetricHandle>(new CacheShardUsageMetricHandle(entry));
        }
        family.Remove(it->second.gauge);
        metrics.erase(it);
    }

    auto& gauge = family.Add(MakeShardUsageLabels(key));
    auto entry = std::make_shared<CacheShardUsageMetricEntry>(CacheShardUsageMetricEntry{key, &gauge});
    metrics.emplace(std::move(key), ShardUsageMetricValue{entry, &gauge});
    return std::unique_ptr<CacheShardUsageMetricHandle>(new CacheShardUsageMetricHandle(std::move(entry)));
}

std::vector<CacheShardUsageStats>
collect_cache_shard_usage_stats(StorageType storage_type) {
    auto& family = GetShardUsageMetricFamily(storage_type);
    std::lock_guard<std::mutex> lock(shard_usage_mutex);

    std::vector<CacheShardUsageStats> stats;
    auto bucket = shard_usage_metrics.find(storage_type);
    if (bucket == shard_usage_metrics.end()) {
        return stats;
    }
    auto& metrics = bucket->second;
    for (auto it = metrics.begin(); it != metrics.end();) {
        auto entry = it->second.entry.lock();
        if (entry == nullptr) {
            family.Remove(it->second.gauge);
            it = metrics.erase(it);
            continue;
        }
        stats.push_back(
            CacheShardUsageStats{entry->key.cell_data_type, entry->key.shard, storage_type, entry->gauge->Value()});
        ++it;
    }
    return stats;
}

std::optional<double>
cache_shard_usage_bytes_value(CellDataType type, const std::string& shard, StorageType storage_type) {
    if (shard.empty()) {
        return std::nullopt;
    }
    auto key = MakeShardUsageMetricKey(type, shard);
    auto& family = GetShardUsageMetricFamily(storage_type);
    std::lock_guard<std::mutex> lock(shard_usage_mutex);
    auto bucket = shard_usage_metrics.find(storage_type);
    if (bucket == shard_usage_metrics.end()) {
        return std::nullopt;
    }
    auto& metrics = bucket->second;
    auto it = metrics.find(key);
    if (it == metrics.end()) {
        return std::nullopt;
    }
    if (it->second.entry.expired()) {
        family.Remove(it->second.gauge);
        metrics.erase(it);
        return std::nullopt;
    }
    return it->second.gauge->Value();
}

}  // namespace milvus::cachinglayer::monitor
