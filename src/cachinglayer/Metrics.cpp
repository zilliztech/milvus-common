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

struct ShardDiskUsageMetricKey {
    CellDataType cell_data_type;
    std::string shard;

    bool
    operator<(const ShardDiskUsageMetricKey& other) const {
        if (cell_data_type != other.cell_data_type) {
            return static_cast<int>(cell_data_type) < static_cast<int>(other.cell_data_type);
        }
        return shard < other.shard;
    }
};

struct ShardDiskUsageMetricValue {
    std::weak_ptr<CacheShardDiskUsageMetricEntry> entry;
    prometheus::Gauge* gauge{nullptr};
};

std::mutex shard_disk_usage_mutex;
std::map<ShardDiskUsageMetricKey, ShardDiskUsageMetricValue> shard_disk_usage_metrics;

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

ShardDiskUsageMetricKey
MakeShardDiskUsageMetricKey(CellDataType type, const std::string& shard) {
    static_cast<void>(CellDataTypeLabel(type));
    return {type, shard};
}

std::map<std::string, std::string>
MakeShardDiskUsageLabels(const ShardDiskUsageMetricKey& key) {
    return {{"data_type", CellDataTypeLabel(key.cell_data_type)}, {"shard", key.shard}};
}

}  // namespace

struct CacheShardDiskUsageMetricEntry {
    ShardDiskUsageMetricKey key;
    prometheus::Gauge* gauge{nullptr};
};

CacheShardDiskUsageMetricHandle::CacheShardDiskUsageMetricHandle(std::shared_ptr<CacheShardDiskUsageMetricEntry> entry)
    : entry_(std::move(entry)) {
}

CacheShardDiskUsageMetricHandle::~CacheShardDiskUsageMetricHandle() = default;

void
CacheShardDiskUsageMetricHandle::Increment(double value) {
    std::lock_guard<std::mutex> lock(shard_disk_usage_mutex);
    if (entry_->gauge != nullptr) {
        entry_->gauge->Increment(value);
    }
}

void
CacheShardDiskUsageMetricHandle::Decrement(double value) {
    std::lock_guard<std::mutex> lock(shard_disk_usage_mutex);
    if (entry_->gauge != nullptr) {
        entry_->gauge->Decrement(value);
    }
}

double
CacheShardDiskUsageMetricHandle::Value() const {
    std::lock_guard<std::mutex> lock(shard_disk_usage_mutex);
    if (entry_->gauge == nullptr) {
        return 0;
    }
    return entry_->gauge->Value();
}

std::unique_ptr<CacheShardDiskUsageMetricHandle>
create_cache_shard_disk_usage_metric_handle(CellDataType type, const std::string& shard) {
    if (shard.empty()) {
        return nullptr;
    }
    auto key = MakeShardDiskUsageMetricKey(type, shard);

    std::lock_guard<std::mutex> lock(shard_disk_usage_mutex);
    auto it = shard_disk_usage_metrics.find(key);
    if (it != shard_disk_usage_metrics.end()) {
        if (auto entry = it->second.entry.lock()) {
            return std::unique_ptr<CacheShardDiskUsageMetricHandle>(new CacheShardDiskUsageMetricHandle(entry));
        }
        internal_cache_shard_disk_usage_bytes_family.Remove(it->second.gauge);
        shard_disk_usage_metrics.erase(it);
    }

    auto& gauge = internal_cache_shard_disk_usage_bytes_family.Add(MakeShardDiskUsageLabels(key));
    auto entry = std::make_shared<CacheShardDiskUsageMetricEntry>(CacheShardDiskUsageMetricEntry{key, &gauge});
    shard_disk_usage_metrics.emplace(std::move(key), ShardDiskUsageMetricValue{entry, &gauge});
    return std::unique_ptr<CacheShardDiskUsageMetricHandle>(new CacheShardDiskUsageMetricHandle(std::move(entry)));
}

std::vector<CacheShardDiskUsageStats>
collect_cache_shard_disk_usage_stats() {
    std::lock_guard<std::mutex> lock(shard_disk_usage_mutex);

    std::vector<CacheShardDiskUsageStats> stats;
    for (auto it = shard_disk_usage_metrics.begin(); it != shard_disk_usage_metrics.end();) {
        auto entry = it->second.entry.lock();
        if (entry == nullptr) {
            internal_cache_shard_disk_usage_bytes_family.Remove(it->second.gauge);
            it = shard_disk_usage_metrics.erase(it);
            continue;
        }
        stats.push_back(CacheShardDiskUsageStats{entry->key.cell_data_type, entry->key.shard, entry->gauge->Value()});
        ++it;
    }
    return stats;
}

std::optional<double>
cache_shard_disk_usage_bytes_value(CellDataType type, const std::string& shard) {
    if (shard.empty()) {
        return std::nullopt;
    }
    auto key = MakeShardDiskUsageMetricKey(type, shard);
    std::lock_guard<std::mutex> lock(shard_disk_usage_mutex);
    auto it = shard_disk_usage_metrics.find(key);
    if (it == shard_disk_usage_metrics.end()) {
        return std::nullopt;
    }
    if (it->second.entry.expired()) {
        internal_cache_shard_disk_usage_bytes_family.Remove(it->second.gauge);
        shard_disk_usage_metrics.erase(it);
        return std::nullopt;
    }
    return it->second.gauge->Value();
}

}  // namespace milvus::cachinglayer::monitor
