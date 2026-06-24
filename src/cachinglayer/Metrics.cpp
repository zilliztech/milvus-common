#include "cachinglayer/Metrics.h"

#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

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
DEFINE_PROMETHEUS_GAUGE_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_loading_bytes, "[cpp]cache loading bytes");
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

using ShardDiskUsageMetricKey = std::pair<std::string, std::string>;

struct ShardDiskUsageMetricEntry {
    prometheus::Gauge* gauge{nullptr};
    size_t ref_count{0};
};

std::mutex shard_disk_usage_mutex;
std::map<ShardDiskUsageMetricKey, ShardDiskUsageMetricEntry> shard_disk_usage_metrics;

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
    return {CellDataTypeLabel(type), shard};
}

std::map<std::string, std::string>
MakeShardDiskUsageLabels(const ShardDiskUsageMetricKey& key) {
    return {{"data_type", key.first}, {"shard", key.second}};
}

void
ReleaseShardDiskUsageMetricHandle(const ShardDiskUsageMetricKey& key) {
    std::lock_guard<std::mutex> lock(shard_disk_usage_mutex);
    auto it = shard_disk_usage_metrics.find(key);
    if (it == shard_disk_usage_metrics.end()) {
        return;
    }
    if (--it->second.ref_count > 0) {
        return;
    }
    internal_cache_shard_disk_usage_bytes_family.Remove(it->second.gauge);
    shard_disk_usage_metrics.erase(it);
}

}  // namespace

CacheShardDiskUsageMetricHandle::CacheShardDiskUsageMetricHandle(std::string data_type_label, std::string shard)
    : data_type_label_(std::move(data_type_label)), shard_(std::move(shard)) {
}

CacheShardDiskUsageMetricHandle::~CacheShardDiskUsageMetricHandle() {
    ReleaseShardDiskUsageMetricHandle({data_type_label_, shard_});
}

void
CacheShardDiskUsageMetricHandle::Increment(double value) {
    std::lock_guard<std::mutex> lock(shard_disk_usage_mutex);
    auto it = shard_disk_usage_metrics.find({data_type_label_, shard_});
    if (it != shard_disk_usage_metrics.end()) {
        it->second.gauge->Increment(value);
    }
}

void
CacheShardDiskUsageMetricHandle::Decrement(double value) {
    std::lock_guard<std::mutex> lock(shard_disk_usage_mutex);
    auto it = shard_disk_usage_metrics.find({data_type_label_, shard_});
    if (it != shard_disk_usage_metrics.end()) {
        it->second.gauge->Decrement(value);
    }
}

double
CacheShardDiskUsageMetricHandle::Value() const {
    std::lock_guard<std::mutex> lock(shard_disk_usage_mutex);
    auto it = shard_disk_usage_metrics.find({data_type_label_, shard_});
    if (it == shard_disk_usage_metrics.end()) {
        return 0;
    }
    return it->second.gauge->Value();
}

std::unique_ptr<CacheShardDiskUsageMetricHandle>
create_cache_shard_disk_usage_metric_handle(CellDataType type, const std::string& shard) {
    auto key = MakeShardDiskUsageMetricKey(type, shard);
    if (key.second.empty()) {
        return nullptr;
    }
    auto handle =
        std::unique_ptr<CacheShardDiskUsageMetricHandle>(new CacheShardDiskUsageMetricHandle(key.first, key.second));

    std::lock_guard<std::mutex> lock(shard_disk_usage_mutex);
    auto it = shard_disk_usage_metrics.find(key);
    if (it != shard_disk_usage_metrics.end()) {
        ++it->second.ref_count;
        return handle;
    }

    auto& gauge = internal_cache_shard_disk_usage_bytes_family.Add(MakeShardDiskUsageLabels(key));
    shard_disk_usage_metrics.emplace(std::move(key), ShardDiskUsageMetricEntry{&gauge, 1});
    return handle;
}

std::optional<double>
cache_shard_disk_usage_bytes_value(CellDataType type, const std::string& shard) {
    auto key = MakeShardDiskUsageMetricKey(type, shard);
    if (key.second.empty()) {
        return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(shard_disk_usage_mutex);
    auto it = shard_disk_usage_metrics.find(key);
    if (it == shard_disk_usage_metrics.end()) {
        return std::nullopt;
    }
    return it->second.gauge->Value();
}

}  // namespace milvus::cachinglayer::monitor
