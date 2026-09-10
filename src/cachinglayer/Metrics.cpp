#include "cachinglayer/Metrics.h"

#include <atomic>
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
std::map<CellDataType, prometheus::Gauge*> aggregate_shard_disk_usage_gauges;
std::atomic<bool> aggregate_shard_disk_usage{false};
// Protected by shard_disk_usage_mutex; the first handle also freezes the mode.
bool shard_disk_usage_mode_initialized = false;

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
    CacheShardDiskUsageMetricEntry(ShardDiskUsageMetricKey key, prometheus::Gauge* gauge, bool aggregate)
        : key(std::move(key)), gauge(gauge), aggregate(aggregate) {
    }

    ShardDiskUsageMetricKey key;
    prometheus::Gauge* gauge{nullptr};
    const bool aggregate;
    std::atomic<double> bytes{0};
    // Aggregate business ownership must not include temporary snapshot refs.
    // Protected by shard_disk_usage_mutex.
    size_t handles{0};
};

CacheShardDiskUsageMetricHandle::CacheShardDiskUsageMetricHandle(std::shared_ptr<CacheShardDiskUsageMetricEntry> entry)
    : entry_(std::move(entry)) {
    if (entry_->aggregate) {
        ++entry_->handles;
    }
}

CacheShardDiskUsageMetricHandle::~CacheShardDiskUsageMetricHandle() {
    if (!entry_->aggregate) {
        return;
    }
    std::lock_guard<std::mutex> lock(shard_disk_usage_mutex);
    if (--entry_->handles != 0) {
        return;
    }
    // CacheCell normally refunds all bytes before its handle is destroyed.
    // Retire any remaining contribution without resetting the shared Gauge.
    entry_->gauge->Decrement(entry_->bytes.exchange(0, std::memory_order_relaxed));
    const auto it = shard_disk_usage_metrics.find(entry_->key);
    if (it != shard_disk_usage_metrics.end() && it->second.entry.lock() == entry_) {
        shard_disk_usage_metrics.erase(it);
    }
}

void
CacheShardDiskUsageMetricHandle::Increment(double value) {
    entry_->bytes.fetch_add(value, std::memory_order_relaxed);
    entry_->gauge->Increment(value);
}

void
CacheShardDiskUsageMetricHandle::Decrement(double value) {
    entry_->bytes.fetch_sub(value, std::memory_order_relaxed);
    entry_->gauge->Decrement(value);
}

double
CacheShardDiskUsageMetricHandle::Value() const {
    return entry_->bytes.load(std::memory_order_relaxed);
}

bool
set_cache_shard_disk_usage_metrics_mode(bool aggregate) {
    std::lock_guard<std::mutex> lock(shard_disk_usage_mutex);
    if (shard_disk_usage_mode_initialized && aggregate_shard_disk_usage.load() != aggregate) {
        return false;
    }
    aggregate_shard_disk_usage.store(aggregate);
    shard_disk_usage_mode_initialized = true;
    return true;
}

bool
cache_shard_disk_usage_metrics_aggregate() {
    return aggregate_shard_disk_usage.load();
}

std::unique_ptr<CacheShardDiskUsageMetricHandle>
create_cache_shard_disk_usage_metric_handle(CellDataType type, const std::string& shard) {
    if (shard.empty()) {
        return nullptr;
    }
    auto key = MakeShardDiskUsageMetricKey(type, shard);

    std::lock_guard<std::mutex> lock(shard_disk_usage_mutex);
    shard_disk_usage_mode_initialized = true;
    const bool aggregate = aggregate_shard_disk_usage.load();
    auto it = shard_disk_usage_metrics.find(key);
    if (it != shard_disk_usage_metrics.end()) {
        if (auto entry = it->second.entry.lock()) {
            return std::unique_ptr<CacheShardDiskUsageMetricHandle>(new CacheShardDiskUsageMetricHandle(entry));
        }
        if (!aggregate) {
            internal_cache_shard_disk_usage_bytes_family.Remove(it->second.gauge);
        }
        shard_disk_usage_metrics.erase(it);
    }

    prometheus::Gauge* gauge;
    if (aggregate) {
        auto& shared_gauge = aggregate_shard_disk_usage_gauges[type];
        if (shared_gauge == nullptr) {
            shared_gauge = &internal_cache_shard_disk_usage_bytes_family.Add(
                {{"data_type", CellDataTypeLabel(type)}, {"shard", "all"}});
        }
        gauge = shared_gauge;
    } else {
        gauge = &internal_cache_shard_disk_usage_bytes_family.Add(MakeShardDiskUsageLabels(key));
    }
    auto entry = std::make_shared<CacheShardDiskUsageMetricEntry>(key, gauge, aggregate);
    shard_disk_usage_metrics.emplace(std::move(key), ShardDiskUsageMetricValue{entry, gauge});
    return std::unique_ptr<CacheShardDiskUsageMetricHandle>(new CacheShardDiskUsageMetricHandle(std::move(entry)));
}

std::vector<CacheShardDiskUsageStats>
collect_cache_shard_disk_usage_stats() {
    std::lock_guard<std::mutex> lock(shard_disk_usage_mutex);

    std::vector<CacheShardDiskUsageStats> stats;
    stats.reserve(shard_disk_usage_metrics.size());
    for (auto it = shard_disk_usage_metrics.begin(); it != shard_disk_usage_metrics.end();) {
        auto entry = it->second.entry.lock();
        if (entry == nullptr) {
            if (!aggregate_shard_disk_usage.load()) {
                internal_cache_shard_disk_usage_bytes_family.Remove(it->second.gauge);
            }
            it = shard_disk_usage_metrics.erase(it);
            continue;
        }
        stats.push_back(CacheShardDiskUsageStats{entry->key.cell_data_type, entry->key.shard,
                                                 entry->bytes.load(std::memory_order_relaxed)});
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
    auto entry = it->second.entry.lock();
    if (entry == nullptr) {
        if (!aggregate_shard_disk_usage.load()) {
            internal_cache_shard_disk_usage_bytes_family.Remove(it->second.gauge);
        }
        shard_disk_usage_metrics.erase(it);
        return std::nullopt;
    }
    return entry->bytes.load(std::memory_order_relaxed);
}

}  // namespace milvus::cachinglayer::monitor
