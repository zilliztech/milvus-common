#include "cachinglayer/Metrics.h"

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
DEFINE_PROMETHEUS_GAUGE_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_slot_count, "[cpp]cache slot count");
DEFINE_PROMETHEUS_GAUGE_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_cell_count, "[cpp]cache cell count");
DEFINE_PROMETHEUS_GAUGE_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_loaded_bytes, "[cpp]cache loaded bytes");
DEFINE_PROMETHEUS_GAUGE_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_loading_bytes, "[cpp]cache loading bytes");
DEFINE_PROMETHEUS_GAUGE_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_cell_loading_count,
                                                           "[cpp]cache cell loading count");
DEFINE_PROMETHEUS_GAUGE_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_cell_loaded_count,
                                                           "[cpp]cache cell loaded count");

/* Metrics for Cache Cell Access */
DEFINE_PROMETHEUS_COUNTER_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_access_event_total,
                                                             "[cpp]cache access event total");
DEFINE_PROMETHEUS_COUNTER_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_cell_access_hit_total,
                                                             "[cpp]cache cell access hit total");
DEFINE_PROMETHEUS_COUNTER_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_cell_access_miss_total,
                                                             "[cpp]cache cell access miss total");

/* Metrics for Cache Cell Loading */
DEFINE_PROMETHEUS_COUNTER_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_load_event_success_total,
                                                             "[cpp]cache load event success total");
DEFINE_PROMETHEUS_COUNTER_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_load_event_fail_total,
                                                             "[cpp]cache load event fail total");
DEFINE_PROMETHEUS_HISTOGRAM_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_load_latency_microseconds,
                                                               milvus::monitor::secondsBuckets,
                                                               "[cpp]cache load latency microseconds");
DEFINE_PROMETHEUS_COUNTER_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_cell_load_success_total,
                                                             "[cpp]cache cell load success total");
DEFINE_PROMETHEUS_COUNTER_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_cell_load_fail_total,
                                                             "[cpp]cache cell load fail total");

/* Metrics for Cache Cell Eviction */
DEFINE_PROMETHEUS_COUNTER_METRIC_WITH_LOCATION(internal_cache_eviction_event_total, "[cpp]cache eviction event total");
DEFINE_PROMETHEUS_COUNTER_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_cell_evicted_total,
                                                             "[cpp]cache cell evicted total");
DEFINE_PROMETHEUS_COUNTER_METRIC_WITH_LOCATION(internal_cache_evicted_bytes_total, "[cpp]cache evicted bytes total");
DEFINE_PROMETHEUS_HISTOGRAM_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_cell_lifetime_seconds,
                                                               milvus::monitor::secondsBuckets,
                                                               "[cpp]cache cell lifetime seconds");

}  // namespace milvus::cachinglayer::monitor
