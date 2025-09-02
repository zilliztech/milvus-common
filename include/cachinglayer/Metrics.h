#pragma once

#include "cachinglayer/Utils.h"
#include "common/PrometheusClient.h"

namespace milvus::cachinglayer::monitor {

/* Caching Layer Metrics Declaration Helpers */
#define DECLARE_PROMETHEUS_GAUGE_METRIC_WITH_DATA_TYPE_AND_LOCATION(metric_name) \
    DECLARE_PROMETHEUS_GAUGE_FAMILY(metric_name);                                \
    DECLARE_PROMETHEUS_GAUGE(metric_name##_vector_field_memory);                 \
    DECLARE_PROMETHEUS_GAUGE(metric_name##_vector_index_memory);                 \
    DECLARE_PROMETHEUS_GAUGE(metric_name##_scalar_field_memory);                 \
    DECLARE_PROMETHEUS_GAUGE(metric_name##_scalar_index_memory);                 \
    DECLARE_PROMETHEUS_GAUGE(metric_name##_other_memory);                        \
    DECLARE_PROMETHEUS_GAUGE(metric_name##_vector_field_disk);                   \
    DECLARE_PROMETHEUS_GAUGE(metric_name##_vector_index_disk);                   \
    DECLARE_PROMETHEUS_GAUGE(metric_name##_scalar_field_disk);                   \
    DECLARE_PROMETHEUS_GAUGE(metric_name##_scalar_index_disk);                   \
    DECLARE_PROMETHEUS_GAUGE(metric_name##_other_disk);                          \
    DECLARE_PROMETHEUS_GAUGE(metric_name##_vector_field_mixed);                  \
    DECLARE_PROMETHEUS_GAUGE(metric_name##_vector_index_mixed);                  \
    DECLARE_PROMETHEUS_GAUGE(metric_name##_scalar_field_mixed);                  \
    DECLARE_PROMETHEUS_GAUGE(metric_name##_scalar_index_mixed);                  \
    DECLARE_PROMETHEUS_GAUGE(metric_name##_other_mixed);

#define DECLARE_PROMETHEUS_COUNTER_METRIC_WITH_DATA_TYPE_AND_LOCATION(metric_name) \
    DECLARE_PROMETHEUS_COUNTER_FAMILY(metric_name);                                \
    DECLARE_PROMETHEUS_COUNTER(metric_name##_vector_field_memory);                 \
    DECLARE_PROMETHEUS_COUNTER(metric_name##_vector_index_memory);                 \
    DECLARE_PROMETHEUS_COUNTER(metric_name##_scalar_field_memory);                 \
    DECLARE_PROMETHEUS_COUNTER(metric_name##_scalar_index_memory);                 \
    DECLARE_PROMETHEUS_COUNTER(metric_name##_other_memory);                        \
    DECLARE_PROMETHEUS_COUNTER(metric_name##_vector_field_disk);                   \
    DECLARE_PROMETHEUS_COUNTER(metric_name##_vector_index_disk);                   \
    DECLARE_PROMETHEUS_COUNTER(metric_name##_scalar_field_disk);                   \
    DECLARE_PROMETHEUS_COUNTER(metric_name##_scalar_index_disk);                   \
    DECLARE_PROMETHEUS_COUNTER(metric_name##_other_disk);                          \
    DECLARE_PROMETHEUS_COUNTER(metric_name##_vector_field_mixed);                  \
    DECLARE_PROMETHEUS_COUNTER(metric_name##_vector_index_mixed);                  \
    DECLARE_PROMETHEUS_COUNTER(metric_name##_scalar_field_mixed);                  \
    DECLARE_PROMETHEUS_COUNTER(metric_name##_scalar_index_mixed);                  \
    DECLARE_PROMETHEUS_COUNTER(metric_name##_other_mixed);

#define DECLARE_PROMETHEUS_HISTOGRAM_METRIC_WITH_DATA_TYPE_AND_LOCATION(metric_name) \
    DECLARE_PROMETHEUS_HISTOGRAM_FAMILY(metric_name);                                \
    DECLARE_PROMETHEUS_HISTOGRAM(metric_name##_vector_field_memory);                 \
    DECLARE_PROMETHEUS_HISTOGRAM(metric_name##_vector_index_memory);                 \
    DECLARE_PROMETHEUS_HISTOGRAM(metric_name##_scalar_field_memory);                 \
    DECLARE_PROMETHEUS_HISTOGRAM(metric_name##_scalar_index_memory);                 \
    DECLARE_PROMETHEUS_HISTOGRAM(metric_name##_other_memory);                        \
    DECLARE_PROMETHEUS_HISTOGRAM(metric_name##_vector_field_disk);                   \
    DECLARE_PROMETHEUS_HISTOGRAM(metric_name##_vector_index_disk);                   \
    DECLARE_PROMETHEUS_HISTOGRAM(metric_name##_scalar_field_disk);                   \
    DECLARE_PROMETHEUS_HISTOGRAM(metric_name##_scalar_index_disk);                   \
    DECLARE_PROMETHEUS_HISTOGRAM(metric_name##_other_disk);                          \
    DECLARE_PROMETHEUS_HISTOGRAM(metric_name##_vector_field_mixed);                  \
    DECLARE_PROMETHEUS_HISTOGRAM(metric_name##_vector_index_mixed);                  \
    DECLARE_PROMETHEUS_HISTOGRAM(metric_name##_scalar_field_mixed);                  \
    DECLARE_PROMETHEUS_HISTOGRAM(metric_name##_scalar_index_mixed);                  \
    DECLARE_PROMETHEUS_HISTOGRAM(metric_name##_other_mixed);

#define DECLARE_PROMETHEUS_COUNTER_METRIC_WITH_LOCATION(metric_name) \
    DECLARE_PROMETHEUS_COUNTER_FAMILY(metric_name);                  \
    DECLARE_PROMETHEUS_COUNTER(metric_name##_memory);                \
    DECLARE_PROMETHEUS_COUNTER(metric_name##_disk);                  \
    DECLARE_PROMETHEUS_COUNTER(metric_name##_mixed);

#define DECLARE_PROMETHEUS_GAUGE_METRIC_WITH_LOCATION(metric_name) \
    DECLARE_PROMETHEUS_GAUGE_FAMILY(metric_name);                  \
    DECLARE_PROMETHEUS_GAUGE(metric_name##_memory);                \
    DECLARE_PROMETHEUS_GAUGE(metric_name##_disk);                  \
    DECLARE_PROMETHEUS_GAUGE(metric_name##_mixed);

/* Caching Layer Metrics Label Helpers */
#define DEFINE_LABEL_MAP_WITH_DATA_TYPE_AND_LOCATION(data_type, storage_type)                           \
    std::map<std::string, std::string> label_##data_type##_##storage_type = {{"data_type", #data_type}, \
                                                                             {"location", #storage_type}};
#define DEFINE_LABEL_MAP_WITH_LOCATION(storage_type) \
    std::map<std::string, std::string> label_##storage_type = {{"location", #storage_type}};

/* Caching Layer Metrics Definition Helpers */
#define DEFINE_PROMETHEUS_GAUGE_METRIC_WITH_DATA_TYPE_AND_LOCATION(metric_family, desc)                     \
    DEFINE_PROMETHEUS_GAUGE_FAMILY(metric_family, desc);                                                    \
    DEFINE_PROMETHEUS_GAUGE(metric_family##_vector_field_memory, metric_family, label_vector_field_memory); \
    DEFINE_PROMETHEUS_GAUGE(metric_family##_vector_index_memory, metric_family, label_vector_index_memory); \
    DEFINE_PROMETHEUS_GAUGE(metric_family##_scalar_field_memory, metric_family, label_scalar_field_memory); \
    DEFINE_PROMETHEUS_GAUGE(metric_family##_scalar_index_memory, metric_family, label_scalar_index_memory); \
    DEFINE_PROMETHEUS_GAUGE(metric_family##_other_memory, metric_family, label_other_memory);               \
    DEFINE_PROMETHEUS_GAUGE(metric_family##_vector_field_disk, metric_family, label_vector_field_disk);     \
    DEFINE_PROMETHEUS_GAUGE(metric_family##_vector_index_disk, metric_family, label_vector_index_disk);     \
    DEFINE_PROMETHEUS_GAUGE(metric_family##_scalar_field_disk, metric_family, label_scalar_field_disk);     \
    DEFINE_PROMETHEUS_GAUGE(metric_family##_scalar_index_disk, metric_family, label_scalar_index_disk);     \
    DEFINE_PROMETHEUS_GAUGE(metric_family##_other_disk, metric_family, label_other_disk);                   \
    DEFINE_PROMETHEUS_GAUGE(metric_family##_vector_field_mixed, metric_family, label_vector_field_mixed);   \
    DEFINE_PROMETHEUS_GAUGE(metric_family##_vector_index_mixed, metric_family, label_vector_index_mixed);   \
    DEFINE_PROMETHEUS_GAUGE(metric_family##_scalar_field_mixed, metric_family, label_scalar_field_mixed);   \
    DEFINE_PROMETHEUS_GAUGE(metric_family##_scalar_index_mixed, metric_family, label_scalar_index_mixed);   \
    DEFINE_PROMETHEUS_GAUGE(metric_family##_other_mixed, metric_family, label_other_mixed);

#define DEFINE_PROMETHEUS_COUNTER_METRIC_WITH_DATA_TYPE_AND_LOCATION(metric_family, desc)                     \
    DEFINE_PROMETHEUS_COUNTER_FAMILY(metric_family, desc);                                                    \
    DEFINE_PROMETHEUS_COUNTER(metric_family##_vector_field_memory, metric_family, label_vector_field_memory); \
    DEFINE_PROMETHEUS_COUNTER(metric_family##_vector_index_memory, metric_family, label_vector_index_memory); \
    DEFINE_PROMETHEUS_COUNTER(metric_family##_scalar_field_memory, metric_family, label_scalar_field_memory); \
    DEFINE_PROMETHEUS_COUNTER(metric_family##_scalar_index_memory, metric_family, label_scalar_index_memory); \
    DEFINE_PROMETHEUS_COUNTER(metric_family##_other_memory, metric_family, label_other_memory);               \
    DEFINE_PROMETHEUS_COUNTER(metric_family##_vector_field_disk, metric_family, label_vector_field_disk);     \
    DEFINE_PROMETHEUS_COUNTER(metric_family##_vector_index_disk, metric_family, label_vector_index_disk);     \
    DEFINE_PROMETHEUS_COUNTER(metric_family##_scalar_field_disk, metric_family, label_scalar_field_disk);     \
    DEFINE_PROMETHEUS_COUNTER(metric_family##_scalar_index_disk, metric_family, label_scalar_index_disk);     \
    DEFINE_PROMETHEUS_COUNTER(metric_family##_other_disk, metric_family, label_other_disk);                   \
    DEFINE_PROMETHEUS_COUNTER(metric_family##_vector_field_mixed, metric_family, label_vector_field_mixed);   \
    DEFINE_PROMETHEUS_COUNTER(metric_family##_vector_index_mixed, metric_family, label_vector_index_mixed);   \
    DEFINE_PROMETHEUS_COUNTER(metric_family##_scalar_field_mixed, metric_family, label_scalar_field_mixed);   \
    DEFINE_PROMETHEUS_COUNTER(metric_family##_scalar_index_mixed, metric_family, label_scalar_index_mixed);   \
    DEFINE_PROMETHEUS_COUNTER(metric_family##_other_mixed, metric_family, label_other_mixed);

#define DEFINE_PROMETHEUS_HISTOGRAM_METRIC_WITH_DATA_TYPE_AND_LOCATION(metric_family, buckets, desc)                \
    DEFINE_PROMETHEUS_HISTOGRAM_FAMILY(metric_family, desc);                                                        \
    DEFINE_PROMETHEUS_HISTOGRAM_WITH_BUCKETS(metric_family##_vector_field_memory, metric_family,                    \
                                             label_vector_field_memory, buckets);                                   \
    DEFINE_PROMETHEUS_HISTOGRAM_WITH_BUCKETS(metric_family##_vector_index_memory, metric_family,                    \
                                             label_vector_index_memory, buckets);                                   \
    DEFINE_PROMETHEUS_HISTOGRAM_WITH_BUCKETS(metric_family##_scalar_field_memory, metric_family,                    \
                                             label_scalar_field_memory, buckets);                                   \
    DEFINE_PROMETHEUS_HISTOGRAM_WITH_BUCKETS(metric_family##_scalar_index_memory, metric_family,                    \
                                             label_scalar_index_memory, buckets);                                   \
    DEFINE_PROMETHEUS_HISTOGRAM_WITH_BUCKETS(metric_family##_other_memory, metric_family, label_other_memory,       \
                                             buckets);                                                              \
    DEFINE_PROMETHEUS_HISTOGRAM_WITH_BUCKETS(metric_family##_vector_field_disk, metric_family,                      \
                                             label_vector_field_disk, buckets);                                     \
    DEFINE_PROMETHEUS_HISTOGRAM_WITH_BUCKETS(metric_family##_vector_index_disk, metric_family,                      \
                                             label_vector_index_disk, buckets);                                     \
    DEFINE_PROMETHEUS_HISTOGRAM_WITH_BUCKETS(metric_family##_scalar_field_disk, metric_family,                      \
                                             label_scalar_field_disk, buckets);                                     \
    DEFINE_PROMETHEUS_HISTOGRAM_WITH_BUCKETS(metric_family##_scalar_index_disk, metric_family,                      \
                                             label_scalar_index_disk, buckets);                                     \
    DEFINE_PROMETHEUS_HISTOGRAM_WITH_BUCKETS(metric_family##_other_disk, metric_family, label_other_disk, buckets); \
    DEFINE_PROMETHEUS_HISTOGRAM_WITH_BUCKETS(metric_family##_vector_field_mixed, metric_family,                     \
                                             label_vector_field_mixed, buckets);                                    \
    DEFINE_PROMETHEUS_HISTOGRAM_WITH_BUCKETS(metric_family##_vector_index_mixed, metric_family,                     \
                                             label_vector_index_mixed, buckets);                                    \
    DEFINE_PROMETHEUS_HISTOGRAM_WITH_BUCKETS(metric_family##_scalar_field_mixed, metric_family,                     \
                                             label_scalar_field_mixed, buckets);                                    \
    DEFINE_PROMETHEUS_HISTOGRAM_WITH_BUCKETS(metric_family##_scalar_index_mixed, metric_family,                     \
                                             label_scalar_index_mixed, buckets);                                    \
    DEFINE_PROMETHEUS_HISTOGRAM_WITH_BUCKETS(metric_family##_other_mixed, metric_family, label_other_mixed, buckets);

#define DEFINE_PROMETHEUS_GAUGE_METRIC_WITH_LOCATION(metric_family, desc)         \
    DEFINE_PROMETHEUS_GAUGE_FAMILY(metric_family, desc);                          \
    DEFINE_PROMETHEUS_GAUGE(metric_family##_memory, metric_family, label_memory); \
    DEFINE_PROMETHEUS_GAUGE(metric_family##_disk, metric_family, label_disk);     \
    DEFINE_PROMETHEUS_GAUGE(metric_family##_mixed, metric_family, label_mixed);

#define DEFINE_PROMETHEUS_COUNTER_METRIC_WITH_LOCATION(metric_family, desc)         \
    DEFINE_PROMETHEUS_COUNTER_FAMILY(metric_family, desc);                          \
    DEFINE_PROMETHEUS_COUNTER(metric_family##_memory, metric_family, label_memory); \
    DEFINE_PROMETHEUS_COUNTER(metric_family##_disk, metric_family, label_disk);     \
    DEFINE_PROMETHEUS_COUNTER(metric_family##_mixed, metric_family, label_mixed);

/* Caching Layer Metrics Access Helpers */
#define DEFINE_METRIC_HELPER_WITH_DATA_TYPE_AND_LOCATION(metric_type, metric_name)     \
    static inline metric_type& metric_name(CellDataType t, StorageType loc) {          \
        switch (loc) {                                                                 \
            case StorageType::MEMORY:                                                  \
                switch (t) {                                                           \
                    case CellDataType::VECTOR_FIELD:                                   \
                        return internal_##metric_name##_vector_field_memory;           \
                    case CellDataType::VECTOR_INDEX:                                   \
                        return internal_##metric_name##_vector_index_memory;           \
                    case CellDataType::SCALAR_FIELD:                                   \
                        return internal_##metric_name##_scalar_field_memory;           \
                    case CellDataType::SCALAR_INDEX:                                   \
                        return internal_##metric_name##_scalar_index_memory;           \
                    case CellDataType::OTHER:                                          \
                        return internal_##metric_name##_other_memory;                  \
                    default:                                                           \
                        ThrowInfo(ErrorCode::UnexpectedError, "Unknown CellDataType"); \
                }                                                                      \
                break;                                                                 \
            case StorageType::DISK:                                                    \
                switch (t) {                                                           \
                    case CellDataType::VECTOR_FIELD:                                   \
                        return internal_##metric_name##_vector_field_disk;             \
                    case CellDataType::VECTOR_INDEX:                                   \
                        return internal_##metric_name##_vector_index_disk;             \
                    case CellDataType::SCALAR_FIELD:                                   \
                        return internal_##metric_name##_scalar_field_disk;             \
                    case CellDataType::SCALAR_INDEX:                                   \
                        return internal_##metric_name##_scalar_index_disk;             \
                    case CellDataType::OTHER:                                          \
                        return internal_##metric_name##_other_disk;                    \
                    default:                                                           \
                        ThrowInfo(ErrorCode::UnexpectedError, "Unknown CellDataType"); \
                }                                                                      \
                break;                                                                 \
            case StorageType::MIXED:                                                   \
                switch (t) {                                                           \
                    case CellDataType::VECTOR_FIELD:                                   \
                        return internal_##metric_name##_vector_field_mixed;            \
                    case CellDataType::VECTOR_INDEX:                                   \
                        return internal_##metric_name##_vector_index_mixed;            \
                    case CellDataType::SCALAR_FIELD:                                   \
                        return internal_##metric_name##_scalar_field_mixed;            \
                    case CellDataType::SCALAR_INDEX:                                   \
                        return internal_##metric_name##_scalar_index_mixed;            \
                    case CellDataType::OTHER:                                          \
                        return internal_##metric_name##_other_mixed;                   \
                    default:                                                           \
                        ThrowInfo(ErrorCode::UnexpectedError, "Unknown CellDataType"); \
                }                                                                      \
                break;                                                                 \
            default:                                                                   \
                ThrowInfo(ErrorCode::UnexpectedError, "Unknown StorageType");          \
        }                                                                              \
    }

#define DEFINE_METRIC_HELPER_WITH_LOCATION(metric_type, metric_name)          \
    static inline metric_type& metric_name(StorageType loc) {                 \
        switch (loc) {                                                        \
            case StorageType::MEMORY:                                         \
                return internal_##metric_name##_memory;                       \
            case StorageType::DISK:                                           \
                return internal_##metric_name##_disk;                         \
            case StorageType::MIXED:                                          \
                return internal_##metric_name##_mixed;                        \
            default:                                                          \
                ThrowInfo(ErrorCode::UnexpectedError, "Unknown StorageType"); \
        }                                                                     \
    }

/* Metrics for Cache Resource Usage */
DECLARE_PROMETHEUS_GAUGE_METRIC_WITH_LOCATION(internal_cache_capacity_bytes);
DECLARE_PROMETHEUS_GAUGE_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_slot_count);
DECLARE_PROMETHEUS_GAUGE_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_cell_count);
DECLARE_PROMETHEUS_GAUGE_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_loaded_bytes);
DECLARE_PROMETHEUS_GAUGE_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_loading_bytes);
DECLARE_PROMETHEUS_GAUGE_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_cell_loading_count);
DECLARE_PROMETHEUS_GAUGE_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_cell_loaded_count);

/* Metrics for Cache Cell Access */
DECLARE_PROMETHEUS_COUNTER_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_access_event_total);
DECLARE_PROMETHEUS_COUNTER_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_cell_access_hit_total);
DECLARE_PROMETHEUS_COUNTER_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_cell_access_miss_total);

/* Metrics for Cache Cell Loading */
DECLARE_PROMETHEUS_COUNTER_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_load_event_success_total);
DECLARE_PROMETHEUS_COUNTER_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_load_event_fail_total);
DECLARE_PROMETHEUS_HISTOGRAM_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_load_latency_microseconds);
DECLARE_PROMETHEUS_COUNTER_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_cell_load_success_total);
DECLARE_PROMETHEUS_COUNTER_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_cell_load_fail_total);

/* Metrics for Cache Cell Eviction */
DECLARE_PROMETHEUS_COUNTER_METRIC_WITH_LOCATION(internal_cache_eviction_event_total);
DECLARE_PROMETHEUS_COUNTER_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_cell_evicted_total)
DECLARE_PROMETHEUS_COUNTER_METRIC_WITH_LOCATION(internal_cache_evicted_bytes_total);
DECLARE_PROMETHEUS_HISTOGRAM_METRIC_WITH_DATA_TYPE_AND_LOCATION(internal_cache_cell_lifetime_seconds);

DEFINE_METRIC_HELPER_WITH_LOCATION(prometheus::Gauge, cache_capacity_bytes);
DEFINE_METRIC_HELPER_WITH_DATA_TYPE_AND_LOCATION(prometheus::Gauge, cache_slot_count);
DEFINE_METRIC_HELPER_WITH_DATA_TYPE_AND_LOCATION(prometheus::Gauge, cache_cell_count);
DEFINE_METRIC_HELPER_WITH_DATA_TYPE_AND_LOCATION(prometheus::Gauge, cache_loading_bytes);
DEFINE_METRIC_HELPER_WITH_DATA_TYPE_AND_LOCATION(prometheus::Gauge, cache_cell_loading_count);
DEFINE_METRIC_HELPER_WITH_DATA_TYPE_AND_LOCATION(prometheus::Gauge, cache_loaded_bytes);
DEFINE_METRIC_HELPER_WITH_DATA_TYPE_AND_LOCATION(prometheus::Gauge, cache_cell_loaded_count);

DEFINE_METRIC_HELPER_WITH_DATA_TYPE_AND_LOCATION(prometheus::Counter, cache_access_event_total);
DEFINE_METRIC_HELPER_WITH_DATA_TYPE_AND_LOCATION(prometheus::Counter, cache_cell_access_hit_total);
DEFINE_METRIC_HELPER_WITH_DATA_TYPE_AND_LOCATION(prometheus::Counter, cache_cell_access_miss_total);

DEFINE_METRIC_HELPER_WITH_DATA_TYPE_AND_LOCATION(prometheus::Counter, cache_cell_load_success_total);
DEFINE_METRIC_HELPER_WITH_DATA_TYPE_AND_LOCATION(prometheus::Counter, cache_cell_load_fail_total);
DEFINE_METRIC_HELPER_WITH_DATA_TYPE_AND_LOCATION(prometheus::Counter, cache_load_event_success_total);
DEFINE_METRIC_HELPER_WITH_DATA_TYPE_AND_LOCATION(prometheus::Counter, cache_load_event_fail_total);
DEFINE_METRIC_HELPER_WITH_DATA_TYPE_AND_LOCATION(prometheus::Histogram, cache_load_latency_microseconds);

DEFINE_METRIC_HELPER_WITH_LOCATION(prometheus::Counter, cache_eviction_event_total);
DEFINE_METRIC_HELPER_WITH_DATA_TYPE_AND_LOCATION(prometheus::Counter, cache_cell_evicted_total);
DEFINE_METRIC_HELPER_WITH_LOCATION(prometheus::Counter, cache_evicted_bytes_total);
DEFINE_METRIC_HELPER_WITH_DATA_TYPE_AND_LOCATION(prometheus::Histogram, cache_cell_lifetime_seconds);

}  // namespace milvus::cachinglayer::monitor
