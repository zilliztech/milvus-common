#include "common/common_monitor.h"

namespace milvus::monitor {

const prometheus::Histogram::BucketBoundaries secondsBuckets = {
    std::chrono::duration<float>(std::chrono::microseconds(10)).count(),
    std::chrono::duration<float>(std::chrono::microseconds(50)).count(),
    std::chrono::duration<float>(std::chrono::microseconds(100)).count(),
    std::chrono::duration<float>(std::chrono::microseconds(250)).count(),
    std::chrono::duration<float>(std::chrono::microseconds(500)).count(),
    std::chrono::duration<float>(std::chrono::milliseconds(1)).count(),
    std::chrono::duration<float>(std::chrono::milliseconds(5)).count(),
    std::chrono::duration<float>(std::chrono::milliseconds(10)).count(),
    std::chrono::duration<float>(std::chrono::milliseconds(20)).count(),
    std::chrono::duration<float>(std::chrono::milliseconds(50)).count(),
    std::chrono::duration<float>(std::chrono::milliseconds(100)).count(),
    std::chrono::duration<float>(std::chrono::milliseconds(200)).count(),
    std::chrono::duration<float>(std::chrono::milliseconds(500)).count(),
    std::chrono::duration<float>(std::chrono::seconds(1)).count(),
    std::chrono::duration<float>(std::chrono::seconds(2)).count(),
    std::chrono::duration<float>(std::chrono::seconds(5)).count(),
    std::chrono::duration<float>(std::chrono::seconds(10)).count(),
};

const prometheus::Histogram::BucketBoundaries buckets = {1,
                                                         2,
                                                         4,
                                                         8,
                                                         16,
                                                         32,
                                                         64,
                                                         128,
                                                         256,
                                                         512,
                                                         1024,
                                                         2048,
                                                         4096,
                                                         8192,
                                                         16384,
                                                         32768,
                                                         65536};

const prometheus::Histogram::BucketBoundaries bytesBuckets = {
    1024,         // 1k
    8192,         // 8k
    65536,        // 64k
    262144,       // 256k
    524288,       // 512k
    1048576,      // 1M
    4194304,      // 4M
    8388608,      // 8M
    16777216,     // 16M
    67108864,     // 64M
    134217728,    // 128M
    268435456,    // 256M
    536870912,    // 512M
    1073741824};  // 1G

const prometheus::Histogram::BucketBoundaries ratioBuckets = {
    0.0,  0.05, 0.1,  0.15, 0.2,  0.25, 0.3,  0.35, 0.4,  0.45, 0.5,
    0.55, 0.6,  0.65, 0.7,  0.75, 0.8,  0.85, 0.9,  0.95, 1.0};x

// --- caching layer metrics ---
// TODO(tiered storage 1): choose better buckets.

std::map<std::string, std::string> cacheMemoryLabel = {{"location", "memory"}};
std::map<std::string, std::string> cacheDiskLabel = {{"location", "disk"}};
std::map<std::string, std::string> cacheMixedLabel = {{"location", "mixed"}};

// Cache slot count
DEFINE_PROMETHEUS_GAUGE_FAMILY(internal_cache_slot_count,
                               "[cpp]cache slot count");
DEFINE_PROMETHEUS_GAUGE(internal_cache_slot_count_memory,
                        internal_cache_slot_count,
                        cacheMemoryLabel);
DEFINE_PROMETHEUS_GAUGE(internal_cache_slot_count_disk,
                        internal_cache_slot_count,
                        cacheDiskLabel);
DEFINE_PROMETHEUS_GAUGE(internal_cache_slot_count_mixed,
                        internal_cache_slot_count,
                        cacheMixedLabel);

// Cache cell count
DEFINE_PROMETHEUS_GAUGE_FAMILY(internal_cache_cell_count,
                               "[cpp]cache cell count");
DEFINE_PROMETHEUS_GAUGE(internal_cache_cell_count_memory,
                        internal_cache_cell_count,
                        cacheMemoryLabel);
DEFINE_PROMETHEUS_GAUGE(internal_cache_cell_count_disk,
                        internal_cache_cell_count,
                        cacheDiskLabel);
DEFINE_PROMETHEUS_GAUGE(internal_cache_cell_count_mixed,
                        internal_cache_cell_count,
                        cacheMixedLabel);

// Cache cell loaded count
DEFINE_PROMETHEUS_GAUGE_FAMILY(internal_cache_cell_loaded_count,
                               "[cpp]cache cell loaded count");
DEFINE_PROMETHEUS_GAUGE(internal_cache_cell_loaded_count_memory,
                        internal_cache_cell_loaded_count,
                        cacheMemoryLabel);
DEFINE_PROMETHEUS_GAUGE(internal_cache_cell_loaded_count_disk,
                        internal_cache_cell_loaded_count,
                        cacheDiskLabel);
DEFINE_PROMETHEUS_GAUGE(internal_cache_cell_loaded_count_mixed,
                        internal_cache_cell_loaded_count,
                        cacheMixedLabel);

// Cache load latency histogram
DEFINE_PROMETHEUS_HISTOGRAM_FAMILY(internal_cache_load_latency,
                                   "[cpp]cache load latency histogram");
DEFINE_PROMETHEUS_HISTOGRAM_WITH_BUCKETS(internal_cache_load_latency_memory,
                                         internal_cache_load_latency,
                                         cacheMemoryLabel,
                                         secondsBuckets);
DEFINE_PROMETHEUS_HISTOGRAM_WITH_BUCKETS(internal_cache_load_latency_disk,
                                         internal_cache_load_latency,
                                         cacheDiskLabel,
                                         secondsBuckets);
DEFINE_PROMETHEUS_HISTOGRAM_WITH_BUCKETS(internal_cache_load_latency_mixed,
                                         internal_cache_load_latency,
                                         cacheMixedLabel,
                                         secondsBuckets);

// Cache hit rate (represented by hit/miss counters)
std::map<std::string, std::string> cacheHitMemoryLabels = {
    {"result", "hit"}, {"location", "memory"}};
std::map<std::string, std::string> cacheHitDiskLabels = {{"result", "hit"},
                                                         {"location", "disk"}};
std::map<std::string, std::string> cacheHitMixedLabels = {
    {"result", "hit"}, {"location", "mixed"}};
std::map<std::string, std::string> cacheMissMemoryLabels = {
    {"result", "miss"}, {"location", "memory"}};
std::map<std::string, std::string> cacheMissDiskLabels = {{"result", "miss"},
                                                          {"location", "disk"}};
std::map<std::string, std::string> cacheMissMixedLabels = {
    {"result", "miss"}, {"location", "mixed"}};
DEFINE_PROMETHEUS_COUNTER_FAMILY(internal_cache_op_result_count,
                                 "[cpp]cache operation result count");
DEFINE_PROMETHEUS_COUNTER(internal_cache_op_result_count_hit_memory,
                          internal_cache_op_result_count,
                          cacheHitMemoryLabels);
DEFINE_PROMETHEUS_COUNTER(internal_cache_op_result_count_hit_disk,
                          internal_cache_op_result_count,
                          cacheHitDiskLabels);
DEFINE_PROMETHEUS_COUNTER(internal_cache_op_result_count_hit_mixed,
                          internal_cache_op_result_count,
                          cacheHitMixedLabels);
DEFINE_PROMETHEUS_COUNTER(internal_cache_op_result_count_miss_memory,
                          internal_cache_op_result_count,
                          cacheMissMemoryLabels);
DEFINE_PROMETHEUS_COUNTER(internal_cache_op_result_count_miss_disk,
                          internal_cache_op_result_count,
                          cacheMissDiskLabels);
DEFINE_PROMETHEUS_COUNTER(internal_cache_op_result_count_miss_mixed,
                          internal_cache_op_result_count,
                          cacheMissMixedLabels);

// Cache usage (bytes)
DEFINE_PROMETHEUS_GAUGE_FAMILY(internal_cache_used_bytes,
                               "[cpp]currently used bytes in cache");
DEFINE_PROMETHEUS_GAUGE(internal_cache_used_bytes_memory,
                        internal_cache_used_bytes,
                        cacheMemoryLabel);
DEFINE_PROMETHEUS_GAUGE(internal_cache_used_bytes_disk,
                        internal_cache_used_bytes,
                        cacheDiskLabel);
DEFINE_PROMETHEUS_GAUGE(internal_cache_used_bytes_mixed,
                        internal_cache_used_bytes,
                        cacheMixedLabel);

DEFINE_PROMETHEUS_GAUGE_FAMILY(internal_cache_capacity_bytes,
                               "[cpp]total capacity bytes of cache");
DEFINE_PROMETHEUS_GAUGE(internal_cache_capacity_bytes_memory,
                        internal_cache_capacity_bytes,
                        cacheMemoryLabel);
DEFINE_PROMETHEUS_GAUGE(internal_cache_capacity_bytes_disk,
                        internal_cache_capacity_bytes,
                        cacheDiskLabel);
DEFINE_PROMETHEUS_GAUGE(internal_cache_capacity_bytes_mixed,
                        internal_cache_capacity_bytes,
                        cacheMixedLabel);

// Eviction count and resource size
DEFINE_PROMETHEUS_COUNTER_FAMILY(internal_cache_eviction_count,
                                 "[cpp]cache eviction count");
DEFINE_PROMETHEUS_COUNTER(internal_cache_eviction_count_memory,
                          internal_cache_eviction_count,
                          cacheMemoryLabel);
DEFINE_PROMETHEUS_COUNTER(internal_cache_eviction_count_disk,
                          internal_cache_eviction_count,
                          cacheDiskLabel);
DEFINE_PROMETHEUS_COUNTER(internal_cache_eviction_count_mixed,
                          internal_cache_eviction_count,
                          cacheMixedLabel);

DEFINE_PROMETHEUS_COUNTER_FAMILY(internal_cache_evicted_bytes,
                                 "[cpp]total bytes evicted from cache");
DEFINE_PROMETHEUS_COUNTER(internal_cache_evicted_bytes_memory,
                          internal_cache_evicted_bytes,
                          cacheMemoryLabel);
DEFINE_PROMETHEUS_COUNTER(internal_cache_evicted_bytes_disk,
                          internal_cache_evicted_bytes,
                          cacheDiskLabel);
DEFINE_PROMETHEUS_COUNTER(internal_cache_evicted_bytes_mixed,
                          internal_cache_evicted_bytes,
                          cacheMixedLabel);

// Cache item lifetime histogram
DEFINE_PROMETHEUS_HISTOGRAM_FAMILY(internal_cache_item_lifetime_seconds,
                                   "[cpp]cache item lifetime histogram");
DEFINE_PROMETHEUS_HISTOGRAM_WITH_BUCKETS(
    internal_cache_item_lifetime_seconds_memory,
    internal_cache_item_lifetime_seconds,
    cacheMemoryLabel,
    secondsBuckets);
DEFINE_PROMETHEUS_HISTOGRAM_WITH_BUCKETS(
    internal_cache_item_lifetime_seconds_disk,
    internal_cache_item_lifetime_seconds,
    cacheDiskLabel,
    secondsBuckets);
DEFINE_PROMETHEUS_HISTOGRAM_WITH_BUCKETS(
    internal_cache_item_lifetime_seconds_mixed,
    internal_cache_item_lifetime_seconds,
    cacheMixedLabel,
    secondsBuckets);

// Load error rate (represented by success/fail counters)
std::map<std::string, std::string> cacheLoadSuccessMemoryLabels = {
    {"status", "success"}, {"location", "memory"}};
std::map<std::string, std::string> cacheLoadSuccessDiskLabels = {
    {"status", "success"}, {"location", "disk"}};
std::map<std::string, std::string> cacheLoadSuccessMixedLabels = {
    {"status", "success"}, {"location", "mixed"}};
std::map<std::string, std::string> cacheLoadFailMemoryLabels = {
    {"status", "fail"}, {"location", "memory"}};
std::map<std::string, std::string> cacheLoadFailDiskLabels = {
    {"status", "fail"}, {"location", "disk"}};
std::map<std::string, std::string> cacheLoadFailMixedLabels = {
    {"status", "fail"}, {"location", "mixed"}};
DEFINE_PROMETHEUS_COUNTER_FAMILY(internal_cache_load_count,
                                 "[cpp]cache load operation count");
DEFINE_PROMETHEUS_COUNTER(internal_cache_load_count_success_memory,
                          internal_cache_load_count,
                          cacheLoadSuccessMemoryLabels);
DEFINE_PROMETHEUS_COUNTER(internal_cache_load_count_success_disk,
                          internal_cache_load_count,
                          cacheLoadSuccessDiskLabels);
DEFINE_PROMETHEUS_COUNTER(internal_cache_load_count_success_mixed,
                          internal_cache_load_count,
                          cacheLoadSuccessMixedLabels);
DEFINE_PROMETHEUS_COUNTER(internal_cache_load_count_fail_memory,
                          internal_cache_load_count,
                          cacheLoadFailMemoryLabels);
DEFINE_PROMETHEUS_COUNTER(internal_cache_load_count_fail_disk,
                          internal_cache_load_count,
                          cacheLoadFailDiskLabels);
DEFINE_PROMETHEUS_COUNTER(internal_cache_load_count_fail_mixed,
                          internal_cache_load_count,
                          cacheLoadFailMixedLabels);

// Cache system memory overhead (bytes)
DEFINE_PROMETHEUS_GAUGE_FAMILY(internal_cache_memory_overhead_bytes,
                               "[cpp]cache system memory overhead in bytes");
DEFINE_PROMETHEUS_GAUGE(internal_cache_memory_overhead_bytes_memory,
                        internal_cache_memory_overhead_bytes,
                        cacheMemoryLabel);
DEFINE_PROMETHEUS_GAUGE(internal_cache_memory_overhead_bytes_disk,
                        internal_cache_memory_overhead_bytes,
                        cacheDiskLabel);
DEFINE_PROMETHEUS_GAUGE(internal_cache_memory_overhead_bytes_mixed,
                        internal_cache_memory_overhead_bytes,
                        cacheMixedLabel);

// --- caching layer metrics end ---
}
