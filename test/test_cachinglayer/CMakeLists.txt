# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License

find_package(GTest CONFIG REQUIRED)
find_package(Threads REQUIRED)

set(CACHINGLAYER_TEST_FILES
    ../init_gtest.cpp
    test_dlist.cpp
    test_cache_slot.cpp
    test_cache_shard_metrics.cpp
    test_tiered_storage_config.cpp
    test_loading_overhead_policy.cpp
    test_loading_overhead_group.cpp
    test_utils.cpp
)

add_executable(cachinglayer_test
    ${CACHINGLAYER_TEST_FILES}
)

# The metrics mode is fixed for the process lifetime, so aggregate tests use a
# separate executable with explicit startup configuration.
add_executable(cachinglayer_aggregate_metrics_test
    aggregate_metrics_main.cpp
    test_cache_slot.cpp
    test_cache_shard_metrics.cpp
)

foreach(test_target cachinglayer_test cachinglayer_aggregate_metrics_test)
    target_link_libraries(${test_target} PRIVATE
        GTest::gtest
        GTest::gmock
        milvus-common
        ${COMMON_LINKER_LIBS}
        Threads::Threads
    )
    if(NOT APPLE)
        target_link_libraries(${test_target} PRIVATE atomic gomp)
    endif()
endforeach()

add_test(NAME cachinglayer_test COMMAND cachinglayer_test)
add_test(NAME cachinglayer_aggregate_metrics_test COMMAND cachinglayer_aggregate_metrics_test
    --gtest_filter=CacheShardDiskUsageMetricTest.*:CacheSlotShardDiskUsageTest.*)

install(TARGETS cachinglayer_test cachinglayer_aggregate_metrics_test DESTINATION unittest)
