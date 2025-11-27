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
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

set(CACHINGLAYER_TEST_FILES
    ../init_gtest.cpp
    test_dlist.cpp
    test_cache_slot.cpp
    test_utils.cpp
)

add_executable(cachinglayer_test
    ${CACHINGLAYER_TEST_FILES}
)

# Ensure SSE4.2 CRC intrinsics are available to match Folly build (x86_64)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND NOT MSVC)
    target_compile_options(cachinglayer_test PRIVATE -msse4.2)
endif()

target_link_libraries(cachinglayer_test
    GTest::gtest
    GTest::gmock
    milvus-common
    ${COMMON_LINKER_LIBS}
    pthread
    atomic
    gomp
    openblas
)

install(TARGETS cachinglayer_test DESTINATION unittest)
