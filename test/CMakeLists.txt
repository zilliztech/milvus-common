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

add_subdirectory(test_cachinglayer)

find_package(GTest CONFIG REQUIRED)
find_package(BLAS REQUIRED)

set(ALL_TEST_FILES
    init_gtest.cpp
    TracerTest.cpp
    StreamTest.cpp
)

add_executable(all_tests
    ${ALL_TEST_FILES}
)


target_link_libraries(all_tests
    GTest::gtest
    GTest::gmock
    milvus-common
    ${COMMON_LINKER_LIBS}
    Threads::Threads
    ${BLAS_LIBRARIES}
)
if(NOT APPLE)
    target_link_libraries(all_tests atomic gomp)
endif()

add_test(NAME all_tests COMMAND all_tests)

install(TARGETS all_tests DESTINATION unittest)
