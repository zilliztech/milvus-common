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


set(CACHINGLAYER_TEST_FILES
    ../init_gtest.cpp
    test_dlist.cpp
    test_cache_slot.cpp
)

add_executable(cachinglayer_test
    ${CACHINGLAYER_TEST_FILES}
)

target_link_libraries(cachinglayer_test
    gtest 
    gmock
    milvus-common
    pthread
    atomic
    ${CONAN_LIBS}
)

install(TARGETS cachinglayer_test DESTINATION unittest)
