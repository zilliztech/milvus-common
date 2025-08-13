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

cmake_minimum_required( VERSION 3.18 )
project(milvus_common CXX C)

option(ENABLE_UNIT_TESTS "Enable unit tests" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(MILVUS_COMMON_WORKSPACE ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${MILVUS_COMMON_WORKSPACE}/include)

find_package(opentelemetry-cpp REQUIRED)
find_package(folly REQUIRED)
find_package(gflags REQUIRED)
find_package(glog REQUIRED)
find_package(fmt REQUIRED)
find_package(prometheus-cpp REQUIRED)

list(APPEND COMMON_LINKER_LIBS glog::glog)
list(APPEND COMMON_LINKER_LIBS prometheus-cpp::core prometheus-cpp::push)
list(APPEND COMMON_LINKER_LIBS fmt::fmt-header-only)
list(APPEND COMMON_LINKER_LIBS Folly::folly)
list(APPEND COMMON_LINKER_LIBS gflags::gflags)

list(APPEND COMMON_LINKER_LIBS opentelemetry-cpp::opentelemetry_trace)
list(APPEND COMMON_LINKER_LIBS opentelemetry-cpp::opentelemetry_exporter_ostream_span)
list(APPEND COMMON_LINKER_LIBS opentelemetry-cpp::opentelemetry_exporter_otlp_grpc)
list(APPEND COMMON_LINKER_LIBS opentelemetry-cpp::opentelemetry_exporter_otlp_http)

file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cc)
if (NOT WITH_AIO)
   list(REMOVE_ITEM src/knowhere/aio_context_pool.cc)
endif()
add_library(milvus-common SHARED ${SRC_FILES})

# Ensure SSE4.2 CRC intrinsics are available to match Folly build (x86_64)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND NOT MSVC)
    target_compile_options(milvus-common PUBLIC -msse4.2)
endif()

if (WITH_JAEGER)
    # list(APPEND COMMON_LINKER_LIBS opentelemetry-cpp::opentelemetry_exporter_jaeger_trace)
    # above commented out is expected as a workaround, we will enable jaeger only when combined with Milvus
    # use static linking since macos does not support opentelemetry shared library
    # do not link jaeger here to avoid double static linking, which will cause static object odr
    # and for macoos, we need to make the compiler happy to ignore the undefined symbol
    if (APPLE)
        target_link_options(milvus-common PRIVATE "-undefined" "dynamic_lookup")
    endif()
    add_definitions(-DWITH_JAEGER)
endif()

target_link_libraries(milvus-common PUBLIC 
    ${COMMON_LINKER_LIBS}
)

if(ENABLE_UNIT_TESTS)
    add_subdirectory(test)
endif()

install(TARGETS milvus-common
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY "${MILVUS_COMMON_WORKSPACE}/include/"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include")
