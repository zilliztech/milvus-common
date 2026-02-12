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
option(ENABLE_SYNCPOINT "Enable sync point for testing" OFF)

set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED on )

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

set(MILVUS_COMMON_WORKSPACE ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${MILVUS_COMMON_WORKSPACE}/include)

set(CMAKE_CXX_FLAGS "-Wall -fPIC ${CMAKE_CXX_FLAGS}")

if (WITH_ASAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -fno-stack-protector -fno-var-tracking")
endif()


find_package(opentelemetry-cpp REQUIRED)
find_package(folly REQUIRED)
set(GFLAGS_USE_TARGET_NAMESPACE TRUE)
find_package(gflags REQUIRED)
find_package(glog REQUIRED)
find_package(fmt REQUIRED)
find_package(prometheus-cpp REQUIRED)

# OpenMP is required for thread pool OMP setter
if(APPLE)
    # Apple Clang needs Homebrew libomp: brew install libomp
    find_program(_BREW_CMD brew)
    if(_BREW_CMD)
        execute_process(COMMAND ${_BREW_CMD} --prefix libomp OUTPUT_VARIABLE LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
    if(LIBOMP_PREFIX)
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
endif()
find_package(OpenMP REQUIRED)

if(APPLE)
    # On macOS, transitive find_package calls (CURL, ZLIB, etc.) resolve to the
    # SDK sysroot (e.g. .../MacOSX.sdk/usr/include). When CMake adds that path
    # as -I or -isystem, it places C standard library headers ahead of libc++'s
    # C++ wrapper headers (cinttypes, cstdio, etc.), breaking compilation.
    # Strip SDK sysroot include dirs from all imported targets since the compiler
    # already searches them by default.
    execute_process(COMMAND xcrun --show-sdk-path OUTPUT_VARIABLE _SDK_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(_SDK_INCLUDE "${_SDK_PATH}/usr/include")
    get_property(_all_targets DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY IMPORTED_TARGETS)
    foreach(_tgt IN LISTS _all_targets)
        foreach(_prop INTERFACE_INCLUDE_DIRECTORIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
            get_target_property(_dirs ${_tgt} ${_prop})
            if(_dirs)
                list(REMOVE_ITEM _dirs "${_SDK_INCLUDE}")
                # Also handle versioned SDK paths (e.g. MacOSX15.sdk)
                foreach(_d IN LISTS _dirs)
                    if(_d MATCHES "/SDKs/MacOSX[0-9]*\\.sdk/usr/include$")
                        list(REMOVE_ITEM _dirs "${_d}")
                    endif()
                endforeach()
                set_target_properties(${_tgt} PROPERTIES ${_prop} "${_dirs}")
            endif()
        endforeach()
    endforeach()
endif()

list(APPEND COMMON_LINKER_LIBS glog::glog)
list(APPEND COMMON_LINKER_LIBS prometheus-cpp::core prometheus-cpp::push)
list(APPEND COMMON_LINKER_LIBS fmt::fmt)
list(APPEND COMMON_LINKER_LIBS Folly::folly)
list(APPEND COMMON_LINKER_LIBS gflags::gflags)

list(APPEND COMMON_LINKER_LIBS opentelemetry-cpp::opentelemetry_trace)
list(APPEND COMMON_LINKER_LIBS opentelemetry-cpp::opentelemetry_exporter_ostream_span)
list(APPEND COMMON_LINKER_LIBS opentelemetry-cpp::opentelemetry_exporter_otlp_http)
if(TARGET opentelemetry-cpp::opentelemetry_exporter_otlp_grpc)
    list(APPEND COMMON_LINKER_LIBS opentelemetry-cpp::opentelemetry_exporter_otlp_grpc)
    set(HAVE_OTLP_GRPC_EXPORTER ON)
endif()

file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cc)

# Check for libaio library and conditionally include aio_context_pool.cc
find_library(LIBAIO_LIBRARY NAMES aio)
if(NOT LIBAIO_LIBRARY)
    list(REMOVE_ITEM SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/knowhere/aio_context_pool.cc)
    message(STATUS "libaio not found, excluding aio_context_pool.cc from build")
else()
    message(STATUS "libaio found: ${LIBAIO_LIBRARY}, including aio_context_pool.cc in build")
    list(APPEND COMMON_LINKER_LIBS ${LIBAIO_LIBRARY})
endif()

add_library(milvus-common SHARED ${SRC_FILES})

# Use std::shared_ptr etc. instead of opentelemetry::nostd:: equivalents
target_compile_definitions(milvus-common PUBLIC OPENTELEMETRY_STL_VERSION=2017)

if(APPLE)
    target_compile_definitions(milvus-common PUBLIC BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED)
endif()

if(HAVE_OTLP_GRPC_EXPORTER)
    target_compile_definitions(milvus-common PUBLIC HAVE_OTLP_GRPC_EXPORTER)
endif()

if (ENABLE_SYNCPOINT)
    add_definitions(-DENABLE_SYNCPOINT)
endif()

target_link_libraries(milvus-common PUBLIC
    ${COMMON_LINKER_LIBS}
)

target_link_libraries(milvus-common PUBLIC OpenMP::OpenMP_CXX)

if(WITH_COMMON_UT)
    # Enable unit-test-only helpers for caching layer
    target_compile_definitions(milvus-common PRIVATE MCL_ENABLE_TEST_CGROUP_OVERRIDE)

    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(test)
endif()

install(TARGETS milvus-common
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY "${MILVUS_COMMON_WORKSPACE}/include/"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include")
