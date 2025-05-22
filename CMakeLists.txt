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

set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED on )

# 添加 PIC 编译选项
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )

set(CMAKE_BUILD_TYPE Release)

set(MILVUS_COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${MILVUS_COMMON_INCLUDE_DIRS})

list( APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR}/conan )
include( ${CMAKE_BINARY_DIR}/conan/conanbuildinfo.cmake )
set( CONAN_DISABLE_CHECK_COMPILER ON )
conan_basic_setup( KEEP_RPATHS )

include( Utils )

add_subdirectory(common)
add_subdirectory(cachinglayer)
add_subdirectory(log)

add_library(milvus_base SHARED
    $<TARGET_OBJECTS:milvus_internal_common>
    $<TARGET_OBJECTS:milvus_cachinglayer>
    $<TARGET_OBJECTS:milvus_log>
)

if ( BUILD_BASE_TEST STREQUAL "ON" )
    add_subdirectory(test)
endif()

list(APPEND MILVUS_BASE_LINKER_LIBS ${CONAN_LIBS})

target_link_libraries(milvus_base PRIVATE ${MILVUS_BASE_LINKER_LIBS})

install(TARGETS milvus_base
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
