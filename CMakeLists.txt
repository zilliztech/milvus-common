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
add_subdirectory(monitor)
add_subdirectory(cachinglayer)
add_subdirectory(log)

add_library(milvus_common SHARED
    $<TARGET_OBJECTS:milvus_internal_common>
    $<TARGET_OBJECTS:milvus_monitor>
    $<TARGET_OBJECTS:milvus_cachinglayer>
    $<TARGET_OBJECTS:milvus_log>
)

if ( BUILD_UNIT_TEST STREQUAL "ON" )
    add_subdirectory(test)
endif()

list(APPEND MILVUS_COMMON_LINKER_LIBS ${CONAN_LIBS})

target_link_libraries(milvus_common PRIVATE ${MILVUS_COMMON_LINKER_LIBS})
