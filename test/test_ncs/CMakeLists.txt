# NCS tests

find_package(GTest CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

set(IN_MEM_NCS_TEST_FILES
    ../init_gtest.cpp
    test_inmem_ncs.cpp
)

add_executable(inmem_ncs_test
    ${IN_MEM_NCS_TEST_FILES}
)

set(REDIS_NCS_TEST_FILES
    ../init_gtest.cpp
    test_redis_ncs.cpp
)

add_executable(redis_ncs_test
    ${REDIS_NCS_TEST_FILES}
)

# Ensure SSE4.2 CRC intrinsics are available to match Folly build (x86_64)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND NOT MSVC)
    target_compile_options(inmem_ncs_test PRIVATE -msse4.2)
    target_compile_options(redis_ncs_test PRIVATE -msse4.2)
endif()

target_link_libraries(inmem_ncs_test
    GTest::gtest
    GTest::gmock
    milvus-common
    ${COMMON_LINKER_LIBS}
    pthread
    atomic
    gomp
    openblas
)

target_link_libraries(redis_ncs_test
    GTest::gtest
    GTest::gmock
    milvus-common
    ${COMMON_LINKER_LIBS}
    pthread
    atomic
    gomp
    openblas
)


install(TARGETS inmem_ncs_test redis_ncs_test DESTINATION unittest)

