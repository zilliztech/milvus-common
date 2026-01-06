# NCS tests

find_package(GTest CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

# Unified NCS test for all implementations (InMemory, Redis)
set(NCS_ALL_TEST_FILES
    ../init_gtest.cpp
    test_ncs_all.cpp
)

add_executable(ncs_all_test
    ${NCS_ALL_TEST_FILES}
)

# Ensure SSE4.2 CRC intrinsics are available to match Folly build (x86_64)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND NOT MSVC)
    target_compile_options(ncs_all_test PRIVATE -msse4.2)
endif()

target_link_libraries(ncs_all_test
    GTest::gtest
    GTest::gmock
    milvus-common
    ${COMMON_LINKER_LIBS}
    pthread
    atomic
    gomp
    openblas
)

install(TARGETS ncs_all_test DESTINATION unittest)
