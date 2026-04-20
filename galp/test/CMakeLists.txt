add_executable(galp_tests
        thrust_memcpy_test.cu
        thrust_core_test.cu
        galp_test.cu
        reader_test.cu
)


target_link_libraries(galp_tests PRIVATE
        galp_core
        GTest::gtest_main
        FastLanes::core
)

target_compile_definitions(galp_tests PRIVATE FLS_GALP_SOURCE_DIR="${GALP_DATA_ROOT}")

apply_warnings(galp_tests)

gtest_discover_tests(galp_tests
        DISCOVERY_TIMEOUT 60
)