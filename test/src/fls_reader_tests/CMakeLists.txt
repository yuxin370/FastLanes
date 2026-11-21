# ────────────────────────────────────────────────────────
# |                      FastLanes                       |
# ────────────────────────────────────────────────────────
# test/src/fls_reader_tests/CMakeLists.txt
# ────────────────────────────────────────────────────────
add_executable(
        fls_reader_tests
        #[[]]
        file_range_test.cpp
        incomplete_vectors_test.cpp
        inlined_footer_test.cpp
        rowgroup_size_test.cpp
        verify_fastlanes_files_test.cpp)

target_link_libraries(fls_reader_tests PUBLIC gtest_main gmock_main FastLanes msvc_heap_guard)
fls_msvc_test_setup(fls_reader_tests)
gtest_discover_tests(fls_reader_tests DISCOVERY_TIMEOUT 60)

fls_enable_sanitizers(fls_reader_tests)


