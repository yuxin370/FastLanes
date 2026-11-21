add_executable(test_fsst test_fsst.cpp)
target_link_libraries(test_fsst PUBLIC gtest_main gmock_main FastLanes)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test_fsst-results")
gtest_discover_tests(test_fsst WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test_fsst-results")

add_executable(test_fls_fsst test_fls_fsst.cpp)
target_link_libraries(test_fls_fsst PUBLIC gtest_main gmock_main FastLanes)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test_fls_fsst-results")
gtest_discover_tests(test_fls_fsst WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test_fls_fsst-results")
