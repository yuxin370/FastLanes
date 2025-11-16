# C++ Example : --------------------------------------------------------------------------------------------------------
add_executable(cpp_example ${CMAKE_CURRENT_SOURCE_DIR}/cpp_example.cpp)
target_link_libraries(cpp_example PUBLIC FastLanes)

add_executable(jpeg_loader_example ${CMAKE_CURRENT_SOURCE_DIR}/jpeg_loader_example.cpp)
target_link_libraries(jpeg_loader_example PUBLIC FastLanes)

add_executable(jpeg_loader_example_simple ${CMAKE_CURRENT_SOURCE_DIR}/jpeg_loader_example_simple.cpp)
target_link_libraries(jpeg_loader_example_simple PUBLIC FastLanes)

add_executable(jpeg_decoding_test ${CMAKE_CURRENT_SOURCE_DIR}/jpeg_decoding_test.cpp)
target_link_libraries(jpeg_decoding_test PUBLIC FastLanes)

add_executable(jpeg_decoding_dir_test ${CMAKE_CURRENT_SOURCE_DIR}/jpeg_decoding_dir_test.cpp)
target_link_libraries(jpeg_decoding_dir_test PUBLIC FastLanes)

add_executable(jpeg_gpu_load_rgb_test ${CMAKE_CURRENT_SOURCE_DIR}/jpeg_gpu_load_rgb_test.cpp)
target_link_libraries(jpeg_gpu_load_rgb_test PUBLIC FastLanes)


if (WIN32 AND FLS_BUILD_SHARED_LIBS)
    add_custom_command(
            TARGET cpp_example
            POST_BUILD
            COMMAND ${CMAKE_COMMAND}
            -E copy_if_different
            $<TARGET_RUNTIME_DLLS:cpp_example>
            $<TARGET_FILE_DIR:cpp_example>
    )
endif ()
