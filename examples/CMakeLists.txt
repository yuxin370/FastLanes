# C++ Example : --------------------------------------------------------------------------------------------------------
add_executable(cpp_example ${CMAKE_CURRENT_SOURCE_DIR}/cpp_example.cpp)
target_link_libraries(cpp_example PUBLIC FastLanes)

add_executable(cpp_csv_to_fls ${CMAKE_CURRENT_SOURCE_DIR}/cpp_csv_to_fls.cpp)
target_link_libraries(cpp_csv_to_fls PUBLIC FastLanes)

add_executable(cpp_decompress_bench ${CMAKE_CURRENT_SOURCE_DIR}/cpp_decompress_bench.cpp)
target_link_libraries(cpp_decompress_bench PUBLIC FastLanes)

if (WIN32 AND FLS_BUILD_SHARED_LIBS)
    add_custom_command(
            TARGET cpp_example
            POST_BUILD
            COMMAND ${CMAKE_COMMAND}
            -E copy_if_different
            $<TARGET_RUNTIME_DLLS:cpp_example>
            $<TARGET_FILE_DIR:cpp_example>
    )
    add_custom_command(
            TARGET cpp_csv_to_fls
            POST_BUILD
            COMMAND ${CMAKE_COMMAND}
            -E copy_if_different
            $<TARGET_RUNTIME_DLLS:cpp_csv_to_fls>
            $<TARGET_FILE_DIR:cpp_csv_to_fls>
    )
    add_custom_command(
            TARGET cpp_decompress_bench
            POST_BUILD
            COMMAND ${CMAKE_COMMAND}
            -E copy_if_different
            $<TARGET_RUNTIME_DLLS:cpp_decompress_bench>
            $<TARGET_FILE_DIR:cpp_decompress_bench>
    )
endif ()
