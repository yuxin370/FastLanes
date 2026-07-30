add_executable(galp_jpeg_dct_tool
        jpeg_dct_tool.cpp
)

add_executable(galp_jpeg_dct_policy_bench
        jpeg_dct_policy_bench.cpp
)

add_executable(galp_jpeg_dct_rowgroup_read_bench
        jpeg_dct_rowgroup_read_bench.cu
)

target_link_libraries(galp_jpeg_dct_tool PRIVATE
        galp_core
)

target_link_libraries(galp_jpeg_dct_policy_bench PRIVATE
        galp_core
)

target_link_libraries(galp_jpeg_dct_rowgroup_read_bench PRIVATE
        galp_core
)

target_include_directories(galp_jpeg_dct_tool PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

target_include_directories(galp_jpeg_dct_policy_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

target_include_directories(galp_jpeg_dct_rowgroup_read_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

apply_warnings(galp_jpeg_dct_tool)
apply_warnings(galp_jpeg_dct_policy_bench)
apply_warnings(galp_jpeg_dct_rowgroup_read_bench)
