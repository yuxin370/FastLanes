add_executable(galp_jpeg_dct_tool
        jpeg_dct_tool.cpp
)

add_executable(galp_block_major_access_tool
        block_major_access_tool.cpp
)

add_executable(galp_block_major_plan_audit
        block_major_plan_audit.cpp
)

add_executable(galp_jpeg_dct_policy_bench
        jpeg_dct_policy_bench.cpp
)

add_executable(galp_compact_v3_runtime_audit
        compact_v3_runtime_audit.cpp
)
set_source_files_properties(compact_v3_runtime_audit.cpp PROPERTIES LANGUAGE CUDA)

add_executable(galp_jpeg_dct_rowgroup_read_bench
        jpeg_dct_rowgroup_read_bench.cu
)

target_link_libraries(galp_jpeg_dct_tool PRIVATE
        galp_core
		galp_compact_format
)

target_link_libraries(galp_block_major_access_tool PRIVATE
        galp_jpeg_dct_storage
)

target_link_libraries(galp_block_major_plan_audit PRIVATE
        galp_core
)

target_link_libraries(galp_jpeg_dct_policy_bench PRIVATE
        galp_core
)

target_link_libraries(galp_compact_v3_runtime_audit PRIVATE
        galp_core
)

target_link_libraries(galp_jpeg_dct_rowgroup_read_bench PRIVATE
        galp_core
)

target_include_directories(galp_jpeg_dct_tool PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

target_include_directories(galp_jpeg_dct_policy_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

target_include_directories(galp_compact_v3_runtime_audit PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

target_include_directories(galp_jpeg_dct_rowgroup_read_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

apply_warnings(galp_jpeg_dct_tool)
apply_warnings(galp_block_major_access_tool)
apply_warnings(galp_block_major_plan_audit)
apply_warnings(galp_jpeg_dct_policy_bench)
apply_warnings(galp_compact_v3_runtime_audit)
apply_warnings(galp_jpeg_dct_rowgroup_read_bench)
