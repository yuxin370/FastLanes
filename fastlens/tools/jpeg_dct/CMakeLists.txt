add_executable(fastlens_jpeg_dct_tool
        jpeg_dct_tool.cpp
)

add_executable(fastlens_block_major_access_tool
        block_major_access_tool.cpp
)

add_executable(fastlens_block_major_plan_audit
        block_major_plan_audit.cpp
)

add_executable(fastlens_jpeg_dct_policy_bench
        jpeg_dct_policy_bench.cpp
)

add_executable(fastlens_compact_v3_runtime_audit
        compact_v3_runtime_audit.cpp
)
set_source_files_properties(compact_v3_runtime_audit.cpp PROPERTIES LANGUAGE CUDA)

add_executable(fastlens_jpeg_dct_rowgroup_read_bench
        jpeg_dct_rowgroup_read_bench.cu
)

target_link_libraries(fastlens_jpeg_dct_tool PRIVATE
        fastlens_core
		fastlens_compact_format
)

target_link_libraries(fastlens_block_major_access_tool PRIVATE
        fastlens_jpeg_dct_storage
)

target_link_libraries(fastlens_block_major_plan_audit PRIVATE
        fastlens_core
)

target_link_libraries(fastlens_jpeg_dct_policy_bench PRIVATE
        fastlens_core
)

target_link_libraries(fastlens_compact_v3_runtime_audit PRIVATE
        fastlens_core
)

target_link_libraries(fastlens_jpeg_dct_rowgroup_read_bench PRIVATE
        fastlens_core
)

target_include_directories(fastlens_jpeg_dct_tool PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

target_include_directories(fastlens_jpeg_dct_policy_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

target_include_directories(fastlens_compact_v3_runtime_audit PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

target_include_directories(fastlens_jpeg_dct_rowgroup_read_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

apply_warnings(fastlens_jpeg_dct_tool)
apply_warnings(fastlens_block_major_access_tool)
apply_warnings(fastlens_block_major_plan_audit)
apply_warnings(fastlens_jpeg_dct_policy_bench)
apply_warnings(fastlens_compact_v3_runtime_audit)
apply_warnings(fastlens_jpeg_dct_rowgroup_read_bench)
