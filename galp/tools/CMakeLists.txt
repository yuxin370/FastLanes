add_executable(galp_cli
        galp_cli.cu
)

target_link_libraries(galp_cli PRIVATE
        galp_benchmark_support
)
target_include_directories(galp_cli PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/include)

apply_warnings(galp_cli)
