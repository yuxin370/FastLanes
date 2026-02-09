add_executable(galp_cli
        galp_cli.cu
)

target_link_libraries(galp_cli PRIVATE
        galp_core
)

apply_warnings(galp_cli)
