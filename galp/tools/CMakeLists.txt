add_subdirectory(metadata)

if (GALP_WITH_JPEG_DCT)
        add_subdirectory(jpeg_dct)
endif ()

add_executable(galp_cli
        galp_cli/galp_cli.cu
)

target_link_libraries(galp_cli PRIVATE
        galp_benchmark_support
)
target_include_directories(galp_cli PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src)

apply_warnings(galp_cli)

if (BUILD_TESTING)
        add_test(NAME GalpCliHelp COMMAND galp_cli --help)
        set_tests_properties(GalpCliHelp PROPERTIES LABELS "galp;cli;smoke")
        if (Python3_Interpreter_FOUND)
                add_test(
                        NAME GalpPipelineBenchmarkSummaryScript
                        COMMAND ${CMAKE_COMMAND}
                                -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}
                                -DREPO_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/../..
                                -DTEST_TMP=${CMAKE_CURRENT_BINARY_DIR}/pipeline_summary_script_test
                                -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/GalpPipelineBenchmarkSummaryScript.cmake
                )
                set_tests_properties(GalpPipelineBenchmarkSummaryScript PROPERTIES LABELS "galp;cli;smoke")
        endif ()
        if (GALP_WITH_JPEG_DCT)
                add_test(NAME GalpCliPipelineBenchmarkHelp COMMAND galp_cli pipeline_benchmark --help)
                set_tests_properties(GalpCliPipelineBenchmarkHelp PROPERTIES
                        LABELS "galp;cli;smoke"
                        PASS_REGULAR_EXPRESSION "--dct-coeffs")
        endif ()
endif ()
