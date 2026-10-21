add_subdirectory(metadata)

if (FASTLENS_WITH_JPEG_DCT)
        add_subdirectory(jpeg_dct)
endif ()

add_executable(fastlens_cli
        fastlens_cli/fastlens_cli.cu
)

target_link_libraries(fastlens_cli PRIVATE
        fastlens_benchmark_support
)
target_include_directories(fastlens_cli PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src)

apply_warnings(fastlens_cli)

if (BUILD_TESTING)
        add_test(NAME FastLensCliHelp COMMAND fastlens_cli --help)
        set_tests_properties(FastLensCliHelp PROPERTIES LABELS "fastlens;cli;smoke")
        if (Python3_Interpreter_FOUND)
                add_test(
                        NAME FastLensPipelineBenchmarkSummaryScript
                        COMMAND ${CMAKE_COMMAND}
                                -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}
                                -DREPO_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/../..
                                -DTEST_TMP=${CMAKE_CURRENT_BINARY_DIR}/pipeline_summary_script_test
                                -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/FastLensPipelineBenchmarkSummaryScript.cmake
                )
                set_tests_properties(FastLensPipelineBenchmarkSummaryScript PROPERTIES LABELS "fastlens;cli;smoke")
        endif ()
        if (FASTLENS_WITH_JPEG_DCT)
                add_test(NAME FastLensCliPipelineBenchmarkHelp COMMAND fastlens_cli pipeline_benchmark --help)
                set_tests_properties(FastLensCliPipelineBenchmarkHelp PROPERTIES
                        LABELS "fastlens;cli;smoke"
                        PASS_REGULAR_EXPRESSION "--dct-coeffs")
        endif ()
endif ()
