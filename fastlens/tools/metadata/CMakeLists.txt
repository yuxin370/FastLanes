add_library(fastlens_metadata_interning STATIC
        interned_table_descriptor.cpp
)
target_link_libraries(fastlens_metadata_interning PUBLIC FastLanes::core)
target_include_directories(fastlens_metadata_interning PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
apply_warnings(fastlens_metadata_interning)

add_executable(fastlens_fls_metadata_tool
        fls_metadata_tool.cpp
)

target_link_libraries(fastlens_fls_metadata_tool PRIVATE
        FastLanes::core
        fastlens_metadata_interning
)

apply_warnings(fastlens_fls_metadata_tool)

add_executable(fastlens_fls_metadata_benchmark
        fls_metadata_benchmark.cpp
)
target_link_libraries(fastlens_fls_metadata_benchmark PRIVATE FastLanes::core)
apply_warnings(fastlens_fls_metadata_benchmark)

if (BUILD_TESTING)
        add_test(NAME FastLensFlsMetadataToolHelp COMMAND fastlens_fls_metadata_tool --help)
        set_tests_properties(FastLensFlsMetadataToolHelp PROPERTIES LABELS "fastlens;cli;metadata;smoke")
        add_test(NAME FastLensFlsMetadataBenchmarkHelp COMMAND fastlens_fls_metadata_benchmark --help)
        set_tests_properties(FastLensFlsMetadataBenchmarkHelp PROPERTIES LABELS "fastlens;cli;metadata;smoke")

        add_executable(fastlens_metadata_interning_test interned_table_descriptor_test.cpp)
        target_link_libraries(fastlens_metadata_interning_test PRIVATE
                FastLanes::core
                fastlens_metadata_interning
        )
        apply_warnings(fastlens_metadata_interning_test)
        add_test(NAME FastLensMetadataInterningRoundTrip COMMAND fastlens_metadata_interning_test)
        set_tests_properties(FastLensMetadataInterningRoundTrip PROPERTIES LABELS "fastlens;metadata;unit")

        if (Python3_Interpreter_FOUND)
                add_test(NAME FastLensMetadataBenchmarkRunnerHelp
                        COMMAND ${Python3_EXECUTABLE}
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../benchmarks/metadata/run.py --help)
                add_test(NAME FastLensMetadataDatasetBreakdownHelp
                        COMMAND ${Python3_EXECUTABLE}
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../benchmarks/metadata/breakdown_dataset.py --help)
                set_tests_properties(
                        FastLensMetadataBenchmarkRunnerHelp
                        FastLensMetadataDatasetBreakdownHelp
                        PROPERTIES LABELS "fastlens;metadata;cli;smoke")
        endif ()
endif ()
