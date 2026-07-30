add_library(galp_metadata_interning STATIC
        interned_table_descriptor.cpp
)
target_link_libraries(galp_metadata_interning PUBLIC FastLanes::core)
target_include_directories(galp_metadata_interning PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
apply_warnings(galp_metadata_interning)

add_executable(galp_fls_metadata_tool
        fls_metadata_tool.cpp
)

target_link_libraries(galp_fls_metadata_tool PRIVATE
        FastLanes::core
        galp_metadata_interning
)

apply_warnings(galp_fls_metadata_tool)

add_executable(galp_fls_metadata_benchmark
        fls_metadata_benchmark.cpp
)
target_link_libraries(galp_fls_metadata_benchmark PRIVATE FastLanes::core)
apply_warnings(galp_fls_metadata_benchmark)

if (BUILD_TESTING)
        add_test(NAME GalpFlsMetadataToolHelp COMMAND galp_fls_metadata_tool --help)
        set_tests_properties(GalpFlsMetadataToolHelp PROPERTIES LABELS "galp;cli;metadata;smoke")
        add_test(NAME GalpFlsMetadataBenchmarkHelp COMMAND galp_fls_metadata_benchmark --help)
        set_tests_properties(GalpFlsMetadataBenchmarkHelp PROPERTIES LABELS "galp;cli;metadata;smoke")

        add_executable(galp_metadata_interning_test interned_table_descriptor_test.cpp)
        target_link_libraries(galp_metadata_interning_test PRIVATE
                FastLanes::core
                galp_metadata_interning
        )
        apply_warnings(galp_metadata_interning_test)
        add_test(NAME GalpMetadataInterningRoundTrip COMMAND galp_metadata_interning_test)
        set_tests_properties(GalpMetadataInterningRoundTrip PROPERTIES LABELS "galp;metadata;unit")

        if (Python3_Interpreter_FOUND)
                add_test(NAME GalpMetadataBenchmarkRunnerHelp
                        COMMAND ${Python3_EXECUTABLE}
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../benchmarks/metadata/run.py --help)
                add_test(NAME GalpMetadataDatasetBreakdownHelp
                        COMMAND ${Python3_EXECUTABLE}
                                ${CMAKE_CURRENT_SOURCE_DIR}/../../benchmarks/metadata/breakdown_dataset.py --help)
                set_tests_properties(
                        GalpMetadataBenchmarkRunnerHelp
                        GalpMetadataDatasetBreakdownHelp
                        PROPERTIES LABELS "galp;metadata;cli;smoke")
        endif ()
endif ()
