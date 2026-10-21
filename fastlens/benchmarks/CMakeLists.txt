add_executable(fastlens_transfer_contention development/transfer_contention.cpp)
target_include_directories(fastlens_transfer_contention PRIVATE ../src ${CUDAToolkit_INCLUDE_DIRS})
target_compile_definitions(fastlens_transfer_contention PRIVATE FASTLENS_MEMORY_DIAGNOSTICS=1)
target_link_libraries(fastlens_transfer_contention PRIVATE CUDA::cudart)

if (FASTLENS_WITH_JPEG_DCT AND UNIX)
        # reader.cuh includes device functors; compile the host-side interposition with CUDA headers active.
        set_source_files_properties(paper/shared/input_boundary_controls.cpp PROPERTIES LANGUAGE CUDA)
        add_library(fastlens_input_boundary_controls SHARED paper/shared/input_boundary_controls.cpp)
        apply_warnings(fastlens_input_boundary_controls)
        set_target_properties(fastlens_input_boundary_controls PROPERTIES OUTPUT_NAME input_boundary_controls)
        target_include_directories(fastlens_input_boundary_controls SYSTEM PRIVATE
                ../src ../include ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(fastlens_input_boundary_controls PRIVATE FastLanes::headers ${CMAKE_DL_LIBS})
        add_executable(fastlens_dct_storage dct_models/dctnet_storage.cpp)
        target_include_directories(fastlens_dct_storage PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
        target_link_libraries(fastlens_dct_storage PRIVATE fastlens_core fastlens_compact_format)
        apply_warnings(fastlens_dct_storage)
endif ()

if (FASTLENS_BUILD_TESTS AND FASTLENS_BUILD_TORCH)
        foreach (suite system_dct_major training_pls dct_retokenization baselines/dct_baselines)
                get_filename_component(suite_name "${suite}" NAME)
                add_test(NAME FastLensBenchmark_${suite_name}
                        COMMAND ${Python3_EXECUTABLE} -B -m unittest discover
                                -s fastlens/benchmarks/${suite}/tests -t .)
                set_tests_properties(FastLensBenchmark_${suite_name} PROPERTIES
                        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                        LABELS "fastlens;benchmark;cpu")
        endforeach ()
        add_test(NAME FastLensBenchmark_rgbnomore
                COMMAND ${Python3_EXECUTABLE} -B -m unittest
                        fastlens.benchmarks.baselines.system_rgbnomore.tests.test_system_benchmark
                        fastlens.benchmarks.baselines.system_rgbnomore.tests.test_training_benchmark
                        fastlens.benchmarks.baselines.system_rgbnomore.tests.test_equal_image_epoch_benchmark
                        fastlens.benchmarks.baselines.system_rgbnomore.tests.test_prepare_imagenet512_v3_train
                        fastlens.benchmarks.baselines.system_rgbnomore.tests.test_rgbnomore_fingerprint_cache)
        set_tests_properties(FastLensBenchmark_rgbnomore PROPERTIES
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                LABELS "fastlens;benchmark;cpu")
        add_test(NAME FastLensBenchmark_profiling
                COMMAND ${Python3_EXECUTABLE} -B -m unittest
                        fastlens.benchmarks.profiling.tests.test_profiling_evidence)
        set_tests_properties(FastLensBenchmark_profiling PROPERTIES
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                LABELS "fastlens;benchmark;cpu")
        add_test(NAME FastLensBenchmark_coefficient_masks
                COMMAND ${Python3_EXECUTABLE} -B -m unittest discover
                        -s fastlens/benchmarks/coefficient_mask_evaluator/tests -t .)
        set_tests_properties(FastLensBenchmark_coefficient_masks PROPERTIES
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                LABELS "fastlens;benchmark;cpu")
        add_test(NAME FastLensBenchmark_training_gpu
                COMMAND ${Python3_EXECUTABLE} -B -m unittest
                        fastlens.benchmarks.baselines.system_rgbnomore.tests.test_training_gpu)
        set_tests_properties(FastLensBenchmark_training_gpu PROPERTIES
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                LABELS "fastlens;benchmark;gpu")
endif ()

## ────────────────────────────────────────────────
## Generated bindings library
## ────────────────────────────────────────────────
if (NOT Python3_Interpreter_FOUND)
        message(FATAL_ERROR "Python3 is required to generate FASTLENS benchmark bindings")
endif ()

set(FASTLENS_GENERATED_BENCHMARK_DIR "${CMAKE_BINARY_DIR}/generated/fastlens/benchmarks")
set(FASTLENS_GENERATED_BINDINGS_DIR "${FASTLENS_GENERATED_BENCHMARK_DIR}/bindings")
set(FASTLENS_GENERATED_INCLUDE_DIR "${FASTLENS_GENERATED_BENCHMARK_DIR}/include")
set(FASTLENS_GENERATED_HEADER_DIR "${FASTLENS_GENERATED_INCLUDE_DIR}/fastlens_bench/generated")
set(FASTLENS_GENERATED_STAMP "${FASTLENS_GENERATED_BENCHMARK_DIR}/generated-bindings.stamp")

set(FASTLENS_GENERATED_BINDINGS_MANIFEST
        "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/codegen/generated_bindings_manifest.json")
file(READ "${FASTLENS_GENERATED_BINDINGS_MANIFEST}" FASTLENS_GENERATED_BINDINGS_MANIFEST_CONTENT)
string(JSON FASTLENS_GENERATED_BINDING_COUNT LENGTH "${FASTLENS_GENERATED_BINDINGS_MANIFEST_CONTENT}" bindings)
if (FASTLENS_GENERATED_BINDING_COUNT LESS 1)
        message(FATAL_ERROR "FASTLENS generated binding manifest is empty")
endif ()
math(EXPR FASTLENS_GENERATED_BINDING_LAST "${FASTLENS_GENERATED_BINDING_COUNT} - 1")
set(FASTLENS_GENERATED_BINDING_FILENAMES)
foreach (generated_index RANGE 0 ${FASTLENS_GENERATED_BINDING_LAST})
        string(JSON generated_file GET
                "${FASTLENS_GENERATED_BINDINGS_MANIFEST_CONTENT}" bindings ${generated_index})
        list(APPEND FASTLENS_GENERATED_BINDING_FILENAMES "${generated_file}")
endforeach ()

set(FASTLENS_GENERATED_BINDING_COMPILED_FILENAMES ${FASTLENS_GENERATED_BINDING_FILENAMES})

set(FASTLENS_GENERATED_BINDINGS_OUTPUTS)
foreach (generated_file IN LISTS FASTLENS_GENERATED_BINDING_FILENAMES)
        list(APPEND FASTLENS_GENERATED_BINDINGS_OUTPUTS "${FASTLENS_GENERATED_BINDINGS_DIR}/${generated_file}")
endforeach ()

set(FASTLENS_GENERATED_BINDINGS_SOURCES)
foreach (generated_file IN LISTS FASTLENS_GENERATED_BINDING_COMPILED_FILENAMES)
        list(APPEND FASTLENS_GENERATED_BINDINGS_SOURCES "${FASTLENS_GENERATED_BINDINGS_DIR}/${generated_file}")
endforeach ()

string(JSON FASTLENS_GENERATED_HEADER_COUNT LENGTH "${FASTLENS_GENERATED_BINDINGS_MANIFEST_CONTENT}" headers)
if (FASTLENS_GENERATED_HEADER_COUNT LESS 1)
        message(FATAL_ERROR "FASTLENS generated header manifest is empty")
endif ()
math(EXPR FASTLENS_GENERATED_HEADER_LAST "${FASTLENS_GENERATED_HEADER_COUNT} - 1")
set(FASTLENS_GENERATED_HEADERS)
foreach (generated_index RANGE 0 ${FASTLENS_GENERATED_HEADER_LAST})
        string(JSON generated_file GET
                "${FASTLENS_GENERATED_BINDINGS_MANIFEST_CONTENT}" headers ${generated_index})
        list(APPEND FASTLENS_GENERATED_HEADERS "${FASTLENS_GENERATED_HEADER_DIR}/${generated_file}")
endforeach ()

set(FASTLENS_GENERATE_KERNEL_BINDINGS_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/codegen/generate_kernel_bindings.py")

add_custom_command(
        OUTPUT "${FASTLENS_GENERATED_STAMP}"
        BYPRODUCTS ${FASTLENS_GENERATED_BINDINGS_OUTPUTS} ${FASTLENS_GENERATED_HEADERS}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${FASTLENS_GENERATED_BINDINGS_DIR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${FASTLENS_GENERATED_HEADER_DIR}"
        COMMAND ${Python3_EXECUTABLE}
                "${FASTLENS_GENERATE_KERNEL_BINDINGS_SCRIPT}"
                --out-dir "${FASTLENS_GENERATED_BINDINGS_DIR}"
                --header-out-dir "${FASTLENS_GENERATED_HEADER_DIR}"
                --logging-level 40
        COMMAND ${CMAKE_COMMAND} -E touch "${FASTLENS_GENERATED_STAMP}"
        DEPENDS
                "${FASTLENS_GENERATE_KERNEL_BINDINGS_SCRIPT}"
                "${FASTLENS_GENERATED_BINDINGS_MANIFEST}"
        COMMENT "Generating FASTLENS benchmark bindings"
        VERBATIM)

add_custom_target(fastlens_generate_benchmark_bindings DEPENDS "${FASTLENS_GENERATED_STAMP}")

set_source_files_properties(
        ${FASTLENS_GENERATED_BINDINGS_OUTPUTS}
        ${FASTLENS_GENERATED_HEADERS}
        PROPERTIES GENERATED TRUE)

add_library(fastlens_generated_bindings STATIC ${FASTLENS_GENERATED_BINDINGS_SOURCES})
add_dependencies(fastlens_generated_bindings fastlens_generate_benchmark_bindings)
target_include_directories(fastlens_generated_bindings PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${FASTLENS_GENERATED_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(fastlens_generated_bindings PRIVATE
        CUDA::cudart
        Thrust::Thrust
        fastlens_engine
        fastlens_alp_support)
apply_warnings(fastlens_generated_bindings)
set_target_properties(fastlens_generated_bindings PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if (FASTLENS_WITH_NVCOMP)
	add_executable(compressor_bench
		development/compressor_bench.cu
		baselines/nvcomp/benchmark_compressors.cu
		baselines/nvcomp/nvcomp_compressors.cu
		baselines/nvcomp/types.cu)
        apply_warnings(compressor_bench)
        target_include_directories(compressor_bench PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/baselines/nvcomp/include
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${FASTLENS_GENERATED_INCLUDE_DIR}
                ${CMAKE_CURRENT_SOURCE_DIR}/../src
                ${CMAKE_CURRENT_SOURCE_DIR}/../tools/data/include)
        target_link_libraries(compressor_bench PRIVATE
                fastlens_engine
                fastlens_alp_support
                nvcomp::nvcomp)
        target_compile_definitions(compressor_bench PRIVATE FLS_FASTLENS_SOURCE_DIR="${FASTLENS_DATA_ROOT}")
endif ()

add_executable(micro_bench development/micro_bench.cu)
apply_warnings(micro_bench)
target_include_directories(micro_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${FASTLENS_GENERATED_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(micro_bench PRIVATE
        fastlens_engine
        fastlens_generated_bindings
        fastlens_alp_support)
