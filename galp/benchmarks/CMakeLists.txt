## ────────────────────────────────────────────────
## Generated bindings library
## ────────────────────────────────────────────────
if (NOT Python3_Interpreter_FOUND)
        message(FATAL_ERROR "Python3 is required to generate GALP benchmark bindings")
endif ()

set(GALP_GENERATED_BENCHMARK_DIR "${CMAKE_BINARY_DIR}/generated/galp/benchmarks")
set(GALP_GENERATED_BINDINGS_DIR "${GALP_GENERATED_BENCHMARK_DIR}/bindings")
set(GALP_GENERATED_INCLUDE_DIR "${GALP_GENERATED_BENCHMARK_DIR}/include")
set(GALP_GENERATED_HEADER_DIR "${GALP_GENERATED_INCLUDE_DIR}/galp_bench/generated")
set(GALP_GENERATED_STAMP "${GALP_GENERATED_BENCHMARK_DIR}/generated-bindings.stamp")

set(GALP_GENERATED_BINDINGS_MANIFEST
        "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/codegen/generated_bindings_manifest.json")
file(READ "${GALP_GENERATED_BINDINGS_MANIFEST}" GALP_GENERATED_BINDINGS_MANIFEST_CONTENT)
string(JSON GALP_GENERATED_BINDING_COUNT LENGTH "${GALP_GENERATED_BINDINGS_MANIFEST_CONTENT}" bindings)
if (GALP_GENERATED_BINDING_COUNT LESS 1)
        message(FATAL_ERROR "GALP generated binding manifest is empty")
endif ()
math(EXPR GALP_GENERATED_BINDING_LAST "${GALP_GENERATED_BINDING_COUNT} - 1")
set(GALP_GENERATED_BINDING_FILENAMES)
foreach (generated_index RANGE 0 ${GALP_GENERATED_BINDING_LAST})
        string(JSON generated_file GET
                "${GALP_GENERATED_BINDINGS_MANIFEST_CONTENT}" bindings ${generated_index})
        list(APPEND GALP_GENERATED_BINDING_FILENAMES "${generated_file}")
endforeach ()

set(GALP_GENERATED_BINDING_COMPILED_FILENAMES ${GALP_GENERATED_BINDING_FILENAMES})

set(GALP_GENERATED_BINDINGS_OUTPUTS)
foreach (generated_file IN LISTS GALP_GENERATED_BINDING_FILENAMES)
        list(APPEND GALP_GENERATED_BINDINGS_OUTPUTS "${GALP_GENERATED_BINDINGS_DIR}/${generated_file}")
endforeach ()

set(GALP_GENERATED_BINDINGS_SOURCES)
foreach (generated_file IN LISTS GALP_GENERATED_BINDING_COMPILED_FILENAMES)
        list(APPEND GALP_GENERATED_BINDINGS_SOURCES "${GALP_GENERATED_BINDINGS_DIR}/${generated_file}")
endforeach ()

string(JSON GALP_GENERATED_HEADER_COUNT LENGTH "${GALP_GENERATED_BINDINGS_MANIFEST_CONTENT}" headers)
if (GALP_GENERATED_HEADER_COUNT LESS 1)
        message(FATAL_ERROR "GALP generated header manifest is empty")
endif ()
math(EXPR GALP_GENERATED_HEADER_LAST "${GALP_GENERATED_HEADER_COUNT} - 1")
set(GALP_GENERATED_HEADERS)
foreach (generated_index RANGE 0 ${GALP_GENERATED_HEADER_LAST})
        string(JSON generated_file GET
                "${GALP_GENERATED_BINDINGS_MANIFEST_CONTENT}" headers ${generated_index})
        list(APPEND GALP_GENERATED_HEADERS "${GALP_GENERATED_HEADER_DIR}/${generated_file}")
endforeach ()

set(GALP_GENERATE_KERNEL_BINDINGS_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/codegen/generate_kernel_bindings.py")

add_custom_command(
        OUTPUT "${GALP_GENERATED_STAMP}"
        BYPRODUCTS ${GALP_GENERATED_BINDINGS_OUTPUTS} ${GALP_GENERATED_HEADERS}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${GALP_GENERATED_BINDINGS_DIR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${GALP_GENERATED_HEADER_DIR}"
        COMMAND ${Python3_EXECUTABLE}
                "${GALP_GENERATE_KERNEL_BINDINGS_SCRIPT}"
                --out-dir "${GALP_GENERATED_BINDINGS_DIR}"
                --header-out-dir "${GALP_GENERATED_HEADER_DIR}"
                --logging-level 40
        COMMAND ${CMAKE_COMMAND} -E touch "${GALP_GENERATED_STAMP}"
        DEPENDS
                "${GALP_GENERATE_KERNEL_BINDINGS_SCRIPT}"
                "${GALP_GENERATED_BINDINGS_MANIFEST}"
        COMMENT "Generating GALP benchmark bindings"
        VERBATIM)

add_custom_target(galp_generate_benchmark_bindings DEPENDS "${GALP_GENERATED_STAMP}")

set_source_files_properties(
        ${GALP_GENERATED_BINDINGS_OUTPUTS}
        ${GALP_GENERATED_HEADERS}
        PROPERTIES GENERATED TRUE)

add_library(galp_generated_bindings STATIC ${GALP_GENERATED_BINDINGS_SOURCES})
add_dependencies(galp_generated_bindings galp_generate_benchmark_bindings)
target_include_directories(galp_generated_bindings PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${GALP_GENERATED_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(galp_generated_bindings PRIVATE
        CUDA::cudart
        Thrust::Thrust
        galp_engine
        galp_alp_support)
apply_warnings(galp_generated_bindings)
set_target_properties(galp_generated_bindings PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Build-only compatibility target for older local commands.
add_custom_target(generated-bindings DEPENDS galp_generated_bindings)
add_custom_target(generated_bindings DEPENDS galp_generated_bindings)


if (GALP_WITH_NVCOMP)
	add_executable(compressor_bench
		compressor_bench.cu
		nvcomp/benchmark_compressors.cu
		nvcomp/nvcomp_compressors.cu
		nvcomp/types.cu)
        apply_warnings(compressor_bench)
        target_include_directories(compressor_bench PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${GALP_GENERATED_INCLUDE_DIR}
                ${CMAKE_CURRENT_SOURCE_DIR}/../src
                ${CMAKE_CURRENT_SOURCE_DIR}/../tools/data/include)
        target_link_libraries(compressor_bench PRIVATE
                galp_engine
                galp_alp_support
                nvcomp::nvcomp)
        target_compile_definitions(compressor_bench PRIVATE FLS_GALP_SOURCE_DIR="${GALP_DATA_ROOT}")

        add_custom_target(compressors_benchmarks DEPENDS compressor_bench)
        add_custom_target(compressors-benchmarks DEPENDS compressor_bench)
endif ()

add_executable(micro_bench micro_bench.cu)
apply_warnings(micro_bench)
target_include_directories(micro_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${GALP_GENERATED_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(micro_bench PRIVATE
        galp_engine
        galp_generated_bindings
        galp_alp_support)

add_custom_target(micro_benchmarks DEPENDS micro_bench)
add_custom_target(micro-benchmarks DEPENDS micro_bench)
