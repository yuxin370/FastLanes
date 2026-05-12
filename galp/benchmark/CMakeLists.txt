## ────────────────────────────────────────────────
## Generated-bindings library
## ────────────────────────────────────────────────
set(GALP_GENERATED_BINDINGS_SOURCES
        generated-bindings/alp-double-decompress_column-bindings.cu
        generated-bindings/alp-double-query_column-bindings.cu
        generated-bindings/alp-float-decompress_column-bindings.cu
        generated-bindings/alp-float-query_column-bindings.cu
        generated-bindings/alpextended-double-decompress_column-bindings.cu
        generated-bindings/alpextended-double-query_column-bindings.cu
        generated-bindings/alpextended-float-decompress_column-bindings.cu
        generated-bindings/alpextended-float-query_column-bindings.cu
        generated-bindings/bp-uint32_t-decompress_column-bindings.cu
        generated-bindings/bp-uint32_t-query_column-bindings.cu
        generated-bindings/bp-uint64_t-decompress_column-bindings.cu
        generated-bindings/bp-uint64_t-query_column-bindings.cu
        generated-bindings/ffor-uint32_t-compute_column-bindings.cu
        generated-bindings/ffor-uint32_t-decompress_column-bindings.cu
        generated-bindings/ffor-uint32_t-query_column-bindings.cu
        generated-bindings/ffor-uint64_t-compute_column-bindings.cu
        generated-bindings/ffor-uint64_t-decompress_column-bindings.cu
        generated-bindings/ffor-uint64_t-query_column-bindings.cu
        generated-bindings/slpatch-uint32_t-decompress_column-bindings.cu
        generated-bindings/slpatch-uint64_t-decompress_column-bindings.cu
        generated-bindings/slpatch-int16_t-decompress_column-bindings.cu
        generated-bindings/freq-uint32_t-decompress_column-bindings.cu
        generated-bindings/freq-uint64_t-decompress_column-bindings.cu
        generated-bindings/freq-int8_t-decompress_column-bindings.cu
        generated-bindings/freq-int16_t-decompress_column-bindings.cu
        generated-bindings/freqextended-uint32_t-decompress_column-bindings.cu
        generated-bindings/freqextended-uint64_t-decompress_column-bindings.cu
        generated-bindings/freqextended-int8_t-decompress_column-bindings.cu
        generated-bindings/freqextended-int16_t-decompress_column-bindings.cu
        generated-bindings/dict-uint32_t-decompress_column-bindings.cu
        generated-bindings/dict-uint64_t-decompress_column-bindings.cu
        generated-bindings/dictslpatch-uint32_t-decompress_column-bindings.cu
        generated-bindings/dictslpatch-uint64_t-decompress_column-bindings.cu
        generated-bindings/rle-uint32_t-decompress_column-bindings.cu
        generated-bindings/rle-uint64_t-decompress_column-bindings.cu
        generated-bindings/constant-uint32_t-decompress_column-bindings.cu
        generated-bindings/constant-uint64_t-decompress_column-bindings.cu
        generated-bindings/crossrle-uint32_t-decompress_column-bindings.cu
        generated-bindings/crossrle-uint64_t-decompress_column-bindings.cu
        generated-bindings/crossrleextended-uint32_t-decompress_column-bindings.cu
        generated-bindings/crossrleextended-uint64_t-decompress_column-bindings.cu
        generated-bindings/crossrlelanemask-uint32_t-decompress_column-bindings.cu
        generated-bindings/crossrlelanemask-uint64_t-decompress_column-bindings.cu
        # generated-bindings/dictshfl32-uint32_t-decompress_column-bindings.cu
        # generated-bindings/dictshfl32-uint64_t-decompress_column-bindings.cu
)

# Keep the experimental multi-column binding sources in-tree, but do not compile
# them as part of generated-bindings. They instantiate very large CUDA kernels
# and can make nvcc stall on query_multi_column translation units.
# GALP_ENABLE_MULTI_COLUMN=ON is rejected during configure in galp/CMakeLists.txt.

add_library(generated-bindings STATIC ${GALP_GENERATED_BINDINGS_SOURCES})
target_include_directories(generated-bindings PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(generated-bindings PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/include
        ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(generated-bindings PUBLIC CUDA::cudart Thrust::Thrust galp_core)
apply_warnings(generated-bindings)
set_target_properties(generated-bindings PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


if (GALP_WITH_NVCOMP)
        add_executable(compressors-benchmarks compressors-benchmarks.cu nvcomp/benchmark-compressors.cu nvcomp/nvcomp-compressors.cu)
        apply_warnings(compressors-benchmarks)
        target_include_directories(compressors-benchmarks PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_CURRENT_SOURCE_DIR}/../src/include)
        target_link_libraries(compressors-benchmarks PRIVATE galp_core nvcomp::nvcomp)
        target_compile_definitions(compressors-benchmarks PRIVATE FLS_GALP_SOURCE_DIR="${GALP_DATA_ROOT}")
endif ()

add_executable(micro-benchmarks micro-benchmarks.cu)
apply_warnings(micro-benchmarks)
target_compile_definitions(micro-benchmarks PRIVATE GALP_ENABLE_MULTI_COLUMN=0)
target_include_directories(micro-benchmarks PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/include)
target_link_libraries(micro-benchmarks PRIVATE galp_core generated-bindings)
