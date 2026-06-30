add_library(galp_benchmark_support STATIC
        table.cu
)
if (GALP_WITH_JPEG_DCT)
        target_sources(galp_benchmark_support PRIVATE
                pipeline.cu)
endif ()
set_target_properties(galp_benchmark_support PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(galp_benchmark_support PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(galp_benchmark_support PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src
        ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(galp_benchmark_support PUBLIC
        galp_core)
apply_warnings(galp_benchmark_support)
