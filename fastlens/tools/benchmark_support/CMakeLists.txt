add_library(fastlens_benchmark_support STATIC
        table.cu
)
if (FASTLENS_WITH_JPEG_DCT)
        target_sources(fastlens_benchmark_support PRIVATE
                pipeline.cu)
endif ()
set_target_properties(fastlens_benchmark_support PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(fastlens_benchmark_support PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(fastlens_benchmark_support PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src
        ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(fastlens_benchmark_support PUBLIC
        fastlens_engine)
if (FASTLENS_WITH_JPEG_DCT)
        target_link_libraries(fastlens_benchmark_support PUBLIC fastlens_jpeg_dct_device)
endif ()
apply_warnings(fastlens_benchmark_support)
