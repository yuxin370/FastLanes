if (NOT DEFINED GALP_ROOT)
        message(FATAL_ERROR "GALP_ROOT is required")
endif ()

set(paths
        "${GALP_ROOT}/src/jpeg/jpeg_dct_planner.cpp"
        "${GALP_ROOT}/src/jpeg/jpeg_dct_device_bridge.cpp"
        "${GALP_ROOT}/src/jpeg/jpeg_dct_device.cu"
        "${GALP_ROOT}/src/jpeg/jpeg_dct_gather_kernels.cu"
        "${GALP_ROOT}/src/jpeg/jpeg_dct_transform_kernels.cu"
        "${GALP_ROOT}/src/jpeg/jpeg_dct_device_runtime.hpp"
        "${GALP_ROOT}/src/jpeg/jpeg_dct_cuda_internal.cuh"
        "${GALP_ROOT}/tools/benchmark_support/pipeline.cu"
        "${GALP_ROOT}/tools/benchmark_support/include/galp_tools/benchmark_support/pipeline.cuh"
)

foreach (path IN LISTS paths)
        if (NOT EXISTS "${path}")
                message(FATAL_ERROR "JPEG DCT sync check path does not exist: ${path}")
        endif ()
        file(READ "${path}" content)
        foreach (forbidden_sync cudaDeviceSynchronize cudaStreamSynchronize)
                if (content MATCHES "${forbidden_sync}[ \t\r\n]*\\(")
                        message(FATAL_ERROR
                                "JPEG DCT pushdown hot path must not call ${forbidden_sync}(): ${path}")
                endif ()
        endforeach ()
endforeach ()
