if (NOT DEFINED FASTLENS_ROOT)
        message(FATAL_ERROR "FASTLENS_ROOT is required")
endif ()

set(cxx_paths
        "${FASTLENS_ROOT}/include/fastlens/jpeg_dct.hpp"
        "${FASTLENS_ROOT}/include/fastlens/direct_dct.hpp"
        "${FASTLENS_ROOT}/src/jpeg/jpeg_dct_planner.cpp"
        "${FASTLENS_ROOT}/src/jpeg/jpeg_dct_device_bridge.cpp"
        "${FASTLENS_ROOT}/src/jpeg/jpeg_dct_device.cu"
        "${FASTLENS_ROOT}/src/jpeg/jpeg_dct_gather_kernels.cu"
        "${FASTLENS_ROOT}/src/jpeg/jpeg_dct_transform_kernels.cu"
        "${FASTLENS_ROOT}/src/jpeg/jpeg_dct_device_runtime.hpp"
        "${FASTLENS_ROOT}/src/jpeg/jpeg_dct_cuda_internal.cuh"
        "${FASTLENS_ROOT}/src/api/direct_dct.cpp"
        "${FASTLENS_ROOT}/torch/direct_dct_torch.cpp"
)
if (DEFINED FASTLENS_D2H_EXTRA_SOURCE)
        list(APPEND cxx_paths "${FASTLENS_D2H_EXTRA_SOURCE}")
endif ()

# The benchmark pipeline also promises asynchronous submission, but explicitly
# supports host materialization. Apply only the synchronization check there.
set(sync_only_paths
        "${FASTLENS_ROOT}/tools/benchmark_support/pipeline.cu"
        "${FASTLENS_ROOT}/tools/benchmark_support/include/fastlens_tools/benchmark_support/pipeline.cuh"
)

foreach (path IN LISTS cxx_paths sync_only_paths)
        if (NOT EXISTS "${path}")
                message(FATAL_ERROR "Direct-DCT no-D2H check path does not exist: ${path}")
        endif ()
        file(READ "${path}" content)
        foreach (forbidden_call cudaDeviceSynchronize cudaStreamSynchronize)
                if (content MATCHES "${forbidden_call}[ \t\r\n]*\\(")
                        message(FATAL_ERROR
                                "Direct-DCT runtime/export path must not call ${forbidden_call}(): ${path}")
                endif ()
        endforeach ()
        list(FIND cxx_paths "${path}" runtime_path_index)
        if (NOT runtime_path_index EQUAL -1)
                foreach (forbidden_token cudaMemcpyDeviceToHost cudaMemcpyDefault cudaMemcpyDtoH)
                        if (content MATCHES "${forbidden_token}")
                                message(FATAL_ERROR
                                        "Direct-DCT runtime/export path must not contain ${forbidden_token}: ${path}")
                        endif ()
                endforeach ()
        endif ()
endforeach ()
