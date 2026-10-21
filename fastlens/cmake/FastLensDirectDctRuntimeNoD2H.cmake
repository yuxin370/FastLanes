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

foreach (path IN LISTS cxx_paths)
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
        foreach (forbidden_token cudaMemcpyDeviceToHost cudaMemcpyDefault cudaMemcpyDtoH)
                if (content MATCHES "${forbidden_token}")
                        message(FATAL_ERROR
                                "Direct-DCT runtime/export path must not contain ${forbidden_token}: ${path}")
                endif ()
        endforeach ()
endforeach ()

set(python_paths
        "${FASTLENS_ROOT}/examples/direct_dct_torch_end_to_end_demo.py"
        "${FASTLENS_ROOT}/torch/direct_dct_torch_runtime_smoke.py"
)

foreach (path IN LISTS python_paths)
        if (NOT EXISTS "${path}")
                message(FATAL_ERROR "Direct-DCT Python no-D2H check path does not exist: ${path}")
        endif ()
        file(READ "${path}" content)
        foreach (forbidden_pattern "\\.cpu[ \t\r\n]*\\(" "\\.numpy[ \t\r\n]*\\(" "\\.tolist[ \t\r\n]*\\(")
                if (content MATCHES "${forbidden_pattern}")
                        message(FATAL_ERROR
                                "Direct-DCT demo must not materialize coefficient tensors on host: ${path}")
                endif ()
        endforeach ()
endforeach ()
