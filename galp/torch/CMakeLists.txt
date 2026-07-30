find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

if (NOT Torch_DIR)
        execute_process(
                COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
                OUTPUT_VARIABLE GALP_TORCH_CMAKE_PREFIX_PATH
                RESULT_VARIABLE GALP_TORCH_PREFIX_RESULT
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
        )
        if (GALP_TORCH_PREFIX_RESULT EQUAL 0 AND GALP_TORCH_CMAKE_PREFIX_PATH)
                list(APPEND CMAKE_PREFIX_PATH "${GALP_TORCH_CMAKE_PREFIX_PATH}")
        endif ()
endif ()

if (NOT TARGET CUDA::nvToolsExt)
        add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
endif ()

if (CMAKE_CUDA_ARCHITECTURES AND NOT TORCH_CUDA_ARCH_LIST AND NOT DEFINED ENV{TORCH_CUDA_ARCH_LIST})
        set(GALP_TORCH_CUDA_ARCH_LIST)
        foreach (GALP_CUDA_ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES)
                if (GALP_CUDA_ARCH MATCHES "^([0-9]+)([a-z]?)$")
                        set(GALP_CUDA_ARCH_NUM "${CMAKE_MATCH_1}")
                        set(GALP_CUDA_ARCH_SUFFIX "${CMAKE_MATCH_2}")
                        string(LENGTH "${GALP_CUDA_ARCH_NUM}" GALP_CUDA_ARCH_NUM_LENGTH)
                        math(EXPR GALP_CUDA_ARCH_MAJOR_LENGTH "${GALP_CUDA_ARCH_NUM_LENGTH} - 1")
                        string(SUBSTRING "${GALP_CUDA_ARCH_NUM}" 0 ${GALP_CUDA_ARCH_MAJOR_LENGTH} GALP_CUDA_ARCH_MAJOR)
                        string(SUBSTRING "${GALP_CUDA_ARCH_NUM}" ${GALP_CUDA_ARCH_MAJOR_LENGTH} 1 GALP_CUDA_ARCH_MINOR)
                        list(APPEND GALP_TORCH_CUDA_ARCH_LIST
                                "${GALP_CUDA_ARCH_MAJOR}.${GALP_CUDA_ARCH_MINOR}${GALP_CUDA_ARCH_SUFFIX}")
                endif ()
        endforeach ()
        if (GALP_TORCH_CUDA_ARCH_LIST)
                list(JOIN GALP_TORCH_CUDA_ARCH_LIST " " TORCH_CUDA_ARCH_LIST)
        endif ()
endif ()

find_package(Torch REQUIRED)

find_library(GALP_TORCH_PYTHON_LIBRARY
        NAMES torch_python
        HINTS "${TORCH_INSTALL_PREFIX}/lib"
)

Python3_add_library(_galp_direct_dct MODULE WITH_SOABI
        direct_dct_torch.cpp
)

target_include_directories(_galp_direct_dct PRIVATE
        ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(_galp_direct_dct PRIVATE
        Galp::jpeg_dct_device
        ${TORCH_LIBRARIES}
)
if (GALP_TORCH_PYTHON_LIBRARY)
        target_link_libraries(_galp_direct_dct PRIVATE ${GALP_TORCH_PYTHON_LIBRARY})
endif ()

target_compile_features(_galp_direct_dct PRIVATE cxx_std_20)

if (TORCH_CXX_FLAGS)
        separate_arguments(GALP_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
        target_compile_options(_galp_direct_dct PRIVATE ${GALP_TORCH_CXX_FLAGS})
endif ()

set_target_properties(_galp_direct_dct PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/galp/torch
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/galp/torch
)

if (BUILD_TESTING)
        add_test(
                NAME GalpDirectDctTorchImport
                COMMAND ${CMAKE_COMMAND} -E env
                        "PYTHONPATH=$<TARGET_FILE_DIR:_galp_direct_dct>"
                        ${Python3_EXECUTABLE}
                        -c "import torch; import _galp_direct_dct as m; print(torch.__version__); print(torch.cuda.is_available()); assert m.DirectDctReader; assert m.DirectDctBatch"
        )
        set_tests_properties(GalpDirectDctTorchImport PROPERTIES
                LABELS "galp;jpeg-dct;direct-dct;torch;smoke"
        )
        add_test(
                NAME GalpDirectDctTorchRuntimeSmoke
                COMMAND ${CMAKE_COMMAND} -E env
                        "PYTHONPATH=$<TARGET_FILE_DIR:_galp_direct_dct>"
                        ${Python3_EXECUTABLE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/direct_dct_torch_runtime_smoke.py
        )
        set_tests_properties(GalpDirectDctTorchRuntimeSmoke PROPERTIES
                LABELS "galp;jpeg-dct;direct-dct;torch;gpu;smoke"
                SKIP_RETURN_CODE 77
        )
endif ()
