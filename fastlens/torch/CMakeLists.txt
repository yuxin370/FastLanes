find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

if (NOT Torch_DIR)
        execute_process(
                COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
                OUTPUT_VARIABLE FASTLENS_TORCH_CMAKE_PREFIX_PATH
                RESULT_VARIABLE FASTLENS_TORCH_PREFIX_RESULT
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
        )
        if (FASTLENS_TORCH_PREFIX_RESULT EQUAL 0 AND FASTLENS_TORCH_CMAKE_PREFIX_PATH)
                list(APPEND CMAKE_PREFIX_PATH "${FASTLENS_TORCH_CMAKE_PREFIX_PATH}")
        endif ()
endif ()

if (NOT TARGET CUDA::nvToolsExt)
        add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
endif ()

if (CMAKE_CUDA_ARCHITECTURES AND NOT TORCH_CUDA_ARCH_LIST AND NOT DEFINED ENV{TORCH_CUDA_ARCH_LIST})
        set(FASTLENS_TORCH_CUDA_ARCH_LIST)
        foreach (FASTLENS_CUDA_ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES)
                if (FASTLENS_CUDA_ARCH MATCHES "^([0-9]+)([a-z]?)$")
                        set(FASTLENS_CUDA_ARCH_NUM "${CMAKE_MATCH_1}")
                        set(FASTLENS_CUDA_ARCH_SUFFIX "${CMAKE_MATCH_2}")
                        string(LENGTH "${FASTLENS_CUDA_ARCH_NUM}" FASTLENS_CUDA_ARCH_NUM_LENGTH)
                        math(EXPR FASTLENS_CUDA_ARCH_MAJOR_LENGTH "${FASTLENS_CUDA_ARCH_NUM_LENGTH} - 1")
                        string(SUBSTRING "${FASTLENS_CUDA_ARCH_NUM}" 0 ${FASTLENS_CUDA_ARCH_MAJOR_LENGTH} FASTLENS_CUDA_ARCH_MAJOR)
                        string(SUBSTRING "${FASTLENS_CUDA_ARCH_NUM}" ${FASTLENS_CUDA_ARCH_MAJOR_LENGTH} 1 FASTLENS_CUDA_ARCH_MINOR)
                        list(APPEND FASTLENS_TORCH_CUDA_ARCH_LIST
                                "${FASTLENS_CUDA_ARCH_MAJOR}.${FASTLENS_CUDA_ARCH_MINOR}${FASTLENS_CUDA_ARCH_SUFFIX}")
                endif ()
        endforeach ()
        if (FASTLENS_TORCH_CUDA_ARCH_LIST)
                list(JOIN FASTLENS_TORCH_CUDA_ARCH_LIST " " TORCH_CUDA_ARCH_LIST)
        endif ()
endif ()

# Torch's legacy FindCUDA must use the toolkit already selected for FastLens.
set(CUDA_TOOLKIT_ROOT_DIR "${CUDAToolkit_TARGET_DIR}")
set(CUDA_NVCC_EXECUTABLE "${CMAKE_CUDA_COMPILER}")
find_package(Torch REQUIRED)

find_library(FASTLENS_TORCH_PYTHON_LIBRARY
        NAMES torch_python
        HINTS "${TORCH_INSTALL_PREFIX}/lib"
)

Python3_add_library(_fastlens_direct_dct MODULE WITH_SOABI
        direct_dct_torch.cpp
		direct_dct_pls_torch.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/../src/direct_dct/direct_dct_metrics.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/../src/direct_dct/native_batch_lifetime.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/../src/direct_dct/physical_layout_planner.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/direct_dct/profile_registry.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/direct_dct/resolved_execution_policy.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/direct_dct/native_pipeline_state.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/direct_dct/native_logical_batch_pipeline.cpp
)

target_include_directories(_fastlens_direct_dct PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(_fastlens_direct_dct PRIVATE
        FastLens::jpeg_dct_device
        ${TORCH_LIBRARIES}
)
if (FASTLENS_TORCH_PYTHON_LIBRARY)
        target_link_libraries(_fastlens_direct_dct PRIVATE ${FASTLENS_TORCH_PYTHON_LIBRARY})
endif ()

target_compile_features(_fastlens_direct_dct PRIVATE cxx_std_20)

if (TORCH_CXX_FLAGS)
        separate_arguments(FASTLENS_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
        target_compile_options(_fastlens_direct_dct PRIVATE ${FASTLENS_TORCH_CXX_FLAGS})
endif ()

set_target_properties(_fastlens_direct_dct PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/fastlens/torch
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/fastlens/torch
)

if (FASTLENS_INSTALL_TORCH_PACKAGE)
	set_target_properties(_fastlens_direct_dct PROPERTIES
		INSTALL_RPATH "$ORIGIN/../../torch/lib")
	install(TARGETS _fastlens_direct_dct
		LIBRARY DESTINATION fastlens/torch
		RUNTIME DESTINATION fastlens/torch)
	install(FILES
		${CMAKE_CURRENT_SOURCE_DIR}/../__init__.py
		DESTINATION fastlens)
	install(FILES
		${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
		${CMAKE_CURRENT_SOURCE_DIR}/direct_dct.py
		${CMAKE_CURRENT_SOURCE_DIR}/experimental.py
		DESTINATION fastlens/torch)
	install(DIRECTORY
		${CMAKE_CURRENT_SOURCE_DIR}/../profiles
		${CMAKE_CURRENT_SOURCE_DIR}/../diagnostics
		DESTINATION fastlens
		FILES_MATCHING PATTERN "*.py"
		PATTERN "__pycache__" EXCLUDE)
endif ()

if (BUILD_TESTING)
	# Private Phase-4 deterministic outstanding-consumer harness. The module
	# submits its gate, sentinel-read consumer, and completion marker in one
	# native call and is never linked into or installed with _fastlens_direct_dct.
	Python3_add_library(_fastlens_phase4_outstanding_test MODULE WITH_SOABI
		phase4_native_outstanding_harness.cu)
	target_include_directories(_fastlens_phase4_outstanding_test PRIVATE ${TORCH_INCLUDE_DIRS})
	target_link_libraries(_fastlens_phase4_outstanding_test PRIVATE CUDA::cudart)
	if (FASTLENS_TORCH_PYTHON_LIBRARY)
		target_link_libraries(_fastlens_phase4_outstanding_test PRIVATE ${FASTLENS_TORCH_PYTHON_LIBRARY})
	endif ()
	target_compile_features(_fastlens_phase4_outstanding_test PRIVATE cxx_std_20 cuda_std_20)
	set_target_properties(_fastlens_phase4_outstanding_test PROPERTIES
		LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/fastlens/torch
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/fastlens/torch)

	# Private Phase-2 A/B probe. It links the shadow executor into a separate
	# module and cannot become an authority for the production Torch binding.
	Python3_add_library(_fastlens_phase2_native_ab MODULE WITH_SOABI
		phase2_native_ab.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/../src/direct_dct/profile_registry.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/../src/direct_dct/resolved_execution_policy.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/../src/direct_dct/native_pipeline_state.cpp
	)
	target_include_directories(_fastlens_phase2_native_ab PRIVATE
		${TORCH_INCLUDE_DIRS}
		${CMAKE_CURRENT_SOURCE_DIR}/../src
	)
	target_link_libraries(_fastlens_phase2_native_ab PRIVATE
		FastLens::jpeg_dct_device
		${TORCH_LIBRARIES}
	)
	if (FASTLENS_TORCH_PYTHON_LIBRARY)
		target_link_libraries(_fastlens_phase2_native_ab PRIVATE ${FASTLENS_TORCH_PYTHON_LIBRARY})
	endif ()
	target_compile_features(_fastlens_phase2_native_ab PRIVATE cxx_std_20)
	if (TORCH_CXX_FLAGS)
		target_compile_options(_fastlens_phase2_native_ab PRIVATE ${FASTLENS_TORCH_CXX_FLAGS})
	endif ()
	set_target_properties(_fastlens_phase2_native_ab PROPERTIES
		LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/fastlens/torch
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/fastlens/torch
	)

        add_test(
                NAME FastLensDirectDctTorchImport
                COMMAND ${CMAKE_COMMAND} -E env
                        "PYTHONPATH=$<TARGET_FILE_DIR:_fastlens_direct_dct>"
                        ${Python3_EXECUTABLE}
                        -c "import torch; import _fastlens_direct_dct as m; print(torch.__version__); print(torch.cuda.is_available()); assert m.DirectDctReader; assert m.DirectDctPipeline; assert m.DirectDctBatch; assert m.DirectDctPlsPipeline; assert m.DirectDctPlsPool; assert m.DirectDctPlsMicrobatch; assert m.DIRECT_DCT_PROFILE_SCHEMA == 'galp-direct-dct-profile-v1'; assert m.DIRECT_DCT_METRICS_SCHEMA == 'galp-direct-dct-metrics-v2'; assert 'rgbnomore-training-pls-v1' in m.available_direct_dct_profiles()"
        )
        set_tests_properties(FastLensDirectDctTorchImport PROPERTIES
                LABELS "fastlens;jpeg-dct;direct-dct;torch;smoke"
        )
        add_test(
                NAME FastLensDirectDctTorchRuntimeSmoke
                COMMAND ${CMAKE_COMMAND} -E env
                        "PYTHONPATH=$<TARGET_FILE_DIR:_fastlens_direct_dct>"
                        ${Python3_EXECUTABLE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/direct_dct_torch_runtime_smoke.py
        )
        set_tests_properties(FastLensDirectDctTorchRuntimeSmoke PROPERTIES
                LABELS "fastlens;jpeg-dct;direct-dct;torch;gpu;smoke"
                SKIP_RETURN_CODE 77
        )
        set(FASTLENS_PLS_SMOKE_FIXTURE_ARGS)
        if (TARGET fastlens_jpeg_dct_tool AND TARGET fastlens_block_major_access_tool)
                set(FASTLENS_PLS_SMOKE_FIXTURE_ARGS
                        --jpeg-tool $<TARGET_FILE:fastlens_jpeg_dct_tool>
                        --access-tool $<TARGET_FILE:fastlens_block_major_access_tool>)
        endif ()
        add_test(
                NAME FastLensDirectDctPlsTorchRuntimeSmoke
                COMMAND ${CMAKE_COMMAND} -E env
                        "PYTHONPATH=${PROJECT_SOURCE_DIR}:$<TARGET_FILE_DIR:_fastlens_direct_dct>"
                        ${Python3_EXECUTABLE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/direct_dct_pls_torch_runtime_smoke.py
                        ${FASTLENS_PLS_SMOKE_FIXTURE_ARGS}
        )
        set_tests_properties(FastLensDirectDctPlsTorchRuntimeSmoke PROPERTIES
                LABELS "fastlens;jpeg-dct;direct-dct;pls;torch;gpu;smoke"
                SKIP_RETURN_CODE 77
                TIMEOUT 300
        )
endif ()
