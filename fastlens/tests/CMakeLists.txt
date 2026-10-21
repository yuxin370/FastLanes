add_executable(fastlens_tests
        thrust_memcpy_test.cu
        thrust_core_test.cu
        fastlens_test.cu
		delta_test.cu
        reader_test.cu
        device_arena_test.cu
        workset_selected_vectors_test.cu
)

# Fault injection uses CUDA declarations but its own runtime symbols, so these
# ownership tests run on CPU and do not depend on a working GPU/driver.
add_executable(fastlens_cuda_transfer_failure_tests cuda_transfer_failure_test.cpp)
target_include_directories(fastlens_cuda_transfer_failure_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(fastlens_cuda_transfer_failure_tests PRIVATE GTest::gtest_main)
apply_warnings(fastlens_cuda_transfer_failure_tests)
gtest_discover_tests(fastlens_cuda_transfer_failure_tests
		DISCOVERY_TIMEOUT 60 PROPERTIES LABELS "fastlens-cpu")

add_executable(fastlens_compact_format_tests
        compact_descriptor_v3_test.cpp)
target_link_libraries(fastlens_compact_format_tests PRIVATE
        fastlens_compact_format
        GTest::gtest_main
        FastLanes::core)
target_include_directories(fastlens_compact_format_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src)
apply_warnings(fastlens_compact_format_tests)
gtest_discover_tests(fastlens_compact_format_tests
        DISCOVERY_TIMEOUT 60
	PROPERTIES LABELS "fastlens-cpu")
if (Python3_Interpreter_FOUND)
	add_test(
		NAME FastLensPublicDirectDctApiPython
		COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_SOURCE_DIR}
		        ${Python3_EXECUTABLE} -B -m unittest fastlens.torch.tests.test_public_direct_dct_api)
	set_tests_properties(FastLensPublicDirectDctApiPython PROPERTIES LABELS "fastlens;direct-dct;cpu")
        add_test(NAME FastLensExperimentalDirectDctPlsApiPython
                COMMAND ${Python3_EXECUTABLE} -B -m unittest
                        fastlens.torch.tests.test_experimental_direct_dct_pls_api)
        set_tests_properties(FastLensExperimentalDirectDctPlsApiPython PROPERTIES
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" LABELS "fastlens;direct-dct;cpu")
        add_test(
                NAME FastLensCompactV3AcceptancePython
                COMMAND ${Python3_EXECUTABLE} -B -m unittest
                        fastlens.tools.jpeg_dct.tests.test_compact_v3_acceptance)
        set_tests_properties(FastLensCompactV3AcceptancePython PROPERTIES
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" LABELS "fastlens;compact-v3;cpu")
        set_tests_properties(FastLensCompactV3AcceptancePython PROPERTIES
                ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}")
        add_test(NAME FastLensCodegenTemporaryDirectoryPython
                COMMAND ${Python3_EXECUTABLE} -B -m unittest
                        fastlens.scripts.codegen.tests.test_codegen_tmp_safety)
        set_tests_properties(FastLensCodegenTemporaryDirectoryPython PROPERTIES
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" LABELS "fastlens;codegen;cpu")
endif ()
if (FASTLENS_WITH_JPEG_DCT)
	add_library(fastlens_direct_dct_shadow_contract STATIC
			${CMAKE_CURRENT_SOURCE_DIR}/../src/direct_dct/direct_dct_metrics.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/../src/direct_dct/native_batch_lifetime.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/../src/direct_dct/physical_layout_planner.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/../src/direct_dct/profile_registry.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/../src/direct_dct/resolved_execution_policy.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/../src/direct_dct/native_pipeline_state.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/../src/direct_dct/native_logical_batch_pipeline.cpp)
	target_link_libraries(fastlens_direct_dct_shadow_contract PUBLIC
			fastlens_jpeg_dct_device)
	target_include_directories(fastlens_direct_dct_shadow_contract PUBLIC
			${CMAKE_CURRENT_SOURCE_DIR}/../src)
	apply_warnings(fastlens_direct_dct_shadow_contract)

	add_executable(fastlens_direct_dct_contract_tests
			direct_dct_contract_test.cpp)
	target_link_libraries(fastlens_direct_dct_contract_tests PRIVATE
			fastlens_direct_dct_shadow_contract
			GTest::gtest_main)
	apply_warnings(fastlens_direct_dct_contract_tests)
	gtest_discover_tests(fastlens_direct_dct_contract_tests
			DISCOVERY_TIMEOUT 60
			PROPERTIES LABELS "fastlens-cpu")

	add_executable(fastlens_native_logical_batch_pipeline_tests
			direct_dct_metrics_test.cpp
			native_batch_lifetime_test.cpp
			native_logical_batch_pipeline_test.cpp
			physical_layout_planner_test.cpp)
	target_link_libraries(fastlens_native_logical_batch_pipeline_tests PRIVATE
			fastlens_direct_dct_shadow_contract
			GTest::gtest_main)
	apply_warnings(fastlens_native_logical_batch_pipeline_tests)
	gtest_discover_tests(fastlens_native_logical_batch_pipeline_tests
			DISCOVERY_TIMEOUT 60
			PROPERTIES LABELS "fastlens-cpu")

        add_executable(fastlens_jpeg_dct_decode_memory_tests
                jpeg_dct_decode_memory_test.cpp)
        target_link_libraries(fastlens_jpeg_dct_decode_memory_tests PRIVATE
                fastlens_jpeg_dct_storage
                GTest::gtest_main)
        target_include_directories(fastlens_jpeg_dct_decode_memory_tests PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/../src)
        apply_warnings(fastlens_jpeg_dct_decode_memory_tests)
        gtest_discover_tests(fastlens_jpeg_dct_decode_memory_tests
                DISCOVERY_TIMEOUT 60
                PROPERTIES LABELS "fastlens-cpu")

        add_executable(fastlens_jpeg_dct_exact_verifier_tests
                jpeg_dct_exact_verifier_test.cpp)
        target_link_libraries(fastlens_jpeg_dct_exact_verifier_tests PRIVATE
                fastlens_core
                GTest::gtest_main
                JPEG::JPEG)
        target_include_directories(fastlens_jpeg_dct_exact_verifier_tests PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/../src)
        apply_warnings(fastlens_jpeg_dct_exact_verifier_tests)
        gtest_discover_tests(fastlens_jpeg_dct_exact_verifier_tests
                DISCOVERY_TIMEOUT 60
                PROPERTIES LABELS "fastlens-cpu")

        add_executable(fastlens_block_major_access_tests
                jpeg_dct_block_major_access_test.cpp)
        target_link_libraries(fastlens_block_major_access_tests PRIVATE
				fastlens_core
                GTest::gtest_main
                FastLanes::core)
        target_include_directories(fastlens_block_major_access_tests PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/../src)
        apply_warnings(fastlens_block_major_access_tests)
        gtest_discover_tests(fastlens_block_major_access_tests
                DISCOVERY_TIMEOUT 60
                PROPERTIES LABELS "fastlens-cpu")

		add_executable(fastlens_direct_dct_pls_tests
		        direct_dct_pls_test.cpp)
		target_link_libraries(fastlens_direct_dct_pls_tests PRIVATE
		        fastlens_jpeg_dct_device
		        GTest::gtest_main)
		target_include_directories(fastlens_direct_dct_pls_tests PRIVATE
		        ${CMAKE_CURRENT_SOURCE_DIR}/../src)
		apply_warnings(fastlens_direct_dct_pls_tests)
		gtest_discover_tests(fastlens_direct_dct_pls_tests
		        DISCOVERY_TIMEOUT 60
		        PROPERTIES LABELS "fastlens-cpu")

        target_sources(fastlens_tests PRIVATE
                jpeg_dct_order_test.cpp
                jpeg_dct_test.cpp
        )
endif ()

add_executable(fastlens_public_api_smoke
        public_api_smoke.cpp
)
if (FASTLENS_WITH_JPEG_DCT)
        target_sources(fastlens_public_api_smoke PRIVATE
		direct_dct_advanced_header_smoke.cpp
		direct_dct_diagnostics_boundary_smoke.cpp
		direct_dct_legacy_header_smoke.cpp
		legacy_umbrella_header_smoke.cpp
                jpeg_dct_format_header_smoke.cpp
                jpeg_dct_storage_header_smoke.cpp
                jpeg_dct_device_header_smoke.cpp
                jpeg_dct_diagnostics_header_smoke.cpp)
endif ()

set(FASTLENS_TEST_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
file(MAKE_DIRECTORY
        "${FASTLENS_TEST_DATA_DIR}/floats"
        "${FASTLENS_TEST_DATA_DIR}/doubles")

target_link_libraries(fastlens_tests PRIVATE
        fastlens_core
        fastlens_alp_support
        GTest::gtest_main
        FastLanes::core
)
target_include_directories(fastlens_tests PRIVATE
        ${FastLanes_SOURCE_DIR}/data/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${CMAKE_CURRENT_SOURCE_DIR}/../tools/benchmark_support/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../tools/data/include)

target_compile_definitions(fastlens_tests PRIVATE
        FLS_CMAKE_SOURCE_DIR="${FastLanes_SOURCE_DIR}"
        FLS_FASTLENS_SOURCE_DIR="${FASTLENS_DATA_ROOT}"
        FASTLENS_TEST_DATA_DIR="${FASTLENS_TEST_DATA_DIR}"
)
if (FASTLENS_WITH_JPEG_DCT)
        target_link_libraries(fastlens_tests PRIVATE
                fastlens_benchmark_support
                JPEG::JPEG)
endif ()

apply_warnings(fastlens_tests)

if (FASTLENS_WITH_JPEG_DCT)
        target_link_libraries(fastlens_public_api_smoke PRIVATE FastLens::jpeg_dct_device)
else ()
        target_link_libraries(fastlens_public_api_smoke PRIVATE FastLens::engine)
endif ()
apply_warnings(fastlens_public_api_smoke)

add_test(NAME FastLensPublicApiSmoke COMMAND fastlens_public_api_smoke)
set_tests_properties(FastLensPublicApiSmoke PROPERTIES LABELS "fastlens;public-api;smoke")
add_test(
        NAME FastLensPublicApiBoundaries
	COMMAND ${CMAKE_COMMAND}
	        -DFASTLENS_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/..
	        -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/FastLensPublicApiBoundaryCheck.cmake
)
set_tests_properties(FastLensPublicApiBoundaries PROPERTIES LABELS "fastlens;public-api;boundaries")

if (FASTLENS_WITH_JPEG_DCT)
        add_test(
                NAME FastLensJpegDctNoDeviceSynchronize
                COMMAND ${CMAKE_COMMAND}
	                -DFASTLENS_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/..
	                -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/FastLensJpegDctNoDeviceSynchronize.cmake
        )
        set_tests_properties(FastLensJpegDctNoDeviceSynchronize PROPERTIES LABELS "fastlens;jpeg-dct;static")
        add_test(
                NAME FastLensDirectDctRuntimeNoD2H
                COMMAND ${CMAKE_COMMAND}
	                -DFASTLENS_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/..
	                -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/FastLensDirectDctRuntimeNoD2H.cmake
        )
        set_tests_properties(FastLensDirectDctRuntimeNoD2H PROPERTIES LABELS "fastlens;jpeg-dct;direct-dct;static")
        add_test(
                NAME FastLensDirectDctRuntimeRejectsD2H
                COMMAND ${CMAKE_COMMAND}
	                -DFASTLENS_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/..
	                -DFASTLENS_D2H_NEGATIVE_FIXTURE=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/direct_dct_d2h_fixture.cu
	                -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/FastLensDirectDctRuntimeD2HNegativeTest.cmake
        )
        set_tests_properties(
                FastLensDirectDctRuntimeRejectsD2H
                PROPERTIES LABELS "fastlens;jpeg-dct;direct-dct;static")
endif ()

if (FASTLENS_ENABLE_INSTALL)
        set(FASTLENS_PACKAGE_CONSUMER_CONFIGURE_ARGS)
        if (CMAKE_CUDA_ARCHITECTURES)
                list(APPEND FASTLENS_PACKAGE_CONSUMER_CONFIGURE_ARGS
                        -DFASTLENS_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES})
        endif ()
        if (CMAKE_CUDA_COMPILER)
                list(APPEND FASTLENS_PACKAGE_CONSUMER_CONFIGURE_ARGS
                        -DFASTLENS_CUDA_COMPILER=${CMAKE_CUDA_COMPILER})
        endif ()
        if (CMAKE_CUDA_HOST_COMPILER)
                list(APPEND FASTLENS_PACKAGE_CONSUMER_CONFIGURE_ARGS
                        -DFASTLENS_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER})
        endif ()
        if (CMAKE_CXX_COMPILER)
                list(APPEND FASTLENS_PACKAGE_CONSUMER_CONFIGURE_ARGS
                        -DFASTLENS_CXX_COMPILER=${CMAKE_CXX_COMPILER})
        endif ()
        if (FASTLENS_WITH_JPEG_DCT AND JPEG_INCLUDE_DIR AND JPEG_LIBRARY)
                list(APPEND FASTLENS_PACKAGE_CONSUMER_CONFIGURE_ARGS
                        -DFASTLENS_JPEG_INCLUDE_DIR=${JPEG_INCLUDE_DIR}
                        -DFASTLENS_JPEG_LIBRARY=${JPEG_LIBRARY})
        endif ()
        if (THRUST_INCLUDE_DIR)
                list(APPEND FASTLENS_PACKAGE_CONSUMER_CONFIGURE_ARGS
                        -DFASTLENS_THRUST_INCLUDE_DIR=${THRUST_INCLUDE_DIR})
        endif ()
        add_test(
                NAME FastLensPackageConsumerSmoke
                COMMAND ${CMAKE_COMMAND}
	                -DFASTLENS_PACKAGE_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/../src
	                -DFASTLENS_CONSUMER_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/package-consumer
	                ${FASTLENS_PACKAGE_CONSUMER_CONFIGURE_ARGS}
	                -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/FastLensPackageConsumerSmoke.cmake
        )
        set_tests_properties(FastLensPackageConsumerSmoke PROPERTIES LABELS "fastlens;public-api;package")
endif ()

gtest_discover_tests(fastlens_tests
        TEST_FILTER "DeviceArenaBackingRegions.*:WorksetSelectedVectors.*"
        DISCOVERY_TIMEOUT 60
        PROPERTIES LABELS "fastlens-workset-cpu"
)
gtest_discover_tests(fastlens_tests
	TEST_FILTER "CompactSegment.*:GPUArray.*"
	DISCOVERY_TIMEOUT 60
	PROPERTIES LABELS "fastlens-cpu"
)
gtest_discover_tests(fastlens_tests
		TEST_FILTER "ThrustMemcpy.*:ThrustCore.*:DeviceUtils.*:DevicePool.*:DeltaUnsumer.*:DeltaDecode.*:JpegDct.*:JpegDctMixedExpressions.*:JpegDctOrder.*"
        DISCOVERY_TIMEOUT 60
        PROPERTIES LABELS "fastlens-gpu-cuda"
)
gtest_discover_tests(fastlens_tests
	        TEST_FILTER "Materialize.*:Reader.*:ExternalDictionaryI16.*"
        DISCOVERY_TIMEOUT 60
        PROPERTIES LABELS "fastlens-gpu-reader"
)
gtest_discover_tests(fastlens_tests
        TEST_FILTER "ALP.*:FASTLENS.*"
        DISCOVERY_TIMEOUT 60
        PROPERTIES LABELS "fastlens-gpu-integration"
)
