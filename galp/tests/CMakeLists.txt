add_executable(galp_tests
        thrust_memcpy_test.cu
        thrust_core_test.cu
        galp_test.cu
		delta_test.cu
        reader_test.cu
        device_arena_test.cu
        workset_selected_vectors_test.cu
)

add_executable(galp_compact_format_tests
        compact_descriptor_v3_test.cpp)
target_link_libraries(galp_compact_format_tests PRIVATE
        galp_compact_format
        GTest::gtest_main
        FastLanes::core)
target_include_directories(galp_compact_format_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src)
apply_warnings(galp_compact_format_tests)
gtest_discover_tests(galp_compact_format_tests
        DISCOVERY_TIMEOUT 60
        PROPERTIES LABELS "galp;compact-v3;cpu")
if (Python3_Interpreter_FOUND)
        add_test(
                NAME GalpCompactV3AcceptancePython
                COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/test_compact_v3_acceptance.py)
        set_tests_properties(GalpCompactV3AcceptancePython PROPERTIES LABELS "galp;compact-v3;cpu")
        add_test(
                NAME GalpPrepareImagenet512ParallelismPython
                COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/test_prepare_imagenet512_v3_train.py)
        set_tests_properties(GalpPrepareImagenet512ParallelismPython PROPERTIES LABELS "galp;jpeg-dct;pipeline;cpu")
endif ()
if (GALP_WITH_JPEG_DCT)
	add_library(galp_direct_dct_shadow_contract STATIC
			${CMAKE_CURRENT_SOURCE_DIR}/../src/direct_dct/direct_dct_metrics.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/../src/direct_dct/native_batch_lifetime.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/../src/direct_dct/physical_layout_planner.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/../src/direct_dct/profile_registry.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/../src/direct_dct/resolved_execution_policy.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/../src/direct_dct/native_pipeline_state.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/../src/direct_dct/native_logical_batch_pipeline.cpp)
	target_link_libraries(galp_direct_dct_shadow_contract PUBLIC
			galp_jpeg_dct_device)
	target_include_directories(galp_direct_dct_shadow_contract PUBLIC
			${CMAKE_CURRENT_SOURCE_DIR}/../src)
	apply_warnings(galp_direct_dct_shadow_contract)

	add_executable(galp_direct_dct_contract_tests
			direct_dct_contract_test.cpp)
	target_link_libraries(galp_direct_dct_contract_tests PRIVATE
			galp_direct_dct_shadow_contract
			GTest::gtest_main)
	apply_warnings(galp_direct_dct_contract_tests)
	gtest_discover_tests(galp_direct_dct_contract_tests
			DISCOVERY_TIMEOUT 60
			PROPERTIES LABELS "galp;direct-dct;contract;cpu;shadow")

	add_executable(galp_native_logical_batch_pipeline_tests
			direct_dct_metrics_test.cpp
			native_batch_lifetime_test.cpp
			native_logical_batch_pipeline_test.cpp
			physical_layout_planner_test.cpp)
	target_link_libraries(galp_native_logical_batch_pipeline_tests PRIVATE
			galp_direct_dct_shadow_contract
			GTest::gtest_main)
	apply_warnings(galp_native_logical_batch_pipeline_tests)
	gtest_discover_tests(galp_native_logical_batch_pipeline_tests
			DISCOVERY_TIMEOUT 60
			PROPERTIES LABELS "galp;direct-dct;pipeline;cpu;shadow")

        add_executable(galp_jpeg_dct_decode_memory_tests
                jpeg_dct_decode_memory_test.cpp)
        target_link_libraries(galp_jpeg_dct_decode_memory_tests PRIVATE
                galp_jpeg_dct_storage
                GTest::gtest_main)
        target_include_directories(galp_jpeg_dct_decode_memory_tests PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/../src)
        apply_warnings(galp_jpeg_dct_decode_memory_tests)
        gtest_discover_tests(galp_jpeg_dct_decode_memory_tests
                DISCOVERY_TIMEOUT 60
                PROPERTIES LABELS "galp;jpeg-dct;memory;cpu")

        add_executable(galp_jpeg_dct_exact_verifier_tests
                jpeg_dct_exact_verifier_test.cpp)
        target_link_libraries(galp_jpeg_dct_exact_verifier_tests PRIVATE
                galp_core
                GTest::gtest_main
                JPEG::JPEG)
        target_include_directories(galp_jpeg_dct_exact_verifier_tests PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/../src)
        apply_warnings(galp_jpeg_dct_exact_verifier_tests)
        gtest_discover_tests(galp_jpeg_dct_exact_verifier_tests
                DISCOVERY_TIMEOUT 60
                PROPERTIES LABELS "galp;jpeg-dct;exact-verifier;cpu")

        add_executable(galp_block_major_access_tests
                jpeg_dct_block_major_access_test.cpp)
        target_link_libraries(galp_block_major_access_tests PRIVATE
				galp_core
                GTest::gtest_main
                FastLanes::core)
        target_include_directories(galp_block_major_access_tests PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/../src)
        apply_warnings(galp_block_major_access_tests)
        gtest_discover_tests(galp_block_major_access_tests
                DISCOVERY_TIMEOUT 60
                PROPERTIES LABELS "galp;jpeg-dct;block-major-access;cpu")

		add_executable(galp_direct_dct_pls_tests
		        direct_dct_pls_test.cpp)
		target_link_libraries(galp_direct_dct_pls_tests PRIVATE
		        galp_jpeg_dct_device
		        GTest::gtest_main)
		target_include_directories(galp_direct_dct_pls_tests PRIVATE
		        ${CMAKE_CURRENT_SOURCE_DIR}/../src)
		apply_warnings(galp_direct_dct_pls_tests)
		gtest_discover_tests(galp_direct_dct_pls_tests
		        DISCOVERY_TIMEOUT 60
		        PROPERTIES LABELS "galp;jpeg-dct;pls;cpu")

        target_sources(galp_tests PRIVATE
                jpeg_dct_order_test.cpp
                jpeg_dct_test.cpp
        )
endif ()

add_executable(galp_public_api_smoke
        public_api_smoke.cpp
)
if (GALP_WITH_JPEG_DCT)
        target_sources(galp_public_api_smoke PRIVATE
		direct_dct_advanced_header_smoke.cpp
		direct_dct_diagnostics_boundary_smoke.cpp
		direct_dct_legacy_header_smoke.cpp
		legacy_umbrella_header_smoke.cpp
                jpeg_dct_format_header_smoke.cpp
                jpeg_dct_storage_header_smoke.cpp
                jpeg_dct_device_header_smoke.cpp
                jpeg_dct_diagnostics_header_smoke.cpp)
endif ()

set(GALP_TEST_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
file(MAKE_DIRECTORY
        "${GALP_TEST_DATA_DIR}/floats"
        "${GALP_TEST_DATA_DIR}/doubles")

target_link_libraries(galp_tests PRIVATE
        galp_core
        galp_alp_support
        GTest::gtest_main
        FastLanes::core
)
target_include_directories(galp_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${CMAKE_CURRENT_SOURCE_DIR}/../tools/benchmark_support/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../tools/data/include)

target_compile_definitions(galp_tests PRIVATE
        FLS_GALP_SOURCE_DIR="${GALP_DATA_ROOT}"
        GALP_TEST_DATA_DIR="${GALP_TEST_DATA_DIR}"
)
if (GALP_WITH_JPEG_DCT)
        target_link_libraries(galp_tests PRIVATE
                galp_benchmark_support
                JPEG::JPEG)
endif ()

apply_warnings(galp_tests)

if (GALP_WITH_JPEG_DCT)
        target_link_libraries(galp_public_api_smoke PRIVATE Galp::jpeg_dct_device)
else ()
        target_link_libraries(galp_public_api_smoke PRIVATE Galp::engine)
endif ()
apply_warnings(galp_public_api_smoke)

add_test(NAME GalpPublicApiSmoke COMMAND galp_public_api_smoke)
set_tests_properties(GalpPublicApiSmoke PROPERTIES LABELS "galp;public-api;smoke")
add_test(
        NAME GalpPublicApiBoundaries
	COMMAND ${CMAKE_COMMAND}
	        -DGALP_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/..
	        -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/GalpPublicApiBoundaryCheck.cmake
)
set_tests_properties(GalpPublicApiBoundaries PROPERTIES LABELS "galp;public-api;boundaries")

if (GALP_WITH_JPEG_DCT)
        add_test(
                NAME GalpJpegDctNoDeviceSynchronize
                COMMAND ${CMAKE_COMMAND}
	                -DGALP_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/..
	                -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/GalpJpegDctNoDeviceSynchronize.cmake
        )
        set_tests_properties(GalpJpegDctNoDeviceSynchronize PROPERTIES LABELS "galp;jpeg-dct;static")
        add_test(
                NAME GalpDirectDctRuntimeNoD2H
                COMMAND ${CMAKE_COMMAND}
	                -DGALP_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/..
	                -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/GalpDirectDctRuntimeNoD2H.cmake
        )
        set_tests_properties(GalpDirectDctRuntimeNoD2H PROPERTIES LABELS "galp;jpeg-dct;direct-dct;static")
        add_test(
                NAME GalpDirectDctRuntimeRejectsD2H
                COMMAND ${CMAKE_COMMAND}
	                -DGALP_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/..
	                -DGALP_D2H_NEGATIVE_FIXTURE=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/direct_dct_d2h_fixture.cu
	                -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/GalpDirectDctRuntimeD2HNegativeTest.cmake
        )
        set_tests_properties(
                GalpDirectDctRuntimeRejectsD2H
                PROPERTIES LABELS "galp;jpeg-dct;direct-dct;static")
endif ()

if (GALP_ENABLE_INSTALL)
        set(GALP_PACKAGE_CONSUMER_CONFIGURE_ARGS)
        if (CMAKE_CUDA_ARCHITECTURES)
                list(APPEND GALP_PACKAGE_CONSUMER_CONFIGURE_ARGS
                        -DGALP_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES})
        endif ()
        if (CMAKE_CUDA_COMPILER)
                list(APPEND GALP_PACKAGE_CONSUMER_CONFIGURE_ARGS
                        -DGALP_CUDA_COMPILER=${CMAKE_CUDA_COMPILER})
        endif ()
        if (CMAKE_CUDA_HOST_COMPILER)
                list(APPEND GALP_PACKAGE_CONSUMER_CONFIGURE_ARGS
                        -DGALP_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER})
        endif ()
        if (CMAKE_CXX_COMPILER)
                list(APPEND GALP_PACKAGE_CONSUMER_CONFIGURE_ARGS
                        -DGALP_CXX_COMPILER=${CMAKE_CXX_COMPILER})
        endif ()
        if (GALP_WITH_JPEG_DCT AND JPEG_INCLUDE_DIR AND JPEG_LIBRARY)
                list(APPEND GALP_PACKAGE_CONSUMER_CONFIGURE_ARGS
                        -DGALP_JPEG_INCLUDE_DIR=${JPEG_INCLUDE_DIR}
                        -DGALP_JPEG_LIBRARY=${JPEG_LIBRARY})
        endif ()
        if (THRUST_INCLUDE_DIR)
                list(APPEND GALP_PACKAGE_CONSUMER_CONFIGURE_ARGS
                        -DGALP_THRUST_INCLUDE_DIR=${THRUST_INCLUDE_DIR})
        endif ()
        add_test(
                NAME GalpPackageConsumerSmoke
                COMMAND ${CMAKE_COMMAND}
	                -DGALP_PACKAGE_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/../src
	                -DGALP_CONSUMER_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/package-consumer
	                ${GALP_PACKAGE_CONSUMER_CONFIGURE_ARGS}
	                -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/GalpPackageConsumerSmoke.cmake
        )
        set_tests_properties(GalpPackageConsumerSmoke PROPERTIES LABELS "galp;public-api;package")
endif ()

gtest_discover_tests(galp_tests
        TEST_FILTER "DeviceArenaBackingRegions.*:WorksetSelectedVectors.*"
        DISCOVERY_TIMEOUT 60
        PROPERTIES LABELS "galp-workset-cpu"
)
gtest_discover_tests(galp_tests
		TEST_FILTER "ThrustMemcpy.*:ThrustCore.*:DeviceUtils.*:DeltaUnsumer.*:DeltaDecode.*:JpegDct.*:JpegDctMixedExpressions.*:JpegDctOrder.*"
        DISCOVERY_TIMEOUT 60
        PROPERTIES LABELS "galp-gpu-cuda"
)
gtest_discover_tests(galp_tests
	        TEST_FILTER "Materialize.*:Reader.*:ExternalDictionaryI16.*"
        DISCOVERY_TIMEOUT 60
        PROPERTIES LABELS "galp-gpu-reader"
)
gtest_discover_tests(galp_tests
        TEST_FILTER "ALP.*:GALP.*"
        DISCOVERY_TIMEOUT 60
        PROPERTIES LABELS "galp-gpu-integration"
)
