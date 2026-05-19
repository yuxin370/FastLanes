add_executable(galp_tests
        thrust_memcpy_test.cu
        thrust_core_test.cu
        galp_test.cu
        reader_test.cu
        jpeg_dct_order_test.cpp
)
if (GALP_WITH_JPEG_DCT)
        target_sources(galp_tests PRIVATE
                jpeg_dct_test.cpp
        )
endif ()

add_executable(galp_public_api_smoke
        public_api_smoke.cpp
)

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
        ${CMAKE_CURRENT_SOURCE_DIR}/../tools/data/include)

target_compile_definitions(galp_tests PRIVATE
        FLS_GALP_SOURCE_DIR="${GALP_DATA_ROOT}"
        GALP_TEST_DATA_DIR="${GALP_TEST_DATA_DIR}"
)
if (GALP_WITH_JPEG_DCT)
        target_link_libraries(galp_tests PRIVATE JPEG::JPEG)
endif ()

apply_warnings(galp_tests)

target_link_libraries(galp_public_api_smoke PRIVATE
        Galp::core
)
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

if (GALP_ENABLE_INSTALL)
        add_test(
                NAME GalpPackageConsumerSmoke
                COMMAND ${CMAKE_COMMAND}
	                -DGALP_PACKAGE_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/../src
	                -DGALP_CONSUMER_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/package-consumer
	                -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/GalpPackageConsumerSmoke.cmake
        )
        set_tests_properties(GalpPackageConsumerSmoke PROPERTIES LABELS "galp;public-api;package")
endif ()

gtest_discover_tests(galp_tests
        TEST_FILTER "ThrustMemcpy.*:ThrustCore.*:DeviceUtils.*:JpegDct.*:JpegDctOrder.*"
        DISCOVERY_TIMEOUT 60
        PROPERTIES LABELS "galp-gpu-cuda"
)
gtest_discover_tests(galp_tests
        TEST_FILTER "Materialize.*:Reader.*"
        DISCOVERY_TIMEOUT 60
        PROPERTIES LABELS "galp-gpu-reader"
)
gtest_discover_tests(galp_tests
        TEST_FILTER "ALP.*:GALP.*"
        DISCOVERY_TIMEOUT 60
        PROPERTIES LABELS "galp-gpu-integration"
)
