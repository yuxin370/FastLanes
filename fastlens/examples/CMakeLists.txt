add_subdirectory(public_api_reader)
if (FASTLENS_BUILD_TESTS AND Python3_Interpreter_FOUND)
        add_test(NAME FastLensImageOrderExamplePython
                COMMAND ${Python3_EXECUTABLE} -B -m unittest
                        fastlens.examples.image_order_benchmark.tests.test_image_order_benchmark)
        set_tests_properties(FastLensImageOrderExamplePython PROPERTIES
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" LABELS "fastlens;example;cpu")
endif ()
if (FASTLENS_WITH_JPEG_DCT)
        add_subdirectory(jpeg_dct_device_batch)
endif ()
