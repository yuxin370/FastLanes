add_subdirectory(public_api_reader)
if (GALP_BUILD_TESTS AND Python3_Interpreter_FOUND)
        add_test(NAME GalpImageOrderExamplePython
                COMMAND ${Python3_EXECUTABLE} -B -m unittest
                        galp.examples.image_order_benchmark.tests.test_image_order_benchmark)
        set_tests_properties(GalpImageOrderExamplePython PROPERTIES
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" LABELS "galp;example;cpu")
endif ()
if (GALP_WITH_JPEG_DCT)
        add_subdirectory(jpeg_dct_device_batch)
endif ()
