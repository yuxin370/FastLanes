if (NOT DEFINED FASTLENS_ROOT)
        message(FATAL_ERROR "FASTLENS_ROOT is required")
endif ()
if (NOT DEFINED FASTLENS_D2H_NEGATIVE_FIXTURE)
        message(FATAL_ERROR "FASTLENS_D2H_NEGATIVE_FIXTURE is required")
endif ()

execute_process(
        COMMAND ${CMAKE_COMMAND}
                -DFASTLENS_ROOT=${FASTLENS_ROOT}
                -DFASTLENS_D2H_EXTRA_SOURCE=${FASTLENS_D2H_NEGATIVE_FIXTURE}
                -P ${FASTLENS_ROOT}/cmake/FastLensDirectDctRuntimeNoD2H.cmake
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error)

if (result EQUAL 0)
        message(FATAL_ERROR "Direct-DCT no-D2H gate accepted an injected device-to-host transfer")
endif ()
if (NOT "${output}\n${error}" MATCHES "cudaMemcpyDeviceToHost")
        message(FATAL_ERROR
                "Direct-DCT no-D2H negative test failed for an unrelated reason:\n${output}\n${error}")
endif ()
