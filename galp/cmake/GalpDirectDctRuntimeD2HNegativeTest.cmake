if (NOT DEFINED GALP_ROOT)
        message(FATAL_ERROR "GALP_ROOT is required")
endif ()
if (NOT DEFINED GALP_D2H_NEGATIVE_FIXTURE)
        message(FATAL_ERROR "GALP_D2H_NEGATIVE_FIXTURE is required")
endif ()

execute_process(
        COMMAND ${CMAKE_COMMAND}
                -DGALP_ROOT=${GALP_ROOT}
                -DGALP_D2H_EXTRA_SOURCE=${GALP_D2H_NEGATIVE_FIXTURE}
                -P ${GALP_ROOT}/cmake/GalpDirectDctRuntimeNoD2H.cmake
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
