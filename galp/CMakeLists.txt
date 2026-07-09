enable_language(CUDA)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ────────────────────────────────────────────────
# 0. Global settings
# ────────────────────────────────────────────────
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

option(GALP_BUILD_TESTS "Build GALP tests" ${BUILD_TESTING})
option(GALP_BUILD_BENCHMARKS "Build GALP benchmarks" OFF)
option(GALP_BUILD_TOOLS "Build GALP command-line tools" ON)
option(GALP_BUILD_EXAMPLES "Build GALP examples" OFF)
option(GALP_BUILD_TORCH "Build optional PyTorch direct-DCT runtime bindings" OFF)
option(GALP_WITH_NVCOMP "Build nvCOMP compressor comparison targets" OFF)
option(GALP_WITH_JPEG_DCT "Build JPEG DCT dataset reader/compressor" ON)
if (DEFINED FLS_ENABLE_INSTALL)
    set(GALP_ENABLE_INSTALL_DEFAULT ${FLS_ENABLE_INSTALL})
else ()
    set(GALP_ENABLE_INSTALL_DEFAULT ON)
endif ()
option(GALP_ENABLE_INSTALL "Generate GALP install and package targets" ${GALP_ENABLE_INSTALL_DEFAULT})

# ────────────────────────────────────────────────
# 1. CUDA Toolkit
# ────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

if (GALP_WITH_JPEG_DCT)
    find_package(JPEG QUIET)
    if (NOT JPEG_FOUND)
        find_path(GALP_JPEG_INCLUDE_DIR NAMES jpeglib.h)
        find_library(GALP_JPEG_LIBRARY NAMES jpeg)
        if (GALP_JPEG_INCLUDE_DIR AND GALP_JPEG_LIBRARY)
            add_library(JPEG::JPEG UNKNOWN IMPORTED)
            set_target_properties(JPEG::JPEG PROPERTIES
                    IMPORTED_LOCATION "${GALP_JPEG_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${GALP_JPEG_INCLUDE_DIR}")
            set(JPEG_FOUND ON)
        endif ()
    endif ()
    if (NOT JPEG_FOUND)
        message(WARNING
                "GALP_WITH_JPEG_DCT=ON but CMake could not find the libjpeg-compatible API "
                "(jpeglib.h and libjpeg.so). Disabling the JPEG DCT reader/compressor. "
                "The libturbojpeg0-dev package provides the TurboJPEG API only and is not sufficient for "
                "jpeg_read_coefficients. Install libjpeg-turbo8-dev or libjpeg-dev, or pass "
                "-DJPEG_INCLUDE_DIR=/path/to/include -DJPEG_LIBRARY=/path/to/libjpeg.so.")
        set(GALP_WITH_JPEG_DCT OFF)
    endif ()
endif ()

# ────────────────────────────────────────────────
# 2. nvCOMP (optional; required only for compressor comparison targets)
# ────────────────────────────────────────────────
if (GALP_WITH_NVCOMP)
    find_package(nvcomp CONFIG REQUIRED)

    get_target_property(NVCOMP_VERSION nvcomp::nvcomp VERSION)
    get_target_property(NVCOMP_INC_DIRS nvcomp::nvcomp INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(NVCOMP_LIBS nvcomp::nvcomp INTERFACE_LINK_LIBRARIES)

    message(STATUS "nvCOMP version ..........: ${NVCOMP_VERSION}")
    message(STATUS "nvCOMP include dirs .....: ${NVCOMP_INC_DIRS}")
    message(STATUS "nvCOMP libs .............: ${NVCOMP_LIBS}")
endif ()

# ────────────────────────────────────────────────
# 3. Thrust / CCCL – prefer toolkit copy, fall back to CCCL ≥ 2.3
# ────────────────────────────────────────────────
find_path(THRUST_INCLUDE_DIR
        NAMES thrust/version.h
        thrust/device_uvector.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS})

if (NOT THRUST_INCLUDE_DIR)
    message(FATAL_ERROR "Need Thrust ≥ 2.2 (with device_uvector.h) in CUDA Toolkit")
endif ()

add_library(Thrust::Thrust INTERFACE IMPORTED)
target_include_directories(Thrust::Thrust INTERFACE ${THRUST_INCLUDE_DIR})

# ────────────────────────────────────────────────
# 4. User-tunable options & warnings
# ────────────────────────────────────────────────
option(GALP_VERBOSE_BUILD "Print every compile/link command" OFF)
if (GALP_VERBOSE_BUILD)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif ()

set(COMMON_WARNINGS "")
set(CLANG_ONLY_WARNINGS
        -Wno-c++98-compat-local-type-template-args
        -Wno-c++98-compat-pedantic
        -Wno-c++98-compat
        -Wno-padded
        -Wno-float-equal
        -Wno-global-constructors
        -Wno-exit-time-destructors
        -Wno-float-conversion
        -Wno-sign-conversion
        -Wno-implicit-int-float-conversion
        -Wno-unused-parameter
        -Wno-unused-variable
)

# set(NVCC_ANALYZE_FLAGS
#     --resource-usage
#     --keep
#     --keep-dir=${CMAKE_BINARY_DIR}/nvcc_keep
#     -Xptxas=-v
# )


macro(apply_warnings tgt)
	    target_compile_options(${tgt} PRIVATE
	            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo -src-in-ptx --diag-suppress=550>
	            $<$<COMPILE_LANGUAGE:CXX>:${COMMON_WARNINGS}>
	            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Clang>>:${CLANG_ONLY_WARNINGS}>
	            $<$<COMPILE_LANGUAGE:CUDA>:${COMMON_WARNINGS}>)
endmacro()

# Sample data root for tests and benchmarks. Attach the definition to those
# targets instead of the core library to keep source paths out of installed
# artifacts.
set(GALP_DATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH
    "Root directory containing data/floats and data/doubles for galp tests and benchmarks")

find_package(Python3 COMPONENTS Interpreter QUIET)
if (Python3_Interpreter_FOUND)
        set(GALP_INCLUDE_BOUNDARY_CHECK_COMMAND
                COMMAND ${Python3_EXECUTABLE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/check_include_boundaries.py
                        --repo-root ${CMAKE_CURRENT_SOURCE_DIR}/..
                        --galp-root ${CMAKE_CURRENT_SOURCE_DIR})
        add_custom_target(galp_include_boundary_checks
                ${GALP_INCLUDE_BOUNDARY_CHECK_COMMAND}
                COMMENT "Checking GALP include boundaries"
                VERBATIM)
else ()
        set(GALP_INCLUDE_BOUNDARY_CHECK_COMMAND)
        message(STATUS "Python3 interpreter not found; galp_include_boundary_checks target is unavailable")
endif ()

add_custom_target(galp_api_boundary_checks
	COMMAND ${CMAKE_COMMAND}
	        -DGALP_ROOT=${CMAKE_CURRENT_SOURCE_DIR}
	        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GalpPublicApiBoundaryCheck.cmake
        ${GALP_INCLUDE_BOUNDARY_CHECK_COMMAND}
        COMMENT "Checking GALP public API and include boundaries"
        VERBATIM)

# ────────────────────────────────────────────────
# 5. GALP core static library
# ────────────────────────────────────────────────
add_subdirectory(src)

# ────────────────────────────────────────────────
# 6. Optional extension libraries
# ────────────────────────────────────────────────
add_subdirectory(extensions/alp)

# ────────────────────────────────────────────────
# 7. Shared benchmark support
# ────────────────────────────────────────────────
if (GALP_BUILD_TOOLS OR (GALP_BUILD_TESTS AND GALP_WITH_JPEG_DCT))
    add_subdirectory(tools/benchmark_support)
endif ()

# ────────────────────────────────────────────────
# 8. Tests
# ────────────────────────────────────────────────
if (GALP_BUILD_TESTS)
    add_subdirectory(tests)
endif ()

## ────────────────────────────────────────────────
## 9. Benchmarks
## ────────────────────────────────────────────────
if (GALP_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif ()

## ────────────────────────────────────────────────
## 10. Tools
## ────────────────────────────────────────────────
if (GALP_BUILD_TOOLS)
    add_subdirectory(tools)
endif ()

## ────────────────────────────────────────────────
## 11. Examples
## ────────────────────────────────────────────────
if (GALP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

## ────────────────────────────────────────────────
## 12. Optional PyTorch bindings
## ────────────────────────────────────────────────
if (GALP_BUILD_TORCH)
    if (NOT GALP_WITH_JPEG_DCT)
        message(FATAL_ERROR "GALP_BUILD_TORCH requires GALP_WITH_JPEG_DCT=ON")
    endif ()
    add_subdirectory(torch)
endif ()
