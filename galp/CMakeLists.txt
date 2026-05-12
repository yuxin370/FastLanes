enable_language(CUDA)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ────────────────────────────────────────────────
# 0. Global settings
# ────────────────────────────────────────────────
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

option(GALP_BUILD_TESTS "Build GALP tests" ${BUILD_TESTING})
option(GALP_BUILD_BENCHMARKS "Build GALP benchmarks" OFF)
option(GALP_BUILD_TOOLS "Build GALP command-line tools" ON)
option(GALP_WITH_NVCOMP "Build nvCOMP compressor comparison targets" OFF)
option(GALP_ENABLE_MULTI_COLUMN "Reserved; experimental multi-column generated bindings are currently disabled" OFF)
if (GALP_ENABLE_MULTI_COLUMN)
    message(FATAL_ERROR
            "GALP_ENABLE_MULTI_COLUMN is currently disabled because query_multi_column generated CUDA sources "
            "have prohibitive compile time. Leave GALP_ENABLE_MULTI_COLUMN=OFF; multi-column benchmarks are "
            "not part of the supported GALP build.")
endif ()
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

add_custom_target(galp_api_boundary_checks
	COMMAND ${CMAKE_COMMAND}
	        -DGALP_ROOT=${CMAKE_CURRENT_SOURCE_DIR}
	        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GalpPublicApiBoundaryCheck.cmake
        COMMENT "Checking GALP public API include and namespace boundaries"
        VERBATIM)

# ────────────────────────────────────────────────
# 5. GALP core static library
# ────────────────────────────────────────────────
add_subdirectory(src)

# ────────────────────────────────────────────────
# 8. tests
# ────────────────────────────────────────────────
if (GALP_BUILD_TESTS)
    add_subdirectory(test)
endif ()

## ────────────────────────────────────────────────
## 7. Benchmarks
## ────────────────────────────────────────────────
if (GALP_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif ()

## ────────────────────────────────────────────────
## 8. Tools
## ────────────────────────────────────────────────
if (GALP_BUILD_TOOLS)
    add_subdirectory(tools)
endif ()
