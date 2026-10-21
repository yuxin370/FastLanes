include_guard(GLOBAL)
include(CheckCXXCompilerFlag)

# Flags : --------------------------------------------------------------------------------------------------------------
message("---------------------------------------------------------------------------------------------------------")
message("-- FLS: detecting flags.")

if (MSVC)
    # MSVC auto-targets the host architecture. No manual arch flags needed.
    message(STATUS "-- FLS: MSVC build — no arch flags applied.")

    # Warning level: /W3 for initial port (tighten to /W4 /WX later).
    # /Zc:__cplusplus is needed so __cplusplus reports the correct standard version.
    # /bigobj is needed for large translation units (e.g. materializer.cpp).
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /Zc:__cplusplus /bigobj /MP")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3")
else ()
    # First, check if running on GitHub Actions
    if (DEFINED ENV{GITHUB_ACTIONS} AND "$ENV{GITHUB_ACTIONS}" STREQUAL "true")
        message(STATUS "Running on GitHub Actions runner.")
        set(IS_GITHUB_ACTIONS ON)
    endif ()

    # Check if the compiler supports -mavx512dq
    check_cxx_compiler_flag("-mavx512dq" COMPILER_SUPPORTS_AVX512DQ)

    # Detect AVX-512DQ hardware support via /proc/cpuinfo
    set(HAS_HW_AVX512DQ OFF)
    if (EXISTS "/proc/cpuinfo")
        file(READ "/proc/cpuinfo" CPUINFO_CONTENT)
        if (CPUINFO_CONTENT MATCHES "avx512dq")
            set(HAS_HW_AVX512DQ ON)
        endif ()
    endif ()

    if (COMPILER_SUPPORTS_AVX512DQ AND HAS_HW_AVX512DQ AND NOT IS_GITHUB_ACTIONS)
        message(STATUS "Compiler and hardware both support AVX-512DQ. Adding flag '-mavx512dq'.")
        set(FLAGS "-mavx512dq")
    elseif (COMPILER_SUPPORTS_AVX512DQ AND HAS_HW_AVX512DQ AND IS_GITHUB_ACTIONS)
        message(WARNING "Hardware supports AVX-512DQ, but not adding on GitHub Actions runner.")
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i[3-6]86")
        message(STATUS "Setting '-march=native' for x86 processors without AVX-512DQ.")
        set(FLAGS "-march=native")
    else ()
        message(STATUS "No instruction set flags applied.")
    endif ()

    # Append the computed FLAGS to the compiler flags if defined
    if (FLAGS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")
    endif ()

    # Flags for warnings and errors (common to Clang and GCC):
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wnon-virtual-dtor -Wunused -Wpedantic -Woverloaded-virtual")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")

    # Clang-only warning flags (GCC equivalents are covered by clang-tidy):
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow -Wconversion -Winconsistent-missing-override -Wshorten-64-to-32")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wshadow -Wconversion")
    endif ()
    # GCC-only warning flags:
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wsuggest-override")
    endif ()

    # On Windows (Clang, not MSVC), vectorize-width pragmas may not be honoured
    # because the CI runners lack AVX-512.  Demote the transform-warning to a
    # non-fatal warning so -Werror does not reject advisory hints.
    if (WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-pass-failed")
    endif ()
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("---------------------------------------------------------------------------------------------------------")
    message("-- FLS: Debug mode enabled. Adding -g and -O0.")
    if (NOT MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
        set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0")
    endif ()
endif ()
