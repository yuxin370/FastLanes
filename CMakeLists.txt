cmake_minimum_required(VERSION 3.22)

if (NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows" AND NOT DEFINED CMAKE_CXX_COMPILER)
    find_program(CLANG_CXX NAMES clang++ REQUIRED)
    find_program(CLANG_C NAMES clang REQUIRED)
    set(CMAKE_C_COMPILER "${CLANG_C}" CACHE STRING "C Compiler")
    set(CMAKE_CXX_COMPILER "${CLANG_CXX}" CACHE STRING "C++ Compiler")
endif ()

project(FastLens LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_definitions(NOMINMAX)

include(CTest)
option(FASTLENS_BUILD_TESTS "Build FastLens tests" ${BUILD_TESTING})

# Keep the compiler settings used for the published experiments in both projects.
include(third_party/fastlanes/cmake/compiler_flags.cmake)
include(third_party/fastlanes/cmake/enable_sanitizer.cmake)
set(FLS_BUILD_GTEST ${FASTLENS_BUILD_TESTS})
add_subdirectory(third_party/fastlanes)
if (FASTLENS_BUILD_TESTS)
    enable_testing()
    include(GoogleTest)
endif ()
add_subdirectory(fastlens)
