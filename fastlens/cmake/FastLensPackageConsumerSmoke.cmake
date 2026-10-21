if (NOT DEFINED FASTLENS_PACKAGE_PREFIX)
	message(FATAL_ERROR "FASTLENS_PACKAGE_PREFIX is required")
endif ()

if (NOT DEFINED FASTLENS_CONSUMER_BINARY_DIR)
	set(FASTLENS_CONSUMER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/fastlens-package-consumer")
endif ()

set(FASTLENS_CONSUMER_SOURCE_DIR "${FASTLENS_CONSUMER_BINARY_DIR}/src")
set(FASTLENS_CONSUMER_BUILD_DIR "${FASTLENS_CONSUMER_BINARY_DIR}/build")

# This smoke test intentionally models a clean downstream configure. Reusing a
# stale consumer cache can retain compilers or CUDA architectures from a
# different parent configuration and obscure package-export failures.
file(REMOVE_RECURSE "${FASTLENS_CONSUMER_BUILD_DIR}")
file(MAKE_DIRECTORY "${FASTLENS_CONSUMER_SOURCE_DIR}")
file(WRITE "${FASTLENS_CONSUMER_SOURCE_DIR}/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.22)
project(FastLensConsumerSmoke LANGUAGES CXX CUDA)

find_package(FastLens CONFIG REQUIRED)

add_executable(fastlens_consumer_smoke main.cpp)
target_compile_features(fastlens_consumer_smoke PRIVATE cxx_std_20)
target_link_libraries(fastlens_consumer_smoke PRIVATE FastLens::core)
]=])

file(WRITE "${FASTLENS_CONSUMER_SOURCE_DIR}/main.cpp" [=[
#include <fastlens/config.hpp>
#include <fastlens/fastlens.hpp>
#include <fastlens/jpeg_dct.hpp>

#include <filesystem>

int main() {
	fastlens::DecompressOptions options {};
	options.write_output = false;
	if (options.write_output) {
		return 1;
	}

	fastlens::Reader reader(std::filesystem::path {"__fastlens_package_consumer_missing__"} / "data.fls");
	if (reader.path().empty()) {
		return 2;
	}

	fastlens::Table table;
	if (!table.empty()) {
		return 3;
	}
#if FASTLENS_WITH_JPEG_DCT
	static_assert(FASTLENS_WITH_JPEG_DCT == 1);
#else
	static_assert(FASTLENS_WITH_JPEG_DCT == 0);
#endif
	return table.rowgroup_count() == 0 && table.total_columns() == 0 ? 0 : 4;
}
]=])

set(FASTLENS_CONSUMER_CONFIGURE_COMMAND
	${CMAKE_COMMAND}
	-S "${FASTLENS_CONSUMER_SOURCE_DIR}"
	-B "${FASTLENS_CONSUMER_BUILD_DIR}"
	-DCMAKE_PREFIX_PATH=${FASTLENS_PACKAGE_PREFIX})
if (DEFINED FASTLENS_CUDA_ARCHITECTURES AND NOT FASTLENS_CUDA_ARCHITECTURES STREQUAL "")
	list(APPEND FASTLENS_CONSUMER_CONFIGURE_COMMAND
		-DCMAKE_CUDA_ARCHITECTURES=${FASTLENS_CUDA_ARCHITECTURES})
endif ()
if (DEFINED FASTLENS_CUDA_COMPILER AND NOT FASTLENS_CUDA_COMPILER STREQUAL "")
	list(APPEND FASTLENS_CONSUMER_CONFIGURE_COMMAND
		-DCMAKE_CUDA_COMPILER=${FASTLENS_CUDA_COMPILER})
endif ()
if (DEFINED FASTLENS_CUDA_HOST_COMPILER AND NOT FASTLENS_CUDA_HOST_COMPILER STREQUAL "")
	list(APPEND FASTLENS_CONSUMER_CONFIGURE_COMMAND
		-DCMAKE_CUDA_HOST_COMPILER=${FASTLENS_CUDA_HOST_COMPILER})
endif ()
if (DEFINED FASTLENS_CXX_COMPILER AND NOT FASTLENS_CXX_COMPILER STREQUAL "")
	list(APPEND FASTLENS_CONSUMER_CONFIGURE_COMMAND
		-DCMAKE_CXX_COMPILER=${FASTLENS_CXX_COMPILER})
endif ()
if (DEFINED FASTLENS_JPEG_INCLUDE_DIR AND NOT FASTLENS_JPEG_INCLUDE_DIR STREQUAL "")
	list(APPEND FASTLENS_CONSUMER_CONFIGURE_COMMAND
		-DJPEG_INCLUDE_DIR=${FASTLENS_JPEG_INCLUDE_DIR})
endif ()
if (DEFINED FASTLENS_JPEG_LIBRARY AND NOT FASTLENS_JPEG_LIBRARY STREQUAL "")
	list(APPEND FASTLENS_CONSUMER_CONFIGURE_COMMAND
		-DJPEG_LIBRARY=${FASTLENS_JPEG_LIBRARY})
endif ()
if (DEFINED FASTLENS_THRUST_INCLUDE_DIR AND NOT FASTLENS_THRUST_INCLUDE_DIR STREQUAL "")
	list(APPEND FASTLENS_CONSUMER_CONFIGURE_COMMAND
		-DFASTLENS_THRUST_INCLUDE_DIR=${FASTLENS_THRUST_INCLUDE_DIR})
endif ()

execute_process(
	COMMAND ${FASTLENS_CONSUMER_CONFIGURE_COMMAND}
	RESULT_VARIABLE configure_result)
if (NOT configure_result EQUAL 0)
	message(FATAL_ERROR "FASTLENS package consumer configure failed")
endif ()

execute_process(
	COMMAND ${CMAKE_COMMAND} --build "${FASTLENS_CONSUMER_BUILD_DIR}" --parallel
	RESULT_VARIABLE build_result)
if (NOT build_result EQUAL 0)
	message(FATAL_ERROR "FASTLENS package consumer build failed")
endif ()
