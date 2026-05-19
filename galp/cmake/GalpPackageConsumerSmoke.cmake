if (NOT DEFINED GALP_PACKAGE_PREFIX)
	message(FATAL_ERROR "GALP_PACKAGE_PREFIX is required")
endif ()

if (NOT DEFINED GALP_CONSUMER_BINARY_DIR)
	set(GALP_CONSUMER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/galp-package-consumer")
endif ()

set(GALP_CONSUMER_SOURCE_DIR "${GALP_CONSUMER_BINARY_DIR}/src")
set(GALP_CONSUMER_BUILD_DIR "${GALP_CONSUMER_BINARY_DIR}/build")

file(MAKE_DIRECTORY "${GALP_CONSUMER_SOURCE_DIR}")
file(WRITE "${GALP_CONSUMER_SOURCE_DIR}/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.22)
project(GalpConsumerSmoke LANGUAGES CXX CUDA)

find_package(Galp CONFIG REQUIRED)

add_executable(galp_consumer_smoke main.cpp)
target_compile_features(galp_consumer_smoke PRIVATE cxx_std_20)
target_link_libraries(galp_consumer_smoke PRIVATE Galp::core)
]=])

file(WRITE "${GALP_CONSUMER_SOURCE_DIR}/main.cpp" [=[
#include <galp/config.hpp>
#include <galp/galp.hpp>
#include <galp/jpeg_dct.hpp>

#include <filesystem>

int main() {
	galp::DecompressOptions options {};
	options.write_output = false;
	if (options.write_output) {
		return 1;
	}

	galp::Reader reader(std::filesystem::path {"__galp_package_consumer_missing__"} / "data.fls");
	if (reader.path().empty()) {
		return 2;
	}

	galp::Table table;
	if (!table.empty()) {
		return 3;
	}
#if GALP_WITH_JPEG_DCT
	static_assert(GALP_WITH_JPEG_DCT == 1);
#else
	static_assert(GALP_WITH_JPEG_DCT == 0);
#endif
	return table.rowgroup_count() == 0 && table.total_columns() == 0 ? 0 : 4;
}
]=])

execute_process(
	COMMAND ${CMAKE_COMMAND}
	        -S "${GALP_CONSUMER_SOURCE_DIR}"
	        -B "${GALP_CONSUMER_BUILD_DIR}"
	        -DCMAKE_PREFIX_PATH=${GALP_PACKAGE_PREFIX}
	RESULT_VARIABLE configure_result)
if (NOT configure_result EQUAL 0)
	message(FATAL_ERROR "GALP package consumer configure failed")
endif ()

execute_process(
	COMMAND ${CMAKE_COMMAND} --build "${GALP_CONSUMER_BUILD_DIR}" --parallel
	RESULT_VARIABLE build_result)
if (NOT build_result EQUAL 0)
	message(FATAL_ERROR "GALP package consumer build failed")
endif ()
