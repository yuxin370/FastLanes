// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/test/galp_test.cu
// ────────────────────────────────────────────────────────
#include "alp/alp-bindings.cuh"
#include "data/fastlanes_data.hpp"
#include "engine/device-utils.cuh"
#include "engine/kernels/dispatch.cuh"
#include "decompression/alp.cuh"
#include "compression/columns/all.cuh"
#include "galp_support/data/generate_binaries.hpp"
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <type_traits>
#include <utility> // for std::pair
#include <vector>

template <typename T>
std::vector<T> read_file(const std::string& path) {
	// Open file in binary mode, positioned at end to get its size
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		throw std::runtime_error("Could not open file: " + path);
	}

	// Determine file size in bytes
	std::streamsize bytes = file.tellg();
	if (bytes < 0) {
		throw std::runtime_error("Could not determine file size: " + path);
	}

	// Ensure the file contains an integral number of T elements
	if (bytes % sizeof(T) != 0) {
		throw std::runtime_error("File size (" + std::to_string(bytes) + " bytes) is not a multiple of element size (" +
		                         std::to_string(sizeof(T)) + " bytes)");
	}

	// Calculate number of elements
	std::size_t count = static_cast<std::size_t>(bytes / sizeof(T));

	// Seek back to beginning and read all data into a vector
	file.seekg(0, std::ios::beg);
	std::vector<T> data(count);
	if (!file.read(reinterpret_cast<char*>(data.data()), bytes)) {
		throw std::runtime_error("Error reading file: " + path);
	}

	return data;
}

// -----------------------------------------------------------------------------
// CSV loader – light‑weight, header‑only
// -----------------------------------------------------------------------------
template <typename T>
std::vector<T> read_csv(const std::filesystem::path& path) {
	std::ifstream file(path);
	if (!file) {
		throw std::runtime_error("Could not open csv file: " + path.string());
	}

	std::vector<T> data;
	std::string    line;
	while (std::getline(file, line)) {
		std::stringstream ss(line);
		std::string       cell;
		while (std::getline(ss, cell, ',')) {
			if (!cell.empty()) {
				data.push_back(static_cast<T>(std::stod(cell)));
			}
		}
	}
	return data;
}

template <typename T>
bool check_if_device_buffers_are_equal(const T* a, const T* b, const size_t n_values) {
	// Convert to uint8_t as we don't want to compare floats (-nan == -nan =>
	// false)
	thrust::device_ptr<const uint8_t> d_a(reinterpret_cast<const uint8_t*>(a));
	thrust::device_ptr<const uint8_t> d_b(reinterpret_cast<const uint8_t*>(b));

	return thrust::equal(d_a, d_a + n_values * sizeof(T), d_b);
}

std::filesystem::path generated_data_root(const char* test_name) {
	auto root = std::filesystem::path {GALP_TEST_DATA_DIR} / test_name;
	std::filesystem::create_directories(root / "floats");
	std::filesystem::create_directories(root / "doubles");
	return root;
}

std::string cuda_unavailable_reason() {
	int        device_count = 0;
	const auto status       = cudaGetDeviceCount(&device_count);
	if (status != cudaSuccess) {
		return std::string("CUDA device not available for GALP test: ") + cudaGetErrorString(status);
	}
	if (device_count <= 0) {
		return "CUDA device not available for GALP test: device count is zero";
	}
	return {};
}

template <typename T, unsigned UNPACK_N_VECTORS>
using ALPDecompressor = typename galp::codec::device::ALPDecompressor<
    T,
    UNPACK_N_VECTORS,
    galp::codec::device::
        BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, 1, galp::codec::device::ALPFunctor<T, UNPACK_N_VECTORS>>,
    galp::codec::device::StatefulALPExceptionPatcher<T, UNPACK_N_VECTORS, 1>,
    galp::codec::device::ALPColumn<T>>;

template <typename T, unsigned UNPACK_N_VECTORS>
using ALPExtendedDecompressor = typename galp::codec::device::ALPDecompressor<
    T,
    UNPACK_N_VECTORS,
    galp::codec::device::
        BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, 1, galp::codec::device::ALPFunctor<T, UNPACK_N_VECTORS>>,
    galp::codec::device::PrefetchAllALPExceptionPatcher<T, UNPACK_N_VECTORS, 1>,
    galp::codec::device::ALPExtendedColumn<T>>;

struct CLIArgs;
template <typename T>
inline void test_alp(const std::filesystem::path& path) {
	auto data_vec = read_file<T>(path);

	galp::codec::host::ALPColumn<T> column = galp::codec::alp::encode(data_vec.data(), data_vec.size(), true);
	GPUArray<T>                     d_decompression_result(data_vec.size());
	constexpr int32_t               UNPACK_N_VECTORS = 1;
	const ThreadblockMapping<T>     mapping(UNPACK_N_VECTORS, column.get_n_vecs());

	galp::codec::device::ALPColumn<T> d_column = column.copy_to_device();
	galp::kernels::device::decompress_column<T,
	                                         UNPACK_N_VECTORS,
	                                         1,
	                                         ALPDecompressor<T, UNPACK_N_VECTORS>,
	                                         galp::codec::device::ALPColumn<T>>
	    <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(d_column, d_decompression_result.get());

	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	galp::codec::host::free_column(d_column);

	bool        kernel_successful = false;
	GPUArray<T> d_input(data_vec.size(), data_vec.data());

	kernel_successful =
	    check_if_device_buffers_are_equal<T>(d_decompression_result.get(), d_input.get(), column.get_n_values());

	EXPECT_TRUE(kernel_successful);

	galp::codec::host::free_column(column);
}

template <typename T>
inline void test_galp_data(const std::vector<T>& data_vec) {
	galp::codec::host::ALPColumn<T>           column = galp::codec::alp::encode(data_vec.data(), data_vec.size(), true);
	galp::codec::host::ALPExtendedColumn<T>   column_extended = column.create_extended_column();
	galp::codec::device::ALPExtendedColumn<T> d_column        = column_extended.copy_to_device();
	GPUArray<T>                               d_decompression_result(data_vec.size());
	constexpr int32_t                         UNPACK_N_VECTORS = 1;
	const ThreadblockMapping<T>               mapping(UNPACK_N_VECTORS, column.get_n_vecs());
	galp::kernels::device::decompress_column<T,
	                                         UNPACK_N_VECTORS,
	                                         1,
	                                         ALPExtendedDecompressor<T, UNPACK_N_VECTORS>,
	                                         galp::codec::device::ALPExtendedColumn<T>>
	    <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(d_column, d_decompression_result.get());

	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	double compression_ratio = column_extended.get_compression_ratio();
	std::cout << "compression_ratio : " << compression_ratio << std::endl;
	GPUArray<T> d_input(data_vec.size(), data_vec.data());
	EXPECT_TRUE(
	    check_if_device_buffers_are_equal<T>(d_decompression_result.get(), d_input.get(), column.get_n_values()));

	galp::codec::host::free_column(column);
	galp::codec::host::free_column(column_extended);
	galp::codec::host::free_column(d_column);
}

struct CLIArgs;
template <typename T>
inline void test_galp(const std::filesystem::path& path) {
	test_galp_data<T>(read_file<T>(path));
}

template <typename T>
inline void test_galp_csv(const std::filesystem::path& path) {
	test_galp_data<T>(read_csv<T>(path));
}

TEST(DeviceUtils, ThreadblockMappingCoversTailVectors) {
	for (const size_t unpack_n_vecs : {1U, 2U, 4U, 8U}) {
		for (const size_t n_vecs : {1U, 3U, 7U, 9U, 33U}) {
			const ThreadblockMapping<int8_t> mapping(unpack_n_vecs, n_vecs);
			const size_t                     capacity =
			    unpack_n_vecs * static_cast<size_t>(ThreadblockMapping<int8_t>::N_CONCURRENT_VECTORS_PER_BLOCK);
			const size_t expected = std::max<size_t>(1, (n_vecs + capacity - 1U) / capacity);
			EXPECT_EQ(mapping.n_blocks, expected) << "unpack_n_vecs=" << unpack_n_vecs << " n_vecs=" << n_vecs;
			EXPECT_GE(static_cast<size_t>(mapping.n_blocks) * capacity, n_vecs)
			    << "unpack_n_vecs=" << unpack_n_vecs << " n_vecs=" << n_vecs;
		}
	}
}

TEST(DeviceUtils, MultiVectorTailSplitKeepsFullGroupsAligned) {
	for (const size_t n_vecs : {1U, 3U, 7U, 9U, 33U}) {
		const size_t full_n_vecs = galp::kernels::detail::full_vector_count(n_vecs, 4);
		EXPECT_EQ(full_n_vecs % 4U, 0U);
		EXPECT_LE(full_n_vecs, n_vecs);
		EXPECT_LT(n_vecs - full_n_vecs, 4U);
	}
	EXPECT_EQ(galp::kernels::detail::full_vector_count(8, 4), 8U);
	EXPECT_EQ(galp::kernels::detail::full_vector_count(3, 4), 0U);
	EXPECT_EQ(galp::kernels::detail::full_vector_count(33, 4), 32U);
}

TEST(DeviceUtils, DictSlpatchTailRebindUsesScalarPatchers) {
	using Stateful4  = galp::codec::device::StatefulSLPATCHDictExceptionPatcher<uint32_t, uint16_t, 4, 1>;
	using Stateful1  = galp::codec::device::StatefulSLPATCHDictExceptionPatcher<uint32_t, uint16_t, 1, 1>;
	using Stateless4 = galp::codec::device::StatelessSLPATCHDictExceptionPatcher<uint32_t, uint16_t, 4, 1>;
	using Stateless1 = galp::codec::device::StatelessSLPATCHDictExceptionPatcher<uint32_t, uint16_t, 1, 1>;

	static_assert(std::is_same_v<galp::kernels::detail::ScalarTailDecompressorT<Stateful4>, Stateful1>);
	static_assert(std::is_same_v<galp::kernels::detail::ScalarTailDecompressorT<Stateless4>, Stateless1>);
	SUCCEED();
}

TEST(ALP, TEST_ALP) {
	if (const auto reason = cuda_unavailable_reason(); !reason.empty()) {
		GTEST_SKIP() << reason;
	}

	// -------------------------------------------------------------------------
	// (0) Generate new binaries + discover directories
	// -------------------------------------------------------------------------
	const size_t TOTAL = 25'600 * 1'024;
	const size_t HEAD  = 0;

	const auto data_root   = generated_data_root("alp_test_alp");
	const auto floats_dir  = data_root / "floats";
	const auto doubles_dir = data_root / "doubles";

	auto gen = galp::testdata::generate_write_and_scan(floats_dir, doubles_dir, TOTAL, HEAD);
	for (const auto& path : gen.float_files) {
		test_alp<float>(path);
	}
}

TEST(GALP, TEST_GALP) {
	if (const auto reason = cuda_unavailable_reason(); !reason.empty()) {
		GTEST_SKIP() << reason;
	}

	// -------------------------------------------------------------------------
	// (0) Generate new binaries + discover directories
	// -------------------------------------------------------------------------
	const size_t TOTAL = 25'600 * 1'024;
	const size_t HEAD  = 0;

	const auto data_root   = generated_data_root("galp_test_galp");
	const auto floats_dir  = data_root / "floats";
	const auto doubles_dir = data_root / "doubles";

	auto gen = galp::testdata::generate_write_and_scan(floats_dir, doubles_dir, TOTAL, HEAD);
	for (const auto& path : gen.float_files) {
		test_galp<float>(path);
	}
}

TEST(GALP, TEST_GALP_BY_GALP_DATASET) {
	if (const auto reason = cuda_unavailable_reason(); !reason.empty()) {
		GTEST_SKIP() << reason;
	}

	namespace fs = std::filesystem;

	size_t tested_files = 0;
	for (const auto& dataset_entry : fastlanes::galp::dataset) {
		const auto& dir = dataset_entry.second;
		if (!fs::exists(dir)) {
			continue;
		}
		for (const auto& entry : fs::directory_iterator(dir)) {
			if (entry.is_regular_file() && entry.path().extension() == ".csv") {
				std::cout << "Testing file: " << entry.path().string() << std::endl;
				test_galp_csv<float>(entry.path());
				++tested_files;
			}
		}
	}
	if (tested_files == 0) {
		GTEST_SKIP() << "No GALP CSV dataset files found.";
	}
}
