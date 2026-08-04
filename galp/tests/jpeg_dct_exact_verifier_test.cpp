#include "galp/jpeg_dct_storage.hpp"
#include "jpeg/jpeg_dct_exact_verifier.hpp"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <jpeglib.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

struct FileCloser {
	void operator()(FILE* file) const {
		if (file != nullptr) {
			std::fclose(file);
		}
	}
};

using FilePtr = std::unique_ptr<FILE, FileCloser>;

class TemporaryDirectory {
public:
	TemporaryDirectory() {
		const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
		path_ = std::filesystem::temp_directory_path() / ("galp_jpeg_dct_exact_verifier_" + std::to_string(suffix));
		std::filesystem::create_directories(path_);
	}

	~TemporaryDirectory() {
		std::error_code error;
		std::filesystem::remove_all(path_, error);
	}

	TemporaryDirectory(const TemporaryDirectory&)            = delete;
	TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

	[[nodiscard]] const std::filesystem::path& path() const noexcept {
		return path_;
	}

private:
	std::filesystem::path path_;
};

void write_test_jpeg(const std::filesystem::path& path, const int width, const int height, const int seed) {
	FilePtr file(std::fopen(path.string().c_str(), "wb"));
	if (!file) {
		throw std::runtime_error("failed to create verifier test JPEG");
	}

	jpeg_compress_struct compressor {};
	jpeg_error_mgr       error_manager {};
	compressor.err = jpeg_std_error(&error_manager);
	jpeg_create_compress(&compressor);
	jpeg_stdio_dest(&compressor, file.get());
	compressor.image_width      = static_cast<JDIMENSION>(width);
	compressor.image_height     = static_cast<JDIMENSION>(height);
	compressor.input_components = 3;
	compressor.in_color_space   = JCS_RGB;
	jpeg_set_defaults(&compressor);
	jpeg_set_quality(&compressor, 90, TRUE);
	jpeg_start_compress(&compressor, TRUE);

	std::vector<unsigned char> row(static_cast<size_t>(width) * 3U);
	while (compressor.next_scanline < compressor.image_height) {
		for (int x = 0; x < width; ++x) {
			const auto offset = static_cast<size_t>(x) * 3U;
			const auto y      = static_cast<int>(compressor.next_scanline);
			row[offset + 0U]  = static_cast<unsigned char>((x * 11 + seed * 29) & 0xFF);
			row[offset + 1U]  = static_cast<unsigned char>((y * 19 + seed * 17) & 0xFF);
			row[offset + 2U]  = static_cast<unsigned char>((x * 7 + y * 3 + seed * 13) & 0xFF);
		}
		JSAMPROW row_pointer = row.data();
		jpeg_write_scanlines(&compressor, &row_pointer, 1U);
	}

	jpeg_finish_compress(&compressor);
	jpeg_destroy_compress(&compressor);
}

void expect_same_result(const galp::jpeg::JpegDctExactVerificationResult& expected,
                        const galp::jpeg::JpegDctExactVerificationResult& actual) {
	EXPECT_EQ(actual.source_images, expected.source_images);
	EXPECT_EQ(actual.expected_blocks, expected.expected_blocks);
	EXPECT_EQ(actual.actual_blocks, expected.actual_blocks);
	EXPECT_EQ(actual.missing_blocks, expected.missing_blocks);
	EXPECT_EQ(actual.extra_blocks, expected.extra_blocks);
	EXPECT_EQ(actual.coefficient_mismatches, expected.coefficient_mismatches);
	EXPECT_EQ(actual.max_abs_difference, expected.max_abs_difference);
	EXPECT_EQ(actual.first_mismatch.present, expected.first_mismatch.present);
	EXPECT_EQ(actual.first_mismatch.global_image_index, expected.first_mismatch.global_image_index);
	EXPECT_EQ(actual.first_mismatch.semantic_slot_id, expected.first_mismatch.semantic_slot_id);
	EXPECT_EQ(actual.first_mismatch.block_y, expected.first_mismatch.block_y);
	EXPECT_EQ(actual.first_mismatch.block_x, expected.first_mismatch.block_x);
	EXPECT_EQ(actual.first_mismatch.coefficient, expected.first_mismatch.coefficient);
	EXPECT_EQ(actual.first_mismatch.expected, expected.first_mismatch.expected);
	EXPECT_EQ(actual.first_mismatch.actual, expected.first_mismatch.actual);
	EXPECT_EQ(actual.first_mismatch.kind, expected.first_mismatch.kind);
}

std::string read_binary_file(const std::filesystem::path& path) {
	std::ifstream stream(path, std::ios::binary);
	if (!stream) {
		throw std::runtime_error("failed to open verifier test artifact");
	}
	return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

class JpegDctExactVerifierTest : public testing::Test {
protected:
	void SetUp() override {
		for (size_t index = 0U; index < 4U; ++index) {
			const auto source_path = temporary_.path() / ("source_" + std::to_string(index) + ".jpg");
			write_test_jpeg(source_path, 32, 32, static_cast<int>(index) + 1);
			source_paths_.push_back(source_path);
		}

		galp::jpeg::JpegDctShardOptions options;
		options.shard_images           = 1U;
		options.shard_images_specified = true;
		options.threads                = 1U;
		options.shard_workers          = 1U;
		serial_manifest_               = galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls(
            source_paths_, temporary_.path() / "dataset", {}, options, {});
		ASSERT_EQ(serial_manifest_.shards.size(), source_paths_.size());
		manifest_path_ = temporary_.path() / "dataset" / "manifest.bin";
	}

	TemporaryDirectory                 temporary_;
	std::vector<std::filesystem::path> source_paths_;
	std::filesystem::path              manifest_path_;
	galp::jpeg::JpegDctShardManifest   serial_manifest_;
};

TEST(JpegDctExactVerifier, RejectsWorkerCountsOutsideTheBoundBeforeOpeningTheManifest) {
	const std::filesystem::path nonexistent_manifest = "nonexistent_manifest.bin";
	EXPECT_THROW(galp::jpeg::verify_jpeg_dct_manifest_exact(nonexistent_manifest, {}, 0U), std::invalid_argument);
	EXPECT_THROW(galp::jpeg::verify_jpeg_dct_manifest_exact(
	                 nonexistent_manifest, {}, galp::jpeg::kMaxJpegDctExactVerificationWorkers + 1U),
	             std::invalid_argument);
}

TEST_F(JpegDctExactVerifierTest, SerialAndParallelExactResultsAreIdentical) {
	const auto by_default = galp::jpeg::verify_jpeg_dct_manifest_exact(manifest_path_, source_paths_);
	const auto serial     = galp::jpeg::verify_jpeg_dct_manifest_exact(manifest_path_, source_paths_, 1U);
	const auto parallel   = galp::jpeg::verify_jpeg_dct_manifest_exact(manifest_path_, source_paths_, 4U);
	const auto capped     = galp::jpeg::verify_jpeg_dct_manifest_exact(
        manifest_path_, source_paths_, galp::jpeg::kMaxJpegDctExactVerificationWorkers);

	EXPECT_TRUE(serial.exact());
	EXPECT_EQ(serial.source_images, source_paths_.size());
	EXPECT_GT(serial.expected_blocks, 0U);
	EXPECT_EQ(serial.expected_blocks, serial.actual_blocks);
	EXPECT_FALSE(serial.first_mismatch.present);
	expect_same_result(serial, by_default);
	expect_same_result(serial, parallel);
	expect_same_result(serial, capped);
}

TEST_F(JpegDctExactVerifierTest, ParallelMergeReportsTheGloballyEarliestMismatch) {
	auto mismatched_paths = source_paths_;
	std::swap(mismatched_paths.front(), mismatched_paths.back());

	const auto serial   = galp::jpeg::verify_jpeg_dct_manifest_exact(manifest_path_, mismatched_paths, 1U);
	const auto parallel = galp::jpeg::verify_jpeg_dct_manifest_exact(manifest_path_, mismatched_paths, 4U);

	EXPECT_FALSE(serial.exact());
	ASSERT_TRUE(serial.first_mismatch.present);
	EXPECT_EQ(serial.first_mismatch.global_image_index, 0U);
	EXPECT_EQ(serial.first_mismatch.kind, galp::jpeg::JpegDctExactMismatchKind::kCoefficient);
	expect_same_result(serial, parallel);
}

TEST_F(JpegDctExactVerifierTest, AnySourceReadFailureFailsTheWholeParallelVerification) {
	auto missing_source_paths = source_paths_;
	missing_source_paths[2]   = temporary_.path() / "missing.jpg";
	EXPECT_ANY_THROW(galp::jpeg::verify_jpeg_dct_manifest_exact(manifest_path_, missing_source_paths, 4U));
}

TEST_F(JpegDctExactVerifierTest, RejectsSourceCountMismatch) {
	auto incomplete_paths = source_paths_;
	incomplete_paths.pop_back();
	EXPECT_THROW(galp::jpeg::verify_jpeg_dct_manifest_exact(manifest_path_, incomplete_paths, 4U), std::runtime_error);
}

TEST_F(JpegDctExactVerifierTest, IndependentPipelineControlsPreserveShardBytes) {
	galp::jpeg::JpegDctShardOptions options;
	options.shard_images                         = 1U;
	options.shard_images_specified               = true;
	options.layout_threads                       = 2U;
	options.layout_threads_specified             = true;
	options.shard_decode_threads                 = 2U;
	options.shard_decode_threads_specified       = true;
	options.shard_workers                        = 2U;
	options.encoding_workers_per_shard           = 2U;
	options.encoding_workers_per_shard_specified = true;
	const auto parallel_root                     = temporary_.path() / "parallel";
	const auto parallel_manifest =
	    galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls(source_paths_, parallel_root, {}, options, {});

	ASSERT_EQ(parallel_manifest.shards.size(), serial_manifest_.shards.size());
	for (size_t shard = 0U; shard < serial_manifest_.shards.size(); ++shard) {
		const auto& serial_entry   = serial_manifest_.shards[shard];
		const auto& parallel_entry = parallel_manifest.shards[shard];
		EXPECT_EQ(read_binary_file(temporary_.path() / "dataset" / serial_entry.fls_file_name),
		          read_binary_file(parallel_root / parallel_entry.fls_file_name));
		EXPECT_EQ(read_binary_file(temporary_.path() / "dataset" / serial_entry.metadata_file_name),
		          read_binary_file(parallel_root / parallel_entry.metadata_file_name));
	}
	EXPECT_TRUE(galp::jpeg::verify_jpeg_dct_manifest_exact(parallel_root / "manifest.bin", source_paths_, 4U).exact());
}

TEST_F(JpegDctExactVerifierTest, ConflictingLegacyAndIndependentThreadControlsFail) {
	galp::jpeg::JpegDctShardOptions options;
	options.threads                  = 2U;
	options.threads_specified        = true;
	options.layout_threads           = 3U;
	options.layout_threads_specified = true;
	EXPECT_THROW(galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls(
	                 source_paths_, temporary_.path() / "conflict", {}, options, {}),
	             std::invalid_argument);
}

} // namespace
