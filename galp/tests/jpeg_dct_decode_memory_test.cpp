#include "jpeg/jpeg_dct_decode.hpp"
#include <algorithm>
#include <barrier>
#include <cstdint>
#include <future>
#include <gtest/gtest.h>
#include <numeric>
#include <string>
#include <vector>

namespace {

using galp::jpeg::detail::DecodedDctRow;
using galp::jpeg::detail::DecodedImage;
using galp::jpeg::detail::JpegDctTableBuildStats;

int16_t synthetic_coefficient(const size_t image_index, const size_t block_index, const size_t coefficient) {
	const auto value = (image_index * 131U + block_index * 17U + coefficient * 7U) % 30001U;
	return static_cast<int16_t>(static_cast<int>(value) - 15000);
}

DecodedImage make_synthetic_image(const size_t image_index, const size_t block_count) {
	galp::jpeg::JpegComponentMetadata metadata;
	metadata.semantic_slot_id        = 0;
	metadata.component_index         = 0;
	metadata.local_component_index   = 0;
	metadata.component_id            = 1;
	metadata.width_in_blocks         = static_cast<uint32_t>(block_count);
	metadata.height_in_blocks        = 1;
	metadata.padded_width_in_blocks  = metadata.width_in_blocks;
	metadata.padded_height_in_blocks = metadata.height_in_blocks;
	metadata.h_samp_factor           = 1;
	metadata.v_samp_factor           = 1;

	galp::jpeg::detail::DecodedComponent component;
	component.metadata = metadata;
	component.blocks.resize(block_count);
	component.coord_to_block_index.resize(block_count);
	std::iota(component.coord_to_block_index.begin(), component.coord_to_block_index.end(), size_t {0});
	for (size_t block_index = 0; block_index < block_count; ++block_index) {
		DecodedDctRow row {};
		for (size_t coefficient = 0; coefficient < row.size(); ++coefficient) {
			row[coefficient] = synthetic_coefficient(image_index, block_index, coefficient);
		}
		component.blocks[block_index] = row;
	}

	DecodedImage image;
	image.metadata.source_path  = "synthetic_" + std::to_string(image_index) + ".jpg";
	image.metadata.image_width  = static_cast<uint32_t>(block_count * 8U);
	image.metadata.image_height = 8;
	image.metadata.components.push_back(metadata);
	image.components.push_back(std::move(component));
	return image;
}

std::vector<DecodedImage> make_synthetic_shard(const size_t image_count, const size_t blocks_per_image) {
	std::vector<DecodedImage> images;
	images.reserve(image_count);
	for (size_t image_index = 0; image_index < image_count; ++image_index) {
		images.push_back(make_synthetic_image(image_index, blocks_per_image));
	}
	return images;
}

struct BuildResult {
	JpegDctTableBuildStats stats;
	size_t                 row_count         = 0;
	size_t                 minimum_capacity  = 0;
	size_t                 image_group_count = 0;
	int16_t                first_coefficient = 0;
	int16_t                last_coefficient  = 0;
};

BuildResult build_image_major_shard(const size_t    image_count,
                                    const size_t    blocks_per_image,
                                    std::barrier<>* build_barrier = nullptr) {
	galp::jpeg::JpegDctReaderOptions options;
	options.image_major_spatial_order = galp::jpeg::JpegDctSpatialOrder::kRaster;

	auto images = make_synthetic_shard(image_count, blocks_per_image);
	if (build_barrier != nullptr) {
		build_barrier->arrive_and_wait();
	}
	BuildResult result;
	auto        table = galp::jpeg::detail::make_dataset_table(
        std::move(images), options, nullptr, galp::jpeg::JpegDctPhysicalLayout::kImageMajor, &result.stats);
	result.row_count         = table.row_count;
	result.minimum_capacity  = table.columns.front().capacity();
	result.image_group_count = table.metadata.image_group_index.size();
	for (const auto& column : table.columns) {
		result.minimum_capacity = std::min(result.minimum_capacity, column.capacity());
	}
	if (table.row_count != 0) {
		result.first_coefficient = table.columns.front().front();
		result.last_coefficient  = table.columns.back().back();
	}
	return result;
}

void expect_exact_preallocation_and_early_release(const BuildResult& result,
                                                  const size_t       image_count,
                                                  const size_t       blocks_per_image) {
	const auto expected_rows = image_count * blocks_per_image;
	EXPECT_EQ(result.row_count, expected_rows);
	EXPECT_EQ(result.stats.expected_table_rows, expected_rows);
	EXPECT_EQ(result.stats.initial_decoded_rows, expected_rows);
	EXPECT_EQ(result.stats.released_decoded_rows, expected_rows);
	EXPECT_EQ(result.stats.remaining_decoded_rows, 0U);
	EXPECT_EQ(result.stats.column_capacity_growths, 0U);
	EXPECT_GE(result.minimum_capacity, expected_rows);
	EXPECT_EQ(result.stats.peak_coefficient_row_slots, expected_rows + blocks_per_image);
	EXPECT_EQ(result.image_group_count, image_count);
	EXPECT_EQ(result.first_coefficient, synthetic_coefficient(0, 0, 0));
	EXPECT_EQ(result.last_coefficient,
	          synthetic_coefficient(image_count - 1U, blocks_per_image - 1U, DecodedDctRow {}.size() - 1U));
}

TEST(JpegDctDecodeMemory, LargeImageMajorShardUsesOneAllocationPerCoefficientColumn) {
	constexpr size_t kImageCount     = 9;
	constexpr size_t kBlocksPerImage = 4099;

	const auto result = build_image_major_shard(kImageCount, kBlocksPerImage);
	expect_exact_preallocation_and_early_release(result, kImageCount, kBlocksPerImage);
}

TEST(JpegDctDecodeMemory, ConcurrentLargeShardBuildsKeepTheSameMemoryBound) {
	constexpr size_t kShardWorkers   = 4;
	constexpr size_t kImageCount     = 7;
	constexpr size_t kBlocksPerImage = 4099;

	std::barrier                          build_barrier(static_cast<std::ptrdiff_t>(kShardWorkers));
	std::vector<std::future<BuildResult>> workers;
	workers.reserve(kShardWorkers);
	for (size_t worker = 0; worker < kShardWorkers; ++worker) {
		workers.push_back(std::async(std::launch::async, [&build_barrier] {
			return build_image_major_shard(kImageCount, kBlocksPerImage, &build_barrier);
		}));
	}
	for (auto& worker : workers) {
		expect_exact_preallocation_and_early_release(worker.get(), kImageCount, kBlocksPerImage);
	}
}

TEST(JpegDctDecodeMemory, SpatialMajorAlsoPreallocatesAndReleasesDecodedRows) {
	constexpr size_t kImageCount     = 3;
	constexpr size_t kBlocksPerImage = 1009;
	const auto       expected_rows   = kImageCount * kBlocksPerImage;

	galp::jpeg::JpegDctReaderOptions options;
	options.use_z_curve_block_order = false;
	JpegDctTableBuildStats stats;
	auto table = galp::jpeg::detail::make_dataset_table(make_synthetic_shard(kImageCount, kBlocksPerImage),
	                                                    options,
	                                                    nullptr,
	                                                    galp::jpeg::JpegDctPhysicalLayout::kSpatialMajorImageMinor,
	                                                    &stats);

	EXPECT_EQ(table.row_count, expected_rows);
	EXPECT_EQ(stats.expected_table_rows, expected_rows);
	EXPECT_EQ(stats.released_decoded_rows, expected_rows);
	EXPECT_EQ(stats.remaining_decoded_rows, 0U);
	EXPECT_EQ(stats.column_capacity_growths, 0U);
	for (const auto& column : table.columns) {
		EXPECT_GE(column.capacity(), expected_rows);
	}
}

} // namespace
