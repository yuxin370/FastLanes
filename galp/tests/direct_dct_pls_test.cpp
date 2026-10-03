#include "api/direct_dct_pls_postprocess.hpp"
#include "galp/advanced/direct_dct_pls.hpp"
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

galp::jpeg::DirectDctPlsLayout small_layout() {
	std::vector<galp::jpeg::DirectDctPlsSample> samples;
	std::vector<std::vector<uint32_t>>          positions {{0U, 1U, 2U, 3U}, {4U, 5U, 6U, 7U}, {8U, 9U}};
	for (uint32_t image = 0U; image < 10U; ++image) {
		samples.push_back({image,
		                   image / 4U,
		                   image % 4U,
		                   100U + image,
		                   static_cast<int64_t>(image % 3U),
		                   "train/sample_" + std::to_string(image) + ".JPEG"});
	}
	return {std::move(samples), std::move(positions), 4U};
}

TEST(DirectDctPls, ClosedPoolScheduleMatchesRegisteredPythonSchedule) {
	auto                                    layout = small_layout();
	galp::jpeg::DirectDctPlsScheduleOptions options;
	options.training_seed     = 11997733U;
	options.epoch             = 7U;
	options.segments_per_pool = 2U;
	options.microbatch_images = 2U;
	galp::jpeg::DirectDctPlsEpochSchedule schedule(layout, options);
	ASSERT_EQ(schedule.remaining_pool_count(), 2U);
	const auto first = schedule.next_pool();
	EXPECT_EQ(first.virtual_pls_ids, (std::vector<uint32_t> {0U, 2U}));
	EXPECT_EQ(first.ordered_positions, (std::vector<uint32_t> {0U, 3U, 9U, 8U, 2U, 1U}));
	const auto second = schedule.next_pool();
	EXPECT_EQ(second.virtual_pls_ids, (std::vector<uint32_t> {1U}));
	EXPECT_EQ(second.ordered_positions, (std::vector<uint32_t> {6U, 7U, 5U, 4U}));
	EXPECT_FALSE(schedule.has_next());
}

TEST(DirectDctPls, PhysicalOrderScheduleNeverShufflesAcrossEpochs) {
	auto                                    layout = small_layout();
	galp::jpeg::DirectDctPlsScheduleOptions options;
	options.training_seed     = 11997733U;
	options.segments_per_pool = 2U;
	options.microbatch_images = 2U;
	options.order_policy      = galp::jpeg::DirectDctPlsOrderPolicy::kPhysicalOrder;

	for (const auto epoch : {0U, 7U}) {
		options.epoch = epoch;
		galp::jpeg::DirectDctPlsEpochSchedule schedule(layout, options);
		ASSERT_EQ(schedule.remaining_pool_count(), 2U);
		const auto first = schedule.next_pool();
		EXPECT_EQ(first.virtual_pls_ids, (std::vector<uint32_t> {0U, 1U}));
		EXPECT_EQ(first.ordered_positions, (std::vector<uint32_t> {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U}));
		const auto second = schedule.next_pool();
		EXPECT_EQ(second.virtual_pls_ids, (std::vector<uint32_t> {2U}));
		EXPECT_EQ(second.ordered_positions, (std::vector<uint32_t> {8U, 9U}));
		EXPECT_FALSE(schedule.has_next());
	}
}

TEST(DirectDctPls, DistributedMicrobatchesPreserveSerialOrderAndMixupKeys) {
	auto layout = small_layout();
	galp::jpeg::DirectDctPlsScheduleOptions options;
	options.training_seed     = 11997733U;
	options.epoch             = 7U;
	options.segments_per_pool = 2U;
	options.microbatch_images = 2U;
	galp::jpeg::DirectDctPlsEpochSchedule         serial(layout, options);
	std::vector<galp::jpeg::DirectDctPlsPoolPlan> plans;
	while (serial.has_next())
		plans.push_back(serial.next_pool());
	std::vector<uint32_t> coverage(10U, 0U);
	for (uint32_t rank = 0U; rank < 4U; ++rank) {
		options.rank       = rank;
		options.world_size = 4U;
		galp::jpeg::DirectDctPlsEpochSchedule distributed(layout, options);
		for (const auto& full : plans) {
			const auto local = distributed.next_pool();
			EXPECT_EQ(local.pool_index, full.pool_index);
			EXPECT_EQ(local.first_microbatch_index, full.first_microbatch_index + rank);
			std::vector<uint32_t> expected;
			for (size_t i = rank * 2U; i < full.ordered_positions.size(); i += 8U) {
				for (size_t j = i; j < std::min(i + 2U, full.ordered_positions.size()); ++j) {
					expected.push_back(full.ordered_positions[j]);
				}
			}
			EXPECT_EQ(local.ordered_positions, expected);
			for (auto id : local.ordered_positions)
				++coverage[id];
		}
		EXPECT_FALSE(distributed.has_next());
	}
	EXPECT_EQ(coverage, std::vector<uint32_t>(10U, 1U));
	options.rank = 4U;
	EXPECT_THROW(galp::jpeg::DirectDctPlsEpochSchedule(layout, options), std::invalid_argument);
}

TEST(DirectDctPls, DistributedFullImageNetCoverage) {
	const auto* manifest_path = std::getenv("GALP_DDP_FULL_MANIFEST");
	const auto* mapping_path = std::getenv("GALP_DDP_FULL_MAPPING");
	if (manifest_path == nullptr || mapping_path == nullptr) {
		GTEST_SKIP() << "full dataset metadata was not supplied";
	}
	const auto manifest = galp::jpeg::read_jpeg_dct_shard_manifest(manifest_path);
	const auto layout = galp::jpeg::DirectDctPlsLayout::LoadPremixedCsv(mapping_path, manifest);
	ASSERT_EQ(layout.sample_count(), 1281167U);
	for (uint32_t epoch : {0U, 1U}) {
		galp::jpeg::DirectDctPlsScheduleOptions options;
		options.training_seed = 11997733U;
		options.epoch         = epoch;
		galp::jpeg::DirectDctPlsEpochSchedule         serial(layout, options);
		std::vector<galp::jpeg::DirectDctPlsPoolPlan> plans;
		size_t                                        updates = 0U;
		while (serial.has_next()) {
			plans.push_back(serial.next_pool());
			updates += (plans.back().ordered_positions.size() + 1023U) / 1024U;
		}
		EXPECT_EQ(updates, 1252U);
		std::vector<uint32_t> coverage(layout.sample_count(), 0U);
		for (uint32_t rank = 0U; rank < 8U; ++rank) {
			options.rank       = rank;
			options.world_size = 8U;
			galp::jpeg::DirectDctPlsEpochSchedule schedule(layout, options);
			for (const auto& full : plans) {
				const auto            local = schedule.next_pool();
				std::vector<uint32_t> expected;
				for (size_t begin = rank * 64U; begin < full.ordered_positions.size(); begin += 512U) {
					for (size_t i = begin; i < std::min(begin + 64U, full.ordered_positions.size()); ++i) {
						expected.push_back(full.ordered_positions[i]);
					}
				}
				ASSERT_EQ(local.ordered_positions, expected);
				for (auto id : local.ordered_positions)
					++coverage[id];
			}
			EXPECT_FALSE(schedule.has_next());
		}
		EXPECT_EQ(coverage, std::vector<uint32_t>(layout.sample_count(), 1U));
	}
}

TEST(DirectDctPls, PublishedCropAndFlipMatchTorchReference) {
	auto layout              = small_layout();
	auto sample              = layout.sample(4U);
	sample.logical_sample_id = "train/n00000001/sample.JPEG";
	galp::jpeg::DirectDctPlsScheduleOptions options;
	options.training_seed = 11997733U;
	options.epoch         = 7U;
	options.crop_policy   = galp::jpeg::DirectDctPlsCropPolicy::kPerSample;
	auto per_sample       = galp::jpeg::derive_direct_dct_pls_augmentation(sample, 512U, 512U, options);
	EXPECT_EQ(per_sample.crop_seed, 15113424312524249426ULL);
	EXPECT_EQ(per_sample.flip_seed, 14071561592706787157ULL);
	EXPECT_EQ(per_sample.request.source_crop.x, 48U);
	EXPECT_EQ(per_sample.request.source_crop.y, 0U);
	EXPECT_EQ(per_sample.request.source_crop.width, 448U);
	EXPECT_EQ(per_sample.request.source_crop.height, 448U);
	EXPECT_TRUE(per_sample.request.horizontal_flip);
	EXPECT_EQ(per_sample.request.augmentation_key, "308ff06268081c844f3dcdb13a181379bcf64179d2dadabecabf5c1a75f5f8f5");

	options.crop_policy = galp::jpeg::DirectDctPlsCropPolicy::kPerPls;
	const auto per_pls  = galp::jpeg::derive_direct_dct_pls_augmentation(sample, 512U, 512U, options);
	EXPECT_EQ(per_pls.crop_seed, 16329242979244989997ULL);
	EXPECT_EQ(per_pls.request.source_crop.x, 16U);
	EXPECT_EQ(per_pls.request.source_crop.y, 16U);
	EXPECT_EQ(per_pls.request.source_crop.width, 448U);
	EXPECT_EQ(per_pls.request.source_crop.height, 448U);
	EXPECT_EQ(per_pls.flip_seed, per_sample.flip_seed);
	EXPECT_EQ(per_pls.request.horizontal_flip, per_sample.request.horizontal_flip);
}

TEST(DirectDctPls, PublishedRandAugmentAndMixupDecisionsMatchTorchReference) {
	galp::jpeg::DirectDctPlsSample sample;
	sample.virtual_pls_id    = 1U;
	sample.logical_sample_id = "train/n00000001/sample.JPEG";
	galp::jpeg::DirectDctPlsScheduleOptions options;
	options.training_seed   = 11997733U;
	options.epoch           = 7U;
	const auto augmentation = galp::jpeg::detail::derive_published_randaugment_decision(sample, options);
	EXPECT_EQ(augmentation.operations[0], galp::jpeg::detail::DirectDctPlsRandAugmentOp::kMidfreqAug);
	EXPECT_FLOAT_EQ(augmentation.magnitudes[0], -0.27F);
	EXPECT_EQ(augmentation.operations[1], galp::jpeg::detail::DirectDctPlsRandAugmentOp::kCutout);
	EXPECT_FLOAT_EQ(augmentation.magnitudes[1], 1.8F);
	EXPECT_EQ(augmentation.cutout_center_h[1], 0);
	EXPECT_EQ(augmentation.cutout_center_w[1], 8);

	const auto mixup0 = galp::jpeg::detail::derive_published_mixup_decision(11997733U, 7U, 0U);
	EXPECT_FLOAT_EQ(mixup0.original, 0.8648064136505127F);
	EXPECT_FLOAT_EQ(mixup0.rolled, 0.1351936012506485F);
	const auto mixup64 = galp::jpeg::detail::derive_published_mixup_decision(11997733U, 7U, 64U);
	EXPECT_FLOAT_EQ(mixup64.original, 0.9997560381889343F);
	EXPECT_FLOAT_EQ(mixup64.rolled, 0.00024396694789174944F);
}

TEST(DirectDctPls, PremixedCsvMustMatchPhysicalShardBoundaries) {
	const auto root = std::filesystem::temp_directory_path() / "galp_direct_dct_pls_mapping_test";
	std::filesystem::create_directories(root);
	const auto csv = root / "ordered_mapping.csv";
	{
		std::ofstream output(csv);
		output << "planned_physical_position,virtual_pls_id,position_in_pls,manifest_index,galp_image_id,logical_"
		          "sample_id,label,source_path\n";
		for (uint32_t image = 0U; image < 6U; ++image) {
			const auto pls    = image < 4U ? 0U : 1U;
			const auto in_pls = image < 4U ? image : image - 4U;
			output << image << ',' << pls << ',' << in_pls << ',' << image << ',' << (10U + image) << ",train/sample_"
			       << image << ".JPEG," << (image % 2U) << ",/unused\n";
		}
	}
	galp::jpeg::JpegDctShardManifest manifest;
	manifest.image_count = 6U;
	manifest.shards      = {
        galp::jpeg::JpegDctShardManifestEntry {.shard_id = 0U, .first_global_image_index = 0U, .image_count = 4U},
        galp::jpeg::JpegDctShardManifestEntry {.shard_id = 1U, .first_global_image_index = 4U, .image_count = 2U},
    };
	const auto loaded = galp::jpeg::DirectDctPlsLayout::LoadPremixedCsv(csv, manifest, 4U);
	EXPECT_EQ(loaded.sample_count(), 6U);
	EXPECT_EQ(loaded.pls_count(), 2U);
	EXPECT_EQ(loaded.sample(4U).logical_sample_id, "train/sample_4.JPEG");
	EXPECT_EQ(loaded.positions_in_pls(1U).size(), 2U);
	EXPECT_THROW(static_cast<void>(galp::jpeg::DirectDctPlsLayout::LoadPremixedCsv(
	                 csv, manifest, 4U, "0000000000000000000000000000000000000000000000000000000000000000")),
	             std::runtime_error);
	for (const auto invalid_label : {-1, 2}) {
		std::ofstream output(csv);
		output << "planned_physical_position,virtual_pls_id,position_in_pls,manifest_index,galp_image_id,logical_"
		          "sample_id,label,source_path\n";
		for (uint32_t image = 0U; image < 6U; ++image) {
			const auto pls    = image < 4U ? 0U : 1U;
			const auto in_pls = image < 4U ? image : image - 4U;
			output << image << ',' << pls << ',' << in_pls << ',' << image << ',' << (10U + image) << ",train/sample_"
			       << image << ".JPEG," << (image == 0U ? invalid_label : 0) << ",/unused\n";
		}
		output.close();
		EXPECT_THROW(static_cast<void>(galp::jpeg::DirectDctPlsLayout::LoadPremixedCsv(csv, manifest, 4U, {}, 2U)),
		             std::runtime_error);
	}
	std::filesystem::remove(csv);
	std::filesystem::remove(root);
}

} // namespace
