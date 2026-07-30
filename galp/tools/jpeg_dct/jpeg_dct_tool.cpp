#include "galp/jpeg_dct.hpp"
#include "galp/profiles/rgbnomore.hpp"
#include "galp/sparse_vector_bundle.hpp"
#include "jpeg/jpeg_dct_expression_validation.hpp"
#include "jpeg/jpeg_dct_metadata.hpp"
#include "jpeg/jpeg_dct_shard_reader.hpp"
#include "fls/connection.hpp"
#include "fls/expression/rpn.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/file/file_header.hpp"
#include "fls/footer/table_descriptor.hpp"
#include "fls/footer/table_descriptor_generated.h"
#include "fls/io/file.hpp"
#include "fls/table/memory_table.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <vector>

namespace {

struct Options {
	std::filesystem::path                 output_fls;
	std::filesystem::path                 output_metadata;
	std::filesystem::path                 output_dir;
	std::filesystem::path                 verify_manifest;
	std::filesystem::path                 sparse_bundle_source;
	std::filesystem::path                 sparse_bundle_output;
	std::filesystem::path                 inspect_crop_manifest;
	std::filesystem::path                 benchmark_rowgroup_read_source;
	std::vector<std::filesystem::path>    inputs;
	galp::jpeg::JpegMetadataProfile       metadata_profile = galp::jpeg::JpegMetadataProfile::kDctDatasetOnly;
	galp::jpeg::JpegDctShardPreset        shard_preset     = galp::jpeg::JpegDctShardPreset::kBalanced;
	galp::jpeg::JpegDctPhysicalLayout     physical_layout  = galp::jpeg::JpegDctPhysicalLayout::kSpatialMajorImageMinor;
	galp::jpeg::JpegDctSpatialOrder       spatial_order    = galp::jpeg::JpegDctSpatialOrder::kTiledZ32;
	size_t                                shard_images     = 8192;
	uint32_t                              rowgroup_vectors = 128;
	uint32_t                              rowgroups_per_shard           = 256;
	size_t                                threads                       = 1;
	size_t                                shard_workers                 = 1;
	bool                                  shard_mode                    = false;
	bool                                  threads_specified             = false;
	bool                                  metadata_profile_specified    = false;
	bool                                  shard_images_specified        = false;
	bool                                  rowgroup_vectors_specified    = false;
	bool                                  rowgroups_per_shard_specified = false;
	bool                                  physical_layout_specified      = false;
	bool                                  spatial_order_specified        = false;
	bool                                  verify_mode                    = false;
	bool                                  sparse_bundle_mode             = false;
	bool                                  inspect_encodings_mode          = false;
	bool                                  inspect_crop_plan_mode           = false;
	bool                                  benchmark_rowgroup_read_mode       = false;
	size_t                                benchmark_repeats                  = 5U;
	uint32_t                              verify_image_index             = 0;
	bool                                  verify_image_index_specified   = false;
};

void print_usage(const char* prog) {
	std::cerr
	    << "Usage:\n"
	    << "  " << prog
	    << " --out output.fls --metadata output.metadata.bin "
	       "[--metadata-profile dct|reconstruct|preserve] input.jpg\n"
	    << "  " << prog
	    << " --out output.fls --metadata output.metadata.bin "
	       "[--metadata-profile dct|reconstruct|preserve] input_dir\n"
	    << "  " << prog
	    << " --out output.fls --metadata output.metadata.bin "
	       "[--metadata-profile dct|reconstruct|preserve] input0.jpg [input1.jpg ...]\n"
	    << "  " << prog
	    << " --shard --out-dir output_dct [--preset crop-latency|balanced|throughput|random-access] "
	       "[--physical-layout spatial-major|image-major|image-major-vector-rowgroups] "
		       "[--spatial-order raster|tiled-raster-32|z-order|tiled-z-32] "
	       "[--shard-images N] [--rowgroup-vectors N] [--rowgroups-per-shard N] [--threads N] "
	       "[--shard-workers N] input_dir\n"
	    << "  " << prog
		    << " --verify-manifest manifest.bin [--image-index N] source.jpg\n"
	    << "  " << prog << " --verify-manifest manifest.bin source_dir\n"
	    << "  " << prog << " --build-sparse-bundle input.fls --bundle-output output.svb\n"
	    << "  " << prog << " --inspect-encodings input.fls [input1.fls ...]\n"
	    << "  " << prog << " --inspect-crop-plan manifest.bin [--image-index N]\n"
	    << "  " << prog << " --benchmark-rowgroup-read input.fls [--repeats N]\n"
	    << "Default: ragged DCT block layout and the legacy metadata format.\n"
	    << "  --threads defaults to all available cores when not set.\n"
	    << "  --metadata-profile writes the sectioned metadata format; use reconstruct to persist image dimensions "
	       "and quantization tables.\n"
	    << "  --shard writes manifest.bin plus shard_*.fls and shard_*.meta.bin; default preset is balanced.\n";
}

int verify_manifest_image(const std::filesystem::path& manifest_path,
                          const uint32_t               image_index,
                          const std::filesystem::path& source_path) {
	using BlockKey = std::tuple<uint32_t, uint32_t, uint32_t>;
	using BlockMap = std::map<BlockKey, galp::jpeg::JpegDctCoefficientRow>;

	const auto source_table = galp::jpeg::read_jpeg_dct_file(source_path);
	BlockMap   expected;
	for (const auto& group : source_table.metadata.block_group_index) {
		for (uint32_t row_offset = 0; row_offset < group.row_count; ++row_offset) {
			const auto row = static_cast<size_t>(group.row_start + row_offset);
			if (row >= source_table.row_count) {
				throw std::runtime_error("source JPEG block-group row is outside the coefficient table");
			}
			galp::jpeg::JpegDctCoefficientRow coefficients {};
			for (size_t coefficient = 0; coefficient < coefficients.size(); ++coefficient) {
				coefficients[coefficient] = source_table.columns[coefficient].at(row);
			}
			const BlockKey key {group.semantic_slot_id, group.block_y, group.block_x};
			if (!expected.emplace(key, coefficients).second) {
				throw std::runtime_error("source JPEG contains a duplicate component/block coordinate");
			}
		}
	}

	galp::jpeg::JpegDctShardDatasetReader reader(manifest_path);
	if (image_index >= reader.image_count()) {
		throw std::runtime_error("--image-index is outside the manifest image range");
	}
	const auto materialized = reader.MaterializeImageDct(image_index);
	BlockMap   actual;
	for (const auto& block : materialized.blocks) {
		const BlockKey key {block.semantic_slot_id, block.block_y, block.block_x};
		if (!actual.emplace(key, block.coefficients).second) {
			throw std::runtime_error("manifest image contains a duplicate component/block coordinate");
		}
	}

	size_t                 missing_blocks          = 0;
	size_t                 extra_blocks            = 0;
	size_t                 coefficient_mismatches  = 0;
	int                    max_abs_difference      = 0;
	bool                   have_first_mismatch     = false;
	BlockKey               first_mismatch_key {};
	size_t                 first_mismatch_coeff    = 0;
	int16_t                first_mismatch_expected = 0;
	int16_t                first_mismatch_actual   = 0;
	for (const auto& [key, expected_coefficients] : expected) {
		const auto actual_it = actual.find(key);
		if (actual_it == actual.end()) {
			++missing_blocks;
			continue;
		}
		for (size_t coefficient = 0; coefficient < expected_coefficients.size(); ++coefficient) {
			const auto expected_value = expected_coefficients[coefficient];
			const auto actual_value   = actual_it->second[coefficient];
			if (expected_value == actual_value) {
				continue;
			}
			++coefficient_mismatches;
			const auto difference = std::abs(static_cast<int>(expected_value) - static_cast<int>(actual_value));
			max_abs_difference     = std::max(max_abs_difference, difference);
			if (!have_first_mismatch) {
				have_first_mismatch     = true;
				first_mismatch_key      = key;
				first_mismatch_coeff    = coefficient;
				first_mismatch_expected = expected_value;
				first_mismatch_actual   = actual_value;
			}
		}
	}
	for (const auto& [key, coefficients] : actual) {
		(void)coefficients;
		if (!expected.contains(key)) {
			++extra_blocks;
		}
	}

	const bool exact = missing_blocks == 0 && extra_blocks == 0 && coefficient_mismatches == 0;
	std::cout << "manifest: " << manifest_path << '\n'
	          << "image_index: " << image_index << '\n'
	          << "source: " << source_path << '\n'
	          << "expected_blocks: " << expected.size() << '\n'
	          << "actual_blocks: " << actual.size() << '\n'
	          << "missing_blocks: " << missing_blocks << '\n'
	          << "extra_blocks: " << extra_blocks << '\n'
	          << "coefficient_mismatches: " << coefficient_mismatches << '\n'
	          << "max_abs_difference: " << max_abs_difference << '\n';
	if (have_first_mismatch) {
		const auto [component, block_y, block_x] = first_mismatch_key;
		std::cout << "first_mismatch: component=" << component << " block_y=" << block_y << " block_x=" << block_x
		          << " coefficient=" << first_mismatch_coeff << " expected=" << first_mismatch_expected
		          << " actual=" << first_mismatch_actual << '\n';
	}
	std::cout << "exact: " << (exact ? "true" : "false") << '\n';
	return exact ? 0 : 3;
}

int verify_manifest_dataset(const std::filesystem::path&              manifest_path,
                            const std::vector<std::filesystem::path>& source_paths) {
	using BlockKey = std::tuple<uint32_t, uint32_t, uint32_t>;
	using BlockMap = std::map<BlockKey, galp::jpeg::JpegDctCoefficientRow>;

	galp::jpeg::JpegDctShardDatasetReader reader(manifest_path);
	if (source_paths.size() != reader.image_count()) {
		throw std::runtime_error("source JPEG count does not match the manifest image count");
	}

	size_t total_expected_blocks    = 0;
	size_t total_actual_blocks      = 0;
	size_t missing_blocks           = 0;
	size_t extra_blocks             = 0;
	size_t coefficient_mismatches   = 0;
	int    max_abs_difference       = 0;
	bool   have_first_mismatch      = false;
	uint32_t first_mismatch_image   = 0;
	BlockKey first_mismatch_key {};
	size_t first_mismatch_coeff     = 0;
	int16_t first_mismatch_expected = 0;
	int16_t first_mismatch_actual   = 0;

	for (size_t image_index = 0; image_index < source_paths.size(); ++image_index) {
		const auto source_table = galp::jpeg::read_jpeg_dct_file(source_paths[image_index]);
		BlockMap   expected;
		for (const auto& group : source_table.metadata.block_group_index) {
			for (uint32_t row_offset = 0; row_offset < group.row_count; ++row_offset) {
				const auto row = static_cast<size_t>(group.row_start + row_offset);
				if (row >= source_table.row_count) {
					throw std::runtime_error("source JPEG block-group row is outside the coefficient table");
				}
				galp::jpeg::JpegDctCoefficientRow coefficients {};
				for (size_t coefficient = 0; coefficient < coefficients.size(); ++coefficient) {
					coefficients[coefficient] = source_table.columns[coefficient].at(row);
				}
				if (!expected.emplace(BlockKey {group.semantic_slot_id, group.block_y, group.block_x}, coefficients).second) {
					throw std::runtime_error("source JPEG contains a duplicate component/block coordinate");
				}
			}
		}

		const auto materialized = reader.MaterializeImageDct(static_cast<uint32_t>(image_index));
		BlockMap   actual;
		for (const auto& block : materialized.blocks) {
			if (!actual.emplace(BlockKey {block.semantic_slot_id, block.block_y, block.block_x}, block.coefficients).second) {
				throw std::runtime_error("manifest image contains a duplicate component/block coordinate");
			}
		}
		total_expected_blocks += expected.size();
		total_actual_blocks += actual.size();

		for (const auto& [key, expected_coefficients] : expected) {
			const auto actual_it = actual.find(key);
			if (actual_it == actual.end()) {
				++missing_blocks;
				if (!have_first_mismatch) {
					have_first_mismatch    = true;
					first_mismatch_image    = static_cast<uint32_t>(image_index);
					first_mismatch_key      = key;
				}
				continue;
			}
			for (size_t coefficient = 0; coefficient < expected_coefficients.size(); ++coefficient) {
				const auto expected_value = expected_coefficients[coefficient];
				const auto actual_value   = actual_it->second[coefficient];
				if (expected_value == actual_value) {
					continue;
				}
				++coefficient_mismatches;
				max_abs_difference = std::max(
				    max_abs_difference, std::abs(static_cast<int>(expected_value) - static_cast<int>(actual_value)));
				if (!have_first_mismatch) {
					have_first_mismatch      = true;
					first_mismatch_image      = static_cast<uint32_t>(image_index);
					first_mismatch_key        = key;
					first_mismatch_coeff      = coefficient;
					first_mismatch_expected   = expected_value;
					first_mismatch_actual     = actual_value;
				}
			}
		}
		for (const auto& [key, coefficients] : actual) {
			(void)coefficients;
			if (!expected.contains(key)) {
				++extra_blocks;
				if (!have_first_mismatch) {
					have_first_mismatch = true;
					first_mismatch_image = static_cast<uint32_t>(image_index);
					first_mismatch_key   = key;
				}
			}
		}
	}

	const bool exact = missing_blocks == 0 && extra_blocks == 0 && coefficient_mismatches == 0;
	std::cout << "manifest: " << manifest_path << '\n'
	          << "source_images: " << source_paths.size() << '\n'
	          << "expected_blocks: " << total_expected_blocks << '\n'
	          << "actual_blocks: " << total_actual_blocks << '\n'
	          << "missing_blocks: " << missing_blocks << '\n'
	          << "extra_blocks: " << extra_blocks << '\n'
	          << "coefficient_mismatches: " << coefficient_mismatches << '\n'
	          << "max_abs_difference: " << max_abs_difference << '\n';
	if (have_first_mismatch) {
		const auto [component, block_y, block_x] = first_mismatch_key;
		std::cout << "first_mismatch: image=" << first_mismatch_image << " component=" << component
		          << " block_y=" << block_y << " block_x=" << block_x << " coefficient=" << first_mismatch_coeff
		          << " expected=" << first_mismatch_expected << " actual=" << first_mismatch_actual << '\n';
	}
	std::cout << "exact: " << (exact ? "true" : "false") << '\n';
	return exact ? 0 : 3;
}

bool is_jpeg_path(const std::filesystem::path& path) {
	auto ext = path.extension().string();
	std::transform(ext.begin(), ext.end(), ext.begin(), [](const unsigned char ch) {
		return static_cast<char>(std::tolower(ch));
	});
	return ext == ".jpg" || ext == ".jpeg" || ext == ".jpe";
}

std::vector<std::filesystem::path> expand_inputs(const std::vector<std::filesystem::path>& inputs) {
	std::vector<std::filesystem::path> expanded;
	for (const auto& input : inputs) {
		if (std::filesystem::is_directory(input)) {
			for (const auto& entry : std::filesystem::recursive_directory_iterator(input)) {
				if (entry.is_regular_file() && is_jpeg_path(entry.path())) {
					expanded.push_back(entry.path());
				}
			}
			continue;
		}
		expanded.push_back(input);
	}

	std::sort(expanded.begin(), expanded.end());
	if (expanded.empty()) {
		throw std::runtime_error("no JPEG files found in input path(s)");
	}
	return expanded;
}

uint32_t parse_u32_arg(const std::string_view name, const char* value) {
	const auto parsed = std::stoull(value);
	if (parsed > std::numeric_limits<uint32_t>::max()) {
		throw std::runtime_error(std::string(name) + " is outside uint32_t range");
	}
	return static_cast<uint32_t>(parsed);
}

size_t parse_size_arg(const std::string_view name, const char* value) {
	const auto parsed = std::stoull(value);
	if (parsed > std::numeric_limits<size_t>::max()) {
		throw std::runtime_error(std::string(name) + " is outside size_t range");
	}
	return static_cast<size_t>(parsed);
}

int inspect_encodings(const std::vector<std::filesystem::path>& paths) {
	std::map<std::string, size_t> token_counts;
	size_t                        rowgroup_count = 0;
	size_t                        column_count   = 0;

	for (const auto& path : paths) {
		fastlanes::FileHeader header {};
		fastlanes::FileFooter footer {};
		fastlanes::FileHeader::Load(header, path);
		fastlanes::FileFooter::Load(footer, path);
		auto descriptor =
		    header.settings.inline_footer
		        ? fastlanes::TableDescriptorHandle::FromFileSlice(
		              path, footer.table_descriptor_offset, footer.table_descriptor_size, true)
		        : fastlanes::TableDescriptorHandle::FromFile(path.parent_path() / "table_descriptor.fbb", true);
		const auto* table     = descriptor.Get();
		const auto* rowgroups = table->m_rowgroup_descriptors();
		if (rowgroups == nullptr) {
			throw std::runtime_error("missing rowgroup descriptors in " + path.string());
		}
		for (flatbuffers::uoffset_t rowgroup_index = 0; rowgroup_index < rowgroups->size(); ++rowgroup_index) {
			const auto* rowgroup = rowgroups->Get(rowgroup_index);
			const auto* columns  = rowgroup == nullptr ? nullptr : rowgroup->m_column_descriptors();
			if (columns == nullptr) {
				throw std::runtime_error("missing column descriptors in " + path.string());
			}
			for (flatbuffers::uoffset_t column_index = 0; column_index < columns->size(); ++column_index) {
				const auto* column = columns->Get(column_index);
				const auto* rpn    = column == nullptr ? nullptr : column->encoding_rpn();
				const auto* tokens = rpn == nullptr ? nullptr : rpn->operator_tokens();
				if (tokens == nullptr || tokens->empty()) {
					throw std::runtime_error("missing encoding tokens in " + path.string());
				}
				for (flatbuffers::uoffset_t token_index = 0; token_index < tokens->size(); ++token_index) {
					const auto name = fastlanes::token_to_string(tokens->Get(token_index));
					++token_counts[name];
				}
			}
			++rowgroup_count;
			column_count += columns->size();
		}
	}

	std::cout << "files: " << paths.size() << '\n'
	          << "rowgroups: " << rowgroup_count << '\n'
	          << "columns: " << column_count << '\n'
	          << "token,count\n";
	for (const auto& [token, count] : token_counts) {
		std::cout << token << ',' << count << '\n';
	}
	return 0;
}

int inspect_crop_plan(const std::filesystem::path& manifest_path, const uint32_t image_index) {
	galp::jpeg::JpegDctShardDatasetReader reader(manifest_path);
	if (image_index >= reader.image_count()) {
		throw std::runtime_error("--image-index is outside the manifest image range");
	}
	galp::jpeg::JpegDctDeviceBatchOptions options;
	options.layout                    = galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid;
	options.grid_transform            = galp::profiles::rgbnomore_val_dct_grid_transform();
	options.crop_execution_mode       = galp::jpeg::JpegDctCropExecutionMode::kVectorRangeReadSelectedDecode;
	options.enable_planless_execution = true;
	const auto preview                = reader.PlanDeviceDctBatch(
        std::vector<galp::jpeg::JpegDctImageCropRequest> {{image_index, {}, false, {}, {}}}, options);
	std::cout << "manifest: " << manifest_path << '\n'
	          << "image_index: " << image_index << '\n'
	          << "full_vectors: " << preview.full_vector_count << '\n'
	          << "selected_vectors: " << preview.planned_selected_vector_count << '\n';
	for (const auto& rowgroup_plan : preview.rowgroup_vector_plans) {
		std::cout << "rowgroup: shard=" << rowgroup_plan.rowgroup.shard_id
		          << " index=" << rowgroup_plan.rowgroup.rowgroup_index << " vectors=";
		for (size_t index = 0; index < rowgroup_plan.selected_vectors.size(); ++index) {
			if (index != 0U) {
				std::cout << ',';
			}
			std::cout << rowgroup_plan.selected_vectors[index];
		}
		std::cout << '\n';
	}
	return 0;
}

int benchmark_rowgroup_read(const std::filesystem::path& path, const size_t repeats) {
	if (repeats == 0U) {
		throw std::invalid_argument("--repeats must be greater than zero");
	}
	fastlanes::File       file(path);
	fastlanes::FileHeader header {};
	fastlanes::FileFooter footer {};
	fastlanes::FileHeader::Load(header, file);
	fastlanes::FileFooter::Load(footer, file);
	auto descriptor =
	    header.settings.inline_footer
	        ? fastlanes::TableDescriptorHandle::FromFileSlice(
	              file, footer.table_descriptor_offset, footer.table_descriptor_size, true)
	        : fastlanes::TableDescriptorHandle::FromFile(path.parent_path() / "table_descriptor.fbb", true);
	const auto* rowgroups = descriptor.Get()->m_rowgroup_descriptors();
	if (rowgroups == nullptr || rowgroups->empty()) {
		throw std::runtime_error("rowgroup read benchmark input has no rowgroups");
	}
	size_t max_rowgroup_bytes = 0U;
	for (flatbuffers::uoffset_t rowgroup = 0; rowgroup < rowgroups->size(); ++rowgroup) {
		max_rowgroup_bytes = std::max(max_rowgroup_bytes, static_cast<size_t>(rowgroups->Get(rowgroup)->m_size()));
	}
	std::vector<std::byte> backing(max_rowgroup_bytes);
	for (size_t repeat = 0; repeat < repeats; ++repeat) {
		size_t     bytes = 0U;
		const auto start = std::chrono::steady_clock::now();
		for (flatbuffers::uoffset_t rowgroup = 0; rowgroup < rowgroups->size(); ++rowgroup) {
			const auto* entry = rowgroups->Get(rowgroup);
			file.ReadRangeUnchecked(backing.data(), entry->m_offset(), entry->m_size());
			bytes += static_cast<size_t>(entry->m_size());
		}
		const auto end = std::chrono::steady_clock::now();
		std::cout << "repeat=" << repeat
		          << " wall_ms=" << std::chrono::duration<double, std::milli>(end - start).count() << " bytes=" << bytes
		          << " preads=" << rowgroups->size() << '\n';
	}
	return 0;
}

void apply_shard_preset(Options& options) {
	size_t   preset_shard_images        = 8192;
	uint32_t preset_rowgroup_vectors    = 128;
	uint32_t preset_rowgroups_per_shard = 256;
	switch (options.shard_preset) {
	case galp::jpeg::JpegDctShardPreset::kCropLatency:
		preset_shard_images     = 4096;
		preset_rowgroup_vectors = 64;
		break;
	case galp::jpeg::JpegDctShardPreset::kBalanced:
		break;
	case galp::jpeg::JpegDctShardPreset::kThroughput:
		preset_rowgroup_vectors = 256;
		break;
	case galp::jpeg::JpegDctShardPreset::kRandomAccess:
		preset_shard_images        = 8192;
		preset_rowgroup_vectors    = 128;
		preset_rowgroups_per_shard = 8192;
		break;
	}
	if (!options.shard_images_specified) {
		options.shard_images = preset_shard_images;
	}
	if (!options.rowgroup_vectors_specified) {
		options.rowgroup_vectors = preset_rowgroup_vectors;
	}
	if (!options.rowgroups_per_shard_specified) {
		options.rowgroups_per_shard = preset_rowgroups_per_shard;
	}
	if (!options.physical_layout_specified &&
	    options.shard_preset == galp::jpeg::JpegDctShardPreset::kRandomAccess) {
		options.physical_layout = galp::jpeg::JpegDctPhysicalLayout::kImageMajor;
	}
	if (options.spatial_order_specified) {
		if (options.physical_layout_specified &&
		    !galp::jpeg::is_image_major_physical_layout(options.physical_layout)) {
			throw std::runtime_error("--spatial-order requires the image-major physical layout");
		}
		options.physical_layout           = galp::jpeg::JpegDctPhysicalLayout::kImageMajor;
		options.physical_layout_specified = true;
	}
}

bool parse_args(const int argc, char** argv, Options& options) {
	for (int i = 1; i < argc; ++i) {
		const std::string_view arg = argv[i];
		if (arg == "--shard") {
			options.shard_mode = true;
			continue;
		}
		if (arg == "--verify-manifest" && i + 1 < argc) {
			options.verify_manifest = argv[++i];
			options.verify_mode     = true;
			continue;
		}
		if (arg == "--build-sparse-bundle" && i + 1 < argc) {
			options.sparse_bundle_source = argv[++i];
			options.sparse_bundle_mode   = true;
			continue;
		}
		if (arg == "--inspect-encodings") {
			options.inspect_encodings_mode = true;
			continue;
		}
		if (arg == "--inspect-crop-plan" && i + 1 < argc) {
			options.inspect_crop_manifest = argv[++i];
			options.inspect_crop_plan_mode = true;
			continue;
		}
		if (arg == "--benchmark-rowgroup-read" && i + 1 < argc) {
			options.benchmark_rowgroup_read_source = argv[++i];
			options.benchmark_rowgroup_read_mode = true;
			continue;
		}
		if (arg == "--repeats" && i + 1 < argc) {
			options.benchmark_repeats = parse_size_arg(arg, argv[++i]);
			continue;
		}
		if (arg == "--bundle-output" && i + 1 < argc) {
			options.sparse_bundle_output = argv[++i];
			continue;
		}
		if (arg == "--image-index" && i + 1 < argc) {
			options.verify_image_index = parse_u32_arg(arg, argv[++i]);
			options.verify_image_index_specified = true;
			continue;
		}
		if ((arg == "--out" || arg == "-o") && i + 1 < argc) {
			options.output_fls = argv[++i];
			continue;
		}
		if (arg == "--out-dir" && i + 1 < argc) {
			options.output_dir = argv[++i];
			continue;
		}
		if (arg == "--metadata" && i + 1 < argc) {
			options.output_metadata = argv[++i];
			continue;
		}
		if (arg == "--policy" && i + 1 < argc) {
			const std::string_view policy = argv[++i];
			if (policy != "ragged") {
				throw std::runtime_error("unknown --policy value; only ragged is supported");
			}
			continue;
		}
		if (arg == "--metadata-profile" && i + 1 < argc) {
			const std::string_view profile     = argv[++i];
			options.metadata_profile_specified = true;
			if (profile == "dct") {
				options.metadata_profile = galp::jpeg::JpegMetadataProfile::kDctDatasetOnly;
			} else if (profile == "reconstruct") {
				options.metadata_profile = galp::jpeg::JpegMetadataProfile::kReconstructableJpeg;
			} else if (profile == "preserve") {
				options.metadata_profile = galp::jpeg::JpegMetadataProfile::kPreserveOriginalMarkers;
			} else {
				throw std::runtime_error("unknown --metadata-profile value; expected dct, reconstruct, or preserve");
			}
			continue;
		}
		if (arg == "--preset" && i + 1 < argc) {
			const std::string_view preset = argv[++i];
			if (preset == "crop-latency") {
				options.shard_preset = galp::jpeg::JpegDctShardPreset::kCropLatency;
			} else if (preset == "balanced") {
				options.shard_preset = galp::jpeg::JpegDctShardPreset::kBalanced;
			} else if (preset == "throughput") {
				options.shard_preset = galp::jpeg::JpegDctShardPreset::kThroughput;
			} else if (preset == "random-access") {
				options.shard_preset = galp::jpeg::JpegDctShardPreset::kRandomAccess;
			} else {
				throw std::runtime_error(
				    "unknown --preset value; expected crop-latency, balanced, throughput, or random-access");
			}
			continue;
		}
		if (arg == "--physical-layout" && i + 1 < argc) {
			const std::string_view layout = argv[++i];
			options.physical_layout_specified = true;
			if (layout == "spatial-major") {
				options.physical_layout = galp::jpeg::JpegDctPhysicalLayout::kSpatialMajorImageMinor;
			} else if (layout == "image-major") {
				options.physical_layout = galp::jpeg::JpegDctPhysicalLayout::kImageMajor;
			} else if (layout == "image-major-vector-rowgroups") {
				options.physical_layout = galp::jpeg::JpegDctPhysicalLayout::kImageMajorVectorRowgroups;
			} else {
				throw std::runtime_error(
				    "unknown --physical-layout value; expected spatial-major, image-major, or "
				    "image-major-vector-rowgroups");
			}
			continue;
		}
		if (arg == "--spatial-order" && i + 1 < argc) {
			const std::string_view order = argv[++i];
			options.spatial_order_specified = true;
			if (order == "raster") {
				options.spatial_order = galp::jpeg::JpegDctSpatialOrder::kRaster;
			} else if (order == "tiled-raster-32") {
				options.spatial_order = galp::jpeg::JpegDctSpatialOrder::kTiledRaster32;
			} else if (order == "z-order") {
				options.spatial_order = galp::jpeg::JpegDctSpatialOrder::kZOrder;
			} else if (order == "tiled-z-32") {
				options.spatial_order = galp::jpeg::JpegDctSpatialOrder::kTiledZ32;
			} else {
				throw std::runtime_error(
				    "unknown --spatial-order value; expected raster, tiled-raster-32, z-order, or tiled-z-32");
			}
			continue;
		}
		if (arg == "--shard-images" && i + 1 < argc) {
			options.shard_images           = parse_size_arg(arg, argv[++i]);
			options.shard_images_specified = true;
			continue;
		}
		if (arg == "--rowgroup-vectors" && i + 1 < argc) {
			options.rowgroup_vectors           = parse_u32_arg(arg, argv[++i]);
			options.rowgroup_vectors_specified = true;
			continue;
		}
		if (arg == "--rowgroups-per-shard" && i + 1 < argc) {
			options.rowgroups_per_shard           = parse_u32_arg(arg, argv[++i]);
			options.rowgroups_per_shard_specified = true;
			continue;
		}
		if (arg == "--threads" && i + 1 < argc) {
			options.threads           = parse_size_arg(arg, argv[++i]);
			options.threads_specified = true;
			if (options.threads == 0) {
				throw std::runtime_error("--threads must be greater than zero");
			}
			continue;
		}
		if (arg == "--shard-workers" && i + 1 < argc) {
			options.shard_workers = parse_size_arg(arg, argv[++i]);
			if (options.shard_workers == 0) {
				throw std::runtime_error("--shard-workers must be greater than zero");
			}
			continue;
		}
		if (arg == "--help" || arg == "-h") {
			return false;
		}
		options.inputs.emplace_back(argv[i]);
	}

	apply_shard_preset(options);
	if (!options.threads_specified) {
		// Default to saturating all available cores when --threads is not set.
		const unsigned hw = std::thread::hardware_concurrency();
		options.threads    = hw == 0 ? 1 : static_cast<size_t>(hw);
	}
	if (options.shard_mode) {
		return !options.output_dir.empty() && !options.inputs.empty();
	}
	if (options.verify_mode) {
		return !options.verify_manifest.empty() && !options.inputs.empty();
	}
	if (options.sparse_bundle_mode) {
		return !options.sparse_bundle_source.empty() && !options.sparse_bundle_output.empty();
	}
	if (options.inspect_encodings_mode) {
		return !options.inputs.empty();
	}
	if (options.inspect_crop_plan_mode) {
		return !options.inspect_crop_manifest.empty();
	}
	if (options.benchmark_rowgroup_read_mode) {
		return !options.benchmark_rowgroup_read_source.empty();
	}
	return !options.output_fls.empty() && !options.output_metadata.empty() && !options.inputs.empty();
}

} // namespace

int main(const int argc, char** argv) {
	try {
		Options options;
		if (!parse_args(argc, argv, options)) {
			print_usage(argv[0]);
			return 1;
		}

		if (options.sparse_bundle_mode) {
			galp::format::write_sparse_vector_bundle(options.sparse_bundle_source, options.sparse_bundle_output);
			return 0;
		}
		if (options.inspect_encodings_mode) {
			return inspect_encodings(options.inputs);
		}
		if (options.inspect_crop_plan_mode) {
			return inspect_crop_plan(options.inspect_crop_manifest, options.verify_image_index);
		}
		if (options.benchmark_rowgroup_read_mode) {
			return benchmark_rowgroup_read(options.benchmark_rowgroup_read_source, options.benchmark_repeats);
		}

		options.inputs = expand_inputs(options.inputs);
		if (options.verify_mode) {
			if (options.inputs.size() == 1) {
				return verify_manifest_image(
				    options.verify_manifest, options.verify_image_index, options.inputs.front());
			}
			if (options.verify_image_index_specified) {
				throw std::runtime_error("--image-index cannot be combined with multi-image verification");
			}
			return verify_manifest_dataset(options.verify_manifest, options.inputs);
		}

		galp::jpeg::JpegDctReaderOptions reader_options;
		reader_options.capture_metadata_markers =
		    options.metadata_profile == galp::jpeg::JpegMetadataProfile::kPreserveOriginalMarkers;
		reader_options.image_major_spatial_order = options.spatial_order;
		galp::jpeg::JpegDctMetadataWriterOptions writer_options;
		writer_options.profile = options.metadata_profile;
		if (options.shard_mode) {
			galp::jpeg::JpegDctShardOptions shard_options;
			shard_options.shard_images        = options.shard_images;
			shard_options.rowgroup_vectors    = options.rowgroup_vectors;
			shard_options.rowgroups_per_shard = options.rowgroups_per_shard;
			shard_options.threads             = options.threads;
			shard_options.shard_workers       = options.shard_workers;
			shard_options.preset              = options.shard_preset;
			shard_options.shard_images_specified        = options.shard_images_specified;
			shard_options.rowgroup_vectors_specified    = options.rowgroup_vectors_specified;
			shard_options.rowgroups_per_shard_specified = options.rowgroups_per_shard_specified;
			shard_options.physical_layout              = options.physical_layout;
			shard_options.physical_layout_specified    = options.physical_layout_specified;
			galp::jpeg::compress_jpeg_dct_dataset_to_sharded_fls(
			    options.inputs, options.output_dir, reader_options, shard_options, writer_options);
			return 0;
		}
		auto table = options.inputs.size() == 1 ? galp::jpeg::read_jpeg_dct_file(options.inputs.front(), reader_options)
		                                        : galp::jpeg::read_jpeg_dct_dataset(options.inputs, reader_options);
		if (options.metadata_profile_specified) {
			galp::jpeg::compress_jpeg_dct_to_fls(table, options.output_fls, options.output_metadata, writer_options);
		} else {
			galp::jpeg::compress_jpeg_dct_to_fls(table, options.output_fls, options.output_metadata);
		}
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "galp_jpeg_dct_tool: " << e.what() << '\n';
		return 2;
	}
}
