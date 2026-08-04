#include "fls/cfg/cfg.hpp"
#include "fls/connection.hpp"
#include "fls/table/memory_table.hpp"
#include "format/compact_descriptor_v3.hpp"
#include "galp/jpeg_dct_storage.hpp"
#include "jpeg/jpeg_dct_decode.hpp"
#include "jpeg/jpeg_dct_expression_validation.hpp"
#include "jpeg/jpeg_dct_order.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <future>
#include <iomanip>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace galp::jpeg {

namespace {

using detail::ComponentSlot;
using detail::DecodedImage;
using detail::find_component_for_slot;

std::string dct_column_name(const size_t col) {
	std::ostringstream out;
	out << "dct_zz_" << std::setw(2) << std::setfill('0') << col;
	return out.str();
}

class StagedFileCleanup {
public:
	explicit StagedFileCleanup(std::vector<std::filesystem::path> paths)
	    : paths_(std::move(paths)) {
	}

	~StagedFileCleanup() {
		if (!active_) {
			return;
		}
		for (const auto& path : paths_) {
			std::error_code ignored;
			std::filesystem::remove(path, ignored);
		}
	}

	void release() noexcept {
		active_ = false;
	}

private:
	std::vector<std::filesystem::path> paths_;
	bool                               active_ = true;
};

struct DatasetGenerationPaths {
	std::string           name;
	std::filesystem::path staged_dir;
	std::filesystem::path final_dir;
};

class DatasetGenerationCleanup {
public:
	explicit DatasetGenerationCleanup(DatasetGenerationPaths paths)
	    : paths_(std::move(paths)) {
	}

	~DatasetGenerationCleanup() {
		if (!active_) {
			return;
		}
		std::error_code ignored;
		std::filesystem::remove_all(paths_.staged_dir, ignored);
		if (published_) {
			ignored.clear();
			std::filesystem::remove_all(paths_.final_dir, ignored);
		}
	}

	void mark_published() noexcept {
		published_ = true;
	}

	void release() noexcept {
		active_ = false;
	}

private:
	DatasetGenerationPaths paths_;
	bool                   active_    = true;
	bool                   published_ = false;
};

DatasetGenerationPaths create_dataset_generation(const std::filesystem::path& output_dir) {
	const auto timestamp =
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
	        .count();
	for (size_t attempt = 0; attempt < 1024U; ++attempt) {
		DatasetGenerationPaths paths;
		paths.name       = "generation_" + std::to_string(timestamp) + "_" + std::to_string(attempt);
		paths.final_dir  = output_dir / paths.name;
		paths.staged_dir = output_dir / (paths.name + ".tmp");

		std::error_code error;
		const bool      final_exists = std::filesystem::exists(paths.final_dir, error);
		if (error) {
			throw std::runtime_error("failed to inspect JPEG DCT generation path '" + paths.final_dir.string() +
			                         "': " + error.message());
		}
		if (final_exists) {
			continue;
		}
		if (std::filesystem::create_directory(paths.staged_dir, error)) {
			return paths;
		}
		if (error) {
			throw std::runtime_error("failed to create staged JPEG DCT generation directory '" +
			                         paths.staged_dir.string() + "': " + error.message());
		}
	}
	throw std::runtime_error("failed to allocate a unique JPEG DCT dataset generation");
}

void replace_with_staged_file(const std::filesystem::path& staged_path, const std::filesystem::path& final_path) {
	std::error_code error;
	std::filesystem::rename(staged_path, final_path, error);
	if (error) {
		throw std::runtime_error("failed to commit staged JPEG DCT file '" + staged_path.string() + "' as '" +
		                         final_path.string() + "': " + error.message());
	}
}

void publish_staged_generation(const DatasetGenerationPaths& paths) {
	std::error_code error;
	std::filesystem::rename(paths.staged_dir, paths.final_dir, error);
	if (error) {
		throw std::runtime_error("failed to publish staged JPEG DCT generation '" + paths.staged_dir.string() +
		                         "' as '" + paths.final_dir.string() + "': " + error.message());
	}
}

std::vector<uint64_t> make_block_group_aligned_rowgroups(JpegDctDatasetMetadata& metadata,
                                                         const size_t            row_count,
                                                         const uint32_t          rowgroup_vectors) {
	if (rowgroup_vectors == 0) {
		throw std::runtime_error("JPEG DCT shard rowgroup_vectors must be greater than zero");
	}
	const uint64_t        target_rows = static_cast<uint64_t>(rowgroup_vectors) * fastlanes::CFG::VEC_SZ;
	std::vector<uint64_t> rowgroups;
	uint64_t              current_rowgroup_rows = 0;

	for (auto& group : metadata.block_group_index) {
		if (group.row_count == 0) {
			continue;
		}
		if (current_rowgroup_rows != 0 && current_rowgroup_rows + group.row_count > target_rows) {
			rowgroups.push_back(current_rowgroup_rows);
			current_rowgroup_rows = 0;
		}
		group.fls_rowgroup_index    = static_cast<uint32_t>(rowgroups.size());
		group.row_start_in_rowgroup = static_cast<uint32_t>(current_rowgroup_rows);
		current_rowgroup_rows += group.row_count;
	}
	if (current_rowgroup_rows != 0) {
		rowgroups.push_back(current_rowgroup_rows);
	}

	uint64_t sum = 0;
	for (const auto n_tuples : rowgroups) {
		sum += n_tuples;
	}
	if (sum != row_count) {
		std::ostringstream msg;
		msg << "JPEG DCT rowgroup layout covers " << sum << " rows; expected " << row_count;
		throw std::runtime_error(msg.str());
	}
	return rowgroups;
}

std::vector<uint64_t> make_image_aligned_rowgroups(JpegDctDatasetMetadata& metadata,
                                                   const size_t            row_count,
                                                   const uint32_t          rowgroup_vectors) {
	if (rowgroup_vectors == 0) {
		throw std::runtime_error("JPEG DCT shard rowgroup_vectors must be greater than zero");
	}
	const uint64_t        max_rows = static_cast<uint64_t>(rowgroup_vectors) * fastlanes::CFG::VEC_SZ;
	std::vector<uint64_t> rowgroups;
	rowgroups.reserve(metadata.image_group_index.size());
	uint64_t covered_rows = 0;
	for (auto& image_group : metadata.image_group_index) {
		if (image_group.local_image_index != rowgroups.size()) {
			throw std::runtime_error("JPEG DCT image-major index is not dense and ordered");
		}
		if (image_group.row_count == 0) {
			throw std::runtime_error("JPEG DCT image-major record must contain at least one DCT block");
		}
		if (image_group.row_count > max_rows) {
			std::ostringstream msg;
			msg << "JPEG DCT image-major record " << image_group.local_image_index << " contains "
			    << image_group.row_count << " rows, exceeding rowgroup capacity " << max_rows;
			throw std::runtime_error(msg.str());
		}
		image_group.fls_rowgroup_index    = static_cast<uint32_t>(rowgroups.size());
		image_group.row_start_in_rowgroup = 0;
		rowgroups.push_back(image_group.row_count);
		covered_rows += image_group.row_count;
	}
	if (covered_rows != row_count) {
		std::ostringstream msg;
		msg << "JPEG DCT image-major rowgroups cover " << covered_rows << " rows; expected " << row_count;
		throw std::runtime_error(msg.str());
	}
	return rowgroups;
}

std::vector<uint64_t> make_image_vector_rowgroups(JpegDctDatasetMetadata& metadata, const size_t row_count) {
	constexpr uint64_t vector_rows = fastlanes::CFG::VEC_SZ;
	std::vector<uint64_t> rowgroups;
	uint64_t covered_rows = 0;
	for (size_t image_index = 0; image_index < metadata.image_group_index.size(); ++image_index) {
		auto& image_group = metadata.image_group_index[image_index];
		if (image_group.local_image_index != image_index ||
		    image_group.row_start != covered_rows || image_group.row_count == 0U) {
			throw std::runtime_error("JPEG DCT manifest-v3 image-major index is not dense and ordered");
		}
		if (rowgroups.size() > std::numeric_limits<uint32_t>::max()) {
			throw std::runtime_error("JPEG DCT manifest-v3 rowgroup index exceeds the supported range");
		}
		image_group.fls_rowgroup_index    = static_cast<uint32_t>(rowgroups.size());
		image_group.row_start_in_rowgroup = 0U;
		uint64_t remaining = image_group.row_count;
		while (remaining != 0U) {
			const auto chunk_rows = std::min(vector_rows, remaining);
			rowgroups.push_back(chunk_rows);
			remaining -= chunk_rows;
		}
		covered_rows += image_group.row_count;
	}
	if (covered_rows != row_count) {
		std::ostringstream msg;
		msg << "JPEG DCT manifest-v3 vector rowgroups cover " << covered_rows << " rows; expected " << row_count;
		throw std::runtime_error(msg.str());
	}
	return rowgroups;
}

galp::format::CompactV3BuildOptions make_compact_v3_options(const JpegDctDatasetMetadata& metadata,
	                                                        const size_t rowgroup_count) {
	galp::format::CompactV3BuildOptions options;
	options.vector_size   = fastlanes::CFG::VEC_SZ;
	options.spatial_order = static_cast<uint32_t>(metadata.image_major_spatial_order);
	options.images.reserve(metadata.image_group_index.size());
	for (size_t image_index = 0U; image_index < metadata.image_group_index.size(); ++image_index) {
		const auto& group = metadata.image_group_index[image_index];
		if (group.local_image_index != image_index || image_index >= metadata.images.size()) {
			throw std::runtime_error("JPEG DCT Compact v3 image metadata is not dense and ordered");
		}
		const size_t next_rowgroup = image_index + 1U < metadata.image_group_index.size()
		                                 ? metadata.image_group_index[image_index + 1U].fls_rowgroup_index
		                                 : rowgroup_count;
		if (group.fls_rowgroup_index >= next_rowgroup || next_rowgroup > rowgroup_count) {
			throw std::runtime_error("JPEG DCT Compact v3 image rowgroup range is invalid");
		}
		if (group.row_count == 0U || group.row_count > std::numeric_limits<uint32_t>::max()) {
			throw std::runtime_error("JPEG DCT Compact v3 image row count exceeds uint32 range");
		}
		galp::format::CompactV3ImageInput image;
		image.first_rowgroup     = group.fls_rowgroup_index;
		image.rowgroup_count     = static_cast<uint32_t>(next_rowgroup - group.fls_rowgroup_index);
		image.real_row_count     = group.row_count;
		image.first_physical_row = group.row_start;
		uint64_t component_row_offset = 0U;
		for (const auto& component : metadata.images[image_index].components) {
			if (!component.present) {
				continue;
			}
			if (component_row_offset > std::numeric_limits<uint32_t>::max()) {
				throw std::runtime_error("JPEG DCT Compact v3 component row offset exceeds uint32 range");
			}
			galp::format::CompactV3ComponentInput compact_component;
			compact_component.semantic_slot_id        = component.semantic_slot_id;
			compact_component.width_in_blocks         = component.width_in_blocks;
			compact_component.height_in_blocks        = component.height_in_blocks;
			compact_component.padded_width_in_blocks  = component.padded_width_in_blocks;
			compact_component.padded_height_in_blocks = component.padded_height_in_blocks;
			compact_component.row_offset              = static_cast<uint32_t>(component_row_offset);
			compact_component.component_index = static_cast<uint32_t>(component.component_index);
			image.components.push_back(compact_component);
			component_row_offset +=
			    static_cast<uint64_t>(component.width_in_blocks) * component.height_in_blocks;
		}
		if (component_row_offset != group.row_count) {
			throw std::runtime_error("JPEG DCT Compact v3 component grids do not cover the image record");
		}
		options.images.push_back(std::move(image));
	}
	return options;
}

uint64_t decoded_image_vector_count(const DecodedImage& image) {
	uint64_t image_rows = 0U;
	for (const auto& component : image.components) {
		const auto component_rows =
		    static_cast<uint64_t>(component.metadata.width_in_blocks) * component.metadata.height_in_blocks;
		if (component_rows > std::numeric_limits<uint64_t>::max() - image_rows) {
			throw std::runtime_error("JPEG DCT image-major row count overflow");
		}
		image_rows += component_rows;
	}
	return (image_rows + fastlanes::CFG::VEC_SZ - 1U) / fastlanes::CFG::VEC_SZ;
}

uint32_t image_major_rowgroup_vectors(const std::vector<DecodedImage>& images, const uint32_t configured_minimum) {
	uint64_t max_image_rows = 0;
	for (const auto& image : images) {
		uint64_t image_rows = 0;
		for (const auto& component : image.components) {
			const auto component_rows =
			    static_cast<uint64_t>(component.metadata.width_in_blocks) * component.metadata.height_in_blocks;
			if (component_rows > std::numeric_limits<uint64_t>::max() - image_rows) {
				throw std::runtime_error("JPEG DCT image-major row count overflow");
			}
			image_rows += component_rows;
		}
		max_image_rows = std::max(max_image_rows, image_rows);
	}

	const auto required_vectors = (max_image_rows + fastlanes::CFG::VEC_SZ - 1U) / fastlanes::CFG::VEC_SZ;
	uint64_t   capacity_vectors = std::max<uint64_t>(1, configured_minimum);
	while (capacity_vectors < required_vectors) {
		if (capacity_vectors > std::numeric_limits<uint32_t>::max() / 2U) {
			throw std::runtime_error("JPEG DCT image-major rowgroup capacity exceeds the supported range");
		}
		capacity_vectors *= 2U;
	}
	return static_cast<uint32_t>(capacity_vectors);
}

// Precomputed data for shard-boundary sizing. Building the block orders and per-image slot grid dimensions once (rather
// than re-deriving them for every binary-search probe) turns shard sizing from O(slots * global_max_grid * images) per
// probe into O(images + slots * global_max_grid). For large heterogeneous datasets (e.g. ImageNet, whose largest image
// inflates the global max grid) the old cost dominated the whole shard writer and looked like a hang.
struct ShardSizingContext {
	std::vector<std::vector<detail::MortonBlockCoord>> block_orders; // [slot]
	std::vector<std::vector<uint32_t>>                 slot_width;   // [slot][image] (0 when component absent)
	std::vector<std::vector<uint32_t>>                 slot_height;  // [slot][image] (0 when component absent)
	std::vector<uint32_t>                              max_width;    // [slot]
	std::vector<uint32_t>                              max_height;   // [slot]
	uint64_t                                           target_rows = 0;
};

ShardSizingContext build_shard_sizing_context(const std::vector<DecodedImage>&  layout_images,
                                              const std::vector<ComponentSlot>& global_slots,
                                              const JpegDctReaderOptions&       options,
                                              const uint32_t                    rowgroup_vectors) {
	ShardSizingContext ctx;
	ctx.target_rows      = static_cast<uint64_t>(rowgroup_vectors) * fastlanes::CFG::VEC_SZ;
	const size_t n_slots = global_slots.size();
	ctx.block_orders.resize(n_slots);
	ctx.slot_width.assign(n_slots, std::vector<uint32_t>(layout_images.size(), 0));
	ctx.slot_height.assign(n_slots, std::vector<uint32_t>(layout_images.size(), 0));
	ctx.max_width.resize(n_slots);
	ctx.max_height.resize(n_slots);
	for (size_t slot_idx = 0; slot_idx < n_slots; ++slot_idx) {
		const auto& slot           = global_slots[slot_idx];
		ctx.max_width[slot_idx]    = slot.max_width_in_blocks;
		ctx.max_height[slot_idx]   = slot.max_height_in_blocks;
		ctx.block_orders[slot_idx] = detail::make_block_order(
		    slot.max_width_in_blocks, slot.max_height_in_blocks, options.use_z_curve_block_order);
		for (size_t image_idx = 0; image_idx < layout_images.size(); ++image_idx) {
			const auto* component = find_component_for_slot(layout_images[image_idx], slot);
			if (component != nullptr) {
				ctx.slot_width[slot_idx][image_idx]  = component->metadata.width_in_blocks;
				ctx.slot_height[slot_idx][image_idx] = component->metadata.height_in_blocks;
			}
		}
	}
	return ctx;
}

// Mirrors the greedy packing in make_block_group_aligned_rowgroups. For each slot the per-block-group row count is the
// number of images in [first_image, first_image + image_count) whose component reaches that block. We obtain those
// counts in O(1) via a 2D suffix histogram over (width_in_blocks, height_in_blocks): group_row_count(x, y) is the
// number of images with width > x and height > y. `scratch` is reused across probes to avoid per-call allocation.
size_t estimate_shard_rowgroup_count(const ShardSizingContext& ctx,
                                     const size_t              first_image,
                                     const size_t              image_count,
                                     std::vector<uint64_t>&    scratch) {
	const uint64_t target_rows           = ctx.target_rows;
	size_t         rowgroup_count        = 0;
	uint64_t       current_rowgroup_rows = 0;

	const auto pack_group = [&](const uint64_t group_row_count) {
		if (group_row_count == 0) {
			return;
		}
		if (current_rowgroup_rows != 0 && current_rowgroup_rows + group_row_count > target_rows) {
			++rowgroup_count;
			current_rowgroup_rows = 0;
		}
		current_rowgroup_rows += group_row_count;
	};

	for (size_t slot_idx = 0; slot_idx < ctx.block_orders.size(); ++slot_idx) {
		const uint32_t max_w  = ctx.max_width[slot_idx];
		const uint32_t max_h  = ctx.max_height[slot_idx];
		const size_t   stride = static_cast<size_t>(max_h) + 2;
		scratch.assign((static_cast<size_t>(max_w) + 2) * stride, 0);
		const auto& widths  = ctx.slot_width[slot_idx];
		const auto& heights = ctx.slot_height[slot_idx];
		for (size_t image_idx = first_image; image_idx < first_image + image_count; ++image_idx) {
			const uint32_t w = widths[image_idx];
			const uint32_t h = heights[image_idx];
			if (w == 0 || h == 0) {
				continue;
			}
			const size_t cw = w > max_w ? max_w : w;
			const size_t ch = h > max_h ? max_h : h;
			++scratch[cw * stride + ch];
		}
		// Suffix sum so that scratch[x][y] becomes the number of images with width >= x and height >= y.
		for (size_t x = static_cast<size_t>(max_w) + 1; x-- > 0;) {
			for (size_t y = static_cast<size_t>(max_h) + 1; y-- > 0;) {
				scratch[x * stride + y] +=
				    scratch[(x + 1) * stride + y] + scratch[x * stride + (y + 1)] - scratch[(x + 1) * stride + (y + 1)];
			}
		}
		for (const auto& block_coord : ctx.block_orders[slot_idx]) {
			// group_row_count(x, y) = images with width > x and height > y = suffix at (x + 1, y + 1).
			pack_group(scratch[(static_cast<size_t>(block_coord.x) + 1) * stride + (block_coord.y + 1)]);
		}
	}
	if (current_rowgroup_rows != 0) {
		++rowgroup_count;
	}
	return rowgroup_count;
}

size_t choose_shard_image_count(const ShardSizingContext&  ctx,
                                const size_t               first_image,
                                const size_t               max_image_count,
                                const JpegDctShardOptions& shard_options,
                                std::vector<uint64_t>&     scratch) {
	if (estimate_shard_rowgroup_count(ctx, first_image, max_image_count, scratch) <=
	    shard_options.rowgroups_per_shard) {
		return max_image_count;
	}

	size_t lo = 1;
	size_t hi = max_image_count;
	while (lo < hi) {
		const auto mid = lo + (hi - lo + 1) / 2;
		if (estimate_shard_rowgroup_count(ctx, first_image, mid, scratch) <= shard_options.rowgroups_per_shard) {
			lo = mid;
		} else {
			hi = mid - 1;
		}
	}

	if (estimate_shard_rowgroup_count(ctx, first_image, lo, scratch) > shard_options.rowgroups_per_shard) {
		std::ostringstream msg;
		msg << "JPEG DCT single-image shard at global image index " << first_image << " requires more than "
		    << shard_options.rowgroups_per_shard << " rowgroups; increase --rowgroups-per-shard or --rowgroup-vectors";
		throw std::runtime_error(msg.str());
	}
	return lo;
}

void write_jpeg_dct_fls_data(const JpegDctTable&                  table,
                             const std::filesystem::path&         fls_output_path,
                             const fastlanes::MemoryTableOptions& options       = {},
                             const bool                           inline_footer = false,
                             const fastlanes::EncodingOptions&    encoding_options = {}) {
	std::array<fastlanes::MemoryColumn, 64> columns;
	for (size_t col = 0; col < columns.size(); ++col) {
		columns[col].name = dct_column_name(col);
		columns[col].data = std::span<const int16_t>(table.columns[col].data(), table.columns[col].size());
	}

	const fastlanes::MemoryTable memory_table {
	    std::span<const fastlanes::MemoryColumn>(columns.data(), columns.size())};
	fastlanes::Connection connection;
	fastlanes::load_memory_table(connection, memory_table, options);
	if (inline_footer) {
		connection.inline_footer();
	}
	connection.to_fls(fls_output_path, encoding_options);
}

std::string shard_file_name(const uint32_t shard_id, const std::string_view suffix) {
	std::ostringstream out;
	out << "shard_" << std::setw(6) << std::setfill('0') << shard_id << suffix;
	return out.str();
}

JpegDctShardOptions effective_shard_options(JpegDctShardOptions options) {
	constexpr size_t   kBalancedShardImages       = 8192;
	constexpr uint32_t kBalancedRowgroupVectors   = 128;
	constexpr uint32_t kBalancedRowgroupsPerShard = 256;
	size_t             preset_shard_images        = kBalancedShardImages;
	uint32_t           preset_rowgroup_vectors    = kBalancedRowgroupVectors;
	uint32_t           preset_rowgroups_per_shard = kBalancedRowgroupsPerShard;

	switch (options.preset) {
	case JpegDctShardPreset::kCropLatency:
		preset_shard_images     = 4096;
		preset_rowgroup_vectors = 64;
		break;
	case JpegDctShardPreset::kBalanced:
		break;
	case JpegDctShardPreset::kThroughput:
		preset_rowgroup_vectors = 256;
		break;
	case JpegDctShardPreset::kRandomAccess:
		// One independently addressable rowgroup per image. Keep a large shard
		// so globally shuffled batches still span only a small number of files.
		preset_shard_images        = 8192;
		preset_rowgroup_vectors    = 128;
		preset_rowgroups_per_shard = 8192;
		break;
	}

	if (!options.shard_images_specified && options.shard_images == kBalancedShardImages) {
		options.shard_images = preset_shard_images;
	}
	if (!options.rowgroup_vectors_specified && options.rowgroup_vectors == kBalancedRowgroupVectors) {
		options.rowgroup_vectors = preset_rowgroup_vectors;
	}
	if (!options.rowgroups_per_shard_specified && options.rowgroups_per_shard == kBalancedRowgroupsPerShard) {
		options.rowgroups_per_shard = preset_rowgroups_per_shard;
	}
	if (!options.physical_layout_specified && options.preset == JpegDctShardPreset::kRandomAccess) {
		options.physical_layout = JpegDctPhysicalLayout::kImageMajor;
	}

	const bool legacy_threads_selected = options.threads_specified || options.threads != 1U;
	const bool layout_threads_selected = options.layout_threads_specified || options.layout_threads != 1U;
	const bool decode_threads_selected =
	    options.shard_decode_threads_specified || options.shard_decode_threads != 1U;
	if (legacy_threads_selected) {
		if (layout_threads_selected && options.layout_threads != options.threads) {
			throw std::invalid_argument("legacy threads conflict with layout_threads");
		}
		if (decode_threads_selected && options.shard_decode_threads != options.threads) {
			throw std::invalid_argument("legacy threads conflict with shard_decode_threads");
		}
		options.layout_threads       = options.threads;
		options.shard_decode_threads = options.threads;
	}
	return options;
}

} // namespace
void compress_jpeg_dct_to_fls(const JpegDctTable&                 table,
                              const std::filesystem::path&        fls_output_path,
                              const std::filesystem::path&        metadata_output_path,
                              const JpegDctMetadataWriterOptions& metadata_options) {
	write_jpeg_dct_fls_data(table, fls_output_path);
	write_jpeg_dct_metadata(table.metadata, metadata_output_path, metadata_options);
}

void compress_jpeg_dct_to_fls(const JpegDctTable&          table,
                              const std::filesystem::path& fls_output_path,
                              const std::filesystem::path& metadata_output_path) {
	write_jpeg_dct_fls_data(table, fls_output_path);
	write_jpeg_dct_metadata(table.metadata, metadata_output_path);
}

void compress_jpeg_dct_file_to_fls(const std::filesystem::path&        jpeg_path,
                                   const std::filesystem::path&        fls_output_path,
                                   const std::filesystem::path&        metadata_output_path,
                                   const JpegDctReaderOptions&         options,
                                   const JpegDctMetadataWriterOptions& metadata_options) {
	auto read_options = options;
	if (metadata_options.profile == JpegMetadataProfile::kPreserveOriginalMarkers) {
		read_options.capture_metadata_markers = true;
	}
	compress_jpeg_dct_to_fls(
	    read_jpeg_dct_file(jpeg_path, read_options), fls_output_path, metadata_output_path, metadata_options);
}

void compress_jpeg_dct_file_to_fls(const std::filesystem::path& jpeg_path,
                                   const std::filesystem::path& fls_output_path,
                                   const std::filesystem::path& metadata_output_path,
                                   const JpegDctReaderOptions&  options) {
	compress_jpeg_dct_to_fls(read_jpeg_dct_file(jpeg_path, options), fls_output_path, metadata_output_path);
}

void compress_jpeg_dct_dataset_to_fls(const std::vector<std::filesystem::path>& jpeg_paths,
                                      const std::filesystem::path&              fls_output_path,
                                      const std::filesystem::path&              metadata_output_path,
                                      const JpegDctReaderOptions&               options,
                                      const JpegDctMetadataWriterOptions&       metadata_options) {
	auto read_options = options;
	if (metadata_options.profile == JpegMetadataProfile::kPreserveOriginalMarkers) {
		read_options.capture_metadata_markers = true;
	}
	compress_jpeg_dct_to_fls(
	    read_jpeg_dct_dataset(jpeg_paths, read_options), fls_output_path, metadata_output_path, metadata_options);
}

void compress_jpeg_dct_dataset_to_fls(const std::vector<std::filesystem::path>& jpeg_paths,
                                      const std::filesystem::path&              fls_output_path,
                                      const std::filesystem::path&              metadata_output_path,
                                      const JpegDctReaderOptions&               options) {
	compress_jpeg_dct_to_fls(read_jpeg_dct_dataset(jpeg_paths, options), fls_output_path, metadata_output_path);
}

JpegDctShardManifest compress_jpeg_dct_dataset_to_sharded_fls(const std::vector<std::filesystem::path>& jpeg_paths,
                                                              const std::filesystem::path&              output_dir,
                                                              const JpegDctReaderOptions&               options,
                                                              const JpegDctShardOptions&                shard_options,
                                                              const JpegDctMetadataWriterOptions& metadata_options) {
	auto effective_options = effective_shard_options(shard_options);
	if (jpeg_paths.empty()) {
		throw std::runtime_error("JPEG DCT sharded dataset requires at least one image");
	}
	if (effective_options.shard_images == 0) {
		throw std::runtime_error("JPEG DCT shard_images must be greater than zero");
	}
	if (effective_options.rowgroup_vectors == 0) {
		throw std::runtime_error("JPEG DCT rowgroup_vectors must be greater than zero");
	}
	if (effective_options.rowgroups_per_shard == 0) {
		throw std::runtime_error("JPEG DCT rowgroups_per_shard must be greater than zero");
	}
	if (effective_options.layout_threads == 0) {
		throw std::runtime_error("JPEG DCT layout threads must be greater than zero");
	}
	if (effective_options.shard_decode_threads == 0) {
		throw std::runtime_error("JPEG DCT shard decode threads must be greater than zero");
	}
	if (effective_options.shard_workers == 0) {
		throw std::runtime_error("JPEG DCT shard workers must be greater than zero");
	}
	if (effective_options.encoding_workers_per_shard == 0) {
		throw std::runtime_error("JPEG DCT encoding workers per shard must be greater than zero");
	}
	constexpr size_t kMaxConcurrentPipelineThreads = 256U;
	const auto per_shard_threads =
	    std::max(effective_options.shard_decode_threads, effective_options.encoding_workers_per_shard);
	if (effective_options.layout_threads > kMaxConcurrentPipelineThreads ||
	    effective_options.shard_workers > kMaxConcurrentPipelineThreads / per_shard_threads) {
		throw std::runtime_error("JPEG DCT configured pipeline parallelism exceeds the 256-thread safety bound");
	}

	std::filesystem::create_directories(output_dir);

	auto read_options = options;
	if (metadata_options.profile == JpegMetadataProfile::kPreserveOriginalMarkers) {
		read_options.capture_metadata_markers = true;
	}
	if (effective_options.physical_layout == JpegDctPhysicalLayout::kImageMajorVectorRowgroups) {
		read_options.image_major_spatial_order = JpegDctSpatialOrder::kTiledZ32;
	}
	auto global_layout_images =
	    detail::decode_jpeg_layouts_parallel(jpeg_paths, read_options, effective_options.layout_threads);
	detail::validate_supported_layout_options(read_options);
	detail::validate_decoded_dataset(global_layout_images);
	const auto global_slots = detail::normalize_component_slots(global_layout_images);
	if (effective_options.physical_layout == JpegDctPhysicalLayout::kImageMajor) {
		// An image is the random-access storage atom. Derive a dataset-wide capacity from the decoded JPEG layouts so
		// even unusually large images remain one independently addressable FLS rowgroup. The configured value is a
		// minimum, not a correctness-sensitive knob; the effective value is persisted in the manifest.
		effective_options.rowgroup_vectors =
		    image_major_rowgroup_vectors(global_layout_images, effective_options.rowgroup_vectors);
	} else if (effective_options.physical_layout == JpegDctPhysicalLayout::kImageMajorVectorRowgroups) {
		// The v3 prototype deliberately makes the physical storage atom exactly
		// one FastLanes vector. The caller's configured value is ignored and the
		// effective value is persisted in the manifest for reproducibility.
		effective_options.rowgroup_vectors = 1U;
		// A legacy 256-rowgroup cap would repeat the shared schema dictionary
		// across thousands of tiny shards at 50K-image scale. Keep explicit
		// caller limits, but use a scale-oriented v3 default.
		if (!shard_options.rowgroups_per_shard_specified) {
			effective_options.rowgroups_per_shard = 8192U;
		}
	}

	JpegDctShardManifest manifest;
	manifest.version             = effective_options.physical_layout == JpegDctPhysicalLayout::kImageMajorVectorRowgroups
	                                   ? 3U
	                               : effective_options.physical_layout == JpegDctPhysicalLayout::kImageMajor ? 2U
	                                                                                                          : 1U;
	manifest.rowgroup_vectors    = effective_options.rowgroup_vectors;
	manifest.rowgroups_per_shard = effective_options.rowgroups_per_shard;
	manifest.image_count         = jpeg_paths.size();
	if (manifest.version == 3U) {
		manifest.physical_layout = "image-major-vector-rowgroups";
		manifest.descriptor_kind = "galp-compact-v1";
		manifest.vector_size     = fastlanes::CFG::VEC_SZ;
		manifest.spatial_order_name = "tiled-z32";
		manifest.spatial_order   = read_options.image_major_spatial_order;
	}

	struct ShardWorkItem {
		size_t first_image = 0;
		size_t image_count = 0;
	};
	std::vector<ShardWorkItem>        shard_work_items;
	std::optional<ShardSizingContext> sizing_ctx;
	if (effective_options.physical_layout == JpegDctPhysicalLayout::kSpatialMajorImageMinor) {
		sizing_ctx.emplace(build_shard_sizing_context(
		    global_layout_images, global_slots, read_options, effective_options.rowgroup_vectors));
	}
	std::vector<uint64_t> sizing_scratch;
	for (size_t first_image = 0; first_image < jpeg_paths.size();) {
		const auto max_shard_image_count = std::min(effective_options.shard_images, jpeg_paths.size() - first_image);
		size_t shard_image_count = 0U;
		if (effective_options.physical_layout == JpegDctPhysicalLayout::kImageMajor) {
			shard_image_count =
			    std::min(max_shard_image_count, static_cast<size_t>(effective_options.rowgroups_per_shard));
		} else if (effective_options.physical_layout == JpegDctPhysicalLayout::kImageMajorVectorRowgroups) {
			uint64_t shard_rowgroups = 0U;
			while (shard_image_count < max_shard_image_count) {
				const auto image_vectors = decoded_image_vector_count(global_layout_images[first_image + shard_image_count]);
				if (image_vectors == 0U || image_vectors > effective_options.rowgroups_per_shard) {
					throw std::runtime_error(
					    "JPEG DCT manifest-v3 image requires more vector rowgroups than --rowgroups-per-shard permits");
				}
				if (shard_rowgroups + image_vectors > effective_options.rowgroups_per_shard) {
					break;
				}
				shard_rowgroups += image_vectors;
				++shard_image_count;
			}
		} else {
			shard_image_count = choose_shard_image_count(
			    *sizing_ctx, first_image, max_shard_image_count, effective_options, sizing_scratch);
		}
		if (shard_image_count == 0) {
			throw std::runtime_error("JPEG DCT shard sizing produced an empty shard");
		}
		shard_work_items.push_back(ShardWorkItem {first_image, shard_image_count});
		first_image += shard_image_count;
	}

	struct ShardOutputPaths {
		std::string           fls_name;
		std::string           metadata_name;
		std::filesystem::path fls_staged;
		std::filesystem::path metadata_staged;
	};
	const auto                    generation = create_dataset_generation(output_dir);
	DatasetGenerationCleanup      generation_cleanup(generation);
	std::vector<ShardOutputPaths> shard_output_paths;
	shard_output_paths.reserve(shard_work_items.size());
	for (size_t shard_id = 0; shard_id < shard_work_items.size(); ++shard_id) {
		ShardOutputPaths paths;
		const auto       fls_base      = shard_file_name(static_cast<uint32_t>(shard_id), ".fls");
		const auto       metadata_base = shard_file_name(static_cast<uint32_t>(shard_id), ".meta.bin");
		paths.fls_name                 = (std::filesystem::path(generation.name) / fls_base).generic_string();
		paths.metadata_name            = (std::filesystem::path(generation.name) / metadata_base).generic_string();
		paths.fls_staged               = generation.staged_dir / fls_base;
		paths.metadata_staged          = generation.staged_dir / metadata_base;
		shard_output_paths.push_back(std::move(paths));
	}
	const auto        manifest_path        = output_dir / "manifest.bin";
	const auto        manifest_staged_path = output_dir / ("manifest.bin." + generation.name + ".tmp");
	StagedFileCleanup manifest_cleanup({manifest_staged_path});

	std::vector<JpegDctShardManifestEntry> shard_entries(shard_work_items.size());
	const auto                             process_shard = [&](const size_t shard_id) {
        const auto&                        work              = shard_work_items[shard_id];
        const auto                         first_image       = work.first_image;
        const auto                         shard_image_count = work.image_count;
        const auto                         shard_begin = jpeg_paths.begin() + static_cast<std::ptrdiff_t>(first_image);
        const auto                         shard_end = shard_begin + static_cast<std::ptrdiff_t>(shard_image_count);
        std::vector<std::filesystem::path> shard_paths(shard_begin, shard_end);

        auto shard_images = detail::decode_jpeg_coefficients_parallel(
            shard_paths, read_options, effective_options.shard_decode_threads);
        auto table = detail::make_dataset_table(
            std::move(shard_images), read_options, &global_slots, effective_options.physical_layout);
		if (effective_options.physical_layout == JpegDctPhysicalLayout::kImageMajor) {
			table.rowgroup_n_tuples =
			    make_image_aligned_rowgroups(table.metadata, table.row_count, effective_options.rowgroup_vectors);
		} else if (effective_options.physical_layout == JpegDctPhysicalLayout::kImageMajorVectorRowgroups) {
			table.rowgroup_n_tuples = make_image_vector_rowgroups(table.metadata, table.row_count);
		} else {
			table.rowgroup_n_tuples =
			    make_block_group_aligned_rowgroups(table.metadata, table.row_count, effective_options.rowgroup_vectors);
		}
        if (table.rowgroup_n_tuples.size() > effective_options.rowgroups_per_shard) {
            std::ostringstream msg;
            msg << "JPEG DCT shard produced " << table.rowgroup_n_tuples.size()
                << " rowgroups after layout estimation selected " << shard_image_count
                << " images; configured maximum is " << effective_options.rowgroups_per_shard;
            throw std::runtime_error(msg.str());
        }

        const auto& output_paths = shard_output_paths[shard_id];

        fastlanes::MemoryTableOptions memory_options;
        memory_options.n_vectors_per_rowgroup = effective_options.rowgroup_vectors;
        memory_options.rowgroup_n_tuples =
            std::span<const fastlanes::n_t>(table.rowgroup_n_tuples.data(), table.rowgroup_n_tuples.size());
        // JPEG-DCT production storage always uses the default FastLanes wizard.
        // Tests that need a specific root token exercise MemoryTableOptions
        // directly instead of exposing a force-schema switch in this API.
		fastlanes::EncodingOptions encoding_options;
		encoding_options.worker_count = effective_options.encoding_workers_per_shard;
		galp::format::CompactV3Report compact_report;
		if (manifest.version == 3U) {
			const auto standard_path = std::filesystem::path(output_paths.fls_staged.string() + ".standard.tmp");
			write_jpeg_dct_fls_data(table, standard_path, memory_options, true, encoding_options);
			detail::validate_jpeg_dct_fls_gpu_expressions(standard_path, static_cast<uint32_t>(shard_id));
			compact_report = galp::format::compact_standard_fls_to_v3(
			    standard_path,
			    output_paths.fls_staged,
			    make_compact_v3_options(table.metadata, table.rowgroup_n_tuples.size()));
			std::error_code remove_error;
			std::filesystem::remove(standard_path, remove_error);
			if (remove_error) {
				throw std::runtime_error("failed to remove temporary standard FLS shard: " + remove_error.message());
			}
			static_cast<void>(galp::format::CompactDescriptorV3::Open(output_paths.fls_staged));
		} else {
			write_jpeg_dct_fls_data(table, output_paths.fls_staged, memory_options, true, encoding_options);
			detail::validate_jpeg_dct_fls_gpu_expressions(
			    output_paths.fls_staged, static_cast<uint32_t>(shard_id));
		}
        write_jpeg_dct_metadata(table.metadata, output_paths.metadata_staged, metadata_options);

        JpegDctShardManifestEntry entry;
        entry.shard_id                 = static_cast<uint32_t>(shard_id);
        entry.first_global_image_index = first_image;
        entry.image_count              = static_cast<uint32_t>(shard_image_count);
        entry.real_row_count           = table.real_row_count;
        entry.padding_row_count        = table.padding_row_count;
        entry.physical_row_count       = table.row_count;
        entry.rowgroup_count     = static_cast<uint32_t>(table.rowgroup_n_tuples.size());
        entry.block_group_count  = static_cast<uint32_t>(table.block_group_count);
        entry.fls_file_size      = std::filesystem::file_size(output_paths.fls_staged);
        entry.metadata_file_size = std::filesystem::file_size(output_paths.metadata_staged);
		entry.payload_size             = compact_report.payload_bytes;
		entry.payload_crc64            = compact_report.payload_crc64;
		entry.compact_descriptor_size  = compact_report.compact_descriptor_bytes;
		entry.source_descriptor_size   = compact_report.source_descriptor_bytes;
        entry.fls_file_name      = output_paths.fls_name;
        entry.metadata_file_name = output_paths.metadata_name;
        shard_entries[shard_id]  = std::move(entry);
	};

	const size_t shard_worker_count =
	    std::max<size_t>(1, std::min(effective_options.shard_workers, shard_work_items.size()));
	if (shard_worker_count == 1) {
		for (size_t shard_id = 0; shard_id < shard_work_items.size(); ++shard_id) {
			process_shard(shard_id);
		}
	} else {
		std::vector<std::future<void>> futures;
		futures.reserve(shard_worker_count);
		for (size_t worker = 0; worker < shard_worker_count; ++worker) {
			futures.push_back(std::async(std::launch::async, [&, worker] {
				for (size_t shard_id = worker; shard_id < shard_work_items.size(); shard_id += shard_worker_count) {
					process_shard(shard_id);
				}
			}));
		}
		for (auto& future : futures) {
			future.get();
		}
	}

	manifest.shards = std::move(shard_entries);
	// Build the complete manifest before changing any final path. Unsupported
	// expressions and serialization failures therefore leave the previous
	// committed dataset untouched.
	write_jpeg_dct_shard_manifest(manifest, manifest_staged_path);
	// Publish this immutable generation under names that no committed manifest
	// references yet. Existing readers therefore continue to resolve every shard
	// from the previous generation until the manifest replacement below.
	publish_staged_generation(generation);
	generation_cleanup.mark_published();
	// The manifest replacement is the only visibility/commit point. If it fails,
	// generation_cleanup removes the unreferenced published generation while the
	// previous manifest and all of its shards remain untouched.
	replace_with_staged_file(manifest_staged_path, manifest_path);
	manifest_cleanup.release();
	generation_cleanup.release();
	return manifest;
}

} // namespace galp::jpeg
