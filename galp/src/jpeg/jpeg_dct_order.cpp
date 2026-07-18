#include "jpeg/jpeg_dct_order.hpp"
#include <algorithm>
#include <stdexcept>

namespace galp::jpeg::detail {

namespace {

constexpr uint32_t kSpatialTileBlocks = 32;

uint64_t part_1_by_1(const uint32_t value) {
	uint64_t x = value;
	x          = (x | (x << 16U)) & 0x0000ffff0000ffffULL;
	x          = (x | (x << 8U)) & 0x00ff00ff00ff00ffULL;
	x          = (x | (x << 4U)) & 0x0f0f0f0f0f0f0f0fULL;
	x          = (x | (x << 2U)) & 0x3333333333333333ULL;
	x          = (x | (x << 1U)) & 0x5555555555555555ULL;
	return x;
}

uint64_t rectangle_intersection_count(const uint64_t width,
                                      const uint64_t height,
                                      const uint64_t origin_x,
                                      const uint64_t origin_y,
                                      const uint64_t size) {
	if (origin_x >= width || origin_y >= height) {
		return 0;
	}
	return std::min(size, width - origin_x) * std::min(size, height - origin_y);
}

uint64_t morton_rank_in_rectangle(const uint32_t width,
                                  const uint32_t height,
                                  const uint32_t x,
                                  const uint32_t y) {
	uint64_t size = 1;
	while (size < std::max<uint64_t>(width, height)) {
		size <<= 1U;
	}

	uint64_t rank     = 0;
	uint64_t origin_x = 0;
	uint64_t origin_y = 0;
	while (size > 1) {
		const auto half = size >> 1U;
		const auto qx   = static_cast<uint32_t>(x >= origin_x + half);
		const auto qy   = static_cast<uint32_t>(y >= origin_y + half);
		const auto target_quadrant = qx | (qy << 1U);
		for (uint32_t quadrant = 0; quadrant < target_quadrant; ++quadrant) {
			const auto child_x = origin_x + ((quadrant & 1U) != 0 ? half : 0);
			const auto child_y = origin_y + ((quadrant & 2U) != 0 ? half : 0);
			rank += rectangle_intersection_count(width, height, child_x, child_y, half);
		}
		if (qx != 0) {
			origin_x += half;
		}
		if (qy != 0) {
			origin_y += half;
		}
		size = half;
	}
	return rank;
}

void append_raster_tile(std::vector<MortonBlockCoord>& order,
                        const uint32_t                  tile_x,
                        const uint32_t                  tile_y,
                        const uint32_t                  tile_width,
                        const uint32_t                  tile_height) {
	for (uint32_t local_y = 0; local_y < tile_height; ++local_y) {
		for (uint32_t local_x = 0; local_x < tile_width; ++local_x) {
			const auto x = tile_x + local_x;
			const auto y = tile_y + local_y;
			order.push_back(MortonBlockCoord {x, y, morton_key(local_x, local_y)});
		}
	}
}

void append_z_tile(std::vector<MortonBlockCoord>& order,
                   const uint32_t                  tile_x,
                   const uint32_t                  tile_y,
                   const uint32_t                  tile_width,
                   const uint32_t                  tile_height) {
	const auto begin = order.size();
	append_raster_tile(order, tile_x, tile_y, tile_width, tile_height);
	std::stable_sort(order.begin() + static_cast<std::ptrdiff_t>(begin),
	                 order.end(),
	                 [tile_x, tile_y](const MortonBlockCoord& lhs, const MortonBlockCoord& rhs) {
		                 const auto lhs_key = morton_key(lhs.x - tile_x, lhs.y - tile_y);
		                 const auto rhs_key = morton_key(rhs.x - tile_x, rhs.y - tile_y);
		                 if (lhs_key != rhs_key) {
			                 return lhs_key < rhs_key;
		                 }
		                 if (lhs.y != rhs.y) {
			                 return lhs.y < rhs.y;
		                 }
		                 return lhs.x < rhs.x;
	                 });
}

} // namespace

uint64_t morton_key(const uint32_t x, const uint32_t y) {
	return part_1_by_1(x) | (part_1_by_1(y) << 1U);
}

std::vector<MortonBlockCoord> make_block_order(const uint32_t width_in_blocks,
                                               const uint32_t height_in_blocks,
                                               const bool     use_z_curve_order) {
	return make_block_order(width_in_blocks,
	                        height_in_blocks,
	                        use_z_curve_order ? JpegDctSpatialOrder::kZOrder : JpegDctSpatialOrder::kRaster);
}

std::vector<MortonBlockCoord> make_block_order(const uint32_t            width_in_blocks,
                                               const uint32_t            height_in_blocks,
                                               const JpegDctSpatialOrder spatial_order) {
	if (width_in_blocks == 0 || height_in_blocks == 0) {
		throw std::runtime_error("JPEG component has an empty DCT block grid");
	}

	std::vector<MortonBlockCoord> order;
	order.reserve(static_cast<size_t>(width_in_blocks) * static_cast<size_t>(height_in_blocks));
	switch (spatial_order) {
	case JpegDctSpatialOrder::kRaster:
		append_raster_tile(order, 0, 0, width_in_blocks, height_in_blocks);
		break;
	case JpegDctSpatialOrder::kZOrder:
		append_z_tile(order, 0, 0, width_in_blocks, height_in_blocks);
		break;
	case JpegDctSpatialOrder::kTiledRaster32:
	case JpegDctSpatialOrder::kTiledZ32:
		for (uint32_t tile_y = 0; tile_y < height_in_blocks; tile_y += kSpatialTileBlocks) {
			for (uint32_t tile_x = 0; tile_x < width_in_blocks; tile_x += kSpatialTileBlocks) {
				const auto tile_width  = std::min(kSpatialTileBlocks, width_in_blocks - tile_x);
				const auto tile_height = std::min(kSpatialTileBlocks, height_in_blocks - tile_y);
				if (spatial_order == JpegDctSpatialOrder::kTiledRaster32) {
					append_raster_tile(order, tile_x, tile_y, tile_width, tile_height);
				} else {
					append_z_tile(order, tile_x, tile_y, tile_width, tile_height);
				}
			}
		}
		break;
	default:
		throw std::runtime_error("unknown JPEG DCT spatial order");
	}

	return order;
}

uint64_t block_order_rank(const uint32_t            width_in_blocks,
                          const uint32_t            height_in_blocks,
                          const uint32_t            x,
                          const uint32_t            y,
                          const JpegDctSpatialOrder spatial_order) {
	if (width_in_blocks == 0 || height_in_blocks == 0 || x >= width_in_blocks || y >= height_in_blocks) {
		throw std::runtime_error("JPEG DCT block coordinate is outside the component grid");
	}
	switch (spatial_order) {
	case JpegDctSpatialOrder::kRaster:
		return static_cast<uint64_t>(y) * width_in_blocks + x;
	case JpegDctSpatialOrder::kZOrder:
		return morton_rank_in_rectangle(width_in_blocks, height_in_blocks, x, y);
	case JpegDctSpatialOrder::kTiledRaster32:
	case JpegDctSpatialOrder::kTiledZ32:
	{
		const auto tile_x      = (x / kSpatialTileBlocks) * kSpatialTileBlocks;
		const auto tile_y      = (y / kSpatialTileBlocks) * kSpatialTileBlocks;
		const auto tile_width  = std::min(kSpatialTileBlocks, width_in_blocks - tile_x);
		const auto tile_height = std::min(kSpatialTileBlocks, height_in_blocks - tile_y);
		const auto before_tile_row = static_cast<uint64_t>(tile_y) * width_in_blocks;
		const auto before_tile     = static_cast<uint64_t>(tile_height) * tile_x;
		const auto local_x         = x - tile_x;
		const auto local_y         = y - tile_y;
		const auto within_tile = spatial_order == JpegDctSpatialOrder::kTiledRaster32
		                             ? static_cast<uint64_t>(local_y) * tile_width + local_x
		                             : morton_rank_in_rectangle(tile_width, tile_height, local_x, local_y);
		return before_tile_row + before_tile + within_tile;
	}
	default:
		throw std::runtime_error("unknown JPEG DCT spatial order");
	}
}

} // namespace galp::jpeg::detail
