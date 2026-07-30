#ifndef GALP_JPEG_DCT_ORDER_HPP
#define GALP_JPEG_DCT_ORDER_HPP

#include "galp/jpeg_dct_format.hpp"
#include <array>
#include <cstdint>
#include <vector>

namespace galp::jpeg::detail {

inline constexpr std::array<uint8_t, 64> kZigzagColumnToNaturalIndex {
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,  12, 19, 26, 33, 40, 48,
    41, 34, 27, 20, 13, 6,  7,  14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23,
    30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
};

struct MortonBlockCoord {
	uint32_t x          = 0;
	uint32_t y          = 0;
	uint64_t morton_key = 0;
};

struct BlockRankInterval {
	uint64_t begin = 0;
	uint64_t end   = 0;
};

uint64_t morton_key(uint32_t x, uint32_t y);

std::vector<MortonBlockCoord>
make_block_order(uint32_t width_in_blocks, uint32_t height_in_blocks, bool use_z_curve_order);

std::vector<MortonBlockCoord>
make_block_order(uint32_t width_in_blocks, uint32_t height_in_blocks, JpegDctSpatialOrder spatial_order);

// Dense rank of (x, y) in make_block_order(). This is O(log(max(width,
// height))) for Z modes and O(1) for raster modes, including ragged edge
// tiles, so the hot image-major planner does not need a per-image rank table.
uint64_t block_order_rank(
    uint32_t width_in_blocks, uint32_t height_in_blocks, uint32_t x, uint32_t y, JpegDctSpatialOrder spatial_order);

// Return the dense block-order ranks covered by a source-grid rectangle as
// sorted, merged half-open intervals. Work is proportional to crop rows for
// raster layouts and to the generated quadtree cover for Morton layouts; it
// never enumerates every source block in the crop.
std::vector<BlockRankInterval> block_order_rectangle_rank_intervals(uint32_t            width_in_blocks,
                                                                    uint32_t            height_in_blocks,
                                                                    uint32_t            x,
                                                                    uint32_t            y,
                                                                    uint32_t            width,
                                                                    uint32_t            height,
                                                                    JpegDctSpatialOrder spatial_order);

} // namespace galp::jpeg::detail

#endif // GALP_JPEG_DCT_ORDER_HPP
