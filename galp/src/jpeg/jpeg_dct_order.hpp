#ifndef GALP_JPEG_DCT_ORDER_HPP
#define GALP_JPEG_DCT_ORDER_HPP

#include <array>
#include <cstdint>
#include <vector>

namespace galp::jpeg::detail {

inline constexpr std::array<uint8_t, 64> kZigzagColumnToNaturalIndex {
    0,  1,  8,  16, 9,  2,  3,  10,
    17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
};

struct MortonBlockCoord {
	uint32_t x          = 0;
	uint32_t y          = 0;
	uint64_t morton_key = 0;
};

uint64_t morton_key(uint32_t x, uint32_t y);

std::vector<MortonBlockCoord> make_block_order(uint32_t width_in_blocks,
                                               uint32_t height_in_blocks,
                                               bool     use_z_curve_order);

} // namespace galp::jpeg::detail

#endif // GALP_JPEG_DCT_ORDER_HPP
