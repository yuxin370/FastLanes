#include "jpeg/jpeg_dct_order.hpp"
#include <algorithm>
#include <stdexcept>

namespace galp::jpeg::detail {

namespace {

uint64_t part_1_by_1(const uint32_t value) {
	uint64_t x = value;
	x          = (x | (x << 16U)) & 0x0000ffff0000ffffULL;
	x          = (x | (x << 8U)) & 0x00ff00ff00ff00ffULL;
	x          = (x | (x << 4U)) & 0x0f0f0f0f0f0f0f0fULL;
	x          = (x | (x << 2U)) & 0x3333333333333333ULL;
	x          = (x | (x << 1U)) & 0x5555555555555555ULL;
	return x;
}

} // namespace

uint64_t morton_key(const uint32_t x, const uint32_t y) {
	return part_1_by_1(x) | (part_1_by_1(y) << 1U);
}

std::vector<MortonBlockCoord> make_block_order(const uint32_t width_in_blocks,
                                               const uint32_t height_in_blocks,
                                               const bool     use_z_curve_order) {
	if (width_in_blocks == 0 || height_in_blocks == 0) {
		throw std::runtime_error("JPEG component has an empty DCT block grid");
	}

	std::vector<MortonBlockCoord> order;
	order.reserve(static_cast<size_t>(width_in_blocks) * static_cast<size_t>(height_in_blocks));
	for (uint32_t y = 0; y < height_in_blocks; ++y) {
		for (uint32_t x = 0; x < width_in_blocks; ++x) {
			order.push_back(MortonBlockCoord {x, y, morton_key(x, y)});
		}
	}

	if (use_z_curve_order) {
		std::stable_sort(order.begin(), order.end(), [](const MortonBlockCoord& lhs, const MortonBlockCoord& rhs) {
			if (lhs.morton_key != rhs.morton_key) {
				return lhs.morton_key < rhs.morton_key;
			}
			if (lhs.y != rhs.y) {
				return lhs.y < rhs.y;
			}
			return lhs.x < rhs.x;
		});
	}

	return order;
}

} // namespace galp::jpeg::detail
