#include "jpeg/jpeg_dct_order.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <set>

namespace {

TEST(JpegDctOrder, ZigzagColumnToNaturalIndex) {
	const std::array<uint8_t, 8> expected_prefix {0, 1, 8, 16, 9, 2, 3, 10};
	for (size_t i = 0; i < expected_prefix.size(); ++i) {
		EXPECT_EQ(galp::jpeg::detail::kZigzagColumnToNaturalIndex[i], expected_prefix[i]);
	}
	EXPECT_EQ(galp::jpeg::detail::kZigzagColumnToNaturalIndex[63], 63);
}

TEST(JpegDctOrder, Morton2x2) {
	const auto order = galp::jpeg::detail::make_block_order(2, 2, true);
	ASSERT_EQ(order.size(), 4);
	EXPECT_EQ(order[0].x, 0);
	EXPECT_EQ(order[0].y, 0);
	EXPECT_EQ(order[1].x, 1);
	EXPECT_EQ(order[1].y, 0);
	EXPECT_EQ(order[2].x, 0);
	EXPECT_EQ(order[2].y, 1);
	EXPECT_EQ(order[3].x, 1);
	EXPECT_EQ(order[3].y, 1);
}

TEST(JpegDctOrder, Morton3x5OnlyLegalCoordinates) {
	const auto order = galp::jpeg::detail::make_block_order(3, 5, true);
	ASSERT_EQ(order.size(), 15);

	std::set<std::pair<uint32_t, uint32_t>> seen;
	uint64_t                               previous_key = 0;
	for (size_t i = 0; i < order.size(); ++i) {
		EXPECT_LT(order[i].x, 3);
		EXPECT_LT(order[i].y, 5);
		EXPECT_TRUE(seen.emplace(order[i].x, order[i].y).second);
		if (i != 0) {
			EXPECT_LE(previous_key, order[i].morton_key);
		}
		previous_key = order[i].morton_key;
	}
}

TEST(JpegDctOrder, RasterOrderCanBeRequested) {
	const auto order = galp::jpeg::detail::make_block_order(3, 2, false);
	ASSERT_EQ(order.size(), 6);
	EXPECT_EQ(order[0].x, 0);
	EXPECT_EQ(order[0].y, 0);
	EXPECT_EQ(order[1].x, 1);
	EXPECT_EQ(order[1].y, 0);
	EXPECT_EQ(order[2].x, 2);
	EXPECT_EQ(order[2].y, 0);
	EXPECT_EQ(order[3].x, 0);
	EXPECT_EQ(order[3].y, 1);
}

} // namespace
