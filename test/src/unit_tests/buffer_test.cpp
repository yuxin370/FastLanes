#include "fls/cor/lyt/buf.hpp"
#include "fls/expression/fsst_dictionary_input.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

namespace {

TEST(Buffer, AppendGrowsGeometricallyAndPreservesBytes) {
	fastlanes::Buf               buffer(8);
	const std::array<uint8_t, 5> prefix {1, 2, 3, 4, 5};
	std::array<uint8_t, 100>     suffix {};
	for (std::size_t idx = 0; idx < suffix.size(); ++idx) {
		suffix[idx] = static_cast<uint8_t>(idx);
	}

	buffer.Append(prefix.data(), prefix.size());
	buffer.Append(suffix.data(), suffix.size());

	ASSERT_EQ(buffer.Size(), prefix.size() + suffix.size());
	ASSERT_GE(buffer.Capacity(), buffer.Size());
	EXPECT_TRUE(std::equal(prefix.begin(), prefix.end(), buffer.data()));
	EXPECT_TRUE(std::equal(suffix.begin(), suffix.end(), buffer.data() + prefix.size()));
}

TEST(Buffer, FixedSizeArrayGrowsFromAnEmptyBuffer) {
	fastlanes::Buf buffer(8);
	auto*          values = buffer.GetFixedSizeArray<int64_t>(4U * sizeof(int64_t));
	for (int64_t idx = 0; idx < 4; ++idx) {
		values[idx] = idx * 17;
	}

	ASSERT_EQ(buffer.Size(), 4U * sizeof(int64_t));
	ASSERT_GE(buffer.Capacity(), buffer.Size());
	const auto* stored = reinterpret_cast<const int64_t*>(buffer.data());
	for (int64_t idx = 0; idx < 4; ++idx) {
		EXPECT_EQ(stored[idx], idx * 17);
	}
}

TEST(FsstDictionaryInput, DerivesPointersOnlyAfterGrowableBytesAreStable) {
	fastlanes::FsstDictionaryInput input(8);
	const std::array<uint8_t, 5>   first {1, 2, 3, 4, 5};
	std::array<uint8_t, 40>        second {};
	for (std::size_t index = 0; index < second.size(); ++index) {
		second[index] = static_cast<uint8_t>(index + 11);
	}

	input.Append(fastlanes::fls_string_t(first.data(), first.size()));
	input.Append(fastlanes::fls_string_t(second.data(), second.size()));
	ASSERT_GT(input.Bytes().Capacity(), 8U);
	input.FinalizePointers();

	ASSERT_EQ(input.Count(), 2U);
	EXPECT_EQ(input.Lengths()[0], first.size());
	EXPECT_EQ(input.Lengths()[1], second.size());
	EXPECT_EQ(input.Strings()[0], input.Bytes().data());
	EXPECT_EQ(input.Strings()[1], input.Bytes().data() + first.size());
	EXPECT_TRUE(std::equal(first.begin(), first.end(), input.Strings()[0]));
	EXPECT_TRUE(std::equal(second.begin(), second.end(), input.Strings()[1]));
	EXPECT_GE(input.EncodedCapacityUpperBound(), 2U * input.Bytes().Size());
	EXPECT_EQ(input.OffsetCapacityBytes(), 3U * sizeof(fastlanes::ofs_t));
}

} // namespace
