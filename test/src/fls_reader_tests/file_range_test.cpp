// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// test/src/fls_reader_tests/file_range_test.cpp
// ────────────────────────────────────────────────────────
#include "data/test.hpp"
#include "fls/cor/lyt/buf.hpp"
#include "fls/io/io.hpp"
#include <array>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fastlanes {

TEST(ReadSourceTest, FileAndMemoryShareRangeAndScatterContract) {
	const path    filename(test_dataset::VALID_FLS);
	std::ifstream stream(filename, std::ios::binary);
	ASSERT_TRUE(stream);
	std::vector<char> bytes {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
	ASSERT_GE(bytes.size(), 9U);
	io file   = make_unique<File>(filename);
	io memory = make_unique<ExternalMemory>(std::as_writable_bytes(std::span(bytes)));
	for (const io* input : {&file, &memory}) {
		EXPECT_EQ(IO::get_size(*input), bytes.size());
		Buf buf(1);
		IO::range_read(*input, buf, 3, 5);
		ASSERT_EQ(buf.Size(), 5U);
		EXPECT_EQ(std::memcmp(buf.data(), bytes.data() + 3, 5), 0);
		IO::read(*input, buf);
		EXPECT_EQ(buf.Size(), bytes.size());
		EXPECT_EQ(std::memcmp(buf.data(), bytes.data(), bytes.size()), 0);
		char dst {};
		IO::range_read(*input, &dst, 8, 1);
		EXPECT_EQ(dst, bytes[8]);
		EXPECT_THROW(IO::range_read(*input, &dst, bytes.size(), 1), std::runtime_error);
		EXPECT_THROW(IO::range_read(*input, &dst, 1, std::numeric_limits<n_t>::max()), std::runtime_error);
		EXPECT_THROW(IO::range_read(*input, nullptr, 0, 1), std::invalid_argument);
		EXPECT_NO_THROW(IO::range_read(*input, nullptr, bytes.size(), 0));
	}
	for (const auto source :
	     {ReadSource(*std::get<up<File>>(file)), ReadSource(*std::get<up<ExternalMemory>>(memory))}) {
		char                             direct {}, last {};
		std::array<char, 2>              first {};
		std::array<ScatterReadTarget, 2> scatter {{{first.data(), first.size()}, {&last, 1}}};
		std::array<RangeReadTarget, 2>   ranges {{{&direct, 8, 1}, {nullptr, 0, 3, scatter}}};
		source.ReadRanges(ranges);
		EXPECT_EQ(direct, bytes[8]);
		EXPECT_EQ(first[0], bytes[0]);
		EXPECT_EQ(first[1], bytes[1]);
		EXPECT_EQ(last, bytes[2]);
		ranges[1].size = 4;
		EXPECT_THROW(source.ReadRanges(ranges), std::invalid_argument);
		ranges[1].offset = source.Size();
		EXPECT_THROW(source.ReadRanges(ranges), std::runtime_error);
	}
}

} // namespace fastlanes
