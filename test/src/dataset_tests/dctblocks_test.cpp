// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// test/src/dataset_tests/dctblocks_test.cpp
// ────────────────────────────────────────────────────────
#include "data/DCTBlocks.hpp"
#include "fastlanes.hpp"
#include "fls_tester.hpp"

namespace fastlanes {

#define DCTBLOCKS_TEST(DATASET_VAR)                                                                                   \
	TEST_F(FastLanesReaderTester, DCTBlcoks_##DATASET_VAR) {                                                          \
		const std::vector<n_t> constant_cols {};                                                                       \
		const std::vector<n_t> equal_cols {};                                                                          \
		const std::vector<n_t> one_to_one_mapped_col_indexes {};                                                       \
		AllTest(DCTBlocks::DATASET_VAR, constant_cols, equal_cols, one_to_one_mapped_col_indexes);                    \
	}

DCTBLOCKS_TEST(IMAGENET)
DCTBLOCKS_TEST(MNIST)
DCTBLOCKS_TEST(SVHN)

#undef DCTBLOCKS_TEST

} // namespace fastlanes
