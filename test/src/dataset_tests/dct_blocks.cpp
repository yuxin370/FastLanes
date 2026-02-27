// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// test/src/dataset_tests/dct_blocks.cpp
// ────────────────────────────────────────────────────────
#include "data/dct_blocks.hpp"
#include "fls_tester.hpp"

namespace fastlanes {

#define DCT_BLOCK_TEST(DATASET_VAR)                                                                                    \
	TEST_F(FastLanesReaderTester, dct_blocks_##DATASET_VAR) {                                                          \
		const std::vector<n_t> constant_cols {};                                                                       \
		const std::vector<n_t> equal_cols {};                                                                          \
		const std::vector<n_t> one_to_one_mapped_col_indexes {};                                                       \
		AllTest(dct_blocks::DATASET_VAR, constant_cols, equal_cols, one_to_one_mapped_col_indexes);                    \
	}

DCT_BLOCK_TEST(MNIST)
DCT_BLOCK_TEST(CIFAR10)
DCT_BLOCK_TEST(CELEBA)
DCT_BLOCK_TEST(TINY_IMAGENET)

} // namespace fastlanes
