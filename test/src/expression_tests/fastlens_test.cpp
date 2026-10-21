// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// test/src/expression_tests/fastlens_test.cpp
// ────────────────────────────────────────────────────────
#include "fls_tester.hpp"

namespace fastlanes {

// ——— FLOAT tests ———————————————————————————————————————————————

TEST_F(FastLanesReaderTester, TEST_FASTLENS_FLOAT_ONE_VECTOR) {
	TestCorrectness(fastlens::float_fastlens_one_vector, {OperatorToken::EXP_FASTLENS_FLT});
}

TEST_F(FastLanesReaderTester, TEST_FASTLENS_FLOAT_SINGLE_COLUMN) {
	TestCorrectness(fastlens::float_fastlens_single_column, {OperatorToken::EXP_FASTLENS_FLT});
}

TEST_F(FastLanesReaderTester, TEST_FASTLENS_FLOAT_CONSTANT_ONE_VECTOR) {
	TestCorrectness(fastlens::float_constant_one_vector, {OperatorToken::EXP_FASTLENS_FLT});
}

TEST_F(FastLanesReaderTester, TEST_FASTLENS_FLOAT_CONSTANT_SINGLE_COLUMN) {
	TestCorrectness(fastlens::float_constant_single_column, {OperatorToken::EXP_FASTLENS_FLT});
}

TEST_F(FastLanesReaderTester, TEST_FASTLENS_FLOAT_POSITIVE_INF_ONE_VECTOR) {
	TestCorrectness(fastlens::float_positive_inf_one_vector, {OperatorToken::EXP_FASTLENS_FLT});
}

TEST_F(FastLanesReaderTester, TEST_FASTLENS_FLOAT_POSITIVE_INF_SINGLE_COLUMN) {
	TestCorrectness(fastlens::float_positive_inf_single_column, {OperatorToken::EXP_FASTLENS_FLT});
}

// ——— DOUBLE tests ———————————————————————————————————————————————

TEST_F(FastLanesReaderTester, TEST_FASTLENS_DOUBLE_ONE_VECTOR) {
	TestCorrectness(fastlens::double_fastlens_one_vector, {OperatorToken::EXP_FASTLENS_DBL});
}

TEST_F(FastLanesReaderTester, TEST_FASTLENS_DOUBLE_SINGLE_COLUMN) {
	TestCorrectness(fastlens::double_fastlens_single_column, {OperatorToken::EXP_FASTLENS_DBL});
}

TEST_F(FastLanesReaderTester, TEST_FASTLENS_DOUBLE_CONSTANT_ONE_VECTOR) {
	TestCorrectness(fastlens::double_constant_one_vector, {OperatorToken::EXP_FASTLENS_DBL});
}

TEST_F(FastLanesReaderTester, TEST_FASTLENS_DOUBLE_CONSTANT_SINGLE_COLUMN) {
	TestCorrectness(fastlens::double_constant_single_column, {OperatorToken::EXP_FASTLENS_DBL});
}

TEST_F(FastLanesReaderTester, TEST_FASTLENS_DOUBLE_POSITIVE_INF_ONE_VECTOR) {
	TestCorrectness(fastlens::double_positive_inf_one_vector, {OperatorToken::EXP_FASTLENS_DBL});
}

TEST_F(FastLanesReaderTester, TEST_FASTLENS_DOUBLE_POSITIVE_INF_SINGLE_COLUMN) {
	TestCorrectness(fastlens::double_positive_inf_single_column, {OperatorToken::EXP_FASTLENS_DBL});
}

} // namespace fastlanes
