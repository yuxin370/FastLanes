#ifndef GALP_JPEG_DCT_EXACT_VERIFIER_HPP
#define GALP_JPEG_DCT_EXACT_VERIFIER_HPP

#include "galp/config.hpp"

#if GALP_WITH_JPEG_DCT

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace galp::jpeg {

inline constexpr size_t kMaxJpegDctExactVerificationWorkers = 32U;

enum class JpegDctExactMismatchKind : uint8_t {
	kMissingBlock = 0,
	kExtraBlock   = 1,
	kCoefficient  = 2,
};

struct JpegDctExactMismatch {
	bool                     present            = false;
	uint32_t                 global_image_index = 0;
	uint32_t                 semantic_slot_id   = 0;
	uint32_t                 block_y            = 0;
	uint32_t                 block_x            = 0;
	uint32_t                 coefficient        = 0;
	int16_t                  expected           = 0;
	int16_t                  actual             = 0;
	JpegDctExactMismatchKind kind               = JpegDctExactMismatchKind::kCoefficient;
};

struct JpegDctExactVerificationResult {
	size_t               source_images          = 0;
	size_t               expected_blocks        = 0;
	size_t               actual_blocks          = 0;
	size_t               missing_blocks         = 0;
	size_t               extra_blocks           = 0;
	size_t               coefficient_mismatches = 0;
	int                  max_abs_difference     = 0;
	JpegDctExactMismatch first_mismatch {};

	[[nodiscard]] bool exact() const noexcept {
		return missing_blocks == 0U && extra_blocks == 0U && coefficient_mismatches == 0U;
	}
};

// Verifies every source image against its manifest-backed coefficient blocks.
// Work is partitioned only at manifest shard boundaries. Each worker owns one
// independent dataset reader and visits all images in an assigned shard in
// increasing global-image order.
JpegDctExactVerificationResult verify_jpeg_dct_manifest_exact(const std::filesystem::path&              manifest_path,
                                                              const std::vector<std::filesystem::path>& source_paths,
                                                              size_t verify_workers = 1U);

} // namespace galp::jpeg

#endif // GALP_WITH_JPEG_DCT

#endif // GALP_JPEG_DCT_EXACT_VERIFIER_HPP
