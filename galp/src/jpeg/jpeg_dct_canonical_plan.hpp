#ifndef GALP_JPEG_DCT_CANONICAL_PLAN_HPP
#define GALP_JPEG_DCT_CANONICAL_PLAN_HPP

#include "galp/jpeg_dct_block_major_plan.hpp"
#include <cstdint>
#include <filesystem>

namespace galp::jpeg::detail {

struct JpegDctCanonicalPlanIdentity {
	uint64_t manifest_size          = 0U;
	uint64_t manifest_crc64         = 0U;
	uint64_t descriptor_size        = 0U;
	uint64_t descriptor_crc64       = 0U;
	uint64_t source_size            = 0U;
	uint64_t source_stat_digest     = 0U;
	uint64_t source_payload_crc64   = 0U;
	uint64_t first_global_image     = 0U;
	uint32_t shard_id               = 0U;
	uint32_t image_count            = 0U;
	uint32_t rowgroup_vectors       = 0U;
	uint64_t transform_digest       = 0U;
};

struct JpegDctCanonicalPlanLoadResult {
	JpegDctBlockMajorCompactPlan plan;
	uint64_t sidecar_bytes       = 0U;
	uint64_t plan_audit_digest   = 0U;
	double   load_ms             = 0.0;
	double   validation_ms       = 0.0;
};

struct JpegDctCanonicalPlanEnsureResult {
	uint64_t sidecar_bytes       = 0U;
	uint64_t plan_audit_digest   = 0U;
	bool     reused_existing     = false;
};

[[nodiscard]] uint64_t jpeg_dct_canonical_transform_digest(const JpegDctGridTransformSpec& transform);
[[nodiscard]] uint64_t jpeg_dct_canonical_source_stat_digest(const std::filesystem::path& source_path);
[[nodiscard]] uint64_t jpeg_dct_canonical_plan_audit_digest(
	const JpegDctBlockMajorCompactPlan& plan);
[[nodiscard]] JpegDctCanonicalPlanLoadResult load_jpeg_dct_canonical_plan_template(
	const std::filesystem::path& path,
	const JpegDctCanonicalPlanIdentity& identity);
[[nodiscard]] JpegDctCanonicalPlanEnsureResult ensure_jpeg_dct_canonical_plan_template(
	const std::filesystem::path& path,
	const JpegDctCanonicalPlanIdentity& identity,
	const JpegDctBlockMajorCompactPlan& plan);

} // namespace galp::jpeg::detail

#endif // GALP_JPEG_DCT_CANONICAL_PLAN_HPP
