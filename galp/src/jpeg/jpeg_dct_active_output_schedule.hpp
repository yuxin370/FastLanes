#pragma once

#include "jpeg/jpeg_dct_plan_types.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace galp::jpeg::detail {

inline constexpr uint32_t kJpegDctActiveOutputSchedulePlannerAbi = 1U;
inline constexpr size_t   kJpegDctActiveOutputScheduleMmapCapacityBytes = 16U << 20U;
inline constexpr size_t   kJpegDctActiveOutputScheduleMmapWindowCount = 2U;

// Only fields that can change output-to-workset ownership participate in the
// immutable sidecar digest.  The remaining fields retain the physical I/O
// decision for diagnostics, but coefficient-dependent byte/range choices do
// not invalidate a schedule when rowgroup ownership is unchanged.
struct JpegDctActiveOutputDecisionRecord {
	uint32_t shard_id              = 0U;
	uint32_t rowgroup_index        = 0U;
	uint32_t workset_index         = 0U;
	uint32_t runtime_decision      = 0U;
	uint32_t read_strategy         = 0U;
	uint32_t submission_backend    = 0U;
	uint64_t selected_vector_count = 0U;
	uint64_t full_vector_count     = 0U;
	uint64_t physical_bytes        = 0U;
	uint64_t physical_run_count    = 0U;
};

struct JpegDctActiveOutputScheduleKey {
	std::filesystem::path directory;
	uint32_t              shard_id                     = 0U;
	uint64_t              canonical_plan_digest        = 0U;
	uint64_t              decision_digest              = 0U;
	uint64_t              transform_digest             = 0U;
	uint64_t              decode_workset_capacity_bytes = 0U;
	uint32_t              decode_batch_rowgroups       = 0U;
	uint32_t              double_buffer_policy         = 0U;
	uint32_t              workset_count                = 0U;
	uint32_t              planner_abi                  = kJpegDctActiveOutputSchedulePlannerAbi;
};

struct JpegDctActiveOutputScheduleIoResult {
	std::optional<JpegDctDeviceBlockMajorActiveOutputSchedule> schedule;
	bool        hit              = false;
	bool        rejected         = false;
	bool        persisted        = false;
	uint64_t    sidecar_bytes    = 0U;
	uint64_t    mapped_bytes     = 0U;
	uint64_t    interval_count   = 0U;
	double      load_ms          = 0.0;
	double      validation_ms    = 0.0;
	double      materialize_ms   = 0.0;
	double      persist_ms       = 0.0;
	std::string rejection_reason;
};

[[nodiscard]] uint64_t jpeg_dct_active_output_transform_digest(
    const JpegDctGridTransformSpec& transform) noexcept;

[[nodiscard]] uint64_t jpeg_dct_active_output_decision_digest(
    const std::vector<JpegDctActiveOutputDecisionRecord>& decisions) noexcept;

[[nodiscard]] uint64_t jpeg_dct_active_output_schedule_key_digest(
    const JpegDctActiveOutputScheduleKey& key) noexcept;

[[nodiscard]] std::filesystem::path jpeg_dct_active_output_schedule_path(
    const JpegDctActiveOutputScheduleKey& key);

[[nodiscard]] JpegDctActiveOutputScheduleIoResult load_jpeg_dct_active_output_schedule(
    const JpegDctActiveOutputScheduleKey& key);

[[nodiscard]] JpegDctActiveOutputScheduleIoResult persist_jpeg_dct_active_output_schedule(
    const JpegDctActiveOutputScheduleKey&                   key,
    const JpegDctDeviceBlockMajorActiveOutputSchedule& schedule);

} // namespace galp::jpeg::detail
