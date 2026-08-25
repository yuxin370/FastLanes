#ifndef GALP_JPEG_DCT_DEVICE_HPP
#define GALP_JPEG_DCT_DEVICE_HPP

#include "galp/config.hpp"

#if GALP_WITH_JPEG_DCT

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace galp::jpeg {

inline constexpr size_t kDefaultJpegDctDecodeBatchRowgroups                   = 64;
inline constexpr size_t kDefaultJpegDctDevicePlanCacheCapacity                = 128;
inline constexpr size_t kDefaultJpegDctDeviceRowgroupPrefetchDepth            = 4;
inline constexpr size_t kDefaultJpegDctDeviceRowgroupPrefetchWorkers          = 1;
inline constexpr size_t kDefaultJpegDctDeviceRowgroupPrefetchMinDecodeBatches = 2;
inline constexpr size_t kDefaultJpegDctDeviceDecodeWorksetCapacityBytes       = size_t {512U} * 1024U * 1024U;
inline constexpr uint32_t kDefaultJpegDctBoundedIoUringQueueDepth              = 256U;

class JpegDctShardDatasetReader;
struct JpegDctDeviceCacheStats;
struct JpegDctDeviceExecutionStats;

enum class JpegDctDeviceLayout {
	kImageMajorComponentBlockCoeff,
	kYcbcrDctGrid,
	kTransformedDctGrid,
};

enum class JpegDctGridOutputDataType {
	kInt16,
	kFloat32,
};

struct JpegDctSamplingRatio {
	uint16_t horizontal_numerator   = 1;
	uint16_t horizontal_denominator = 1;
	uint16_t vertical_numerator     = 1;
	uint16_t vertical_denominator   = 1;
};

// Generic parameters for a fused JPEG-DCT grid transform. Application
// profiles provide concrete geometry and numeric policy; the JPEG executor
// only lowers this specification to its batched transform kernels.
struct JpegDctGridTransformSpec {
	uint32_t                          y_output_width_blocks        = 0;
	uint32_t                          y_output_height_blocks       = 0;
	uint32_t                          cbcr_output_width_blocks     = 0;
	uint32_t                          cbcr_output_height_blocks    = 0;
	uint32_t                          crop_reference_width_blocks  = 0;
	uint32_t                          crop_reference_height_blocks = 0;
	uint32_t                          crop_origin_alignment_blocks = 1;
	uint32_t                          chroma_crop_scale_x          = 1;
	uint32_t                          chroma_crop_scale_y          = 1;
	int32_t                           clamp_min                    = std::numeric_limits<int16_t>::min();
	int32_t                           clamp_max                    = std::numeric_limits<int16_t>::max();
	JpegDctGridOutputDataType         output_data_type             = JpegDctGridOutputDataType::kInt16;
	float                             output_add                   = 0.0F;
	float                             output_scale                 = 1.0F;
	bool                              dequantize                   = true;
	bool                              require_all_coefficients     = true;
	bool                              allow_grayscale              = false;
	std::vector<uint32_t>             preferred_small_crop_width_blocks;
	std::vector<uint32_t>             preferred_small_crop_height_blocks;
	std::vector<JpegDctSamplingRatio> allowed_chroma_sampling_ratios;
};

inline constexpr uint8_t kJpegDctYcbcrDctGridTensorY    = 1;
inline constexpr uint8_t kJpegDctYcbcrDctGridTensorCbCr = 2;

struct JpegDctYcbcrDctGridShape {
	std::array<size_t, 6> y    = {0, 1, 0, 0, 8, 8};
	std::array<size_t, 6> cbcr = {0, 2, 0, 0, 8, 8};

	[[nodiscard]] size_t y_count() const noexcept {
		size_t count = 1;
		for (const auto dim : y) {
			count *= dim;
		}
		return count;
	}

	[[nodiscard]] size_t cbcr_count() const noexcept {
		size_t count = 1;
		for (const auto dim : cbcr) {
			count *= dim;
		}
		return count;
	}
};

struct JpegDctCoefficientSelection {
	std::vector<uint8_t> coefficients;

	[[nodiscard]] bool empty() const noexcept {
		return coefficients.empty();
	}
	[[nodiscard]] size_t size() const noexcept {
		return coefficients.empty() ? 64U : coefficients.size();
	}
};

[[nodiscard]] bool parse_jpeg_dct_coefficient_selection(std::string_view spec, JpegDctCoefficientSelection& selection);

struct JpegDctCropBox {
	uint32_t x      = 0;
	uint32_t y      = 0;
	uint32_t width  = 0;
	uint32_t height = 0;
};

struct JpegDctImageCropRequest {
	uint32_t       global_image_index = 0;
	JpegDctCropBox source_crop {};
	bool           horizontal_flip = false;
	std::string    logical_sample_id;
	std::string    augmentation_key;
};

enum class JpegDctSchedulingPolicy {
	kFullyOverlapped,
	kLimitedOverlap,
	kSerial,
};

enum class JpegDctBlockMajorDoubleBufferPolicy {
	kAutomatic,
	kEnabled,
	kDisabled,
};

// Controls the storage/decode granularity for crop requests. All modes
// preserve the same crop transform and output contract.
enum class JpegDctCropExecutionMode {
	kAutomatic,
	kFullRowgroupDecode,
	kRowgroupReadSelectedDecode,
	kVectorRangeReadSelectedDecode,
	kBoundedRangeReadSelectedDecode,
	kBoundedIoUringRangeReadSelectedDecode,
	kBoundedIoUringScheduledRangeReadSelectedDecode,
};

class JpegDctDeviceTransformSubmissionGate {
public:
	void release() {
		{
			std::lock_guard lock(mutex_);
			released_ = true;
		}
		ready_.notify_all();
	}

	void wait() {
		std::unique_lock lock(mutex_);
		ready_.wait(lock, [this] { return released_; });
	}

private:
	std::mutex              mutex_;
	std::condition_variable ready_;
	bool                    released_ = false;
};

struct JpegDctDeviceBatchOptions {
	JpegDctDeviceLayout                     layout = JpegDctDeviceLayout::kImageMajorComponentBlockCoeff;
	std::optional<JpegDctGridTransformSpec> grid_transform;
	size_t                                  cache_capacity_bytes      = 0;
	size_t                                  decode_batch_rowgroups    = kDefaultJpegDctDecodeBatchRowgroups;
	size_t                                  plan_cache_capacity       = kDefaultJpegDctDevicePlanCacheCapacity;
	bool                                    enable_rowgroup_prefetch  = true;
	size_t                                  rowgroup_prefetch_depth   = kDefaultJpegDctDeviceRowgroupPrefetchDepth;
	size_t                                  rowgroup_prefetch_workers = kDefaultJpegDctDeviceRowgroupPrefetchWorkers;
	size_t rowgroup_prefetch_min_decode_batches = kDefaultJpegDctDeviceRowgroupPrefetchMinDecodeBatches;
	JpegDctCoefficientSelection coefficient_selection {};
	bool                        enable_planless_execution   = true;
	JpegDctSchedulingPolicy     scheduling_policy           = JpegDctSchedulingPolicy::kFullyOverlapped;
	size_t                      transform_blocks_per_launch = 0;
	size_t                      transform_ctas_per_launch   = 0;
	bool                        use_low_priority_streams    = false;
	bool                        async_planless_completion   = false;
	std::shared_ptr<JpegDctDeviceTransformSubmissionGate> transform_submission_gate;
	JpegDctBlockMajorDoubleBufferPolicy block_major_double_buffer_policy =
	    JpegDctBlockMajorDoubleBufferPolicy::kAutomatic;
	JpegDctCropExecutionMode    crop_execution_mode         = JpegDctCropExecutionMode::kAutomatic;
	size_t                      decode_workset_capacity_bytes = kDefaultJpegDctDeviceDecodeWorksetCapacityBytes;
	// Integer millionths keep cap-boundary decisions deterministic.  A value
	// of 1'020'000 is amplification 1.02.  Zero local cap inherits the global
	// whole-run/per-shard cap.  The implementation rejects values above 1.10.
	uint32_t                    bounded_read_amplification_ppm = 1'000'000U;
	uint32_t                    bounded_read_local_amplification_ppm = 0U;
	size_t                      bounded_read_max_run_bytes = 0U;
};

struct JpegDctDeviceImageLayout {
	uint32_t global_image_index = 0;
	uint64_t block_offset       = 0;
	uint32_t block_count        = 0;
};

struct JpegDctDeviceBlockMetadata {
	uint32_t request_index      = 0;
	uint32_t global_image_index = 0;
	uint32_t semantic_slot_id   = 0;
	uint32_t block_x            = 0;
	uint32_t block_y            = 0;
};

struct JpegDctDeviceRowgroupMetadata {
	uint32_t shard_id       = 0;
	uint32_t rowgroup_index = 0;
};

// Detailed, read-only selection diagnostics for one planned rowgroup.  This is
// intentionally part of the CPU plan preview rather than execution statistics:
// callers can inspect the exact physical FastLanes vectors without submitting
// GPU work.
struct JpegDctDeviceRowgroupVectorPlan {
	JpegDctDeviceRowgroupMetadata rowgroup;
	size_t                        full_vector_count = 0U;
	std::vector<uint32_t>         selected_vectors;
};

struct JpegDctDeviceBatchPlanPreview {
	JpegDctDeviceLayout                        layout = JpegDctDeviceLayout::kImageMajorComponentBlockCoeff;
	std::vector<JpegDctDeviceImageLayout>      image_layouts;
	std::vector<JpegDctDeviceBlockMetadata>    block_metadata;
	std::vector<JpegDctDeviceRowgroupMetadata> rowgroups;
	std::vector<JpegDctDeviceRowgroupVectorPlan> rowgroup_vector_plans;
	size_t                                     planned_selected_vector_count   = 0;
	size_t                                     estimated_selected_vector_count = 0;
	size_t                                     full_vector_count               = 0;
	size_t                                     planned_saved_vector_count      = 0;
	size_t                                     estimated_saved_vector_count    = 0;
	std::vector<uint8_t>                       selected_coefficients;
	size_t                                     coefficients_per_block             = 64;
	double                                     planned_selected_vector_ratio      = 0.0;
	double                                     estimated_selected_vector_ratio    = 0.0;
	double                                     planning_ms                        = 0.0;
	double                                     resize_weight_build_ms             = 0.0;
	size_t                                     dct_resize_weight_cache_hits       = 0;
	size_t                                     dct_resize_weight_cache_misses     = 0;
	size_t                                     dct_conversion_matrix_cache_hits   = 0;
	size_t                                     dct_conversion_matrix_cache_misses = 0;
	JpegDctYcbcrDctGridShape                   ycbcr_dct_grid_shape;
	bool                                       uses_planless_fixed_transform               = false;
	size_t                                     compact_image_descriptor_count              = 0;
	size_t                                     fixed_transform_component_count             = 0;
	size_t                                     fixed_transform_source_block_count          = 0;
	size_t                                     fixed_transform_output_block_count          = 0;
	size_t                                     host_expanded_transform_items_created       = 0;
	size_t                                     host_output_block_source_lists_created      = 0;
	size_t                                     host_global_transform_sort_items            = 0;
	size_t                                     planless_axis_program_count                 = 0;
	size_t                                     planless_axis_phase_matrix_count            = 0;
	size_t                                     compiled_access_profile_hits                = 0;
	size_t                                     compiled_access_profile_misses              = 0;
	size_t                                     planless_axis_program_bytes                 = 0;
	size_t                                     compact_plan_bytes                          = 0;
	size_t                                     compact_plan_peak_bytes                     = 0;
	size_t                                     coordinate_group_lookup_count               = 0;
	size_t                                     coordinate_group_index_entries              = 0;
	size_t                                     coordinate_group_index_populated            = 0;
	size_t                                     coordinate_group_index_holes                = 0;
	size_t                                     coordinate_group_index_bytes                = 0;
	double                                     coordinate_group_index_density              = 0.0;
	bool                                       exact_batch_plan_cache_enabled              = false;
	size_t                                     compact_reader_image_locator_bytes          = 0;
	size_t                                     compact_reader_shard_index_bytes            = 0;
	size_t                                     compact_reader_layout_dictionary_bytes      = 0;
	size_t                                     compact_reader_quant_table_dictionary_bytes = 0;
	size_t                                     compact_reader_total_bytes                  = 0;
	size_t                                     compact_reader_shard_descriptor_bytes       = 0;
	bool                                       compact_reader_shard_index_derived          = false;
	size_t                                     decode_workset_capacity_bytes               = 0U;
	size_t                                     estimated_max_decode_workset_bytes           = 0U;
	size_t                                     estimated_oversized_decode_rowgroups         = 0U;
	// Dataset-layout-derived upper bound for all planless rational-axis
	// programs accepted by this transform. A complete contract can be reserved
	// on the first execution independently of which crops happen to warm up.
	size_t                                     planless_axis_program_capacity_contract_bytes = 0U;
	bool                                       planless_axis_program_capacity_contract_complete = false;
};

struct JpegDctDeviceBatchPlanEstimate {
	JpegDctDeviceLayout                        layout      = JpegDctDeviceLayout::kImageMajorComponentBlockCoeff;
	size_t                                     block_count = 0;
	std::vector<JpegDctDeviceRowgroupMetadata> rowgroups;
	size_t                                     full_vector_count = 0;
	double                                     planning_ms       = 0.0;
};

class JpegDctDeviceBatchPreparedPlan {
public:
	struct Impl;

	JpegDctDeviceBatchPreparedPlan() noexcept;
	explicit JpegDctDeviceBatchPreparedPlan(std::unique_ptr<Impl> impl) noexcept;
	~JpegDctDeviceBatchPreparedPlan();

	JpegDctDeviceBatchPreparedPlan(const JpegDctDeviceBatchPreparedPlan&)            = delete;
	JpegDctDeviceBatchPreparedPlan& operator=(const JpegDctDeviceBatchPreparedPlan&) = delete;
	JpegDctDeviceBatchPreparedPlan(JpegDctDeviceBatchPreparedPlan&&) noexcept;
	JpegDctDeviceBatchPreparedPlan& operator=(JpegDctDeviceBatchPreparedPlan&&) noexcept;

	[[nodiscard]] bool                                              empty() const noexcept;
	[[nodiscard]] JpegDctDeviceLayout                               layout() const noexcept;
	[[nodiscard]] const std::vector<JpegDctDeviceImageLayout>&      image_layouts() const noexcept;
	[[nodiscard]] const std::vector<JpegDctDeviceBlockMetadata>&    block_metadata() const noexcept;
	[[nodiscard]] const std::vector<JpegDctDeviceRowgroupMetadata>& rowgroups() const noexcept;
	[[nodiscard]] size_t                                            planned_selected_vector_count() const noexcept;
	[[nodiscard]] size_t                                            estimated_selected_vector_count() const noexcept;
	[[nodiscard]] size_t                                            full_vector_count() const noexcept;
	[[nodiscard]] size_t                                            planned_saved_vector_count() const noexcept;
	[[nodiscard]] size_t                                            estimated_saved_vector_count() const noexcept;
	[[nodiscard]] const std::vector<uint8_t>&                       selected_coefficients() const noexcept;
	[[nodiscard]] size_t                                            coefficients_per_block() const noexcept;
	[[nodiscard]] double                                            planned_selected_vector_ratio() const noexcept;
	[[nodiscard]] double                                            estimated_selected_vector_ratio() const noexcept;
	[[nodiscard]] double                                            planning_ms() const noexcept;
	[[nodiscard]] bool                                              uses_planless_fixed_transform() const noexcept;
	[[nodiscard]] bool                                              exact_batch_plan_cache_enabled() const noexcept;
	[[nodiscard]] bool                                              decoded_rowgroup_cache_enabled() const noexcept;
	[[nodiscard]] size_t automatic_sparse_storage_candidate_rowgroup_count() const noexcept;
	[[nodiscard]] size_t automatic_sparse_storage_selected_rowgroup_count() const noexcept;
	[[nodiscard]] size_t automatic_sparse_storage_rejected_rowgroup_count() const noexcept;
	[[nodiscard]] bool                                              host_io_staged() const noexcept;
	[[nodiscard]] size_t                                            host_io_staged_rowgroups() const noexcept;
	[[nodiscard]] double                                            host_io_staging_ms() const noexcept;

private:
	friend class JpegDctShardDatasetReader;
	std::unique_ptr<Impl> impl_;
};

class JpegDctDeviceBatch {
public:
	struct Impl;

	JpegDctDeviceBatch() noexcept;
	explicit JpegDctDeviceBatch(std::unique_ptr<Impl> impl) noexcept;
	~JpegDctDeviceBatch();

	JpegDctDeviceBatch(const JpegDctDeviceBatch&)            = delete;
	JpegDctDeviceBatch& operator=(const JpegDctDeviceBatch&) = delete;
	JpegDctDeviceBatch(JpegDctDeviceBatch&&) noexcept;
	JpegDctDeviceBatch& operator=(JpegDctDeviceBatch&&) noexcept;

	// Synchronous accessors wait for submitted work and propagate completion
	// failures. Async accessors never perform that wait.
	[[nodiscard]] const int16_t*                                    device_coefficients() const;
	[[nodiscard]] const int16_t*                                    y_coefficients() const;
	[[nodiscard]] const int16_t*                                    cbcr_coefficients() const;
	[[nodiscard]] const int16_t*                                    device_coefficients_async() const noexcept;
	[[nodiscard]] const int16_t*                                    y_coefficients_async() const noexcept;
	[[nodiscard]] const int16_t*                                    cbcr_coefficients_async() const noexcept;
	[[nodiscard]] const float*                                      y_float_coefficients() const;
	[[nodiscard]] const float*                                      cbcr_float_coefficients() const;
	[[nodiscard]] const float*                                      y_float_coefficients_async() const noexcept;
	[[nodiscard]] const float*                                      cbcr_float_coefficients_async() const noexcept;
	[[nodiscard]] JpegDctGridOutputDataType                         grid_output_data_type() const noexcept;
	void                                                            synchronize() const;
	[[nodiscard]] size_t                                            coefficient_count() const noexcept;
	[[nodiscard]] size_t                                            coefficient_bytes() const noexcept;
	[[nodiscard]] size_t                                            y_coefficient_count() const noexcept;
	[[nodiscard]] size_t                                            cbcr_coefficient_count() const noexcept;
	[[nodiscard]] size_t                                            coefficients_per_block() const noexcept;
	[[nodiscard]] size_t                                            block_count() const noexcept;
	[[nodiscard]] size_t                                            image_count() const noexcept;
	[[nodiscard]] size_t                                            rowgroup_count() const noexcept;
	[[nodiscard]] int                                               cuda_device() const noexcept;
	[[nodiscard]] JpegDctDeviceCacheStats                           cache_stats() const noexcept;
	[[nodiscard]] JpegDctDeviceExecutionStats                       execution_stats() const;
	// Non-blocking completion probe used by native metrics aggregation. It
	// finalizes existing event-derived timings only when the existing producer
	// completion event is already ready; it never synchronizes a stream/device.
	[[nodiscard]] bool                                              try_finalize_execution_stats() const;
	[[nodiscard]] const JpegDctDeviceCacheStats&                    cache_stats_ref() const noexcept;
	[[nodiscard]] const JpegDctDeviceExecutionStats&                execution_stats_ref() const noexcept;
	[[nodiscard]] JpegDctDeviceLayout                               layout() const noexcept;
	[[nodiscard]] void*                                             cuda_completion_event() const noexcept;
	[[nodiscard]] const std::vector<JpegDctDeviceImageLayout>&      image_layouts() const noexcept;
	[[nodiscard]] const std::vector<JpegDctDeviceBlockMetadata>&    block_metadata() const noexcept;
	[[nodiscard]] const std::vector<JpegDctDeviceRowgroupMetadata>& rowgroups() const noexcept;
	[[nodiscard]] const std::vector<uint8_t>&                       selected_coefficients() const noexcept;
	[[nodiscard]] JpegDctYcbcrDctGridShape                          ycbcr_dct_grid_shape() const noexcept;

private:
	std::unique_ptr<Impl> impl_;
};

} // namespace galp::jpeg

#endif // GALP_WITH_JPEG_DCT

#endif // GALP_JPEG_DCT_DEVICE_HPP
