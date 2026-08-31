#ifndef GALP_ADVANCED_DIRECT_DCT_PLS_HPP
#define GALP_ADVANCED_DIRECT_DCT_PLS_HPP

// Advanced training-data contract. This API intentionally does not participate
// in <galp/stable.hpp> or the deprecated <galp/jpeg_dct.hpp> umbrella.

#include "galp/config.hpp"

#if GALP_WITH_JPEG_DCT

#include "galp/advanced/direct_dct.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace galp::jpeg {

// The production PLS path deliberately models only scientific condition
// fields. Storage scheduling, crop pushdown, and output placement remain
// native-owned implementation details.
enum class DirectDctPlsCropPolicy {
	kPerSample,
	kPerPls,
};

enum class DirectDctPlsOrderPolicy {
	kGlobal,
	kClosedPool,
	// Frozen premixed physical order, unchanged across epochs. M remains a
	// resource/lifetime boundary and does not imply a permutation.
	kPhysicalOrder,
};

struct DirectDctPlsSample {
	uint32_t    global_image_index = 0U; // physical position in the premixed manifest
	uint32_t    virtual_pls_id     = 0U;
	uint32_t    position_in_pls    = 0U;
	uint32_t    source_image_id    = 0U;
	int64_t     label              = 0;
	std::string logical_sample_id;
};

// Frozen logical identity -> physical position mapping. The CSV loader accepts
// the physical_layout_samples/ordered_mapping schema emitted by the existing
// materializer, but parsing and validation are production C++ code.
class DirectDctPlsLayout {
public:
	static DirectDctPlsLayout LoadPremixedCsv(const std::filesystem::path& mapping_csv,
	                                          const JpegDctShardManifest&  manifest,
	                                          uint32_t                     segment_images  = 1024U,
	                                          std::string_view             expected_sha256 = {},
	                                          uint32_t                     model_classes   = 1000U);

	DirectDctPlsLayout(std::vector<DirectDctPlsSample>    samples,
	                   std::vector<std::vector<uint32_t>> positions_by_pls,
	                   uint32_t                           segment_images);

	[[nodiscard]] size_t                    sample_count() const noexcept;
	[[nodiscard]] size_t                    pls_count() const noexcept;
	[[nodiscard]] uint32_t                  segment_images() const noexcept;
	[[nodiscard]] const DirectDctPlsSample& sample(uint32_t physical_position) const;
	[[nodiscard]] std::span<const uint32_t> positions_in_pls(uint32_t virtual_pls_id) const;

private:
	std::vector<DirectDctPlsSample>    samples_;
	std::vector<std::vector<uint32_t>> positions_by_pls_;
	uint32_t                           segment_images_ = 0U;
};

struct DirectDctPlsScheduleOptions {
	uint64_t                training_seed     = 0U;
	uint32_t                epoch             = 0U;
	uint32_t                segments_per_pool = 4U;
	uint32_t                microbatch_images = 64U;
	DirectDctPlsCropPolicy  crop_policy       = DirectDctPlsCropPolicy::kPerPls;
	DirectDctPlsOrderPolicy order_policy      = DirectDctPlsOrderPolicy::kClosedPool;
};

struct DirectDctPlsPoolPlan {
	uint32_t              epoch                  = 0U;
	uint32_t              pool_index             = 0U;
	uint64_t              first_microbatch_index = 0U;
	std::vector<uint32_t> virtual_pls_ids;
	// Physical image positions in model-consumer order. The block-major planner
	// reorders reads by shard/local image and preserves these output slots.
	std::vector<uint32_t> ordered_positions;
};

// Epoch-local and streaming: only one epoch cursor and one pool plan exist at
// a time. Physical-order mode keeps the frozen PLS and sample order intact.
class DirectDctPlsEpochSchedule {
public:
	DirectDctPlsEpochSchedule(const DirectDctPlsLayout& layout, DirectDctPlsScheduleOptions options);

	[[nodiscard]] bool   has_next() const noexcept;
	[[nodiscard]] size_t remaining_pool_count() const noexcept;
	DirectDctPlsPoolPlan next_pool();

private:
	const DirectDctPlsLayout*   layout_ = nullptr;
	DirectDctPlsScheduleOptions options_;
	std::vector<uint32_t>       pls_order_;
	std::vector<uint32_t>       global_order_;
	size_t                      next_pls_              = 0U;
	uint32_t                    next_pool_index_       = 0U;
	uint64_t                    next_microbatch_index_ = 0U;
};

struct DirectDctPlsAugmentationDecision {
	JpegDctImageCropRequest request;
	uint64_t                crop_seed = 0U;
	uint64_t                flip_seed = 0U;
};

// Exact native port of the keyed RandomResizedCrop_DCT(28) and per-sample
// horizontal-flip decisions used by rgbnomore-vitti-dct-published-v1.
DirectDctPlsAugmentationDecision derive_direct_dct_pls_augmentation(const DirectDctPlsSample&          sample,
                                                                    uint32_t                           source_width,
                                                                    uint32_t                           source_height,
                                                                    const DirectDctPlsScheduleOptions& options);

struct DirectDctPlsPipelineOptions {
	DirectDctPlsScheduleOptions schedule;
	JpegDctDeviceBatchOptions   device;
	uint32_t                    segment_images = 1024U;
	uint32_t                    model_classes  = 1000U;
	std::string                 expected_mapping_sha256;
	bool                        enable_published_randaugment = true;
	bool                        enable_published_mixup       = true;
	bool                        require_block_major_planless = true;
};

struct DirectDctPlsTargetTensorDescriptor {
	const float*          data = nullptr;
	std::array<size_t, 2> shape {0U, 0U};
	std::array<size_t, 2> strides {0U, 1U};
	int                   cuda_device = -1;

	[[nodiscard]] bool empty() const noexcept {
		return shape[0] == 0U || shape[1] == 0U;
	}
};

struct DirectDctPlsMicrobatchView {
	DirectDctGridTensorDescriptor      y;
	DirectDctGridTensorDescriptor      cbcr;
	DirectDctPlsTargetTensorDescriptor targets;
	std::span<const uint32_t>          global_image_ids;
	std::span<const int64_t>           labels;
	size_t                             pool_offset = 0U;
	size_t                             image_count = 0U;
};

// Experimental diagnostics for the bounded one-pool lookahead. A context is
// either the pool currently exposed to the consumer or the single pool being
// prepared/held ready by the native pipeline. Retired device backing remains
// governed by NativeBatchLease/NativeBatchCompletion and is not a context.
struct DirectDctPlsPoolPrefetchStats {
	size_t   context_capacity          = 2U;
	size_t   live_context_count        = 0U;
	size_t   peak_live_context_count   = 0U;
	size_t   context_waiter_count      = 0U;
	size_t   peak_context_waiter_count = 0U;
	uint64_t prepare_started_count     = 0U;
	uint64_t prepare_completed_count   = 0U;
	uint64_t activation_count          = 0U;
	uint64_t retired_count             = 0U;
	double   prepare_plan_ms            = 0.0;
	double   prepare_io_ms              = 0.0;
	double   prepare_materialize_ms     = 0.0;
	double   activation_wait_ms         = 0.0;
	double   activation_ms              = 0.0;
};

// Owns one model-ready M-PLS device pool. Tensor views are zero-copy slices;
// shuffle has already been encoded in CUDA transform output placement.
class DirectDctPlsPoolBatch {
public:
	DirectDctPlsPoolBatch() noexcept;
	~DirectDctPlsPoolBatch();
	DirectDctPlsPoolBatch(const DirectDctPlsPoolBatch&)            = delete;
	DirectDctPlsPoolBatch& operator=(const DirectDctPlsPoolBatch&) = delete;
	DirectDctPlsPoolBatch(DirectDctPlsPoolBatch&&) noexcept;
	DirectDctPlsPoolBatch& operator=(DirectDctPlsPoolBatch&&) noexcept;

	[[nodiscard]] uint32_t                     epoch() const noexcept;
	[[nodiscard]] uint32_t                     pool_index() const noexcept;
	[[nodiscard]] size_t                       image_count() const noexcept;
	[[nodiscard]] size_t                       microbatch_count() const noexcept;
	[[nodiscard]] const std::vector<uint32_t>& virtual_pls_ids() const noexcept;
	[[nodiscard]] const DirectDctBatch&        batch() const noexcept;
	[[nodiscard]] void*                        cuda_completion_event() const noexcept;
	[[nodiscard]] DirectDctPlsMicrobatchView   microbatch(size_t index) const;
	// Compatibility marker for callers that explicitly retire a pool. The
	// bounded context permit now follows the pool backing and is released only
	// when NativeBatchLease/NativeBatchCompletion can destroy that backing.
	void retire_context() noexcept;

private:
	friend class DirectDctPlsPipeline;
	DirectDctPlsPoolBatch(DirectDctPlsPoolPlan plan,
	                      DirectDctBatch       batch,
	                      std::vector<int64_t> labels,
	                      uint32_t             microbatch_images,
	                      std::shared_ptr<void> pool_context_owner);
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

// Native production data-plane owner. Python/PyTorch is intentionally absent:
// a framework binding may wrap the returned CUDA descriptors only at the model
// boundary.
class DirectDctPlsPipeline {
public:
	DirectDctPlsPipeline(const std::filesystem::path& manifest_path,
	                     const std::filesystem::path& premixed_mapping_csv,
	                     DirectDctPlsPipelineOptions  options);
	~DirectDctPlsPipeline();
	DirectDctPlsPipeline(const DirectDctPlsPipeline&)            = delete;
	DirectDctPlsPipeline& operator=(const DirectDctPlsPipeline&) = delete;
	DirectDctPlsPipeline(DirectDctPlsPipeline&&) noexcept;
	DirectDctPlsPipeline& operator=(DirectDctPlsPipeline&&) noexcept;

	void                  start_epoch(uint32_t epoch);
	[[nodiscard]] bool    has_next_pool() const noexcept;
	DirectDctPlsPoolBatch next_pool();

	[[nodiscard]] const DirectDctPlsLayout&          layout() const noexcept;
	[[nodiscard]] const DirectDctPlsPipelineOptions& options() const noexcept;
	[[nodiscard]] DirectDctPlsPoolPrefetchStats      prefetch_stats() const noexcept;

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

} // namespace galp::jpeg

#endif // GALP_WITH_JPEG_DCT

#endif // GALP_ADVANCED_DIRECT_DCT_PLS_HPP
