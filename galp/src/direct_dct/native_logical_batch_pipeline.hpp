#ifndef GALP_DIRECT_DCT_NATIVE_LOGICAL_BATCH_PIPELINE_HPP
#define GALP_DIRECT_DCT_NATIVE_LOGICAL_BATCH_PIPELINE_HPP

#include "direct_dct/logical_types.hpp"
#include "direct_dct/native_pipeline_state.hpp"
#include "direct_dct/physical_layout_planner.hpp"
#include "galp/direct_dct.hpp"
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string_view>
#include <vector>

namespace galp::direct_dct {

// Native execution owner. Phase 2 used it only as a shadow executor; Phase 3
// permits the private Torch adapter to delegate to it without exposing it as a
// public API.
class NativeLogicalBatchPipeline final {
public:
	NativeLogicalBatchPipeline(const std::filesystem::path& manifest_path,
	                           std::string_view             semantic_profile_id,
	                           NativePipelineTraceBuffer*   trace = nullptr);
	NativeLogicalBatchPipeline(std::shared_ptr<jpeg::DirectDctRuntime> runtime,
	                           std::string_view                        semantic_profile_id,
	                           jpeg::JpegDctDeviceBatchOptions         options,
	                           NativePipelineTraceBuffer*              trace = nullptr);
	NativeLogicalBatchPipeline(std::shared_ptr<jpeg::DirectDctRuntime> runtime,
	                           const std::filesystem::path&            manifest_path,
	                           std::string_view                        semantic_profile_id,
	                           jpeg::JpegDctDeviceBatchOptions         options,
	                           NativePipelineTraceBuffer*              trace = nullptr);
	~NativeLogicalBatchPipeline();

	NativeLogicalBatchPipeline(const NativeLogicalBatchPipeline&)            = delete;
	NativeLogicalBatchPipeline& operator=(const NativeLogicalBatchPipeline&) = delete;
	NativeLogicalBatchPipeline(NativeLogicalBatchPipeline&&)                 = delete;
	NativeLogicalBatchPipeline& operator=(NativeLogicalBatchPipeline&&)      = delete;

	void                               reset(std::vector<LogicalBatchRequest> requests);
	[[nodiscard]] jpeg::DirectDctBatch next();
	[[nodiscard]] bool                 ready() const;
	[[nodiscard]] bool                 started() const noexcept;
	[[nodiscard]] size_t               prefetched_batch_count() const noexcept;
	[[nodiscard]] NativePipelineState  state() const noexcept;
	[[nodiscard]] NativePipelinePrefetchMetrics prefetch_metrics() const noexcept;
	[[nodiscard]] const std::vector<LogicalBatchPhysicalPlan>& physical_plans() const noexcept;
	size_t                             close() noexcept;

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

} // namespace galp::direct_dct

#endif // GALP_DIRECT_DCT_NATIVE_LOGICAL_BATCH_PIPELINE_HPP
