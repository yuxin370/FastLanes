#include "jpeg/jpeg_dct_device_bridge.hpp"
#include "jpeg/jpeg_dct_device_runtime.hpp"
#include "jpeg/jpeg_dct_plan_types.hpp"
#include "jpeg/jpeg_dct_policy.hpp"
#include <mutex>
#include <utility>

namespace galp::jpeg::detail {

struct JpegDctDeviceHostBridge::Impl {
	void AttachRuntimeResources(JpegDctDeviceBatchPlan& plan, const JpegDctDeviceBatchOptions& options) {
		if (!scratch) {
			scratch = make_jpeg_dct_device_scratch();
		}
		plan.scratch = scratch.get();

		// The compact transformed path has no decoded-rowgroup cache branch.
		// Retire a legacy cache when the requested plan cannot consume it.
		const bool cache_compatible = !plan.uses_planless_fixed_transform &&
		                              (options.layout == JpegDctDeviceLayout::kImageMajorComponentBlockCoeff ||
		                               options.layout == JpegDctDeviceLayout::kTransformedDctGrid) &&
		                              selects_all_coefficients(plan.selected_coefficients);
		plan.cache_enabled = options.cache_capacity_bytes > 0 && cache_compatible;
		if (plan.cache_enabled) {
			if (!cache) {
				cache = make_jpeg_dct_device_cache();
			}
			set_jpeg_dct_device_cache_capacity(*cache, options.cache_capacity_bytes);
			plan.cache = cache.get();
		} else if (cache) {
			set_jpeg_dct_device_cache_capacity(*cache, 0);
			plan.cache = nullptr;
		}
	}

	JpegDctDeviceDecodedRowgroupCachePtr cache;
	JpegDctDeviceScratchPtr              scratch;
	JpegDctDeviceExecutionFencePtr       execution_fence;
	JpegDctHostIoContextPtr              host_io_context = make_jpeg_dct_host_io_context();
	// The lock covers joining the previous asynchronous tail, rebinding all
	// shared cache/scratch state, and submitting the next execution.
	std::mutex execution_mutex;
};

JpegDctDeviceHostBridge::JpegDctDeviceHostBridge()
    : impl_(std::make_unique<Impl>()) {
}

JpegDctDeviceHostBridge::~JpegDctDeviceHostBridge() = default;

void JpegDctDeviceHostBridge::CompileIoPlan(JpegDctDeviceBatchPlan& plan) {
	compile_jpeg_dct_device_batch_io(plan, *impl_->host_io_context);
}

void JpegDctDeviceHostBridge::StageIo(JpegDctDeviceBatchPlan& plan) {
	stage_jpeg_dct_device_batch_io(plan, *impl_->host_io_context);
}

JpegDctDeviceBatch JpegDctDeviceHostBridge::Execute(JpegDctDeviceBatchPlan           plan,
                                                    const JpegDctDeviceBatchOptions& options) {
	std::lock_guard<std::mutex> execution_guard(impl_->execution_mutex);
	if (!impl_->execution_fence) {
		impl_->execution_fence = make_jpeg_dct_device_execution_fence();
	}
	wait_jpeg_dct_device_execution_fence(*impl_->execution_fence);
	impl_->AttachRuntimeResources(plan, options);
	auto batch = execute_jpeg_dct_device_batch_plan(std::move(plan), impl_->execution_fence.get());
	return batch;
}

} // namespace galp::jpeg::detail
