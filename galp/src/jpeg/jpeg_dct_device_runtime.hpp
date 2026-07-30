#ifndef GALP_JPEG_DCT_DEVICE_RUNTIME_HPP
#define GALP_JPEG_DCT_DEVICE_RUNTIME_HPP

#include "jpeg/jpeg_dct_plan_types.hpp"
#include <cstddef>
#include <memory>
#include <string>

namespace galp::jpeg::detail {

struct JpegDctDeviceScratchDeleter {
	void operator()(JpegDctDeviceScratch* scratch) const noexcept;
};
using JpegDctDeviceScratchPtr = std::unique_ptr<JpegDctDeviceScratch, JpegDctDeviceScratchDeleter>;

struct JpegDctDeviceExecutionFence;
struct JpegDctDeviceExecutionFenceDeleter {
	void operator()(JpegDctDeviceExecutionFence* fence) const noexcept;
};
using JpegDctDeviceExecutionFencePtr = std::unique_ptr<JpegDctDeviceExecutionFence, JpegDctDeviceExecutionFenceDeleter>;

struct JpegDctDeviceDecodedRowgroupCacheDeleter {
	void operator()(JpegDctDeviceDecodedRowgroupCache* cache) const noexcept;
};
using JpegDctDeviceDecodedRowgroupCachePtr =
    std::unique_ptr<JpegDctDeviceDecodedRowgroupCache, JpegDctDeviceDecodedRowgroupCacheDeleter>;

struct JpegDctHostIoContext;
struct JpegDctHostIoContextDeleter {
	void operator()(JpegDctHostIoContext* context) const noexcept;
};
using JpegDctHostIoContextPtr = std::unique_ptr<JpegDctHostIoContext, JpegDctHostIoContextDeleter>;

JpegDctDeviceScratchPtr              make_jpeg_dct_device_scratch();
JpegDctDeviceExecutionFencePtr       make_jpeg_dct_device_execution_fence();
JpegDctDeviceDecodedRowgroupCachePtr make_jpeg_dct_device_cache();
JpegDctHostIoContextPtr              make_jpeg_dct_host_io_context();
void set_jpeg_dct_device_cache_capacity(JpegDctDeviceDecodedRowgroupCache& cache, size_t bytes);
void wait_jpeg_dct_device_execution_fence(JpegDctDeviceExecutionFence& fence);
void compile_jpeg_dct_device_batch_io(JpegDctDeviceBatchPlan& plan, JpegDctHostIoContext& context);
void stage_jpeg_dct_device_batch_io(JpegDctDeviceBatchPlan& plan, JpegDctHostIoContext& context);
JpegDctDeviceBatch execute_jpeg_dct_device_batch_plan(JpegDctDeviceBatchPlan       plan,
                                                      JpegDctDeviceExecutionFence* reuse_fence = nullptr);

// Deterministic completion-failure seam used by unit tests. This private
// declaration is not installed as part of the public API.
JpegDctDeviceBatch make_failed_device_batch_for_testing(std::string message);

} // namespace galp::jpeg::detail

#endif // GALP_JPEG_DCT_DEVICE_RUNTIME_HPP
