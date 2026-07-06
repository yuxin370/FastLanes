#include "galp/direct_dct.hpp"

#if GALP_WITH_JPEG_DCT

#include <cuda_runtime.h>
#include <utility>

namespace galp::jpeg {
namespace {

int detect_cuda_device(const int16_t* ptr) noexcept {
	int device = -1;
	if (ptr != nullptr) {
		cudaPointerAttributes attrs {};
		const auto            status = cudaPointerGetAttributes(&attrs, ptr);
		if (status == cudaSuccess) {
			device = attrs.device;
		} else {
			(void)cudaGetLastError();
		}
	}
	if (device < 0) {
		int current_device = -1;
		if (cudaGetDevice(&current_device) == cudaSuccess) {
			device = current_device;
		} else {
			(void)cudaGetLastError();
		}
	}
	return device;
}

std::vector<uint32_t> request_global_image_ids(const std::vector<JpegDctImageCropRequest>& requests) {
	std::vector<uint32_t> image_ids;
	image_ids.reserve(requests.size());
	for (const auto& request : requests) {
		image_ids.push_back(request.global_image_index);
	}
	return image_ids;
}

std::vector<JpegDctImageCropRequest> make_crop_requests(const std::vector<uint32_t>& global_image_ids,
                                                        const JpegDctCropBox&        crop) {
	std::vector<JpegDctImageCropRequest> requests;
	requests.reserve(global_image_ids.size());
	for (const auto image_id : global_image_ids) {
		requests.push_back(JpegDctImageCropRequest {image_id, crop});
	}
	return requests;
}

} // namespace

DirectDctBatch::DirectDctBatch() noexcept = default;

DirectDctBatch::DirectDctBatch(JpegDctDeviceBatch    batch,
                               std::vector<uint32_t> global_image_ids,
                               const int             cuda_device) noexcept
    : batch_(std::move(batch))
    , global_image_ids_(std::move(global_image_ids))
    , cuda_device_(cuda_device) {
}

DirectDctBatch::~DirectDctBatch() = default;

DirectDctBatch::DirectDctBatch(DirectDctBatch&&) noexcept = default;

DirectDctBatch& DirectDctBatch::operator=(DirectDctBatch&&) noexcept = default;

const int16_t* DirectDctBatch::device_data() const noexcept {
	return batch_.device_coefficients();
}

DirectDctTensorDescriptor DirectDctBatch::tensor() const noexcept {
	const auto columns = coefficients_per_block();
	return DirectDctTensorDescriptor {
	    device_data(),
	    {block_count(), columns},
	    {columns, 1U},
	    DirectDctTensorDataType::kInt16,
	    DirectDctTensorDevice::kCuda,
	    cuda_device_,
	};
}

size_t DirectDctBatch::block_count() const noexcept {
	return batch_.block_count();
}

size_t DirectDctBatch::coefficients_per_block() const noexcept {
	return batch_.coefficients_per_block();
}

size_t DirectDctBatch::coefficient_count() const noexcept {
	return batch_.coefficient_count();
}

size_t DirectDctBatch::coefficient_bytes() const noexcept {
	return batch_.coefficient_bytes();
}

size_t DirectDctBatch::image_count() const noexcept {
	return batch_.image_count();
}

int DirectDctBatch::cuda_device() const noexcept {
	return cuda_device_;
}

const std::vector<uint32_t>& DirectDctBatch::global_image_ids() const noexcept {
	return global_image_ids_;
}

const std::vector<JpegDctDeviceImageLayout>& DirectDctBatch::image_layouts() const noexcept {
	return batch_.image_layouts();
}

const std::vector<JpegDctDeviceBlockMetadata>& DirectDctBatch::block_metadata() const noexcept {
	return batch_.block_metadata();
}

const std::vector<JpegDctDeviceRowgroupMetadata>& DirectDctBatch::rowgroups() const noexcept {
	return batch_.rowgroups();
}

const std::vector<uint8_t>& DirectDctBatch::selected_coefficients() const noexcept {
	return batch_.selected_coefficients();
}

JpegDctDeviceCacheStats DirectDctBatch::cache_stats() const noexcept {
	return batch_.cache_stats();
}

JpegDctDeviceExecutionStats DirectDctBatch::execution_stats() const noexcept {
	return batch_.execution_stats();
}

const JpegDctDeviceBatch& DirectDctBatch::device_batch() const noexcept {
	return batch_;
}

DirectDctRuntime::DirectDctRuntime(const std::filesystem::path& manifest_path)
    : reader_(manifest_path) {
}

DirectDctRuntime::~DirectDctRuntime() = default;

DirectDctRuntime::DirectDctRuntime(DirectDctRuntime&&) noexcept = default;

DirectDctRuntime& DirectDctRuntime::operator=(DirectDctRuntime&&) noexcept = default;

uint64_t DirectDctRuntime::image_count() const noexcept {
	return reader_.image_count();
}

JpegImageMetadata DirectDctRuntime::ImageMetadata(const uint32_t global_image_index) const {
	return reader_.ImageMetadata(global_image_index);
}

DirectDctBatch DirectDctRuntime::ReadBatch(const std::vector<JpegDctImageCropRequest>& requests,
                                           const JpegDctDeviceBatchOptions&            options) {
	auto       batch  = reader_.ReadDeviceDctBatch(requests, options);
	auto       ids    = request_global_image_ids(requests);
	const auto device = detect_cuda_device(batch.device_coefficients());
	return DirectDctBatch(std::move(batch), std::move(ids), device);
}

DirectDctBatch DirectDctRuntime::ReadBatch(const std::vector<uint32_t>&     global_image_ids,
                                           const JpegDctCropBox&            crop,
                                           const JpegDctDeviceBatchOptions& options) {
	return ReadBatch(make_crop_requests(global_image_ids, crop), options);
}

JpegDctDeviceBatchPlanPreview DirectDctRuntime::PlanBatch(const std::vector<JpegDctImageCropRequest>& requests,
                                                          const JpegDctDeviceBatchOptions&            options) const {
	return reader_.PlanDeviceDctBatch(requests, options);
}

} // namespace galp::jpeg

#endif // GALP_WITH_JPEG_DCT
