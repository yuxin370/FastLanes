#include "galp/direct_dct.hpp"

#if GALP_WITH_JPEG_DCT

#include <stdexcept>
#include <utility>

namespace galp::jpeg {
namespace {

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

const int16_t* DirectDctBatch::y_device_data() const noexcept {
	return batch_.y_coefficients();
}

const int16_t* DirectDctBatch::cbcr_device_data() const noexcept {
	return batch_.cbcr_coefficients();
}

DirectDctTensorDescriptor DirectDctBatch::tensor() const {
	if (batch_.layout() == JpegDctDeviceLayout::kYcbcrDctGrid ||
	    batch_.layout() == JpegDctDeviceLayout::kYcbcrDctGridFixed) {
		throw std::logic_error(
		    "compact Direct-DCT tensor is not available for Y/CbCr grid layouts; use y_tensor() or cbcr_tensor()");
	}
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

DirectDctGridTensorDescriptor DirectDctBatch::y_tensor() const {
	if (batch_.layout() != JpegDctDeviceLayout::kYcbcrDctGrid &&
	    batch_.layout() != JpegDctDeviceLayout::kYcbcrDctGridFixed) {
		throw std::logic_error(
		    "Y/CbCr DCT grid tensor is only available for Y/CbCr grid layouts; use tensor() for compact layout");
	}
	const auto shape = batch_.ycbcr_dct_grid_shape().y;
	return DirectDctGridTensorDescriptor {
	    y_device_data(),
	    shape,
	    {shape[1] * shape[2] * shape[3] * shape[4] * shape[5],
	     shape[2] * shape[3] * shape[4] * shape[5],
	     shape[3] * shape[4] * shape[5],
	     shape[4] * shape[5],
	     shape[5],
	     1U},
	    DirectDctTensorDataType::kInt16,
	    DirectDctTensorDevice::kCuda,
	    cuda_device_,
	};
}

DirectDctGridTensorDescriptor DirectDctBatch::cbcr_tensor() const {
	if (batch_.layout() != JpegDctDeviceLayout::kYcbcrDctGrid &&
	    batch_.layout() != JpegDctDeviceLayout::kYcbcrDctGridFixed) {
		throw std::logic_error(
		    "Y/CbCr DCT grid tensor is only available for Y/CbCr grid layouts; use tensor() for compact layout");
	}
	const auto shape = batch_.ycbcr_dct_grid_shape().cbcr;
	return DirectDctGridTensorDescriptor {
	    cbcr_device_data(),
	    shape,
	    {shape[1] * shape[2] * shape[3] * shape[4] * shape[5],
	     shape[2] * shape[3] * shape[4] * shape[5],
	     shape[3] * shape[4] * shape[5],
	     shape[4] * shape[5],
	     shape[5],
	     1U},
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

size_t DirectDctBatch::y_coefficient_count() const noexcept {
	return batch_.y_coefficient_count();
}

size_t DirectDctBatch::cbcr_coefficient_count() const noexcept {
	return batch_.cbcr_coefficient_count();
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

const JpegDctDeviceCacheStats& DirectDctBatch::cache_stats_ref() const noexcept {
	return batch_.cache_stats_ref();
}

const JpegDctDeviceExecutionStats& DirectDctBatch::execution_stats_ref() const noexcept {
	return batch_.execution_stats_ref();
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
	const auto device = batch.cuda_device();
	return DirectDctBatch(std::move(batch), std::move(ids), device);
}

DirectDctBatch DirectDctRuntime::ReadBatch(const std::vector<uint32_t>&     global_image_ids,
                                           const JpegDctCropBox&            crop,
                                           const JpegDctDeviceBatchOptions& options) {
	auto       requests = make_crop_requests(global_image_ids, crop);
	auto       batch    = reader_.ReadDeviceDctBatch(requests, options);
	auto       ids      = std::vector<uint32_t>(global_image_ids.begin(), global_image_ids.end());
	const auto device   = batch.cuda_device();
	return DirectDctBatch(std::move(batch), std::move(ids), device);
}

JpegDctDeviceBatchPlanPreview DirectDctRuntime::PlanBatch(const std::vector<JpegDctImageCropRequest>& requests,
                                                          const JpegDctDeviceBatchOptions&            options) const {
	return reader_.PlanDeviceDctBatch(requests, options);
}

} // namespace galp::jpeg

#endif // GALP_WITH_JPEG_DCT
