#include "galp/direct_dct.hpp"

#if GALP_WITH_JPEG_DCT

#include "cuda/memory/cuda_raii.cuh"
#include "cuda/memory/gpu_array.cuh"
#include <algorithm>
#include <cuda_runtime_api.h>
#include <optional>
#include <stdexcept>
#include <string>
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

void check_cuda(const cudaError_t status, const char* operation) {
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
	}
}

size_t grid_image_elements(const DirectDctGridTensorDescriptor& descriptor) {
	return descriptor.strides[0];
}

} // namespace

struct DirectDctBatch::LogicalImpl final {
	std::vector<LogicalSegmentInput>       segments;
	DirectDctGridTensorDescriptor          y_descriptor;
	DirectDctGridTensorDescriptor          cbcr_descriptor;
	// Stream-ordered allocations remember this stream for cudaFreeAsync.
	// Declare the stream first so reverse member destruction releases both
	// outputs before destroying their allocation stream.
	galp::memory::CudaStream               assembly_stream;
	std::optional<GPUArray<float>>         y_output;
	std::optional<GPUArray<float>>         cbcr_output;
	galp::memory::CudaEvent                completion_event;
	std::shared_ptr<DirectDctBatch>         stats_source;
	std::vector<JpegDctDeviceImageLayout>  image_layouts;
	std::vector<JpegDctDeviceBlockMetadata> block_metadata;
	std::vector<JpegDctDeviceRowgroupMetadata> rowgroups;
	JpegDctDeviceCacheStats                empty_cache_stats;
	JpegDctDeviceExecutionStats            empty_execution_stats;
	size_t                                  block_count = 0U;

	[[nodiscard]] bool assembled() const noexcept {
		return static_cast<bool>(completion_event);
	}

	[[nodiscard]] const DirectDctBatch& representative() const {
		if (segments.empty() || !segments.front().source) {
			throw std::logic_error("logical Direct-DCT batch has no physical backing");
		}
		return *segments.front().source;
	}

	void synchronize() const {
		if (completion_event) {
			check_cuda(cudaEventSynchronize(completion_event.get()), "cudaEventSynchronize(logical batch)");
			return;
		}
		representative().synchronize();
	}

	[[nodiscard]] bool ready() const {
		if (!completion_event) {
			return representative().try_finalize_execution_stats();
		}
		const auto status = cudaEventQuery(completion_event.get());
		if (status == cudaErrorNotReady) {
			return false;
		}
		check_cuda(status, "cudaEventQuery(logical batch)");
		return true;
	}
};

DirectDctBatch::DirectDctBatch() noexcept = default;

DirectDctBatch::DirectDctBatch(JpegDctDeviceBatch    batch,
                               std::vector<uint32_t> global_image_ids,
                               std::vector<JpegDctImageCropRequest> transform_requests,
                               const int             cuda_device) noexcept
    : batch_(std::move(batch))
    , global_image_ids_(std::move(global_image_ids))
	, transform_requests_(std::move(transform_requests))
	, cuda_device_(cuda_device) {
}

DirectDctBatch DirectDctBatch::MakeLogicalGridBatch(
    std::vector<LogicalSegmentInput> segments,
    std::vector<uint32_t> global_image_ids,
    std::vector<JpegDctImageCropRequest> transform_requests,
    std::shared_ptr<DirectDctBatch> stats_source) {
	if (segments.empty() || global_image_ids.empty()) {
		throw std::invalid_argument("logical Direct-DCT grid batch requires physical segments");
	}
	if (global_image_ids.size() != transform_requests.size()) {
		throw std::invalid_argument("logical Direct-DCT grid batch metadata cardinality mismatch");
	}
	const auto logical_image_count = global_image_ids.size();
	auto logical                    = std::make_unique<LogicalImpl>();
	logical->segments               = std::move(segments);
	logical->stats_source           = std::move(stats_source);

	const auto& first = *logical->segments.front().source;
	const auto first_y = first.y_tensor_async();
	const auto first_cbcr = first.cbcr_tensor_async();
	if (first_y.dtype != DirectDctTensorDataType::kFloat32 ||
	    first_cbcr.dtype != DirectDctTensorDataType::kFloat32) {
		throw std::invalid_argument("native physical assembly currently requires FP32 transformed-grid output");
	}
	const auto cuda_device = first.cuda_device();
	logical->y_descriptor = first_y;
	logical->cbcr_descriptor = first_cbcr;
	logical->y_descriptor.shape[0] = logical_image_count;
	logical->cbcr_descriptor.shape[0] = logical_image_count;

	for (const auto& segment : logical->segments) {
		if (!segment.source || segment.image_count == 0U ||
		    segment.logical_image_offset + segment.image_count > logical_image_count ||
		    segment.source_image_offset + segment.image_count > segment.source->image_count()) {
			throw std::invalid_argument("logical Direct-DCT segment is outside its source or destination range");
		}
		const auto y = segment.source->y_tensor_async();
		const auto cbcr = segment.source->cbcr_tensor_async();
		if (segment.source->cuda_device() != cuda_device || y.dtype != first_y.dtype ||
		    cbcr.dtype != first_cbcr.dtype ||
		    !std::equal(y.shape.begin() + 1, y.shape.end(), first_y.shape.begin() + 1) ||
		    !std::equal(cbcr.shape.begin() + 1, cbcr.shape.end(), first_cbcr.shape.begin() + 1)) {
			throw std::invalid_argument("logical Direct-DCT segments have incompatible grid contracts");
		}
		const auto& source_layouts = segment.source->image_layouts();
		if (segment.source_image_offset + segment.image_count > source_layouts.size()) {
			throw std::invalid_argument("logical Direct-DCT segment has incomplete image layout metadata");
		}
		for (size_t index = 0U; index < segment.image_count; ++index) {
			auto layout = source_layouts[segment.source_image_offset + index];
			layout.block_offset = logical->block_count;
			logical->block_count += layout.block_count;
			logical->image_layouts.push_back(layout);
		}
	}

	if (logical->segments.size() == 1U) {
		const auto& segment = logical->segments.front();
		logical->y_descriptor.float_data =
		    first_y.float_data + segment.source_image_offset * grid_image_elements(first_y);
		logical->cbcr_descriptor.float_data =
		    first_cbcr.float_data + segment.source_image_offset * grid_image_elements(first_cbcr);
	} else {
		check_cuda(cudaSetDevice(cuda_device), "cudaSetDevice(logical batch assembly)");
		logical->assembly_stream.create(cudaStreamNonBlocking);
		const auto stream = logical->assembly_stream.get();
		const auto y_count = logical_image_count * grid_image_elements(first_y);
		const auto cbcr_count = logical_image_count * grid_image_elements(first_cbcr);
		logical->y_output.emplace(y_count, stream);
		logical->cbcr_output.emplace(cbcr_count, stream);
		logical->y_descriptor.float_data = logical->y_output->get();
		logical->cbcr_descriptor.float_data = logical->cbcr_output->get();
		for (const auto& segment : logical->segments) {
			const auto y = segment.source->y_tensor_async();
			const auto cbcr = segment.source->cbcr_tensor_async();
			if (auto* event = segment.source->cuda_completion_event(); event != nullptr) {
				check_cuda(cudaStreamWaitEvent(stream, static_cast<cudaEvent_t>(event), 0U),
				           "cudaStreamWaitEvent(logical source)");
			}
			const auto y_elements = segment.image_count * grid_image_elements(y);
			const auto cbcr_elements = segment.image_count * grid_image_elements(cbcr);
			check_cuda(cudaMemcpyAsync(
			               logical->y_output->get() + segment.logical_image_offset * grid_image_elements(first_y),
			               y.float_data + segment.source_image_offset * grid_image_elements(y),
			               y_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream),
			           "cudaMemcpyAsync(logical Y segment)");
			check_cuda(cudaMemcpyAsync(
			               logical->cbcr_output->get() +
			                   segment.logical_image_offset * grid_image_elements(first_cbcr),
			               cbcr.float_data + segment.source_image_offset * grid_image_elements(cbcr),
			               cbcr_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream),
			           "cudaMemcpyAsync(logical CbCr segment)");
		}
		logical->completion_event.create_with_flags(cudaEventDisableTiming);
		logical->completion_event.record(stream);
	}

	DirectDctBatch result;
	result.global_image_ids_   = std::move(global_image_ids);
	result.transform_requests_ = std::move(transform_requests);
	result.cuda_device_        = cuda_device;
	result.logical_            = std::move(logical);
	return result;
}

DirectDctBatch::~DirectDctBatch() = default;

DirectDctBatch::DirectDctBatch(DirectDctBatch&&) noexcept = default;

DirectDctBatch& DirectDctBatch::operator=(DirectDctBatch&&) noexcept = default;

const int16_t* DirectDctBatch::device_data() const {
	if (logical_) {
		throw std::logic_error("compact coefficient data is unavailable for a logical grid batch");
	}
	return batch_.device_coefficients();
}

const int16_t* DirectDctBatch::y_device_data() const {
	if (logical_) {
		logical_->synchronize();
		return logical_->y_descriptor.data;
	}
	return batch_.y_coefficients();
}

const int16_t* DirectDctBatch::cbcr_device_data() const {
	if (logical_) {
		logical_->synchronize();
		return logical_->cbcr_descriptor.data;
	}
	return batch_.cbcr_coefficients();
}

const int16_t* DirectDctBatch::device_data_async() const noexcept {
	if (logical_) {
		return nullptr;
	}
	return batch_.device_coefficients_async();
}

const int16_t* DirectDctBatch::y_device_data_async() const noexcept {
	if (logical_) {
		return logical_->y_descriptor.data;
	}
	return batch_.y_coefficients_async();
}

const int16_t* DirectDctBatch::cbcr_device_data_async() const noexcept {
	if (logical_) {
		return logical_->cbcr_descriptor.data;
	}
	return batch_.cbcr_coefficients_async();
}

const float* DirectDctBatch::y_float_device_data() const {
	if (logical_) {
		logical_->synchronize();
		return logical_->y_descriptor.float_data;
	}
	return batch_.y_float_coefficients();
}

const float* DirectDctBatch::cbcr_float_device_data() const {
	if (logical_) {
		logical_->synchronize();
		return logical_->cbcr_descriptor.float_data;
	}
	return batch_.cbcr_float_coefficients();
}

const float* DirectDctBatch::y_float_device_data_async() const noexcept {
	if (logical_) {
		return logical_->y_descriptor.float_data;
	}
	return batch_.y_float_coefficients_async();
}

const float* DirectDctBatch::cbcr_float_device_data_async() const noexcept {
	if (logical_) {
		return logical_->cbcr_descriptor.float_data;
	}
	return batch_.cbcr_float_coefficients_async();
}

void DirectDctBatch::synchronize() const {
	if (logical_) {
		logical_->synchronize();
		return;
	}
	batch_.synchronize();
}

DirectDctTensorDescriptor DirectDctBatch::tensor() const {
	if (logical_) {
		throw std::logic_error("compact Direct-DCT tensor is unavailable for a logical grid batch");
	}
	if (batch_.layout() == JpegDctDeviceLayout::kYcbcrDctGrid ||
	    batch_.layout() == JpegDctDeviceLayout::kTransformedDctGrid) {
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
	if (logical_) {
		logical_->synchronize();
		return logical_->y_descriptor;
	}
	if (batch_.layout() != JpegDctDeviceLayout::kYcbcrDctGrid &&
	    batch_.layout() != JpegDctDeviceLayout::kTransformedDctGrid) {
		throw std::logic_error(
		    "Y/CbCr DCT grid tensor is only available for Y/CbCr grid layouts; use tensor() for compact layout");
	}
	const auto shape = batch_.ycbcr_dct_grid_shape().y;
	const bool float_output = batch_.grid_output_data_type() == JpegDctGridOutputDataType::kFloat32;
	return DirectDctGridTensorDescriptor {
	    float_output ? nullptr : y_device_data(),
	    float_output ? y_float_device_data() : nullptr,
	    shape,
	    {shape[1] * shape[2] * shape[3] * shape[4] * shape[5],
	     shape[2] * shape[3] * shape[4] * shape[5],
	     shape[3] * shape[4] * shape[5],
	     shape[4] * shape[5],
	     shape[5],
	     1U},
	    float_output ? DirectDctTensorDataType::kFloat32 : DirectDctTensorDataType::kInt16,
	    DirectDctTensorDevice::kCuda,
	    cuda_device_,
	};
}

DirectDctGridTensorDescriptor DirectDctBatch::cbcr_tensor() const {
	if (logical_) {
		logical_->synchronize();
		return logical_->cbcr_descriptor;
	}
	if (batch_.layout() != JpegDctDeviceLayout::kYcbcrDctGrid &&
	    batch_.layout() != JpegDctDeviceLayout::kTransformedDctGrid) {
		throw std::logic_error(
		    "Y/CbCr DCT grid tensor is only available for Y/CbCr grid layouts; use tensor() for compact layout");
	}
	const auto shape = batch_.ycbcr_dct_grid_shape().cbcr;
	const bool float_output = batch_.grid_output_data_type() == JpegDctGridOutputDataType::kFloat32;
	return DirectDctGridTensorDescriptor {
	    float_output ? nullptr : cbcr_device_data(),
	    float_output ? cbcr_float_device_data() : nullptr,
	    shape,
	    {shape[1] * shape[2] * shape[3] * shape[4] * shape[5],
	     shape[2] * shape[3] * shape[4] * shape[5],
	     shape[3] * shape[4] * shape[5],
	     shape[4] * shape[5],
	     shape[5],
	     1U},
	    float_output ? DirectDctTensorDataType::kFloat32 : DirectDctTensorDataType::kInt16,
	    DirectDctTensorDevice::kCuda,
	    cuda_device_,
	};
}

DirectDctTensorDescriptor DirectDctBatch::tensor_async() const {
	if (logical_) {
		throw std::logic_error("compact Direct-DCT tensor is unavailable for a logical grid batch");
	}
	if (batch_.layout() == JpegDctDeviceLayout::kYcbcrDctGrid ||
	    batch_.layout() == JpegDctDeviceLayout::kTransformedDctGrid) {
		throw std::logic_error(
		    "compact Direct-DCT tensor is not available for Y/CbCr grid layouts; use y_tensor_async() or cbcr_tensor_async()");
	}
	const auto columns = coefficients_per_block();
	return DirectDctTensorDescriptor {
	    device_data_async(),
	    {block_count(), columns},
	    {columns, 1U},
	    DirectDctTensorDataType::kInt16,
	    DirectDctTensorDevice::kCuda,
	    cuda_device_,
	};
}

DirectDctGridTensorDescriptor DirectDctBatch::y_tensor_async() const {
	if (logical_) {
		return logical_->y_descriptor;
	}
	if (batch_.layout() != JpegDctDeviceLayout::kYcbcrDctGrid &&
	    batch_.layout() != JpegDctDeviceLayout::kTransformedDctGrid) {
		throw std::logic_error(
		    "Y/CbCr DCT grid tensor is only available for Y/CbCr grid layouts; use tensor_async() for compact layout");
	}
	const auto shape = batch_.ycbcr_dct_grid_shape().y;
	const bool float_output = batch_.grid_output_data_type() == JpegDctGridOutputDataType::kFloat32;
	return DirectDctGridTensorDescriptor {
	    float_output ? nullptr : y_device_data_async(),
	    float_output ? y_float_device_data_async() : nullptr,
	    shape,
	    {shape[1] * shape[2] * shape[3] * shape[4] * shape[5],
	     shape[2] * shape[3] * shape[4] * shape[5],
	     shape[3] * shape[4] * shape[5],
	     shape[4] * shape[5],
	     shape[5],
	     1U},
	    float_output ? DirectDctTensorDataType::kFloat32 : DirectDctTensorDataType::kInt16,
	    DirectDctTensorDevice::kCuda,
	    cuda_device_,
	};
}

DirectDctGridTensorDescriptor DirectDctBatch::cbcr_tensor_async() const {
	if (logical_) {
		return logical_->cbcr_descriptor;
	}
	if (batch_.layout() != JpegDctDeviceLayout::kYcbcrDctGrid &&
	    batch_.layout() != JpegDctDeviceLayout::kTransformedDctGrid) {
		throw std::logic_error(
		    "Y/CbCr DCT grid tensor is only available for Y/CbCr grid layouts; use tensor_async() for compact layout");
	}
	const auto shape = batch_.ycbcr_dct_grid_shape().cbcr;
	const bool float_output = batch_.grid_output_data_type() == JpegDctGridOutputDataType::kFloat32;
	return DirectDctGridTensorDescriptor {
	    float_output ? nullptr : cbcr_device_data_async(),
	    float_output ? cbcr_float_device_data_async() : nullptr,
	    shape,
	    {shape[1] * shape[2] * shape[3] * shape[4] * shape[5],
	     shape[2] * shape[3] * shape[4] * shape[5],
	     shape[3] * shape[4] * shape[5],
	     shape[4] * shape[5],
	     shape[5],
	     1U},
	    float_output ? DirectDctTensorDataType::kFloat32 : DirectDctTensorDataType::kInt16,
	    DirectDctTensorDevice::kCuda,
	    cuda_device_,
	};
}

size_t DirectDctBatch::block_count() const noexcept {
	if (logical_) {
		return logical_->block_count;
	}
	return batch_.block_count();
}

size_t DirectDctBatch::coefficients_per_block() const noexcept {
	if (logical_) {
		return logical_->representative().coefficients_per_block();
	}
	return batch_.coefficients_per_block();
}

size_t DirectDctBatch::coefficient_count() const noexcept {
	if (logical_) {
		return logical_->y_descriptor.element_count() + logical_->cbcr_descriptor.element_count();
	}
	return batch_.coefficient_count();
}

size_t DirectDctBatch::coefficient_bytes() const noexcept {
	if (logical_) {
		return coefficient_count() * sizeof(float);
	}
	return batch_.coefficient_bytes();
}

size_t DirectDctBatch::y_coefficient_count() const noexcept {
	if (logical_) {
		return logical_->y_descriptor.element_count();
	}
	return batch_.y_coefficient_count();
}

size_t DirectDctBatch::cbcr_coefficient_count() const noexcept {
	if (logical_) {
		return logical_->cbcr_descriptor.element_count();
	}
	return batch_.cbcr_coefficient_count();
}

size_t DirectDctBatch::image_count() const noexcept {
	if (logical_) {
		return global_image_ids_.size();
	}
	return batch_.image_count();
}

int DirectDctBatch::cuda_device() const noexcept {
	return cuda_device_;
}

void* DirectDctBatch::cuda_completion_event() const noexcept {
	if (logical_) {
		return logical_->completion_event
		           ? static_cast<void*>(logical_->completion_event.get())
		           : logical_->representative().cuda_completion_event();
	}
	return batch_.cuda_completion_event();
}

const std::vector<uint32_t>& DirectDctBatch::global_image_ids() const noexcept {
	return global_image_ids_;
}

const std::vector<JpegDctImageCropRequest>& DirectDctBatch::transform_requests() const noexcept {
	return transform_requests_;
}

const std::vector<JpegDctDeviceImageLayout>& DirectDctBatch::image_layouts() const noexcept {
	if (logical_) {
		return logical_->image_layouts;
	}
	return batch_.image_layouts();
}

const std::vector<JpegDctDeviceBlockMetadata>& DirectDctBatch::block_metadata() const noexcept {
	if (logical_) {
		return logical_->block_metadata;
	}
	return batch_.block_metadata();
}

const std::vector<JpegDctDeviceRowgroupMetadata>& DirectDctBatch::rowgroups() const noexcept {
	if (logical_) {
		return logical_->rowgroups;
	}
	return batch_.rowgroups();
}

const std::vector<uint8_t>& DirectDctBatch::selected_coefficients() const noexcept {
	if (logical_) {
		return logical_->representative().selected_coefficients();
	}
	return batch_.selected_coefficients();
}

JpegDctDeviceCacheStats DirectDctBatch::cache_stats() const noexcept {
	if (logical_) {
		return cache_stats_ref();
	}
	return batch_.cache_stats();
}

JpegDctDeviceExecutionStats DirectDctBatch::execution_stats() const {
	if (logical_) {
		logical_->synchronize();
		return logical_->stats_source ? logical_->stats_source->execution_stats()
		                              : logical_->empty_execution_stats;
	}
	return batch_.execution_stats();
}

bool DirectDctBatch::try_finalize_execution_stats() const {
	if (logical_) {
		if (!logical_->ready()) {
			return false;
		}
		return !logical_->stats_source || logical_->stats_source->try_finalize_execution_stats();
	}
	return batch_.try_finalize_execution_stats();
}

const JpegDctDeviceCacheStats& DirectDctBatch::cache_stats_ref() const noexcept {
	if (logical_) {
		return logical_->stats_source ? logical_->stats_source->cache_stats_ref()
		                              : logical_->empty_cache_stats;
	}
	return batch_.cache_stats_ref();
}

const JpegDctDeviceExecutionStats& DirectDctBatch::execution_stats_ref() const noexcept {
	if (logical_) {
		return logical_->stats_source ? logical_->stats_source->execution_stats_ref()
		                              : logical_->empty_execution_stats;
	}
	return batch_.execution_stats_ref();
}

const JpegDctDeviceBatch& DirectDctBatch::device_batch() const noexcept {
	if (logical_) {
		return logical_->representative().device_batch();
	}
	return batch_;
}

DirectDctPreparedBatch::DirectDctPreparedBatch() noexcept = default;
DirectDctPreparedBatch::~DirectDctPreparedBatch() = default;
DirectDctPreparedBatch::DirectDctPreparedBatch(DirectDctPreparedBatch&&) noexcept = default;
DirectDctPreparedBatch& DirectDctPreparedBatch::operator=(DirectDctPreparedBatch&&) noexcept = default;

DirectDctPreparedBatch::DirectDctPreparedBatch(JpegDctDeviceBatchPreparedPlan         plan,
	                                           std::vector<JpegDctImageCropRequest> requests) noexcept
	: plan_(std::move(plan)), requests_(std::move(requests)) {
}

bool DirectDctPreparedBatch::empty() const noexcept {
	return plan_.empty();
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

JpegDctReaderInitializationStats DirectDctRuntime::InitializationStats() const noexcept {
	return reader_.InitializationStats();
}

JpegImageMetadata DirectDctRuntime::ImageMetadata(const uint32_t global_image_index) const {
	return reader_.ImageMetadata(global_image_index);
}

uint64_t DirectDctRuntime::RowgroupStorageBytes(
    const uint32_t shard_id, const std::vector<uint32_t>& rowgroup_indices) const {
	return reader_.RowgroupStorageBytes(shard_id, rowgroup_indices);
}

DirectDctBatch DirectDctRuntime::ReadBatch(const std::vector<JpegDctImageCropRequest>& requests,
                                           const JpegDctDeviceBatchOptions&            options) {
	return ReadPreparedBatch(PrepareBatch(requests, options));
}

DirectDctPreparedBatch DirectDctRuntime::PrepareBatch(
	const std::vector<JpegDctImageCropRequest>& requests, const JpegDctDeviceBatchOptions& options) {
	return DirectDctPreparedBatch(reader_.PrepareDeviceDctBatch(requests, options), requests);
}

void DirectDctRuntime::StageBatchIo(DirectDctPreparedBatch& prepared) {
	if (prepared.empty()) {
		throw std::invalid_argument("DirectDct prepared batch is empty");
	}
	reader_.StagePreparedDeviceDctBatchIo(prepared.plan_);
}

DirectDctBatch DirectDctRuntime::ReadPreparedBatch(DirectDctPreparedBatch prepared) {
	if (prepared.empty()) {
		throw std::invalid_argument("DirectDct prepared batch is empty");
	}
	StageBatchIo(prepared);
	auto       ids      = request_global_image_ids(prepared.requests_);
	auto       requests = std::move(prepared.requests_);
	auto       batch    = reader_.ReadPreparedDeviceDctBatch(std::move(prepared.plan_));
	const auto device = batch.cuda_device();
	return DirectDctBatch(std::move(batch), std::move(ids), std::move(requests), device);
}

DirectDctBatch DirectDctRuntime::ReadBatch(const std::vector<uint32_t>&     global_image_ids,
                                           const JpegDctCropBox&            crop,
                                           const JpegDctDeviceBatchOptions& options) {
	auto       requests = make_crop_requests(global_image_ids, crop);
	auto       batch    = reader_.ReadDeviceDctBatch(requests, options);
	auto       ids      = std::vector<uint32_t>(global_image_ids.begin(), global_image_ids.end());
	const auto device   = batch.cuda_device();
	return DirectDctBatch(std::move(batch), std::move(ids), std::move(requests), device);
}

JpegDctDeviceBatchPlanPreview DirectDctRuntime::PlanBatch(const std::vector<JpegDctImageCropRequest>& requests,
                                                          const JpegDctDeviceBatchOptions&            options) const {
	return reader_.PlanDeviceDctBatch(requests, options);
}

} // namespace galp::jpeg

#endif // GALP_WITH_JPEG_DCT
