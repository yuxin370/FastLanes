#ifndef GALP_ADVANCED_DIRECT_DCT_HPP
#define GALP_ADVANCED_DIRECT_DCT_HPP

#define GALP_DIRECT_DCT_ADVANCED_API 1

#include "galp/config.hpp"

#if GALP_WITH_JPEG_DCT

#include "galp/jpeg_dct_diagnostics.hpp"
#include "galp/jpeg_dct_storage.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace galp::direct_dct {
class NativeLogicalBatchPipeline;
}

namespace galp::jpeg {

enum class DirectDctTensorDataType {
	kInt16,
	kFloat32,
};

enum class DirectDctTensorDevice {
	kCuda,
};

struct DirectDctTensorDescriptor {
	const int16_t*          data        = nullptr;
	std::array<size_t, 2>   shape       = {0, 0};
	std::array<size_t, 2>   strides     = {0, 1};
	DirectDctTensorDataType dtype       = DirectDctTensorDataType::kInt16;
	DirectDctTensorDevice   device      = DirectDctTensorDevice::kCuda;
	int                     cuda_device = -1;

	[[nodiscard]] size_t rows() const noexcept {
		return shape[0];
	}

	[[nodiscard]] size_t columns() const noexcept {
		return shape[1];
	}

	[[nodiscard]] bool empty() const noexcept {
		return rows() == 0 || columns() == 0;
	}
};

struct DirectDctGridTensorDescriptor {
	const int16_t*          data        = nullptr;
	const float*            float_data  = nullptr;
	std::array<size_t, 6>   shape       = {0, 0, 0, 0, 0, 0};
	std::array<size_t, 6>   strides     = {0, 0, 0, 0, 0, 1};
	DirectDctTensorDataType dtype       = DirectDctTensorDataType::kInt16;
	DirectDctTensorDevice   device      = DirectDctTensorDevice::kCuda;
	int                     cuda_device = -1;

	[[nodiscard]] size_t element_count() const noexcept {
		size_t count = 1;
		for (const auto dim : shape) {
			count *= dim;
		}
		return count;
	}

	[[nodiscard]] bool empty() const noexcept {
		return element_count() == 0;
	}

	[[nodiscard]] const void* raw_data() const noexcept {
		return dtype == DirectDctTensorDataType::kFloat32 ? static_cast<const void*>(float_data)
		                                                    : static_cast<const void*>(data);
	}
};

class DirectDctBatch {
public:
	DirectDctBatch() noexcept;
	~DirectDctBatch();

	DirectDctBatch(const DirectDctBatch&)            = delete;
	DirectDctBatch& operator=(const DirectDctBatch&) = delete;
	DirectDctBatch(DirectDctBatch&&) noexcept;
	DirectDctBatch& operator=(DirectDctBatch&&) noexcept;

	[[nodiscard]] const int16_t*            device_data() const;
	[[nodiscard]] const int16_t*            y_device_data() const;
	[[nodiscard]] const int16_t*            cbcr_device_data() const;
	[[nodiscard]] const int16_t*            device_data_async() const noexcept;
	[[nodiscard]] const int16_t*            y_device_data_async() const noexcept;
	[[nodiscard]] const int16_t*            cbcr_device_data_async() const noexcept;
	[[nodiscard]] const float*              y_float_device_data() const;
	[[nodiscard]] const float*              cbcr_float_device_data() const;
	[[nodiscard]] const float*              y_float_device_data_async() const noexcept;
	[[nodiscard]] const float*              cbcr_float_device_data_async() const noexcept;
	void                                    synchronize() const;
	[[nodiscard]] DirectDctTensorDescriptor tensor() const;
	[[nodiscard]] DirectDctGridTensorDescriptor y_tensor() const;
	[[nodiscard]] DirectDctGridTensorDescriptor cbcr_tensor() const;
	[[nodiscard]] DirectDctTensorDescriptor tensor_async() const;
	[[nodiscard]] DirectDctGridTensorDescriptor y_tensor_async() const;
	[[nodiscard]] DirectDctGridTensorDescriptor cbcr_tensor_async() const;
	[[nodiscard]] size_t                    block_count() const noexcept;
	[[nodiscard]] size_t                    coefficients_per_block() const noexcept;
	[[nodiscard]] size_t                    coefficient_count() const noexcept;
	[[nodiscard]] size_t                    coefficient_bytes() const noexcept;
	[[nodiscard]] size_t                    y_coefficient_count() const noexcept;
	[[nodiscard]] size_t                    cbcr_coefficient_count() const noexcept;
	[[nodiscard]] size_t                    image_count() const noexcept;
	[[nodiscard]] int                       cuda_device() const noexcept;
	[[nodiscard]] void*                     cuda_completion_event() const noexcept;

	[[nodiscard]] const std::vector<uint32_t>&                      global_image_ids() const noexcept;
	[[nodiscard]] const std::vector<JpegDctImageCropRequest>&       transform_requests() const noexcept;
	[[nodiscard]] const std::vector<JpegDctDeviceImageLayout>&      image_layouts() const noexcept;
	[[nodiscard]] const std::vector<JpegDctDeviceBlockMetadata>&    block_metadata() const noexcept;
	[[nodiscard]] const std::vector<JpegDctDeviceRowgroupMetadata>& rowgroups() const noexcept;
	[[nodiscard]] const std::vector<uint8_t>&                       selected_coefficients() const noexcept;
	[[nodiscard]] JpegDctDeviceCacheStats                           cache_stats() const noexcept;
	[[nodiscard]] JpegDctDeviceExecutionStats                       execution_stats() const;
	[[nodiscard]] bool                                              try_finalize_execution_stats() const;
	[[nodiscard]] const JpegDctDeviceCacheStats&                    cache_stats_ref() const noexcept;
	[[nodiscard]] const JpegDctDeviceExecutionStats&                execution_stats_ref() const noexcept;
	[[nodiscard]] const JpegDctDeviceBatch&                         device_batch() const noexcept;

private:
	friend class DirectDctRuntime;
	friend class galp::direct_dct::NativeLogicalBatchPipeline;

	struct LogicalSegmentInput final {
		std::shared_ptr<DirectDctBatch> source;
		size_t                          source_image_offset  = 0U;
		size_t                          image_count          = 0U;
		size_t                          logical_image_offset = 0U;
	};
	struct LogicalImpl;

	static DirectDctBatch MakeLogicalGridBatch(
	    std::vector<LogicalSegmentInput> segments,
	    std::vector<uint32_t> global_image_ids,
	    std::vector<JpegDctImageCropRequest> transform_requests,
	    std::shared_ptr<DirectDctBatch> stats_source);

	DirectDctBatch(JpegDctDeviceBatch batch,
	               std::vector<uint32_t> global_image_ids,
	               std::vector<JpegDctImageCropRequest> transform_requests,
	               int cuda_device) noexcept;

	// Declared before the backing objects so reverse member destruction returns
	// the admission permit only after device storage has actually been released.
	std::shared_ptr<void>   materialized_output_slot_owner_;
	JpegDctDeviceBatch    batch_;
	std::vector<uint32_t> global_image_ids_;
	std::vector<JpegDctImageCropRequest> transform_requests_;
	int                   cuda_device_ = -1;
	std::unique_ptr<LogicalImpl> logical_;
};

// CPU-only result of compiling request geometry into storage and GPU transform
// work. Instances are reader-specific and may be prepared concurrently, then
// submitted in consumer order through DirectDctRuntime::ReadPreparedBatch.
class DirectDctPreparedBatch {
public:
	DirectDctPreparedBatch() noexcept;
	~DirectDctPreparedBatch();
	DirectDctPreparedBatch(const DirectDctPreparedBatch&)            = delete;
	DirectDctPreparedBatch& operator=(const DirectDctPreparedBatch&) = delete;
	DirectDctPreparedBatch(DirectDctPreparedBatch&&) noexcept;
	DirectDctPreparedBatch& operator=(DirectDctPreparedBatch&&) noexcept;
	[[nodiscard]] bool empty() const noexcept;

private:
	friend class DirectDctRuntime;
	DirectDctPreparedBatch(JpegDctDeviceBatchPreparedPlan         plan,
	                       std::vector<JpegDctImageCropRequest> requests) noexcept;
	JpegDctDeviceBatchPreparedPlan         plan_;
	std::vector<JpegDctImageCropRequest> requests_;
};

// Inherits JpegDctShardDatasetReader's concurrency contract: CPU planning may
// overlap, while shared device resource binding/submission is serialized per
// runtime. Returned batches remain asynchronous.
class DirectDctRuntime {
public:
	explicit DirectDctRuntime(const std::filesystem::path& manifest_path);
	~DirectDctRuntime();

	DirectDctRuntime(const DirectDctRuntime&)            = delete;
	DirectDctRuntime& operator=(const DirectDctRuntime&) = delete;
	DirectDctRuntime(DirectDctRuntime&&) noexcept;
	DirectDctRuntime& operator=(DirectDctRuntime&&) noexcept;

	[[nodiscard]] uint64_t          image_count() const noexcept;
	[[nodiscard]] JpegDctReaderInitializationStats InitializationStats() const noexcept;
	[[nodiscard]] JpegImageMetadata ImageMetadata(uint32_t global_image_index) const;
	[[nodiscard]] uint64_t RowgroupStorageBytes(uint32_t                     shard_id,
	                                            const std::vector<uint32_t>& rowgroup_indices) const;
	DirectDctBatch ReadBatch(const std::vector<JpegDctImageCropRequest>& requests,
	                         const JpegDctDeviceBatchOptions&            options = {});
	DirectDctPreparedBatch PrepareBatch(const std::vector<JpegDctImageCropRequest>& requests,
	                                    const JpegDctDeviceBatchOptions&            options = {});
	void StageBatchIo(DirectDctPreparedBatch& prepared);
	DirectDctBatch ReadPreparedBatch(DirectDctPreparedBatch prepared);

	DirectDctBatch ReadBatch(const std::vector<uint32_t>&     global_image_ids,
	                         const JpegDctCropBox&            crop    = {},
	                         const JpegDctDeviceBatchOptions& options = {});

	[[nodiscard]] JpegDctDeviceBatchPlanPreview PlanBatch(const std::vector<JpegDctImageCropRequest>& requests,
	                                                      const JpegDctDeviceBatchOptions& options = {}) const;

private:
	JpegDctShardDatasetReader reader_;
};

} // namespace galp::jpeg

#endif // GALP_WITH_JPEG_DCT

#endif // GALP_ADVANCED_DIRECT_DCT_HPP
