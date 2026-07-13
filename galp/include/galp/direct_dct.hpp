#ifndef GALP_DIRECT_DCT_HPP
#define GALP_DIRECT_DCT_HPP

#include "galp/config.hpp"

#if GALP_WITH_JPEG_DCT

#include "galp/jpeg_dct.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace galp::jpeg {

enum class DirectDctTensorDataType {
	kInt16,
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
};

class DirectDctBatch {
public:
	DirectDctBatch() noexcept;
	~DirectDctBatch();

	DirectDctBatch(const DirectDctBatch&)            = delete;
	DirectDctBatch& operator=(const DirectDctBatch&) = delete;
	DirectDctBatch(DirectDctBatch&&) noexcept;
	DirectDctBatch& operator=(DirectDctBatch&&) noexcept;

	[[nodiscard]] const int16_t*            device_data() const noexcept;
	[[nodiscard]] const int16_t*            y_device_data() const noexcept;
	[[nodiscard]] const int16_t*            cbcr_device_data() const noexcept;
	[[nodiscard]] const int16_t*            device_data_async() const noexcept;
	[[nodiscard]] const int16_t*            y_device_data_async() const noexcept;
	[[nodiscard]] const int16_t*            cbcr_device_data_async() const noexcept;
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
	[[nodiscard]] const std::vector<JpegDctDeviceImageLayout>&      image_layouts() const noexcept;
	[[nodiscard]] const std::vector<JpegDctDeviceBlockMetadata>&    block_metadata() const noexcept;
	[[nodiscard]] const std::vector<JpegDctDeviceRowgroupMetadata>& rowgroups() const noexcept;
	[[nodiscard]] const std::vector<uint8_t>&                       selected_coefficients() const noexcept;
	[[nodiscard]] JpegDctDeviceCacheStats                           cache_stats() const noexcept;
	[[nodiscard]] JpegDctDeviceExecutionStats                       execution_stats() const noexcept;
	[[nodiscard]] const JpegDctDeviceCacheStats&                    cache_stats_ref() const noexcept;
	[[nodiscard]] const JpegDctDeviceExecutionStats&                execution_stats_ref() const noexcept;
	[[nodiscard]] const JpegDctDeviceBatch&                         device_batch() const noexcept;

private:
	friend class DirectDctRuntime;

	DirectDctBatch(JpegDctDeviceBatch batch, std::vector<uint32_t> global_image_ids, int cuda_device) noexcept;

	JpegDctDeviceBatch    batch_;
	std::vector<uint32_t> global_image_ids_;
	int                   cuda_device_ = -1;
};

class DirectDctRuntime {
public:
	explicit DirectDctRuntime(const std::filesystem::path& manifest_path);
	~DirectDctRuntime();

	DirectDctRuntime(const DirectDctRuntime&)            = delete;
	DirectDctRuntime& operator=(const DirectDctRuntime&) = delete;
	DirectDctRuntime(DirectDctRuntime&&) noexcept;
	DirectDctRuntime& operator=(DirectDctRuntime&&) noexcept;

	[[nodiscard]] uint64_t          image_count() const noexcept;
	[[nodiscard]] JpegImageMetadata ImageMetadata(uint32_t global_image_index) const;

	DirectDctBatch ReadBatch(const std::vector<JpegDctImageCropRequest>& requests,
	                         const JpegDctDeviceBatchOptions&            options = {});

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

#endif // GALP_DIRECT_DCT_HPP
