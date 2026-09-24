#include "api/direct_dct_pls_postprocess.hpp"
#include "cuda/memory/cuda_raii.cuh"
#include "cuda/memory/device_pool.cuh"
#include "cuda/memory/gpu_array.cuh"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace galp::jpeg::detail {
namespace {

void check_cuda(const cudaError_t status, const char* operation) {
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string("Direct-DCT PLS CUDA ") + operation +
		                         " failed: " + cudaGetErrorString(status));
	}
}

__device__ int16_t clamp_round(const float value) {
	return static_cast<int16_t>(nearbyintf(fminf(1016.0F, fmaxf(-1024.0F, value))));
}

template <typename Source>
__global__ void convert_to_int16_kernel(const Source* source, int16_t* output, const size_t count) {
	for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
	     index += static_cast<size_t>(blockDim.x) * gridDim.x) {
		if constexpr (std::is_same_v<Source, float>) {
			output[index] = clamp_round(source[index] * 1020.0F - 4.0F);
		} else {
			const auto value = static_cast<int>(source[index]);
			output[index]    = static_cast<int16_t>(value < -1024 ? -1024 : (value > 1016 ? 1016 : value));
		}
	}
}

struct DeviceStats {
	float y_min;
	float y_max;
	float y_mean_abs;
	float c_min;
	float c_max;
};

__global__ void compute_stats_kernel(const int16_t* y, const int16_t* cbcr, DeviceStats* stats, const size_t images) {
	for (size_t image = blockIdx.x * blockDim.x + threadIdx.x; image < images;
	     image += static_cast<size_t>(blockDim.x) * gridDim.x) {
		constexpr size_t y_dc_count = 28U * 28U;
		constexpr size_t c_dc_count = 2U * 14U * 14U;
		constexpr size_t y_elements = y_dc_count * 64U;
		constexpr size_t c_elements = c_dc_count * 64U;
		float            y_min      = 1.0e30F;
		float            y_max      = -1.0e30F;
		float            y_abs      = 0.0F;
		for (size_t dc = 0U; dc < y_dc_count; ++dc) {
			const auto value = static_cast<float>(y[image * y_elements + dc * 64U]);
			y_min            = fminf(y_min, value);
			y_max            = fmaxf(y_max, value);
			y_abs += fabsf(value);
		}
		float c_min = 1.0e30F;
		float c_max = -1.0e30F;
		for (size_t dc = 0U; dc < c_dc_count; ++dc) {
			const auto value = static_cast<float>(cbcr[image * c_elements + dc * 64U]);
			c_min            = fminf(c_min, value);
			c_max            = fmaxf(c_max, value);
		}
		stats[image] = {y_min, y_max, y_abs / static_cast<float>(y_dc_count), c_min, c_max};
	}
}

__device__ float autocontrast(const float value, const float minimum, const float maximum) {
	if (minimum == maximum) {
		return value;
	}
	return -1024.0F + ((value - minimum) / (maximum - minimum)) * 2040.0F;
}

__device__ bool cutout_contains(const int coordinate_h,
                                const int coordinate_w,
                                const int height,
                                const int width,
                                const int center_h,
                                const int center_w,
                                const int pad) {
	const auto lower = max(0, center_h - pad);
	const auto upper = max(0, height - center_h - pad);
	const auto left  = max(0, center_w - pad);
	const auto right = max(0, width - center_w - pad);
	return coordinate_h >= upper && coordinate_h < height - lower && coordinate_w >= left &&
	       coordinate_w < width - right;
}

__device__ size_t component_offset(
    const int channel, const int h, const int w, const int u, const int v, const int height, const int width) {
	return (((static_cast<size_t>(channel) * height + h) * width + w) * 8U + u) * 8U + v;
}

struct ProjectedChannel {
	int   component;
	int   frequency;
	int   source;
	int   transpose;
	float subtract;
	float divide;
};

template <bool Projected = false>
__global__ void apply_randaugment_kernel(const int16_t*                         source,
                                         int16_t*                               output,
                                         const DirectDctPlsRandAugmentDecision* decisions,
                                         const DeviceStats*                     stats,
                                         const size_t                           images,
                                         const int                              stage,
                                         const int                              channels,
                                         const int                              height,
                                         const int                              width,
                                         const bool                             component_is_luma,
                                         const ProjectedChannel*                channel_info = nullptr) {
	const auto elements_per_image = static_cast<size_t>(channels) * height * width * (Projected ? 1U : 64U);
	const auto total              = images * elements_per_image;
	for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
	     index += static_cast<size_t>(blockDim.x) * gridDim.x) {
		const auto image = index / elements_per_image;
		auto       local = index - image * elements_per_image;
		int        u, v, w, h, channel;
		if constexpr (Projected) {
			w = local % width;
			local /= width;
			h       = local % height;
			channel = local / height;
			u       = channel_info[channel].frequency / 8;
			v       = channel_info[channel].frequency % 8;
		} else {
			v = local % 8U;
			local /= 8U;
			u = local % 8U;
			local /= 8U;
			w = local % width;
			local /= width;
			h       = local % height;
			channel = local / height;
		}
		const bool  luma      = Projected ? channel_info[channel].component == 0 : component_is_luma;
		const bool  full_grid = Projected || luma;
		const auto& decision  = decisions[image];
		const auto  operation = decision.operations[stage];
		const auto  magnitude = decision.magnitudes[stage];
		auto        source_h  = h;
		auto        source_w  = w;
		auto        source_u  = u;
		auto        source_v  = v;
		float       sign      = 1.0F;
		bool        zero      = false;

		if (operation == DirectDctPlsRandAugmentOp::kTranslateX ||
		    operation == DirectDctPlsRandAugmentOp::kTranslateY) {
			auto blocks = static_cast<int>(floorf(magnitude / 2.0F)) * 2;
			if constexpr (Projected)
				blocks = blocks * width / 28;
			if (!full_grid) {
				blocks /= 2;
			}
			if (operation == DirectDctPlsRandAugmentOp::kTranslateX) {
				zero     = blocks >= 0 ? w < blocks : w >= width + blocks;
				source_w = (w - blocks) % width;
				if (source_w < 0)
					source_w += width;
			} else {
				zero     = blocks >= 0 ? h < blocks : h >= height + blocks;
				source_h = (h - blocks) % height;
				if (source_h < 0)
					source_h += height;
			}
		} else if (operation == DirectDctPlsRandAugmentOp::kRotate90) {
			if (magnitude > 0.0F) {
				source_h = w;
				source_w = width - 1 - h;
				source_u = v;
				source_v = u;
				sign     = (u & 1) != 0 ? -1.0F : 1.0F;
			} else {
				source_h = height - 1 - w;
				source_w = h;
				source_u = v;
				source_v = u;
				sign     = (v & 1) != 0 ? -1.0F : 1.0F;
			}
		}

		size_t source_local;
		if constexpr (Projected) {
			const auto source_channel =
			    operation == DirectDctPlsRandAugmentOp::kRotate90 ? channel_info[channel].transpose : channel;
			source_local = (static_cast<size_t>(source_channel) * height + source_h) * width + source_w;
		} else {
			source_local = component_offset(channel, source_h, source_w, source_u, source_v, height, width);
		}
		float      value = zero ? 0.0F : static_cast<float>(source[image * elements_per_image + source_local]) * sign;
		const auto dc    = u == 0 && v == 0;
		if (operation == DirectDctPlsRandAugmentOp::kAutoContrast && luma && dc) {
			value = autocontrast(value, stats[image].y_min, stats[image].y_max);
		} else if (operation == DirectDctPlsRandAugmentOp::kPosterize && dc) {
			value = -1024.0F + 4.0F * nearbyintf((value + 1024.0F) / 4.0F);
		} else if (operation == DirectDctPlsRandAugmentOp::kSolarizeAdd && luma && dc && value < 0.0F) {
			value += static_cast<float>(static_cast<int>(magnitude));
		} else if (operation == DirectDctPlsRandAugmentOp::kColor && !luma && dc) {
			value *= 1.0F + magnitude;
		} else if (operation == DirectDctPlsRandAugmentOp::kContrast && luma && dc) {
			value *= 1.0F + magnitude;
		} else if (operation == DirectDctPlsRandAugmentOp::kBrightness && luma && dc) {
			value += stats[image].y_mean_abs * magnitude;
		} else if (operation == DirectDctPlsRandAugmentOp::kMidfreqAug && luma) {
			const auto sigma     = 4.0F - 2.2F * fabsf(magnitude);
			const auto shifted_u = static_cast<float>((u + 4) % 8) - 3.5F;
			const auto shifted_v = static_cast<float>((v + 4) % 8) - 3.5F;
			auto       factor    = expf(-0.5F * (shifted_u * shifted_u + shifted_v * shifted_v) / (sigma * sigma));
			if (magnitude >= 0.0F)
				factor = 1.0F / factor;
			value *= factor;
		} else if (operation == DirectDctPlsRandAugmentOp::kCutout) {
			const auto pad           = static_cast<int>(nearbyintf(magnitude)) & ~1;
			const auto effective_pad = Projected ? pad * height / 28 : (luma ? pad : pad / 2);
			const auto center_h      = Projected
			                               ? decision.cutout_center_h[stage] * height / 28
			                               : (luma ? decision.cutout_center_h[stage] : decision.cutout_center_h[stage] / 2);
			const auto center_w      = Projected
			                               ? decision.cutout_center_w[stage] * width / 28
			                               : (luma ? decision.cutout_center_w[stage] : decision.cutout_center_w[stage] / 2);
			if (cutout_contains(h, w, height, width, center_h, center_w, effective_pad))
				value = 0.0F;
		} else if (operation == DirectDctPlsRandAugmentOp::kAutoSaturation && !luma && dc) {
			value = autocontrast(value, stats[image].c_min, stats[image].c_max);
		} else if (operation == DirectDctPlsRandAugmentOp::kGrayscale && !luma) {
			value = 0.0F;
		} else if (operation == DirectDctPlsRandAugmentOp::kChromaDrop && !luma &&
		           (Projected ? channel_info[channel].component - 1 : channel) == decision.chroma_drop_channel[stage]) {
			value = 0.0F;
		}
		output[index] = clamp_round(value);
	}
}

__global__ void normalize_mixup_kernel(const int16_t*                   source,
                                       float*                           output,
                                       const DirectDctPlsMixupDecision* mixup,
                                       const size_t                     images,
                                       const size_t                     elements_per_image,
                                       const uint32_t                   microbatch_images,
                                       const bool                       enabled) {
	const auto total = images * elements_per_image;
	for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
	     index += static_cast<size_t>(blockDim.x) * gridDim.x) {
		const auto image          = index / elements_per_image;
		const auto element        = index - image * elements_per_image;
		const auto microbatch     = image / microbatch_images;
		const auto start          = microbatch * microbatch_images;
		const auto count          = min(static_cast<size_t>(microbatch_images), images - start);
		const auto partner        = image == start ? start + count - 1U : image - 1U;
		const auto original_value = (static_cast<float>(source[index]) + 4.0F) / 1020.0F;
		if (!enabled) {
			output[index] = original_value;
		} else {
			const auto rolled_value =
			    (static_cast<float>(source[partner * elements_per_image + element]) + 4.0F) / 1020.0F;
			output[index] = original_value * mixup[microbatch].original + rolled_value * mixup[microbatch].rolled;
		}
	}
}

__global__ void projected_to_int16(const float* source, int16_t* target, size_t count) {
	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += size_t(blockDim.x) * gridDim.x)
		target[i] = clamp_round(source[i]);
}

__global__ void projected_stats(const int16_t* source,
                                DeviceStats*   stats,
                                size_t         images,
                                size_t         pixels,
                                size_t         channels,
                                int            y_dc,
                                int            cb_dc,
                                int            cr_dc) {
	__shared__ DeviceStats partial[256];
	for (size_t image = blockIdx.x; image < images; image += gridDim.x) {
		DeviceStats value {1.e30F, -1.e30F, 0.F, 1.e30F, -1.e30F};
		const auto* data = source + image * pixels * channels;
		for (size_t p = threadIdx.x; p < pixels; p += blockDim.x) {
			float y = data[y_dc * pixels + p], cb = data[cb_dc * pixels + p], cr = data[cr_dc * pixels + p];
			value.y_min = fminf(value.y_min, y);
			value.y_max = fmaxf(value.y_max, y);
			value.y_mean_abs += fabsf(y);
			value.c_min = fminf(value.c_min, fminf(cb, cr));
			value.c_max = fmaxf(value.c_max, fmaxf(cb, cr));
		}
		partial[threadIdx.x] = value;
		__syncthreads();
		for (unsigned stride = blockDim.x / 2; stride != 0; stride /= 2) {
			if (threadIdx.x < stride) {
				auto&      left  = partial[threadIdx.x];
				const auto right = partial[threadIdx.x + stride];
				left.y_min       = fminf(left.y_min, right.y_min);
				left.y_max       = fmaxf(left.y_max, right.y_max);
				left.y_mean_abs += right.y_mean_abs;
				left.c_min = fminf(left.c_min, right.c_min);
				left.c_max = fmaxf(left.c_max, right.c_max);
			}
			__syncthreads();
		}
		if (threadIdx.x == 0) {
			value = partial[0];
			// Clamped integer magnitudes sum exactly in FP32 through 16384 pixels.
			// Larger supported grids retain the original sequential rounding order.
			if (pixels > 16384) {
				value.y_mean_abs = 0.F;
				for (size_t p = 0; p < pixels; ++p)
					value.y_mean_abs += fabsf(float(data[y_dc * pixels + p]));
			}
			value.y_mean_abs /= pixels;
			stats[image] = value;
		}
		__syncthreads();
	}
}

__global__ void projected_normalize_mixup(const int16_t*                   source,
                                          float*                           target,
                                          size_t                           images,
                                          size_t                           pixels,
                                          size_t                           input_channels,
                                          size_t                           output_channels,
                                          const ProjectedChannel*          channels,
                                          const DirectDctPlsMixupDecision* mixup,
                                          bool                             enabled) {
	const auto per_image = pixels * output_channels;
	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < images * per_image;
	     i += size_t(blockDim.x) * gridDim.x) {
		const auto  image = i / per_image, local = i % per_image, c = local / pixels, p = local % pixels;
		const auto  channel = channels[c];
		const float value =
		    (float(source[(image * input_channels + channel.source) * pixels + p]) - channel.subtract) / channel.divide;
		const auto  partner = image == 0 ? images - 1 : image - 1;
		const float rolled =
		    (float(source[(partner * input_channels + channel.source) * pixels + p]) - channel.subtract) /
		    channel.divide;
		target[i] = enabled ? value * mixup->original + rolled * mixup->rolled : value;
	}
}

__global__ void mixup_targets_kernel(const int64_t*                   labels,
                                     float*                           targets,
                                     const DirectDctPlsMixupDecision* mixup,
                                     const size_t                     images,
                                     const uint32_t                   classes,
                                     const uint32_t                   microbatch_images,
                                     const bool                       enabled) {
	const auto total = images * static_cast<size_t>(classes);
	for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
	     index += static_cast<size_t>(blockDim.x) * gridDim.x) {
		const auto image      = index / classes;
		const auto category   = static_cast<uint32_t>(index - image * classes);
		const auto microbatch = image / microbatch_images;
		const auto start      = microbatch * microbatch_images;
		const auto count      = min(static_cast<size_t>(microbatch_images), images - start);
		const auto partner    = image == start ? start + count - 1U : image - 1U;
		const auto original   = labels[image] == category ? 1.0F : 0.0F;
		if (!enabled) {
			targets[index] = original;
		} else {
			const auto rolled = labels[partner] == category ? 1.0F : 0.0F;
			targets[index]    = original * mixup[microbatch].original + rolled * mixup[microbatch].rolled;
		}
	}
}

void validate_metadata(std::span<const int64_t>                         labels,
                       std::span<const DirectDctPlsRandAugmentDecision> randaugment,
                       std::span<const DirectDctPlsMixupDecision>       mixup,
                       uint32_t                                         microbatch_images,
                       uint32_t                                         model_classes) {
	if (labels.empty() || randaugment.size() != labels.size() || microbatch_images == 0U || model_classes == 0U) {
		throw std::invalid_argument("invalid Direct-DCT PLS CUDA postprocess cardinality");
	}
	for (const auto label : labels) {
		if (label < 0 || static_cast<uint64_t>(label) >= model_classes) {
			throw std::invalid_argument("Direct-DCT PLS label is outside the configured model class range");
		}
	}
	const auto expected_mixup = (labels.size() + microbatch_images - 1U) / microbatch_images;
	if (mixup.size() != expected_mixup) {
		throw std::invalid_argument("Direct-DCT PLS mixup decision count does not match microbatches");
	}
}

size_t launch_blocks(const size_t count) {
	return std::min<size_t>(65535U, (count + 255U) / 256U);
}

} // namespace

struct DirectDctPlsCudaPostprocess::Stream::Impl {
	int                      device = -1;
	galp::memory::CudaStream stream;

	explicit Impl(const int cuda_device)
	    : device(cuda_device) {
		check_cuda(cudaSetDevice(device), "select postprocess stream device");
		stream.create(cudaStreamNonBlocking);
	}

	~Impl() {
		if (device >= 0) {
			(void)cudaSetDevice(device);
		}
	}
};

DirectDctPlsCudaPostprocess::Stream::Stream(const int cuda_device)
	: impl_(std::make_unique<Impl>(cuda_device)) {
}
DirectDctPlsCudaPostprocess::Stream::~Stream() = default;

struct DirectDctPlsCudaPostprocess::Impl {
	int                                      device = -1;
	std::shared_ptr<Stream>                  stream;
	galp::memory::CudaEvent                  completion;
	std::optional<GPUArray<int16_t>>         y_a;
	std::optional<GPUArray<int16_t>>         y_b;
	std::optional<GPUArray<int16_t>>         c_a;
	std::optional<GPUArray<int16_t>>         c_b;
	std::optional<GPUArray<float>>           y_output;
	std::optional<GPUArray<float>>           c_output;
	std::optional<GPUArray<float>>           targets_output;
	float*                                  external_targets = nullptr;
	std::optional<GPUArray<int64_t>>         labels_device;
	std::optional<GPUArray<DirectDctPlsRandAugmentDecision>> decisions_device;
	std::optional<GPUArray<DirectDctPlsMixupDecision>>       mixup_device;
	std::optional<GPUArray<DeviceStats>>                     stats_device;
	size_t                                                   images  = 0U;
	uint32_t                                                 classes = 0U;
	DirectDctGridTensorDescriptor                            y_descriptor;
	DirectDctGridTensorDescriptor                            c_descriptor;
	DirectDctGridTensorDescriptor                            projected_descriptor;
	std::optional<GPUArray<ProjectedChannel>>                input_channels_device;
	std::optional<GPUArray<ProjectedChannel>>                output_channels_device;

	void project(DirectDctGridTensorDescriptor                    input,
	             std::span<const int64_t>                         labels,
	             std::span<const DirectDctPlsRandAugmentDecision> randaugment,
	             std::span<const DirectDctPlsMixupDecision>       mixup,
	             uint32_t                                         microbatch_images,
	             bool                                             enable_randaugment,
	             bool                                             enable_mixup,
	             std::span<const JpegDctOutputChannel>            inputs,
	             std::span<const JpegDctOutputChannel>            outputs,
	             cudaStream_t                                     stream);

	DirectDctPlsTargetTensorDescriptor target_descriptor;

	~Impl() {
		if (device >= 0) {
			(void)cudaSetDevice(device);
		}
	}
};

void DirectDctPlsCudaPostprocess::Impl::project(DirectDctGridTensorDescriptor                    input,
                                                std::span<const int64_t>                         labels,
                                                std::span<const DirectDctPlsRandAugmentDecision> randaugment,
                                                std::span<const DirectDctPlsMixupDecision>       mixup,
                                                uint32_t                                         microbatch_images,
                                                bool                                             enable_randaugment,
                                                bool                                             enable_mixup,
                                                std::span<const JpegDctOutputChannel>            inputs,
                                                std::span<const JpegDctOutputChannel>            outputs,
                                                cudaStream_t                                     stream) {
	const auto height = input.shape[2], width = input.shape[3], pixels = height * width;
	if (input.shape[0] != images || input.shape[1] != inputs.size() || outputs.size() > inputs.size() ||
	    height != width || height % 28 != 0)
		throw std::invalid_argument("projected PLS requires equal square grids with size a multiple of 28");
	int lookup[3][64];
	for (auto& row : lookup)
		std::fill(std::begin(row), std::end(row), -1);
	for (size_t i = 0; i < inputs.size(); ++i)
		lookup[inputs[i].component][inputs[i].frequency] = i;
	std::vector<ProjectedChannel> in, out;
	for (const auto c : inputs) {
		const int transpose = lookup[c.component][(c.frequency % 8) * 8 + c.frequency / 8];
		if (transpose < 0)
			throw std::invalid_argument("projected PLS lacks a rotation frequency dependency");
		in.push_back({c.component, c.frequency, lookup[c.component][c.frequency], transpose, c.subtract, c.divide});
	}
	for (int c = 0; c < 3; ++c)
		if (lookup[c][0] < 0)
			throw std::invalid_argument("projected PLS lacks DC statistics dependency");
	for (const auto c : outputs) {
		const int index = lookup[c.component][c.frequency];
		if (index < 0)
			throw std::invalid_argument("projected PLS lacks an output frequency");
		out.push_back({c.component, c.frequency, index, 0, c.subtract, c.divide});
	}
	const size_t capacity = std::min<size_t>(microbatch_images, images);
	// The source pool has a single producer and is not published yet. After loading a
	// whole microbatch into scratch, compact its normalized output in place. Since
	// output channels <= dependency channels, writes cannot touch the next unread
	// source microbatch. No second pool-sized float allocation is needed.
	auto* output = const_cast<float*>(input.float_data);
	y_a.emplace(capacity * inputs.size() * pixels, stream);
	y_b.emplace(capacity * inputs.size() * pixels, stream);
	if (!external_targets)
		targets_output.emplace(images * classes, stream);
	auto* targets = external_targets ? external_targets : targets_output->get();
	labels_device.emplace(images, stream);
	decisions_device.emplace(images, stream);
	mixup_device.emplace(mixup.size(), stream);
	stats_device.emplace(capacity, stream);
	input_channels_device.emplace(in.size(), stream);
	output_channels_device.emplace(out.size(), stream);
	check_cuda(
	    cudaMemcpyAsync(labels_device->get(), labels.data(), labels.size_bytes(), cudaMemcpyHostToDevice, stream),
	    "upload labels");
	check_cuda(
	    cudaMemcpyAsync(
	        decisions_device->get(), randaugment.data(), randaugment.size_bytes(), cudaMemcpyHostToDevice, stream),
	    "upload decisions");
	check_cuda(cudaMemcpyAsync(mixup_device->get(), mixup.data(), mixup.size_bytes(), cudaMemcpyHostToDevice, stream),
	           "upload mixup");
	check_cuda(cudaMemcpyAsync(input_channels_device->get(),
	                           in.data(),
	                           in.size() * sizeof(ProjectedChannel),
	                           cudaMemcpyHostToDevice,
	                           stream),
	           "upload dependency channels");
	check_cuda(cudaMemcpyAsync(output_channels_device->get(),
	                           out.data(),
	                           out.size() * sizeof(ProjectedChannel),
	                           cudaMemcpyHostToDevice,
	                           stream),
	           "upload output channels");
	for (size_t offset = 0; offset < images; offset += capacity) {
		const auto count = std::min(capacity, images - offset), elements = count * inputs.size() * pixels;
		projected_to_int16<<<launch_blocks(elements), 256, 0, stream>>>(
		    input.float_data + offset * inputs.size() * pixels, y_a->get(), elements);
		auto* current = y_a->get();
		auto* next    = y_b->get();
		if (enable_randaugment) {
			for (int stage = 0; stage < 2; ++stage) {
				projected_stats<<<count, 256, 0, stream>>>(current,
				                                           stats_device->get(),
				                                           count,
				                                           pixels,
				                                           inputs.size(),
				                                           lookup[0][0],
				                                           lookup[1][0],
				                                           lookup[2][0]);
				apply_randaugment_kernel<true>
				    <<<launch_blocks(elements), 256, 0, stream>>>(current,
				                                                  next,
				                                                  decisions_device->get() + offset,
				                                                  stats_device->get(),
				                                                  count,
				                                                  stage,
				                                                  inputs.size(),
				                                                  height,
				                                                  width,
				                                                  true,
				                                                  input_channels_device->get());
				std::swap(current, next);
			}
		}
		projected_normalize_mixup<<<launch_blocks(count * outputs.size() * pixels), 256, 0, stream>>>(
		    current,
		    output + offset * outputs.size() * pixels,
		    count,
		    pixels,
		    inputs.size(),
		    outputs.size(),
		    output_channels_device->get(),
		    mixup_device->get() + offset / microbatch_images,
		    enable_mixup);
	}
	mixup_targets_kernel<<<launch_blocks(images * classes), 256, 0, stream>>>(
	    labels_device->get(), targets, mixup_device->get(), images, classes, microbatch_images, enable_mixup);
	check_cuda(cudaGetLastError(), "launch projected augmentation and normalization");
	completion.record(stream);
	projected_descriptor            = input;
	projected_descriptor.float_data = output;
	projected_descriptor.shape[1]   = outputs.size();
	projected_descriptor.strides[0] = outputs.size() * pixels;
	target_descriptor               = {targets, {images, classes}, {classes, 1U}, device};
	// Host channel vectors must survive their asynchronous H2D copies only.
	check_cuda(cudaStreamSynchronize(stream), "finish projected postprocess");
}

DirectDctPlsCudaPostprocess::DirectDctPlsCudaPostprocess(
    DirectDctBatch&                                        source,
    const std::span<const int64_t>                         labels,
    const std::span<const DirectDctPlsRandAugmentDecision> randaugment,
    const std::span<const DirectDctPlsMixupDecision>       mixup,
    const uint32_t                                         microbatch_images,
    const uint32_t                                         model_classes,
    const bool                                             enable_randaugment,
    const bool                                             enable_mixup,
    std::shared_ptr<Stream>                                stream,
    std::span<const JpegDctOutputChannel>                  input_channels,
    std::span<const JpegDctOutputChannel>                  output_channels)
    : impl_(std::make_unique<Impl>()) {
	try {
		if (!stream || !stream->impl_) {
			throw std::invalid_argument("Direct-DCT PLS CUDA postprocess requires a shared stream");
		}
		validate_metadata(labels, randaugment, mixup, microbatch_images, model_classes);
		impl_->device  = source.cuda_device();
		impl_->images  = labels.size();
		impl_->classes = model_classes;
		check_cuda(cudaSetDevice(impl_->device), "select device");
		if (stream->impl_->device != impl_->device) {
			throw std::invalid_argument("Direct-DCT PLS postprocess stream device differs from the source batch");
		}
		impl_->stream = std::move(stream);
		impl_->completion.create_with_flags(cudaEventDisableTiming);
		const auto stream = impl_->stream->impl_->stream.get();
		if (source.cuda_completion_event() != nullptr) {
			check_cuda(cudaStreamWaitEvent(stream, static_cast<cudaEvent_t>(source.cuda_completion_event()), 0U),
			           "wait for Direct-DCT transform");
		}

		if (!output_channels.empty()) {
			impl_->project(source.projected_tensor_async(),
			               labels,
			               randaugment,
			               mixup,
			               microbatch_images,
			               enable_randaugment,
			               enable_mixup,
			               input_channels,
			               output_channels,
			               stream);
			return;
		}
		const auto y_source = source.y_tensor_async();
		const auto c_source = source.cbcr_tensor_async();
		if (y_source.shape != std::array<size_t, 6> {labels.size(), 1U, 28U, 28U, 8U, 8U} ||
		    c_source.shape != std::array<size_t, 6> {labels.size(), 2U, 14U, 14U, 8U, 8U}) {
			throw std::runtime_error("Direct-DCT PLS postprocess requires the registered 28/14 training grid");
		}

		constexpr size_t y_per_image = 1U * 28U * 28U * 8U * 8U;
		constexpr size_t c_per_image = 2U * 14U * 14U * 8U * 8U;
		const auto       y_count     = labels.size() * y_per_image;
		const auto       c_count     = labels.size() * c_per_image;
		impl_->y_a.emplace(y_count, stream);
		impl_->y_b.emplace(y_count, stream);
		impl_->c_a.emplace(c_count, stream);
		impl_->c_b.emplace(c_count, stream);
		impl_->y_output.emplace(y_count, stream);
		impl_->c_output.emplace(c_count, stream);
		impl_->targets_output.emplace(labels.size() * model_classes, stream);
		impl_->labels_device.emplace(labels.size(), stream);
		impl_->decisions_device.emplace(randaugment.size(), stream);
		impl_->mixup_device.emplace(mixup.size(), stream);
		impl_->stats_device.emplace(labels.size(), stream);
		check_cuda(cudaMemcpyAsync(impl_->labels_device->get(),
		                           labels.data(),
		                           labels.size_bytes(),
		                           cudaMemcpyHostToDevice,
		                           stream),
		           "upload labels");
		check_cuda(cudaMemcpyAsync(impl_->decisions_device->get(),
		                           randaugment.data(),
		                           randaugment.size_bytes(),
		                           cudaMemcpyHostToDevice,
		                           stream),
		           "upload RandAugment decisions");
		check_cuda(cudaMemcpyAsync(impl_->mixup_device->get(),
		                           mixup.data(),
		                           mixup.size_bytes(),
		                           cudaMemcpyHostToDevice,
		                           stream),
		           "upload Mixup decisions");

		if (y_source.dtype == DirectDctTensorDataType::kFloat32) {
			convert_to_int16_kernel<<<launch_blocks(y_count), 256, 0, stream>>>(
			    y_source.float_data, impl_->y_a->get(), y_count);
			convert_to_int16_kernel<<<launch_blocks(c_count), 256, 0, stream>>>(
			    c_source.float_data, impl_->c_a->get(), c_count);
		} else {
			convert_to_int16_kernel<<<launch_blocks(y_count), 256, 0, stream>>>(
			    y_source.data, impl_->y_a->get(), y_count);
			convert_to_int16_kernel<<<launch_blocks(c_count), 256, 0, stream>>>(
			    c_source.data, impl_->c_a->get(), c_count);
		}
		check_cuda(cudaGetLastError(), "launch input conversion");

		auto* y_current = impl_->y_a->get();
		auto* y_next    = impl_->y_b->get();
		auto* c_current = impl_->c_a->get();
		auto* c_next    = impl_->c_b->get();
		if (enable_randaugment) {
			for (int stage = 0; stage < 2; ++stage) {
				compute_stats_kernel<<<launch_blocks(labels.size()), 256, 0, stream>>>(
				    y_current, c_current, impl_->stats_device->get(), labels.size());
				apply_randaugment_kernel<false>
				    <<<launch_blocks(y_count), 256, 0, stream>>>(y_current,
				                                                 y_next,
				                                                 impl_->decisions_device->get(),
				                                                 impl_->stats_device->get(),
				                                                 labels.size(),
				                                                 stage,
				                                                 1,
				                                                 28,
				                                                 28,
				                                                 true);
				apply_randaugment_kernel<false>
				    <<<launch_blocks(c_count), 256, 0, stream>>>(c_current,
				                                                 c_next,
				                                                 impl_->decisions_device->get(),
				                                                 impl_->stats_device->get(),
				                                                 labels.size(),
				                                                 stage,
				                                                 2,
				                                                 14,
				                                                 14,
				                                                 false);
				std::swap(y_current, y_next);
				std::swap(c_current, c_next);
			}
			check_cuda(cudaGetLastError(), "launch RandAugment");
		}
		normalize_mixup_kernel<<<launch_blocks(y_count), 256, 0, stream>>>(y_current,
		                                                                          impl_->y_output->get(),
		                                                                          impl_->mixup_device->get(),
		                                                                          labels.size(),
		                                                                          y_per_image,
		                                                                          microbatch_images,
		                                                                          enable_mixup);
		normalize_mixup_kernel<<<launch_blocks(c_count), 256, 0, stream>>>(c_current,
		                                                                          impl_->c_output->get(),
		                                                                          impl_->mixup_device->get(),
		                                                                          labels.size(),
		                                                                          c_per_image,
		                                                                          microbatch_images,
		                                                                          enable_mixup);
		mixup_targets_kernel<<<launch_blocks(labels.size() * model_classes), 256, 0, stream>>>(
		    impl_->labels_device->get(),
		    impl_->targets_output->get(),
		    impl_->mixup_device->get(),
		    labels.size(),
		    model_classes,
		    microbatch_images,
		    enable_mixup);
		check_cuda(cudaGetLastError(), "launch normalization and Mixup");
		impl_->completion.record(stream);

		impl_->y_descriptor            = y_source;
		impl_->y_descriptor.data       = nullptr;
		impl_->y_descriptor.float_data = impl_->y_output->get();
		impl_->y_descriptor.dtype      = DirectDctTensorDataType::kFloat32;
		impl_->c_descriptor            = c_source;
		impl_->c_descriptor.data       = nullptr;
		impl_->c_descriptor.float_data = impl_->c_output->get();
		impl_->c_descriptor.dtype      = DirectDctTensorDataType::kFloat32;
		impl_->target_descriptor       = {
		    impl_->targets_output->get(),
            {labels.size(), model_classes},
            {model_classes, 1U},
            impl_->device,
        };
	} catch (...) {
		impl_.reset();
		throw;
	}
}

DirectDctPlsCudaPostprocess::DirectDctPlsCudaPostprocess(DirectDctGridTensorDescriptor source,
                                                         void*                         source_completion_event,
                                                         float*                        targets,
                                                         std::span<const int64_t>      labels,
                                                         std::span<const DirectDctPlsRandAugmentDecision> randaugment,
                                                         std::span<const DirectDctPlsMixupDecision>       mixup,
                                                         uint32_t                              microbatch_images,
                                                         uint32_t                              model_classes,
                                                         std::shared_ptr<Stream>               stream,
                                                         std::span<const JpegDctOutputChannel> input_channels,
                                                         std::span<const JpegDctOutputChannel> output_channels)
    : impl_(std::make_unique<Impl>()) {
	validate_metadata(labels, randaugment, mixup, microbatch_images, model_classes);
	if (!stream || !stream->impl_ || stream->impl_->device != source.cuda_device)
		throw std::invalid_argument("projected augmentation stream must match the input device");
	impl_->device           = source.cuda_device;
	impl_->images           = labels.size();
	impl_->classes          = model_classes;
	impl_->external_targets = targets;
	impl_->stream           = std::move(stream);
	check_cuda(cudaSetDevice(impl_->device), "select projected augmentation device");
	impl_->completion.create_with_flags(cudaEventDisableTiming);
	const auto execution_stream = impl_->stream->impl_->stream.get();
	check_cuda(cudaStreamWaitEvent(execution_stream, static_cast<cudaEvent_t>(source_completion_event), 0U),
	           "wait for projected input upload");
	impl_->project(source,
	               labels,
	               randaugment,
	               mixup,
	               microbatch_images,
	               true,
	               true,
	               input_channels,
	               output_channels,
	               execution_stream);
}

DirectDctPlsCudaPostprocess::~DirectDctPlsCudaPostprocess()                                                 = default;
DirectDctPlsCudaPostprocess::DirectDctPlsCudaPostprocess(DirectDctPlsCudaPostprocess&&) noexcept            = default;
DirectDctPlsCudaPostprocess& DirectDctPlsCudaPostprocess::operator=(DirectDctPlsCudaPostprocess&&) noexcept = default;

DirectDctGridTensorDescriptor DirectDctPlsCudaPostprocess::y_tensor() const noexcept {
	return impl_->y_descriptor;
}
DirectDctGridTensorDescriptor DirectDctPlsCudaPostprocess::cbcr_tensor() const noexcept {
	return impl_->c_descriptor;
}
DirectDctGridTensorDescriptor DirectDctPlsCudaPostprocess::projected_tensor() const noexcept {
	return impl_->projected_descriptor;
}
DirectDctPlsTargetTensorDescriptor DirectDctPlsCudaPostprocess::targets() const noexcept {
	return impl_->target_descriptor;
}
void* DirectDctPlsCudaPostprocess::completion_event() const noexcept {
	return static_cast<void*>(impl_->completion.get());
}

} // namespace galp::jpeg::detail
