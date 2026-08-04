#include "galp/jpeg_dct_block_major_plan.hpp"

#if GALP_WITH_JPEG_DCT

#include "galp/jpeg_dct_block_major_access.hpp"
#include "jpeg/jpeg_dct_metadata.hpp"
#include "fls/cfg/cfg.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace galp::jpeg {
namespace {

[[noreturn]] void fail(const std::string& message) {
	throw std::runtime_error("JpegDctBlockMajorCompactPlanner: " + message);
}

std::filesystem::path sidecar_path(const std::filesystem::path& directory, const uint32_t shard_id) {
	std::array<char, 64> name {};
	const auto count = std::snprintf(name.data(), name.size(), "shard_%06u.block_major_access.bin", shard_id);
	if (count < 0 || static_cast<size_t>(count) >= name.size()) {
		fail("failed to format sidecar name");
	}
	return directory / name.data();
}

uint32_t floor_mul_div(const uint32_t lhs, const uint32_t rhs, const uint32_t divisor) {
	if (divisor == 0U) {
		fail("crop mapping encountered a zero image dimension");
	}
	return static_cast<uint32_t>((static_cast<uint64_t>(lhs) * rhs) / divisor);
}

uint32_t ceil_mul_div(const uint32_t lhs, const uint32_t rhs, const uint32_t divisor) {
	if (divisor == 0U) {
		fail("crop mapping encountered a zero image dimension");
	}
	const auto product = static_cast<uint64_t>(lhs) * rhs;
	return static_cast<uint32_t>((product + divisor - 1U) / divisor);
}

bool any_bit_in_range(const std::vector<uint64_t>& bits, const uint32_t begin, const uint32_t end) {
	if (begin >= end) {
		return false;
	}
	const auto first_word = begin / 64U;
	const auto last_word  = (end - 1U) / 64U;
	const auto first_mask = std::numeric_limits<uint64_t>::max() << (begin % 64U);
	const auto last_mask = end % 64U == 0U
	                           ? std::numeric_limits<uint64_t>::max()
	                           : (uint64_t {1U} << (end % 64U)) - 1U;
	if (first_word == last_word) {
		return (bits[first_word] & first_mask & last_mask) != 0U;
	}
	if ((bits[first_word] & first_mask) != 0U) {
		return true;
	}
	for (auto word = first_word + 1U; word < last_word; ++word) {
		if (bits[word] != 0U) {
			return true;
		}
	}
	return (bits[last_word] & last_mask) != 0U;
}

int32_t floor_div(const int32_t value, const int32_t divisor) {
	if (divisor <= 0) {
		fail("crop alignment divisor must be positive");
	}
	const int32_t quotient  = value / divisor;
	const int32_t remainder = value % divisor;
	return quotient - (remainder < 0 ? 1 : 0);
}

uint32_t closest_aligned_crop_extent(const uint32_t               source_extent,
	                                 const uint32_t               output_extent,
	                                 const uint32_t               reference_extent,
	                                 const std::vector<uint32_t>& preferred_small_extents) {
	if (source_extent == 0U || output_extent == 0U || reference_extent == 0U) {
		return 0U;
	}
	const auto target = static_cast<uint32_t>(
	    std::nearbyint((static_cast<double>(source_extent) * output_extent) / reference_extent));
	if (target <= output_extent && !preferred_small_extents.empty()) {
		uint32_t best      = output_extent;
		uint32_t best_diff = std::numeric_limits<uint32_t>::max();
		for (const auto choice : preferred_small_extents) {
			if (choice == 0U || choice > output_extent) {
				continue;
			}
			const auto difference = choice > target ? choice - target : target - choice;
			if (difference < best_diff) {
				best      = choice;
				best_diff = difference;
			}
		}
		return best;
	}
	auto closest = static_cast<uint32_t>(
	    std::nearbyint(static_cast<double>(target) / output_extent) * output_extent);
	if (closest > source_extent) {
		closest = closest > output_extent ? closest - output_extent : output_extent;
	}
	return std::max<uint32_t>(1U, closest);
}

template <typename T>
uint64_t vector_bytes(const std::vector<T>& values) {
	if (values.size() > std::numeric_limits<uint64_t>::max() / sizeof(T)) {
		fail("compact-plan vector exceeds byte accounting range");
	}
	return static_cast<uint64_t>(values.size()) * sizeof(T);
}

struct ClippedRectangle {
	uint32_t shard_index       = 0U;
	uint32_t semantic_slot_id  = 0U;
	uint32_t local_image_index = 0U;
	uint32_t x0                = 0U;
	uint32_t y0                = 0U;
	uint32_t x1                = 0U;
	uint32_t y1                = 0U;
};

struct SortedRequest {
	uint32_t shard_index       = 0U;
	uint32_t shard_id          = 0U;
	uint32_t local_image_index = 0U;
	uint32_t output_slot       = 0U;
	const JpegDctImageCropRequest* source = nullptr;
};

struct XEvent {
	uint32_t x                = 0U;
	uint32_t local_image_index = 0U;
	int8_t   delta            = 0;
};

} // namespace

struct JpegDctBlockMajorCompactPlanner::Impl {
	struct DescriptorState {
		mutable std::mutex mutex;
		std::shared_ptr<const JpegDctBlockMajorAccessDescriptor> descriptor;
		uint64_t descriptor_bytes = 0U;
		uint64_t expected_descriptor_bytes = 0U;
		uint64_t expected_descriptor_crc64 = 0U;
		double descriptor_open_ms = 0.0;
		double descriptor_validation_ms = 0.0;
	};

	struct Shard {
		JpegDctShardManifestEntry entry;
		std::shared_ptr<DescriptorState> descriptor_state;
	};

	std::filesystem::path manifest_path;
	std::filesystem::path descriptor_directory;
	JpegDctShardManifest manifest;
	std::vector<Shard> shards;
	uint64_t descriptor_cache_byte_bound = 0U;

	[[nodiscard]] std::shared_ptr<const JpegDctBlockMajorAccessDescriptor>
	descriptor_for(const uint32_t shard_index) const {
		if (shard_index >= shards.size()) {
			fail("descriptor shard index is outside the manifest");
		}
		const auto& shard = shards[shard_index];
		std::lock_guard<std::mutex> guard(shard.descriptor_state->mutex);
		if (!shard.descriptor_state->descriptor) {
			const auto open_started = std::chrono::steady_clock::now();
			auto opened = JpegDctBlockMajorAccessDescriptor::Open(
			    sidecar_path(descriptor_directory, shard.entry.shard_id));
			shard.descriptor_state->descriptor_open_ms = std::chrono::duration<double, std::milli>(
			    std::chrono::steady_clock::now() - open_started).count();
			if (opened.descriptor_bytes() != shard.descriptor_state->expected_descriptor_bytes ||
			    opened.descriptor_crc64() != shard.descriptor_state->expected_descriptor_crc64) {
				fail("block-major descriptor does not match its companion index record");
			}
			const auto validation_started = std::chrono::steady_clock::now();
			opened.ValidateSource(manifest_path,
			                      shard.entry,
			                      manifest_path.parent_path() / shard.entry.metadata_file_name,
			                      manifest_path.parent_path() / shard.entry.fls_file_name);
			shard.descriptor_state->descriptor_validation_ms = std::chrono::duration<double, std::milli>(
			    std::chrono::steady_clock::now() - validation_started).count();
			shard.descriptor_state->descriptor_bytes = opened.descriptor_bytes();
			shard.descriptor_state->descriptor =
			    std::make_shared<const JpegDctBlockMajorAccessDescriptor>(std::move(opened));
		}
		return shard.descriptor_state->descriptor;
	}

	[[nodiscard]] uint32_t shard_index_for(const uint32_t global_image_index) const {
		if (global_image_index >= manifest.image_count) {
			fail("request image index is outside the manifest");
		}
		size_t begin = 0U;
		size_t end = shards.size();
		while (begin < end) {
			const auto middle = begin + (end - begin) / 2U;
			const auto& entry = shards[middle].entry;
			if (global_image_index < entry.first_global_image_index) {
				end = middle;
			} else if (global_image_index >= entry.first_global_image_index + entry.image_count) {
				begin = middle + 1U;
			} else {
				return static_cast<uint32_t>(middle);
			}
		}
		fail("request image index is not covered by a manifest shard");
	}

	struct ComponentSet {
		std::array<std::optional<JpegDctBlockMajorAccessComponentRecord>, 3> values;
	};

	[[nodiscard]] ComponentSet components_for(const JpegDctBlockMajorAccessDescriptor& descriptor,
	                                          const uint32_t local_image_index) const {
		ComponentSet result;
		const auto image = descriptor.image(local_image_index);
		for (uint32_t index = 0U; index < image.component_count; ++index) {
			const auto component = descriptor.component(image.first_component + index);
			uint32_t output_slot = component.semantic_slot_id;
			if (output_slot >= result.values.size()) {
				output_slot = component.local_component_index;
			}
			if (output_slot < result.values.size() && !result.values[output_slot].has_value()) {
				result.values[output_slot] = component;
			}
		}
		return result;
	}

	[[nodiscard]] std::pair<std::array<JpegDctBlockMajorSupportRectangle, 3>, std::vector<ClippedRectangle>>
	make_supports(const SortedRequest& request,
	              const std::optional<JpegDctGridTransformSpec>& transform) const {
		const auto& shard = shards[request.shard_index];
		const auto descriptor_owner = descriptor_for(request.shard_index);
		const auto& descriptor = *descriptor_owner;
		const auto image = descriptor.image(request.local_image_index);
		const auto components = components_for(descriptor, request.local_image_index);
		std::array<JpegDctBlockMajorSupportRectangle, 3> supports {};
		std::vector<ClippedRectangle> clipped;
		clipped.reserve(3U);

		auto add_support = [&](const uint32_t output_slot,
		                       const JpegDctBlockMajorAccessComponentRecord& component,
		                       const int32_t x,
		                       const int32_t y,
		                       const uint32_t width,
		                       const uint32_t height) {
			if (output_slot >= supports.size() || width == 0U || height == 0U) {
				return;
			}
			supports[output_slot] = {component.semantic_slot_id,
			                         x,
			                         y,
			                         width,
			                         height,
			                         component.width_in_blocks,
			                         component.height_in_blocks,
			                         component.quant_dictionary_id,
			                         component.h_samp_factor,
			                         component.v_samp_factor,
			                         true};
			const auto clipped_x0 = static_cast<uint32_t>(
			    std::clamp<int64_t>(x, 0, static_cast<int64_t>(component.width_in_blocks)));
			const auto clipped_y0 = static_cast<uint32_t>(
			    std::clamp<int64_t>(y, 0, static_cast<int64_t>(component.height_in_blocks)));
			const auto clipped_x1 = static_cast<uint32_t>(std::clamp<int64_t>(
			    static_cast<int64_t>(x) + width, 0, static_cast<int64_t>(component.width_in_blocks)));
			const auto clipped_y1 = static_cast<uint32_t>(std::clamp<int64_t>(
			    static_cast<int64_t>(y) + height, 0, static_cast<int64_t>(component.height_in_blocks)));
			if (clipped_x0 < clipped_x1 && clipped_y0 < clipped_y1) {
				clipped.push_back({request.shard_index,
				                   component.semantic_slot_id,
				                   request.local_image_index,
				                   clipped_x0,
				                   clipped_y0,
				                   clipped_x1,
				                   clipped_y1});
			}
		};

		const bool explicit_crop = request.source->source_crop.width != 0U &&
		                           request.source->source_crop.height != 0U;
		if (transform.has_value()) {
			if (transform->y_output_width_blocks == 0U || transform->y_output_height_blocks == 0U ||
			    transform->cbcr_output_width_blocks == 0U || transform->cbcr_output_height_blocks == 0U ||
			    transform->crop_reference_width_blocks == 0U || transform->crop_reference_height_blocks == 0U ||
			    transform->crop_origin_alignment_blocks == 0U || transform->chroma_crop_scale_x == 0U ||
			    transform->chroma_crop_scale_y == 0U) {
				fail("grid transform contains zero geometry");
			}
			if (!components.values[0].has_value()) {
				fail("grid transform requires a present Y component");
			}
			if (components.values[1].has_value() != components.values[2].has_value()) {
				fail("grid transform requires both chroma components or neither");
			}
			if (!components.values[1].has_value() && !transform->allow_grayscale) {
				fail("grid transform does not allow grayscale input");
			}
			if (explicit_crop) {
				if (image.image_width == 0U || image.image_height == 0U) {
					fail("explicit crop requires exact image dimensions");
				}
				auto crop = request.source->source_crop;
				if (crop.x >= image.image_width || crop.y >= image.image_height) {
					fail("explicit crop starts outside the source image");
				}
				crop.width  = std::min(crop.width, image.image_width - crop.x);
				crop.height = std::min(crop.height, image.image_height - crop.y);
				for (uint32_t output_slot = 0U; output_slot < components.values.size(); ++output_slot) {
					if (!components.values[output_slot].has_value()) {
						continue;
					}
					const auto& component = *components.values[output_slot];
					const auto x0 = floor_mul_div(crop.x, component.width_in_blocks, image.image_width);
					const auto y0 = floor_mul_div(crop.y, component.height_in_blocks, image.image_height);
					const auto x1 = std::min<uint32_t>(
					    component.width_in_blocks,
					    ceil_mul_div(crop.x + crop.width, component.width_in_blocks, image.image_width));
					const auto y1 = std::min<uint32_t>(
					    component.height_in_blocks,
					    ceil_mul_div(crop.y + crop.height, component.height_in_blocks, image.image_height));
					add_support(output_slot, component, static_cast<int32_t>(x0), static_cast<int32_t>(y0),
					            std::max<uint32_t>(1U, x1 - x0), std::max<uint32_t>(1U, y1 - y0));
				}
			} else {
				const auto& y_component = *components.values[0];
				const auto y_width = closest_aligned_crop_extent(y_component.width_in_blocks,
				                                                  transform->y_output_width_blocks,
				                                                  transform->crop_reference_width_blocks,
				                                                  transform->preferred_small_crop_width_blocks);
				const auto y_height = closest_aligned_crop_extent(y_component.height_in_blocks,
				                                                   transform->y_output_height_blocks,
				                                                   transform->crop_reference_height_blocks,
				                                                   transform->preferred_small_crop_height_blocks);
				const auto alignment = static_cast<int32_t>(transform->crop_origin_alignment_blocks);
				const auto y_x = floor_div(
				                     floor_div(static_cast<int32_t>(y_component.width_in_blocks) -
				                                   static_cast<int32_t>(y_width),
				                               2),
				                     alignment) *
				                 alignment;
				const auto y_y = floor_div(
				                     floor_div(static_cast<int32_t>(y_component.height_in_blocks) -
				                                   static_cast<int32_t>(y_height),
				                               2),
				                     alignment) *
				                 alignment;
				add_support(0U, y_component, y_x, y_y, y_width, y_height);
				if (components.values[1].has_value()) {
					const auto chroma_x = floor_div(y_x, static_cast<int32_t>(transform->chroma_crop_scale_x));
					const auto chroma_y = floor_div(y_y, static_cast<int32_t>(transform->chroma_crop_scale_y));
					const auto chroma_width = std::max<uint32_t>(1U, y_width / transform->chroma_crop_scale_x);
					const auto chroma_height = std::max<uint32_t>(1U, y_height / transform->chroma_crop_scale_y);
					add_support(1U, *components.values[1], chroma_x, chroma_y, chroma_width, chroma_height);
					add_support(2U, *components.values[2], chroma_x, chroma_y, chroma_width, chroma_height);
				}
			}
		} else {
			if (image.image_width == 0U || image.image_height == 0U) {
				fail("crop planning requires exact image dimensions");
			}
			auto crop = request.source->source_crop;
			if (crop.width == 0U || crop.height == 0U) {
				crop = {0U, 0U, image.image_width, image.image_height};
			} else {
				if (crop.x >= image.image_width || crop.y >= image.image_height) {
					fail("crop starts outside the source image");
				}
				crop.width  = std::min(crop.width, image.image_width - crop.x);
				crop.height = std::min(crop.height, image.image_height - crop.y);
			}
			for (uint32_t output_slot = 0U; output_slot < components.values.size(); ++output_slot) {
				if (!components.values[output_slot].has_value()) {
					continue;
				}
				const auto& component = *components.values[output_slot];
				const auto x0 = floor_mul_div(crop.x, component.width_in_blocks, image.image_width);
				const auto y0 = floor_mul_div(crop.y, component.height_in_blocks, image.image_height);
				const auto x1 = std::min<uint32_t>(
				    component.width_in_blocks,
				    ceil_mul_div(crop.x + crop.width, component.width_in_blocks, image.image_width));
				const auto y1 = std::min<uint32_t>(
				    component.height_in_blocks,
				    ceil_mul_div(crop.y + crop.height, component.height_in_blocks, image.image_height));
				add_support(output_slot,
				            component,
				            static_cast<int32_t>(x0),
				            static_cast<int32_t>(y0),
				            x1 - x0,
				            y1 - y0);
			}
		}
		return {supports, clipped};
	}
};

JpegDctBlockMajorCompactPlanner::JpegDctBlockMajorCompactPlanner(
	const std::filesystem::path& manifest_path, const std::filesystem::path& descriptor_directory)
    : impl_(std::make_unique<Impl>()) {
	impl_->manifest_path        = manifest_path;
	impl_->descriptor_directory = descriptor_directory;
	impl_->manifest = detail::read_jpeg_dct_shard_manifest_file(manifest_path);
	if (impl_->manifest.version != 1U || impl_->manifest.shards.empty()) {
		fail("compact planner requires a non-empty manifest-v1 dataset");
	}
	const auto companion_path = descriptor_directory / "manifest.block_major_access.bin";
	if (!std::filesystem::is_regular_file(companion_path)) {
		fail("block-major companion index is missing; select the legacy planner explicitly");
	}
	const auto companion =
	    read_jpeg_dct_block_major_access_index(companion_path, manifest_path, impl_->manifest);
	impl_->shards.reserve(impl_->manifest.shards.size());
	uint64_t expected_first_image = 0U;
	for (size_t shard_index = 0U; shard_index < impl_->manifest.shards.size(); ++shard_index) {
		const auto& entry = impl_->manifest.shards[shard_index];
		const auto& index_record = companion.shards[shard_index];
		if (entry.first_global_image_index != expected_first_image) {
			fail("manifest shard image ranges are not dense and ordered");
		}
		expected_first_image += entry.image_count;
		const auto descriptor_path = sidecar_path(descriptor_directory, entry.shard_id);
		if (!std::filesystem::is_regular_file(descriptor_path)) {
			fail("block-major shard descriptor is missing: " + descriptor_path.string());
		}
		const auto descriptor_bytes = std::filesystem::file_size(descriptor_path);
		if (descriptor_bytes != index_record.descriptor_bytes) {
			fail("block-major descriptor size does not match the companion index");
		}
		if (descriptor_bytes > std::numeric_limits<uint64_t>::max() - impl_->descriptor_cache_byte_bound) {
			fail("block-major descriptor cache byte bound overflow");
		}
		impl_->descriptor_cache_byte_bound += descriptor_bytes;
		Impl::Shard state;
		state.entry            = entry;
		state.descriptor_state = std::make_shared<Impl::DescriptorState>();
		state.descriptor_state->expected_descriptor_bytes = descriptor_bytes;
		state.descriptor_state->expected_descriptor_crc64 = index_record.descriptor_crc64;
		impl_->shards.push_back(std::move(state));
	}
	if (expected_first_image != impl_->manifest.image_count) {
		fail("manifest shard image ranges do not cover the dataset");
	}
}

JpegDctBlockMajorCompactPlanner::~JpegDctBlockMajorCompactPlanner() = default;
JpegDctBlockMajorCompactPlanner::JpegDctBlockMajorCompactPlanner(JpegDctBlockMajorCompactPlanner&&) noexcept =
    default;
JpegDctBlockMajorCompactPlanner& JpegDctBlockMajorCompactPlanner::operator=(
	JpegDctBlockMajorCompactPlanner&&) noexcept = default;

uint64_t JpegDctBlockMajorCompactPlanner::image_count() const noexcept {
	return impl_->manifest.image_count;
}

size_t JpegDctBlockMajorCompactPlanner::loaded_descriptor_count() const noexcept {
	size_t count = 0U;
	for (const auto& shard : impl_->shards) {
		std::lock_guard<std::mutex> guard(shard.descriptor_state->mutex);
		count += shard.descriptor_state->descriptor ? 1U : 0U;
	}
	return count;
}

uint64_t JpegDctBlockMajorCompactPlanner::loaded_descriptor_bytes() const noexcept {
	uint64_t bytes = 0U;
	for (const auto& shard : impl_->shards) {
		std::lock_guard<std::mutex> guard(shard.descriptor_state->mutex);
		bytes += shard.descriptor_state->descriptor_bytes;
	}
	return bytes;
}

uint64_t JpegDctBlockMajorCompactPlanner::descriptor_cache_byte_bound() const noexcept {
	return impl_->descriptor_cache_byte_bound;
}

double JpegDctBlockMajorCompactPlanner::descriptor_open_ms() const noexcept {
	double milliseconds = 0.0;
	for (const auto& shard : impl_->shards) {
		std::lock_guard<std::mutex> guard(shard.descriptor_state->mutex);
		milliseconds += shard.descriptor_state->descriptor_open_ms;
	}
	return milliseconds;
}

double JpegDctBlockMajorCompactPlanner::descriptor_validation_ms() const noexcept {
	double milliseconds = 0.0;
	for (const auto& shard : impl_->shards) {
		std::lock_guard<std::mutex> guard(shard.descriptor_state->mutex);
		milliseconds += shard.descriptor_state->descriptor_validation_ms;
	}
	return milliseconds;
}

JpegDctBlockMajorCompactPlan JpegDctBlockMajorCompactPlanner::Plan(
	const std::vector<JpegDctImageCropRequest>& input_requests,
	const std::optional<JpegDctGridTransformSpec>& grid_transform) const {
	JpegDctBlockMajorCompactPlan plan;
	plan.stats.request_count = input_requests.size();
	if (input_requests.empty()) {
		return plan;
	}
	if (input_requests.size() > std::numeric_limits<uint32_t>::max()) {
		fail("batch request count exceeds compact-plan limits");
	}
	std::vector<SortedRequest> sorted;
	sorted.reserve(input_requests.size());
	bool monotonic = true;
	uint32_t previous_shard = 0U;
	uint32_t previous_local = 0U;
	for (uint32_t output_slot = 0U; output_slot < input_requests.size(); ++output_slot) {
		const auto& request = input_requests[output_slot];
		const auto shard_index = impl_->shard_index_for(request.global_image_index);
		const auto& entry = impl_->shards[shard_index].entry;
		const auto local_image = static_cast<uint32_t>(request.global_image_index - entry.first_global_image_index);
		if (output_slot != 0U && std::tie(shard_index, local_image) < std::tie(previous_shard, previous_local)) {
			monotonic = false;
		}
		previous_shard = shard_index;
		previous_local = local_image;
		sorted.push_back({shard_index, entry.shard_id, local_image, output_slot, &request});
	}
	plan.stats.input_was_shard_local_monotonic = monotonic;
	if (!monotonic) {
		std::stable_sort(sorted.begin(), sorted.end(), [](const auto& lhs, const auto& rhs) {
			return std::tie(lhs.shard_index, lhs.local_image_index, lhs.output_slot) <
			       std::tie(rhs.shard_index, rhs.local_image_index, rhs.output_slot);
		});
		plan.stats.request_sort_items = sorted.size();
	}

	plan.requests.reserve(sorted.size());
	plan.duplicate_output_slots.reserve(sorted.size());
	std::map<uint64_t, std::vector<ClippedRectangle>> rectangles_by_shard_slot;
	std::map<uint64_t, uint16_t> runtime_quant_by_source;
	const auto intern_quant_table = [&](const uint32_t shard_index, const uint16_t source_dictionary_id) {
		if (source_dictionary_id == std::numeric_limits<uint16_t>::max()) {
			fail("present component has no quantization dictionary entry");
		}
		const auto source_key = (static_cast<uint64_t>(shard_index) << 32U) | source_dictionary_id;
		if (const auto found = runtime_quant_by_source.find(source_key); found != runtime_quant_by_source.end()) {
			return found->second;
		}
		const auto descriptor = impl_->descriptor_for(shard_index);
		const auto source = descriptor->quant_table(source_dictionary_id);
		const auto existing = std::find_if(plan.quant_tables.begin(), plan.quant_tables.end(), [&](const auto& table) {
			return table.values == source.values;
		});
		uint16_t runtime_index = 0U;
		if (existing == plan.quant_tables.end()) {
			if (plan.quant_tables.size() >= std::numeric_limits<uint16_t>::max()) {
				fail("batch quantization-table dictionary exceeds compact-plan limits");
			}
			runtime_index = static_cast<uint16_t>(plan.quant_tables.size());
			plan.quant_tables.push_back({source.fingerprint, source.values});
		} else {
			runtime_index = static_cast<uint16_t>(std::distance(plan.quant_tables.begin(), existing));
		}
		runtime_quant_by_source.emplace(source_key, runtime_index);
		return runtime_index;
	};
	uint32_t unique_index = std::numeric_limits<uint32_t>::max();
	uint32_t last_shard = std::numeric_limits<uint32_t>::max();
	uint32_t last_local = std::numeric_limits<uint32_t>::max();
	for (const auto& request : sorted) {
		if (request.shard_index != last_shard || request.local_image_index != last_local) {
			unique_index = static_cast<uint32_t>(plan.unique_images.size());
			plan.unique_images.push_back({request.shard_id,
			                              request.local_image_index,
			                              static_cast<uint32_t>(plan.duplicate_output_slots.size()),
			                              0U});
			if (last_shard != std::numeric_limits<uint32_t>::max()) {
				plan.stats.shard_local_request_runs +=
				    request.shard_index != last_shard || request.local_image_index != last_local + 1U ? 1U : 0U;
			} else {
				plan.stats.shard_local_request_runs = 1U;
			}
			last_shard = request.shard_index;
			last_local = request.local_image_index;
		}
		plan.duplicate_output_slots.push_back(request.output_slot);
		++plan.unique_images[unique_index].fanout_count;
		auto [supports, rectangles] = impl_->make_supports(request, grid_transform);
		for (auto& support : supports) {
			if (support.present) {
				support.quant_table_index = intern_quant_table(request.shard_index, support.quant_table_index);
			}
		}
		plan.stats.support_rectangle_count += rectangles.size();
		plan.requests.push_back({request.source->global_image_index,
		                         request.shard_id,
		                         request.local_image_index,
		                         request.output_slot,
		                         unique_index,
		                         request.source->horizontal_flip,
		                         supports});
		for (auto& rectangle : rectangles) {
			const auto key = (static_cast<uint64_t>(rectangle.shard_index) << 32U) | rectangle.semantic_slot_id;
			rectangles_by_shard_slot[key].push_back(std::move(rectangle));
		}
	}
	plan.stats.unique_image_count     = plan.unique_images.size();
	plan.stats.duplicate_output_count = plan.requests.size() - plan.unique_images.size();

	std::unordered_map<uint64_t, uint32_t> runtime_rank_cell_by_source;
	runtime_rank_cell_by_source.reserve(std::min<size_t>(
	    static_cast<size_t>(std::numeric_limits<uint32_t>::max()), sorted.size() * 16U));
	const auto intern_rank_cell = [&](const uint32_t shard_index, const uint32_t source_cell_id) {
		const auto source_key = (static_cast<uint64_t>(shard_index) << 32U) | source_cell_id;
		if (const auto found = runtime_rank_cell_by_source.find(source_key);
		    found != runtime_rank_cell_by_source.end()) {
			return found->second;
		}
		const auto descriptor = impl_->descriptor_for(shard_index);
		const auto source = descriptor->rank_cell(source_cell_id);
		if (plan.rank_payload.size() > std::numeric_limits<uint32_t>::max() ||
		    source.payload.size() > std::numeric_limits<uint32_t>::max() - plan.rank_payload.size()) {
			fail("batch rank payload exceeds compact-plan limits");
		}
		const auto runtime_index = static_cast<uint32_t>(plan.rank_cells.size());
		const auto payload_offset = static_cast<uint32_t>(plan.rank_payload.size());
		plan.rank_payload.insert(plan.rank_payload.end(), source.payload.begin(), source.payload.end());
		plan.rank_cells.push_back({impl_->shards.at(shard_index).entry.shard_id,
		                           source.cell_id,
		                           source.image_count,
		                           payload_offset,
		                           static_cast<uint32_t>(source.payload.size()),
		                           source.present_count,
		                           source.rank_checkpoint_images,
		                           source.encoding});
		runtime_rank_cell_by_source.emplace(source_key, runtime_index);
		return runtime_index;
	};
	const auto rowgroup_vector_words =
	    (static_cast<size_t>(impl_->manifest.rowgroup_vectors) + 63U) / 64U;
	std::map<uint64_t, std::vector<uint64_t>> rowgroup_vector_bits;
	uint64_t temporary_peak_bytes = vector_bytes(sorted);
	for (const auto& [key, rectangles] : rectangles_by_shard_slot) {
		const auto shard_index = static_cast<uint32_t>(key >> 32U);
		const auto semantic_slot_id = static_cast<uint32_t>(key);
		const auto& shard = impl_->shards.at(shard_index);
		const auto descriptor_owner = impl_->descriptor_for(shard_index);
		const auto& descriptor = *descriptor_owner;
		uint32_t max_y = 0U;
		for (const auto& rectangle : rectangles) {
			max_y = std::max(max_y, rectangle.y1);
		}
		uint32_t max_x = 0U;
		for (const auto& rectangle : rectangles) {
			max_x = std::max(max_x, rectangle.x1);
		}
		if (static_cast<uint64_t>(max_x) * max_y > std::numeric_limits<size_t>::max()) {
			fail("rectangle sweep lookup exceeds host address range");
		}
		std::vector<std::vector<XEvent>> row_events(max_y);
		uint64_t event_count = 0U;
		for (const auto& rectangle : rectangles) {
			for (uint32_t y = rectangle.y0; y < rectangle.y1; ++y) {
				row_events[y].push_back({rectangle.x0, rectangle.local_image_index, 1});
				row_events[y].push_back({rectangle.x1, rectangle.local_image_index, -1});
				event_count += 2U;
			}
		}
		std::vector<std::array<uint32_t, 2>> touched_coordinates;
		std::vector<uint32_t> coordinate_lookup(
		    static_cast<size_t>(max_x) * max_y, std::numeric_limits<uint32_t>::max());
		for (uint32_t y = 0U; y < row_events.size(); ++y) {
			auto& events = row_events[y];
			if (events.empty()) {
				continue;
			}
			std::sort(events.begin(), events.end(), [](const auto& lhs, const auto& rhs) {
				return std::tie(lhs.x, lhs.local_image_index, lhs.delta) <
				       std::tie(rhs.x, rhs.local_image_index, rhs.delta);
			});
			uint64_t active_rectangles = 0U;
			uint32_t previous_x = events.front().x;
			size_t event_index = 0U;
			while (event_index < events.size()) {
				const auto x = events[event_index].x;
				if (previous_x < x && active_rectangles != 0U) {
					for (uint32_t block_x = previous_x; block_x < x; ++block_x) {
						const auto coordinate_index = static_cast<uint32_t>(touched_coordinates.size());
						touched_coordinates.push_back({block_x, y});
						coordinate_lookup[static_cast<size_t>(y) * max_x + block_x] = coordinate_index;
					}
				}
				while (event_index < events.size() && events[event_index].x == x) {
					if (events[event_index].delta > 0) {
						++active_rectangles;
					} else {
						if (active_rectangles == 0U) {
							fail("rectangle-union sweep underflow");
						}
						--active_rectangles;
					}
					++event_index;
				}
				previous_x = x;
			}
			if (active_rectangles != 0U) {
				fail("rectangle-union sweep ended with active rectangles");
			}
		}
		const auto resolved_groups = descriptor.FindGroups(semantic_slot_id, touched_coordinates);
		if (resolved_groups.size() != touched_coordinates.size()) {
			fail("batch group resolver returned the wrong result count");
		}
		for (const auto& group : resolved_groups) {
			if (group.row_count == 0U) {
				fail("active crop coordinate is absent from the block-major topology");
			}
		}
		temporary_peak_bytes = std::max<uint64_t>(
		    temporary_peak_bytes,
		    event_count * sizeof(XEvent) + static_cast<uint64_t>(descriptor.image_count()) * sizeof(uint32_t) +
		        ((static_cast<uint64_t>(descriptor.image_count()) + 63U) / 64U) * sizeof(uint64_t) +
		        vector_bytes(touched_coordinates) + vector_bytes(coordinate_lookup) + vector_bytes(resolved_groups));
		std::vector<uint32_t> active_counts(descriptor.image_count(), 0U);
		std::vector<uint64_t> active_image_bits((descriptor.image_count() + 63U) / 64U, 0U);
		for (uint32_t y = 0U; y < row_events.size(); ++y) {
			auto& events = row_events[y];
			if (events.empty()) {
				continue;
			}
			std::fill(active_counts.begin(), active_counts.end(), 0U);
			std::fill(active_image_bits.begin(), active_image_bits.end(), 0U);
			uint32_t active_image_count = 0U;
			uint32_t previous_x = events.front().x;
			size_t event_index = 0U;
			while (event_index < events.size()) {
				const auto x = events[event_index].x;
				if (previous_x < x && active_image_count != 0U) {
					for (uint32_t block_x = previous_x; block_x < x; ++block_x) {
						const auto coordinate_index = coordinate_lookup[static_cast<size_t>(y) * max_x + block_x];
						if (coordinate_index == std::numeric_limits<uint32_t>::max() ||
						    coordinate_index >= resolved_groups.size()) {
							fail("rectangle sweep coordinate is missing from its resolved lookup");
						}
						const auto& group = resolved_groups[coordinate_index];
						const auto group_vector_begin = group.row_start_in_rowgroup / fastlanes::CFG::VEC_SZ;
						const auto group_vector_end =
						    (static_cast<uint64_t>(group.row_start_in_rowgroup) + group.row_count +
						     fastlanes::CFG::VEC_SZ - 1U) /
						    fastlanes::CFG::VEC_SZ;
						if (group_vector_end > impl_->manifest.rowgroup_vectors ||
						    group_vector_end - group_vector_begin > 128U) {
							fail("group vector range exceeds compact-plan limits");
						}
						const auto rowgroup_key =
						    (static_cast<uint64_t>(shard.entry.shard_id) << 32U) | group.fls_rowgroup_index;
						auto [bits_it, inserted] = rowgroup_vector_bits.try_emplace(rowgroup_key);
						if (inserted) {
							bits_it->second.assign(rowgroup_vector_words, 0U);
						}
						// A group is image-minor ordered and, on the real dataset, spans
						// only a handful of 1024-row vectors. Invert the internal vector
						// boundaries with select() and intersect them with the active
						// image-ID runs. This is exact but avoids one rank decode per
						// fragmented active run.
						uint32_t local_vector_begin = 0U;
						for (uint64_t vector = group_vector_begin; vector < group_vector_end; ++vector) {
							const auto row_end = std::min<uint64_t>(
							    static_cast<uint64_t>(group.row_start_in_rowgroup) + group.row_count,
							    (vector + 1U) * fastlanes::CFG::VEC_SZ);
							const auto rank_end = static_cast<uint32_t>(
							    row_end - static_cast<uint64_t>(group.row_start_in_rowgroup));
							uint32_t local_vector_end = descriptor.image_count();
							if (rank_end < group.row_count) {
								const auto selected_boundary = descriptor.SelectCell(group.rank_cell_id, rank_end);
								if (!selected_boundary.has_value()) {
									fail("group vector boundary cannot be selected from its rank cell");
								}
								local_vector_end = *selected_boundary;
							}
							if (any_bit_in_range(active_image_bits, local_vector_begin, local_vector_end)) {
								const auto compact_vector = static_cast<uint32_t>(vector);
								bits_it->second[compact_vector / 64U] |= uint64_t {1U} << (compact_vector % 64U);
							}
							local_vector_begin = local_vector_end;
						}
						plan.group_bindings.push_back({shard.entry.shard_id,
						                               semantic_slot_id,
						                               block_x,
						                               y,
						                               group.group_id,
						                               group.rank_cell_id,
						                               intern_rank_cell(shard_index, group.rank_cell_id),
						                               group.fls_rowgroup_index,
						                               group.row_start_in_rowgroup,
						                               0U,
						                               0U});
					}
				}
				while (event_index < events.size() && events[event_index].x == x) {
					const auto event = events[event_index++];
					auto& count = active_counts[event.local_image_index];
					if (event.delta > 0) {
						if (count++ == 0U) {
							active_image_bits[event.local_image_index / 64U] |=
							    uint64_t {1U} << (event.local_image_index % 64U);
							++active_image_count;
						}
					} else {
						if (count == 0U) {
							fail("rectangle sweep active-count underflow");
						}
						if (--count == 0U) {
							active_image_bits[event.local_image_index / 64U] &=
							    ~(uint64_t {1U} << (event.local_image_index % 64U));
							--active_image_count;
						}
					}
				}
				previous_x = x;
			}
			if (active_image_count != 0U) {
				fail("rectangle sweep ended with active images");
			}
		}
	}

	plan.rowgroups.reserve(rowgroup_vector_bits.size());
	for (const auto& [key, bits] : rowgroup_vector_bits) {
		const auto shard_id = static_cast<uint32_t>(key >> 32U);
		const auto rowgroup_index = static_cast<uint32_t>(key);
		const auto first_vector_run = static_cast<uint32_t>(plan.vector_runs.size());
		uint32_t selected_vectors = 0U;
		uint32_t vector = 0U;
		while (vector < impl_->manifest.rowgroup_vectors) {
			const auto selected = (bits[vector / 64U] & (uint64_t {1U} << (vector % 64U))) != 0U;
			if (!selected) {
				++vector;
				continue;
			}
			const auto run_begin = vector;
			do {
				++vector;
			} while (vector < impl_->manifest.rowgroup_vectors &&
			         (bits[vector / 64U] & (uint64_t {1U} << (vector % 64U))) != 0U);
			plan.vector_runs.push_back({shard_id, rowgroup_index, run_begin, vector - run_begin});
			selected_vectors += vector - run_begin;
		}
		plan.rowgroups.push_back({shard_id,
		                          rowgroup_index,
		                          first_vector_run,
		                          static_cast<uint32_t>(plan.vector_runs.size()) - first_vector_run,
		                          selected_vectors});
		plan.stats.selected_vectors += selected_vectors;
	}
	plan.stats.touched_block_groups = plan.group_bindings.size();
	plan.stats.group_rank_runs      = plan.rank_runs.size();
	plan.stats.selected_rowgroups   = plan.rowgroups.size();
	plan.stats.selected_vector_runs = plan.vector_runs.size();
	plan.stats.touched_rank_cells   = plan.rank_cells.size();
	plan.stats.rank_payload_bytes   = plan.rank_payload.size();
	plan.stats.touched_quant_tables = plan.quant_tables.size();
	plan.stats.compact_plan_bytes = vector_bytes(plan.requests) + vector_bytes(plan.unique_images) +
	                                vector_bytes(plan.duplicate_output_slots) + vector_bytes(plan.group_bindings) +
	                                vector_bytes(plan.rank_runs) + vector_bytes(plan.rank_cells) +
	                                vector_bytes(plan.rank_payload) + vector_bytes(plan.quant_tables) +
	                                vector_bytes(plan.rowgroups) +
	                                vector_bytes(plan.vector_runs);
	plan.stats.compact_plan_peak_bytes = plan.stats.compact_plan_bytes + temporary_peak_bytes;
	return plan;
}

} // namespace galp::jpeg

#endif // GALP_WITH_JPEG_DCT
