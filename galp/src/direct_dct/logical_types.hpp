#ifndef GALP_DIRECT_DCT_LOGICAL_TYPES_HPP
#define GALP_DIRECT_DCT_LOGICAL_TYPES_HPP

#include "galp/jpeg_dct_device.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace galp::direct_dct {

// Canonical candidate for a logical batch boundary. It deliberately contains
// no shard, rowgroup, physical offset, stream/event, allocator, arena, cache,
// prefetch, CTA, or submission-gate state.
struct LogicalBatchRequest final {
	struct SemanticTransform final {
		std::optional<jpeg::JpegDctCropBox> source_crop;
		bool                                horizontal_flip = false;
		std::string                         logical_sample_id;
		std::string                         augmentation_key;
	};

	struct Sample final {
		uint32_t          image_id = 0U;
		SemanticTransform transform;
	};

	uint64_t            request_identity = 0U;
	uint64_t            batch_ordinal    = 0U;
	std::string         semantic_profile_id;
	size_t              logical_batch_size = 0U;
	bool                partial_tail       = false;
	std::vector<Sample> samples;
};

inline void validate_logical_batch_request(const LogicalBatchRequest& request) {
	if (request.semantic_profile_id.empty()) {
		throw std::invalid_argument("LogicalBatchRequest semantic_profile_id must not be empty");
	}
	if (request.logical_batch_size == 0U) {
		throw std::invalid_argument("LogicalBatchRequest logical_batch_size must be positive");
	}
	if (request.samples.empty()) {
		throw std::invalid_argument("LogicalBatchRequest samples must not be empty");
	}
	if (request.samples.size() > request.logical_batch_size) {
		throw std::invalid_argument("LogicalBatchRequest exceeds its logical batch boundary");
	}
	const bool is_partial = request.samples.size() < request.logical_batch_size;
	if (request.partial_tail != is_partial) {
		throw std::invalid_argument("LogicalBatchRequest partial_tail does not match its logical batch boundary");
	}

	for (const auto& sample : request.samples) {
		if (!sample.transform.source_crop.has_value()) {
			continue;
		}
		const auto& crop = *sample.transform.source_crop;
		if (crop.width == 0U || crop.height == 0U) {
			throw std::invalid_argument("LogicalBatchRequest crop width and height must be positive");
		}
		const auto crop_right  = static_cast<uint64_t>(crop.x) + crop.width;
		const auto crop_bottom = static_cast<uint64_t>(crop.y) + crop.height;
		if (crop_right > std::numeric_limits<uint32_t>::max() || crop_bottom > std::numeric_limits<uint32_t>::max()) {
			throw std::invalid_argument("LogicalBatchRequest crop coordinates overflow uint32");
		}
	}
}

// Lossless compatibility conversion from the legacy semantic request carrier.
// Physical planning and execution policy stay native.
inline LogicalBatchRequest
shadow_convert_legacy_requests(const std::span<const jpeg::JpegDctImageCropRequest> legacy_requests,
                               const size_t                                         logical_batch_size,
                               std::string                                          semantic_profile_id,
                               const uint64_t                                       request_identity = 0U,
                               const uint64_t                                       batch_ordinal    = 0U) {
	LogicalBatchRequest request;
	request.request_identity    = request_identity;
	request.batch_ordinal       = batch_ordinal;
	request.semantic_profile_id = std::move(semantic_profile_id);
	request.logical_batch_size  = logical_batch_size;
	request.partial_tail        = legacy_requests.size() < logical_batch_size;
	request.samples.reserve(legacy_requests.size());

	for (const auto& legacy : legacy_requests) {
		LogicalBatchRequest::SemanticTransform transform;
		const bool                             has_crop = legacy.source_crop.x != 0U || legacy.source_crop.y != 0U ||
		                      legacy.source_crop.width != 0U || legacy.source_crop.height != 0U;
		if (has_crop) {
			transform.source_crop = legacy.source_crop;
		}
		transform.horizontal_flip   = legacy.horizontal_flip;
		transform.logical_sample_id = legacy.logical_sample_id;
		transform.augmentation_key  = legacy.augmentation_key;
		request.samples.push_back(LogicalBatchRequest::Sample {
		    legacy.global_image_index,
		    std::move(transform),
		});
	}

	validate_logical_batch_request(request);
	return request;
}

// Minimal schema only. Explicit shadow/test pipelines may emit it; the Phase-3
// production delegate passes no trace buffer, so it adds no hook, callback,
// lock, allocation, CUDA event, synchronization, or Python interaction.
struct PipelineTraceEvent final {
	enum class Stage : uint8_t {
		kRequestAccepted,
		kPreparing,
		kPlanReady,
		kStaged,
		kAwaitingPredecessor,
		kGateReleased,
		kReadStarted,
		kSubmitted,
		kCompleted,
		kCancelled,
		kFailed,
		kClosed,
	};

	uint64_t request_identity   = 0U;
	uint64_t request_ordinal    = 0U;
	uint64_t batch_ordinal      = 0U;
	Stage    stage              = Stage::kRequestAccepted;
	uint64_t plan_identity_hash = 0U;
	uint64_t io_identity_hash   = 0U;
	uint64_t prepare_ordinal    = 0U;
	uint64_t stage_ordinal      = 0U;
	uint64_t read_ordinal       = 0U;
	uint64_t submission_ordinal = 0U;
	uint64_t completion_ordinal = 0U;
};

// Future lifetime boundary (contract only): the Torch adapter observes and
// forwards the ATen consumer stream and wraps Tensor/Storage. Native code will
// own consumer dependencies, completion interpretation, and reclamation.
// Phase 1 intentionally implements neither side of that migration.

} // namespace galp::direct_dct

#endif // GALP_DIRECT_DCT_LOGICAL_TYPES_HPP
