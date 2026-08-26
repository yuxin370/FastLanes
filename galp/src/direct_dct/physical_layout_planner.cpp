#include "direct_dct/physical_layout_planner.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace galp::direct_dct {
namespace {

void validate_manifest(const jpeg::JpegDctShardManifest& manifest) {
	uint64_t expected_first = 0U;
	for (const auto& shard : manifest.shards) {
		if (shard.image_count == 0U || shard.first_global_image_index != expected_first) {
			throw std::invalid_argument("Direct-DCT manifest shard ranges must be non-empty and contiguous");
		}
		expected_first += shard.image_count;
	}
	if (expected_first != manifest.image_count) {
		throw std::invalid_argument("Direct-DCT manifest image count does not match its shard ranges");
	}
}

} // namespace

PhysicalLayoutPlanner::PhysicalLayoutPlanner(const std::filesystem::path& manifest_path)
    : PhysicalLayoutPlanner(jpeg::read_jpeg_dct_shard_manifest(manifest_path)) {
}

PhysicalLayoutPlanner::PhysicalLayoutPlanner(jpeg::JpegDctShardManifest manifest)
    : manifest_(std::move(manifest)) {
	validate_manifest(manifest_);
}

const jpeg::JpegDctShardManifestEntry& PhysicalLayoutPlanner::locate(const uint32_t image_id) const {
	if (image_id >= manifest_.image_count) {
		throw std::out_of_range("logical Direct-DCT request contains an image outside the manifest");
	}
	const auto found = std::upper_bound(
	    manifest_.shards.begin(), manifest_.shards.end(), image_id,
	    [](const uint32_t value, const jpeg::JpegDctShardManifestEntry& shard) {
		    return static_cast<uint64_t>(value) < shard.first_global_image_index;
	    });
	if (found == manifest_.shards.begin()) {
		throw std::logic_error("Direct-DCT manifest lookup failed");
	}
	const auto& shard = *std::prev(found);
	const auto end = shard.first_global_image_index + shard.image_count;
	if (image_id >= end) {
		throw std::logic_error("Direct-DCT manifest has a gap in its shard ranges");
	}
	return shard;
}

LogicalBatchPhysicalPlan PhysicalLayoutPlanner::plan(const LogicalBatchRequest& request) const {
	if (request.samples.empty()) {
		throw std::invalid_argument("cannot physically plan an empty logical Direct-DCT request");
	}
	LogicalBatchPhysicalPlan result;
	result.request_identity   = request.request_identity;
	result.batch_ordinal      = request.batch_ordinal;
	result.logical_image_count = request.samples.size();
	for (size_t offset = 0U; offset < request.samples.size(); ++offset) {
		const auto image_id = request.samples[offset].image_id;
		const auto& shard = locate(image_id);
		if (!result.segments.empty() && result.segments.back().shard_id == shard.shard_id) {
			auto& segment = result.segments.back();
			++segment.image_count;
			segment.last_global_image_id = image_id;
			continue;
		}
		result.segments.push_back(SegmentPlan {
		    shard.shard_id,
		    offset,
		    1U,
		    image_id,
		    image_id,
		});
	}
	return result;
}

std::vector<LogicalBatchPhysicalPlan>
PhysicalLayoutPlanner::plan(const std::vector<LogicalBatchRequest>& requests) const {
	std::vector<LogicalBatchPhysicalPlan> result;
	result.reserve(requests.size());
	for (const auto& request : requests) {
		result.push_back(plan(request));
	}
	return result;
}

const jpeg::JpegDctShardManifest& PhysicalLayoutPlanner::manifest() const noexcept {
	return manifest_;
}

} // namespace galp::direct_dct
