#include "galp/jpeg_dct.hpp"
#include "galp/jpeg_dct_diagnostics.hpp"
#include "galp/profiles/rgbnomore.hpp"
#include "galp/sparse_read_recipe.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

struct Options {
	std::filesystem::path manifest;
	std::filesystem::path descriptor_directory;
	std::filesystem::path output_json;
	std::filesystem::path sparse_recipe_directory;
	std::filesystem::path gap_replay_json;
	std::filesystem::path canonical_plan_directory;
	uint32_t start = 0U;
	uint32_t count = 128U;
	uint32_t seed  = 20260731U;
	uint32_t decode_batch_rowgroups = 64U;
	uint32_t prefetch_workers       = 2U;
	uint32_t workset_capacity_mib   = 512U;
	uint32_t crop_reference_blocks  = 32U;
	std::string pattern = "sequential";
	std::vector<uint32_t> image_ids;
	bool rgbnomore_production_fp32 = false;
	bool compare_legacy = false;
	bool execute = false;
	bool explicit_crops = false;
	bool require_grayscale = false;
	bool require_cross_shard = false;
};

void usage(const char* program) {
	std::cerr << "Usage: " << program
	          << " MANIFEST --descriptor-dir DIR [--start N] [--count N]"
	             " [--pattern sequential|reverse|random|duplicates] [--seed N]"
	             " [--image-ids N,N,...] [--explicit-crops] [--require-grayscale] [--require-cross-shard]"
	             " [--compare-legacy] [--execute] [--decode-batch-rowgroups N]"
	             " [--prefetch-workers N] [--workset-capacity-mib N] [--crop-reference-blocks N]"
	             " [--rgbnomore-production-fp32]"
	             " [--write-sparse-recipe-dir DIR]"
	             " [--write-canonical-plan-dir DIR]"
	             " [--gap-replay-json PATH]"
	             " [--output-json PATH]\n";
}

uint32_t parse_u32(const std::string_view name, const char* value) {
	const auto parsed = std::stoull(value);
	if (parsed > std::numeric_limits<uint32_t>::max()) {
		throw std::runtime_error(std::string(name) + " is outside uint32_t range");
	}
	return static_cast<uint32_t>(parsed);
}

std::vector<uint32_t> parse_image_ids(const char* value) {
	std::vector<uint32_t> result;
	std::string           input(value);
	size_t                begin = 0U;
	while (begin <= input.size()) {
		const auto end = input.find(',', begin);
		const auto token = input.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
		if (token.empty()) {
			throw std::runtime_error("--image-ids contains an empty item");
		}
		result.push_back(parse_u32("--image-ids", token.c_str()));
		if (end == std::string::npos) {
			break;
		}
		begin = end + 1U;
	}
	if (result.empty()) {
		throw std::runtime_error("--image-ids requires at least one image");
	}
	return result;
}

Options parse_options(const int argc, char** argv) {
	Options options;
	for (int index = 1; index < argc; ++index) {
		const std::string_view argument(argv[index]);
		auto require_value = [&](const std::string_view name) -> const char* {
			if (++index >= argc) {
				throw std::runtime_error(std::string(name) + " requires a value");
			}
			return argv[index];
		};
		if (argument == "--help" || argument == "-h") {
			usage(argv[0]);
			std::exit(0);
		}
		if (argument == "--descriptor-dir") {
			options.descriptor_directory = require_value(argument);
		} else if (argument == "--write-sparse-recipe-dir") {
			options.sparse_recipe_directory = require_value(argument);
		} else if (argument == "--gap-replay-json") {
			options.gap_replay_json = require_value(argument);
		} else if (argument == "--write-canonical-plan-dir") {
			options.canonical_plan_directory = require_value(argument);
		} else if (argument == "--output-json") {
			options.output_json = require_value(argument);
		} else if (argument == "--start") {
			options.start = parse_u32(argument, require_value(argument));
		} else if (argument == "--count") {
			options.count = parse_u32(argument, require_value(argument));
		} else if (argument == "--seed") {
			options.seed = parse_u32(argument, require_value(argument));
		} else if (argument == "--decode-batch-rowgroups") {
			options.decode_batch_rowgroups = parse_u32(argument, require_value(argument));
		} else if (argument == "--prefetch-workers") {
			options.prefetch_workers = parse_u32(argument, require_value(argument));
		} else if (argument == "--workset-capacity-mib") {
			options.workset_capacity_mib = parse_u32(argument, require_value(argument));
		} else if (argument == "--crop-reference-blocks") {
			options.crop_reference_blocks = parse_u32(argument, require_value(argument));
		} else if (argument == "--rgbnomore-production-fp32") {
			options.rgbnomore_production_fp32 = true;
		} else if (argument == "--pattern") {
			options.pattern = require_value(argument);
		} else if (argument == "--image-ids") {
			options.image_ids = parse_image_ids(require_value(argument));
		} else if (argument == "--explicit-crops") {
			options.explicit_crops = true;
		} else if (argument == "--require-grayscale") {
			options.require_grayscale = true;
		} else if (argument == "--require-cross-shard") {
			options.require_cross_shard = true;
		} else if (argument == "--compare-legacy") {
			options.compare_legacy = true;
		} else if (argument == "--execute") {
			options.execute        = true;
			options.compare_legacy = true;
		} else if (!argument.empty() && argument.front() == '-') {
			throw std::runtime_error("unknown option: " + std::string(argument));
		} else if (options.manifest.empty()) {
			options.manifest = argv[index];
		} else {
			throw std::runtime_error("multiple manifest paths were supplied");
		}
	}
	if (options.manifest.empty() || options.descriptor_directory.empty() || options.count == 0U ||
	    options.decode_batch_rowgroups == 0U || options.workset_capacity_mib == 0U ||
	    options.crop_reference_blocks == 0U) {
		usage(argv[0]);
		throw std::runtime_error("manifest, descriptor directory, and positive count are required");
	}
	if (options.pattern != "sequential" && options.pattern != "reverse" && options.pattern != "random" &&
	    options.pattern != "duplicates") {
		throw std::runtime_error("unsupported --pattern");
	}
	if (!options.gap_replay_json.empty() && options.sparse_recipe_directory.empty()) {
		throw std::runtime_error("--gap-replay-json requires --write-sparse-recipe-dir");
	}
	return options;
}

using SelectedVector = std::tuple<uint32_t, uint32_t, uint32_t>;

std::set<SelectedVector> compact_vectors(const galp::jpeg::JpegDctBlockMajorCompactPlan& plan) {
	std::set<SelectedVector> result;
	for (const auto& run : plan.vector_runs) {
		for (uint32_t vector = run.first_vector; vector < run.first_vector + run.vector_count; ++vector) {
			result.emplace(run.shard_id, run.rowgroup_index, vector);
		}
	}
	return result;
}

std::set<SelectedVector> legacy_vectors(const galp::jpeg::JpegDctDeviceBatchPlanPreview& preview) {
	std::set<SelectedVector> result;
	for (const auto& rowgroup : preview.rowgroup_vector_plans) {
		for (const auto vector : rowgroup.selected_vectors) {
			result.emplace(rowgroup.rowgroup.shard_id, rowgroup.rowgroup.rowgroup_index, vector);
		}
	}
	return result;
}

struct HostGrid {
	std::vector<int16_t> y;
	std::vector<int16_t> cbcr;
};

struct DeviceRun {
	HostGrid                                 grid;
	galp::jpeg::JpegDctDeviceExecutionStats stats;
	galp::jpeg::JpegDctDeviceCacheStats     cache_stats;
	double                                   wall_ms = 0.0;
};

DeviceRun execute_device_batch(galp::jpeg::JpegDctShardDatasetReader&                    reader,
                               const std::vector<galp::jpeg::JpegDctImageCropRequest>& requests,
                               const galp::jpeg::JpegDctDeviceBatchOptions&            options) {
	const auto begin = std::chrono::steady_clock::now();
	auto       batch = reader.ReadDeviceDctBatch(requests, options);
	batch.synchronize();
	if (batch.grid_output_data_type() != galp::jpeg::JpegDctGridOutputDataType::kInt16) {
		throw std::runtime_error("block-major execution audit requires int16 transformed-grid output");
	}
	DeviceRun result;
	result.grid.y.resize(batch.y_coefficient_count());
	result.grid.cbcr.resize(batch.cbcr_coefficient_count());
	if (!result.grid.y.empty() &&
	    cudaMemcpy(result.grid.y.data(),
	               batch.y_coefficients(),
	               result.grid.y.size() * sizeof(int16_t),
	               cudaMemcpyDeviceToHost) != cudaSuccess) {
		throw std::runtime_error("failed to copy audited Y coefficients from CUDA");
	}
	if (!result.grid.cbcr.empty() &&
	    cudaMemcpy(result.grid.cbcr.data(),
	               batch.cbcr_coefficients(),
	               result.grid.cbcr.size() * sizeof(int16_t),
	               cudaMemcpyDeviceToHost) != cudaSuccess) {
		throw std::runtime_error("failed to copy audited CbCr coefficients from CUDA");
	}
	result.stats       = batch.execution_stats();
	result.cache_stats = batch.cache_stats();
	result.wall_ms =
	    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
	return result;
}

uint64_t fnv1a_append(uint64_t hash, const void* data, const size_t bytes) {
	const auto* input = static_cast<const uint8_t*>(data);
	for (size_t index = 0U; index < bytes; ++index) {
		hash ^= input[index];
		hash *= UINT64_C(1099511628211);
	}
	return hash;
}

std::string grid_hash(const HostGrid& grid) {
	uint64_t hash = UINT64_C(14695981039346656037);
	hash = fnv1a_append(hash, grid.y.data(), grid.y.size() * sizeof(int16_t));
	const uint8_t separator = 0xA5U;
	hash = fnv1a_append(hash, &separator, sizeof(separator));
	hash = fnv1a_append(hash, grid.cbcr.data(), grid.cbcr.size() * sizeof(int16_t));
	std::ostringstream output;
	output << std::hex << std::setw(16) << std::setfill('0') << hash;
	return output.str();
}

struct Difference {
	size_t  mismatches = 0U;
	int32_t max_abs    = 0;
};

Difference difference(const HostGrid& lhs, const HostGrid& rhs) {
	if (lhs.y.size() != rhs.y.size() || lhs.cbcr.size() != rhs.cbcr.size()) {
		return {std::numeric_limits<size_t>::max(), std::numeric_limits<int32_t>::max()};
	}
	Difference result;
	const auto compare = [&](const std::vector<int16_t>& left, const std::vector<int16_t>& right) {
		for (size_t index = 0U; index < left.size(); ++index) {
			const int32_t delta = static_cast<int32_t>(left[index]) - static_cast<int32_t>(right[index]);
			const int32_t absolute = delta < 0 ? -delta : delta;
			result.mismatches += absolute != 0;
			result.max_abs = std::max(result.max_abs, absolute);
		}
	};
	compare(lhs.y, rhs.y);
	compare(lhs.cbcr, rhs.cbcr);
	return result;
}

constexpr std::array<uint32_t, 5U> kGapReplayCapsBps {100U, 102U, 105U, 110U, 120U};

uint64_t checked_add_u64(const uint64_t lhs, const uint64_t rhs, const char* context) {
	if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
		throw std::overflow_error(context);
	}
	return lhs + rhs;
}

struct GapRef {
	uint64_t bytes          = 0U;
	uint32_t shard_id       = 0U;
	uint32_t rowgroup_index = 0U;
	uint32_t boundary_id    = 0U;
};

bool gap_less(const GapRef& lhs, const GapRef& rhs) {
	return std::tie(lhs.bytes, lhs.shard_id, lhs.rowgroup_index, lhs.boundary_id) <
	       std::tie(rhs.bytes, rhs.shard_id, rhs.rowgroup_index, rhs.boundary_id);
}

struct GapBudgetResult {
	uint32_t cap_bps                = 100U;
	uint64_t additional_budget      = 0U;
	uint64_t merged_gap_bytes       = 0U;
	uint64_t physical_bytes         = 0U;
	uint64_t run_count              = 0U;
	uint64_t merged_boundary_count  = 0U;
};

std::vector<GapBudgetResult> replay_gap_budgets(
	const uint64_t exact_bytes,
	const uint64_t exact_run_count,
	std::vector<GapRef> gaps) {
	std::sort(gaps.begin(), gaps.end(), gap_less);
	std::vector<GapBudgetResult> results;
	results.reserve(kGapReplayCapsBps.size());
	for (const auto cap_bps : kGapReplayCapsBps) {
		const auto extra_percent = static_cast<uint64_t>(cap_bps - 100U);
		if (extra_percent != 0U && exact_bytes > std::numeric_limits<uint64_t>::max() / extra_percent) {
			throw std::overflow_error("gap replay amplification budget overflow");
		}
		GapBudgetResult result;
		result.cap_bps           = cap_bps;
		result.additional_budget = exact_bytes * extra_percent / 100U;
		result.run_count         = exact_run_count;
		for (const auto& gap : gaps) {
			if (gap.bytes > result.additional_budget - result.merged_gap_bytes) {
				break;
			}
			result.merged_gap_bytes = checked_add_u64(
			    result.merged_gap_bytes, gap.bytes, "gap replay merged-byte overflow");
			++result.merged_boundary_count;
			--result.run_count;
		}
		result.physical_bytes = checked_add_u64(
		    exact_bytes, result.merged_gap_bytes, "gap replay physical-byte overflow");
		results.push_back(result);
	}
	return results;
}

uint32_t range_size_log2(uint64_t bytes) {
	if (bytes == 0U) {
		throw std::runtime_error("gap replay encountered an empty exact range");
	}
	uint32_t exponent = 0U;
	while (bytes >>= 1U) {
		++exponent;
	}
	return exponent;
}

uint64_t nearest_rank(std::vector<uint64_t> values, const uint32_t percentile) {
	if (values.empty()) {
		return 0U;
	}
	std::sort(values.begin(), values.end());
	const auto rank = (static_cast<uint64_t>(values.size()) * percentile + 99U) / 100U;
	return values[static_cast<size_t>(std::max<uint64_t>(rank, 1U) - 1U)];
}

void write_range_histogram(
	std::ostream& output,
	const std::map<uint32_t, uint64_t>& histogram) {
	output << '[';
	bool first = true;
	for (const auto& [exponent, count] : histogram) {
		if (!first) {
			output << ',';
		}
		first = false;
		const uint64_t lower = UINT64_C(1) << exponent;
		const uint64_t upper = exponent == 63U
		                           ? std::numeric_limits<uint64_t>::max()
		                           : (UINT64_C(1) << (exponent + 1U)) - 1U;
		output << "{\"min_bytes\":" << lower << ",\"max_bytes\":" << upper
		       << ",\"range_count\":" << count << '}';
	}
	output << ']';
}

void write_gap_quantiles(std::ostream& output, const std::vector<GapRef>& gaps) {
	std::vector<uint64_t> values;
	values.reserve(gaps.size());
	uint64_t maximum = 0U;
	for (const auto& gap : gaps) {
		values.push_back(gap.bytes);
		maximum = std::max(maximum, gap.bytes);
	}
	output << "{\"method\":\"nearest-rank\",\"gap_count\":" << values.size()
	       << ",\"p50_bytes\":" << nearest_rank(values, 50U)
	       << ",\"p90_bytes\":" << nearest_rank(values, 90U)
	       << ",\"p95_bytes\":" << nearest_rank(values, 95U)
	       << ",\"p99_bytes\":" << nearest_rank(values, 99U)
	       << ",\"max_bytes\":" << maximum << '}';
}

void write_gap_caps(
	std::ostream& output,
	const uint64_t exact_bytes,
	const uint64_t exact_run_count,
	const std::vector<GapRef>& gaps) {
	const auto results = replay_gap_budgets(exact_bytes, exact_run_count, gaps);
	output << '[';
	for (size_t index = 0U; index < results.size(); ++index) {
		if (index != 0U) {
			output << ',';
		}
		const auto& result = results[index];
		output << "{\"cap_bps\":" << result.cap_bps
		       << ",\"additional_budget_bytes\":" << result.additional_budget
		       << ",\"merged_gap_bytes\":" << result.merged_gap_bytes
		       << ",\"physical_bytes\":" << result.physical_bytes
		       << ",\"run_count\":" << result.run_count
		       << ",\"merged_boundary_count\":" << result.merged_boundary_count << '}';
	}
	output << ']';
}

struct GapReplayGroup {
	uint64_t exact_bytes        = 0U;
	uint64_t full_bytes         = 0U;
	uint64_t exact_range_count  = 0U;
	std::vector<GapRef> gaps;
	std::map<uint32_t, uint64_t> range_histogram;
};

struct GapReplayRowgroup : GapReplayGroup {
	uint32_t shard_id       = 0U;
	uint32_t rowgroup_index = 0U;
};

class GapReplayAudit {
public:
	void add(
	    const uint32_t shard_id,
	    const galp::format::SparseReadRecipeRowgroupStats& source) {
		if (source.rowgroup_index > std::numeric_limits<uint32_t>::max()) {
			throw std::overflow_error("gap replay rowgroup index exceeds uint32_t");
		}
		GapReplayRowgroup rowgroup;
		rowgroup.shard_id              = shard_id;
		rowgroup.rowgroup_index         = static_cast<uint32_t>(source.rowgroup_index);
		rowgroup.full_bytes             = source.rowgroup_storage_bytes;
		rowgroup.exact_range_count      = source.exact_ranges.size();
		uint64_t previous_end           = 0U;
		for (size_t index = 0U; index < source.exact_ranges.size(); ++index) {
			const auto& range = source.exact_ranges[index];
			const auto offset = static_cast<uint64_t>(range.offset);
			const auto bytes  = static_cast<uint64_t>(range.size);
			if (bytes == 0U || offset > rowgroup.full_bytes || bytes > rowgroup.full_bytes - offset ||
			    (index != 0U && offset <= previous_end)) {
				throw std::runtime_error("gap replay exact ranges are empty, non-monotonic, or out of bounds");
			}
			if (index != 0U) {
				rowgroup.gaps.push_back({
				    offset - previous_end,
				    shard_id,
				    rowgroup.rowgroup_index,
				    static_cast<uint32_t>(index - 1U)});
			}
			previous_end = offset + bytes;
			rowgroup.exact_bytes = checked_add_u64(
			    rowgroup.exact_bytes, bytes, "gap replay exact-byte overflow");
			++rowgroup.range_histogram[range_size_log2(bytes)];
		}
		if (rowgroup.exact_range_count == 0U || rowgroup.exact_bytes > rowgroup.full_bytes) {
			throw std::runtime_error("gap replay rowgroup has no exact ranges or exceeds full storage");
		}
		auto& shard = shards_[shard_id];
		append_group(shard, rowgroup);
		append_group(whole_run_, rowgroup);
		rowgroups_.push_back(std::move(rowgroup));
	}

	[[nodiscard]] size_t rowgroup_count() const { return rowgroups_.size(); }
	[[nodiscard]] uint64_t exact_range_count() const { return whole_run_.exact_range_count; }
	[[nodiscard]] uint64_t exact_bytes() const { return whole_run_.exact_bytes; }

	void write_json(const std::filesystem::path& path) const {
		std::ofstream output(path, std::ios::trunc);
		if (!output) {
			throw std::runtime_error("failed to open gap replay JSON: " + path.string());
		}
		output << std::setprecision(12)
		       << "{\"schema_version\":\"galp_sparse_gap_replay_v1\""
		       << ",\"budget_policy\":\"stable-smallest-gap-first\""
		       << ",\"tie_break\":\"gap_size,shard_id,rowgroup_id,boundary_id\""
		       << ",\"rowgroup_count\":" << rowgroups_.size()
		       << ",\"shard_count\":" << shards_.size()
		       << ",\"whole_run\":";
		write_group(output, whole_run_);
		output << ",\"grouped_pareto_frontier\":";
		write_grouped_frontier(output, whole_run_);
		output << ",\"shards\":[";
		bool first = true;
		for (const auto& [shard_id, shard] : shards_) {
			if (!first) {
				output << ',';
			}
			first = false;
			output << "{\"shard_id\":" << shard_id << ",\"budget_scope\":\"whole-shard\",\"summary\":";
			write_group(output, shard);
			output << '}';
		}
		output << "],\"rowgroups\":[";
		for (size_t index = 0U; index < rowgroups_.size(); ++index) {
			if (index != 0U) {
				output << ',';
			}
			const auto& rowgroup = rowgroups_[index];
			output << "{\"shard_id\":" << rowgroup.shard_id
			       << ",\"rowgroup_index\":" << rowgroup.rowgroup_index
			       << ",\"budget_scope\":\"rowgroup-local\",\"summary\":";
			write_group(output, rowgroup);
			output << '}';
		}
		output << "]}";
		output.close();
		if (!output) {
			throw std::runtime_error("failed to write gap replay JSON: " + path.string());
		}
	}

private:
	static void append_group(GapReplayGroup& destination, const GapReplayGroup& source) {
		destination.exact_bytes = checked_add_u64(
		    destination.exact_bytes, source.exact_bytes, "gap replay aggregate exact-byte overflow");
		destination.full_bytes = checked_add_u64(
		    destination.full_bytes, source.full_bytes, "gap replay aggregate full-byte overflow");
		destination.exact_range_count = checked_add_u64(
		    destination.exact_range_count,
		    source.exact_range_count,
		    "gap replay aggregate range-count overflow");
		destination.gaps.insert(destination.gaps.end(), source.gaps.begin(), source.gaps.end());
		for (const auto& [bucket, count] : source.range_histogram) {
			destination.range_histogram[bucket] = checked_add_u64(
			    destination.range_histogram[bucket], count, "gap replay histogram overflow");
		}
	}

	static void write_group(std::ostream& output, const GapReplayGroup& group) {
		output << "{\"exact_range_count\":" << group.exact_range_count
		       << ",\"exact_bytes\":" << group.exact_bytes
		       << ",\"full_rowgroup_bytes\":" << group.full_bytes
		       << ",\"range_size_histogram_log2\":";
		write_range_histogram(output, group.range_histogram);
		output << ",\"gap_quantiles\":";
		write_gap_quantiles(output, group.gaps);
		output << ",\"caps\":";
		write_gap_caps(output, group.exact_bytes, group.exact_range_count, group.gaps);
		output << '}';
	}

	static void write_grouped_frontier(std::ostream& output, const GapReplayGroup& group) {
		auto gaps = group.gaps;
		std::sort(gaps.begin(), gaps.end(), gap_less);
		uint64_t merged_bytes = 0U;
		uint64_t merged_count = 0U;
		output << "[{\"max_merged_gap_bytes\":0,\"merged_boundary_count\":0"
		       << ",\"physical_bytes\":" << group.exact_bytes
		       << ",\"run_count\":" << group.exact_range_count << '}';
		for (size_t begin = 0U; begin < gaps.size();) {
			size_t end = begin + 1U;
			while (end < gaps.size() && gaps[end].bytes == gaps[begin].bytes) {
				++end;
			}
			for (size_t index = begin; index < end; ++index) {
				merged_bytes = checked_add_u64(
				    merged_bytes, gaps[index].bytes, "gap replay frontier overflow");
			}
			merged_count += end - begin;
			output << ",{\"max_merged_gap_bytes\":" << gaps[begin].bytes
			       << ",\"merged_boundary_count\":" << merged_count
			       << ",\"physical_bytes\":" << checked_add_u64(
			              group.exact_bytes, merged_bytes, "gap replay frontier physical-byte overflow")
			       << ",\"run_count\":" << group.exact_range_count - merged_count << '}';
			begin = end;
		}
		output << ']';
	}

	std::vector<GapReplayRowgroup> rowgroups_;
	std::map<uint32_t, GapReplayGroup> shards_;
	GapReplayGroup whole_run_;
};

} // namespace

int main(const int argc, char** argv) try {
	const auto options = parse_options(argc, argv);
	const auto load_begin = std::chrono::steady_clock::now();
	galp::jpeg::JpegDctBlockMajorCompactPlanner planner(options.manifest, options.descriptor_directory);
	const auto load_end = std::chrono::steady_clock::now();
	if (options.image_ids.empty() &&
	    (options.start >= planner.image_count() || options.count > planner.image_count())) {
		throw std::runtime_error("request range exceeds dataset image count");
	}
	std::vector<uint32_t> image_ids = options.image_ids;
	image_ids.reserve(options.image_ids.empty() ? options.count : options.image_ids.size());
	if (!options.image_ids.empty()) {
		for (const auto image_id : image_ids) {
			if (image_id >= planner.image_count()) {
				throw std::runtime_error("--image-ids contains an image outside the dataset");
			}
		}
	} else if (options.pattern == "random") {
		std::mt19937 generator(options.seed);
		std::uniform_int_distribution<uint32_t> distribution(0U, static_cast<uint32_t>(planner.image_count() - 1U));
		for (uint32_t index = 0U; index < options.count; ++index) {
			image_ids.push_back(distribution(generator));
		}
	} else if (options.pattern == "duplicates") {
		for (uint32_t index = 0U; index < options.count; ++index) {
			image_ids.push_back(options.start + (index / 3U) % std::max<uint32_t>(1U, options.count / 4U));
		}
	} else {
		if (options.start > planner.image_count() - options.count) {
			throw std::runtime_error("sequential request range exceeds dataset image count");
		}
		for (uint32_t index = 0U; index < options.count; ++index) {
			image_ids.push_back(options.start + index);
		}
		if (options.pattern == "reverse") {
			std::reverse(image_ids.begin(), image_ids.end());
		}
	}
	std::vector<galp::jpeg::JpegDctImageCropRequest> requests;
	requests.reserve(image_ids.size());
	std::unique_ptr<galp::jpeg::JpegDctShardDatasetReader> metadata_reader;
	if (options.explicit_crops) {
		metadata_reader = std::make_unique<galp::jpeg::JpegDctShardDatasetReader>(options.manifest);
	}
	for (size_t request_index = 0U; request_index < image_ids.size(); ++request_index) {
		const auto image = image_ids[request_index];
		galp::jpeg::JpegDctCropBox crop;
		if (metadata_reader) {
			const auto metadata = metadata_reader->ImageMetadata(image);
			crop.width  = std::min<uint32_t>(224U, metadata.image_width);
			crop.height = std::min<uint32_t>(224U, metadata.image_height);
			if (crop.width == 0U || crop.height == 0U) {
				throw std::runtime_error("explicit-crop audit found an image with zero dimensions");
			}
			const auto x_slack = metadata.image_width - crop.width;
			const auto y_slack = metadata.image_height - crop.height;
			crop.x = x_slack == 0U
			             ? 0U
			             : static_cast<uint32_t>((static_cast<uint64_t>(image) * 37U + request_index * 17U) %
			                                     (static_cast<uint64_t>(x_slack) + 1U));
			crop.y = y_slack == 0U
			             ? 0U
			             : static_cast<uint32_t>((static_cast<uint64_t>(image) * 29U + request_index * 23U) %
			                                     (static_cast<uint64_t>(y_slack) + 1U));
		}
		requests.push_back({image, crop, options.explicit_crops && request_index % 2U != 0U, {}, {}});
	}
	auto transform = galp::profiles::rgbnomore_val_dct_grid_transform();
	transform.crop_reference_width_blocks = options.crop_reference_blocks;
	transform.crop_reference_height_blocks = options.crop_reference_blocks;
	if (options.rgbnomore_production_fp32) {
		// Match galp/torch/rgbnomore_dct_profile.py exactly.  These fields are
		// part of the canonical sidecar identity even though they do not change
		// the compact selected-vector payload, so the offline builder must use
		// the same profile as production.
		transform.output_data_type = galp::jpeg::JpegDctGridOutputDataType::kFloat32;
		transform.output_add       = 4.0F;
		transform.output_scale     = 1.0F / 1020.0F;
	}
	const auto plan_begin = std::chrono::steady_clock::now();
	const auto plan = planner.Plan(requests, transform);
	const auto plan_end = std::chrono::steady_clock::now();
	const auto compact_selected = compact_vectors(plan);
	size_t canonical_template_shard_count       = 0U;
	size_t canonical_template_reused_shard_count = 0U;
	size_t canonical_template_runtime_hit_count  = 0U;
	size_t canonical_template_sidecar_bytes      = 0U;
	size_t canonical_template_request_count      = 0U;
	size_t canonical_template_rowgroup_count     = 0U;
	size_t canonical_template_vector_run_count   = 0U;
	if (!options.canonical_plan_directory.empty()) {
		std::map<uint32_t, std::vector<galp::jpeg::JpegDctImageCropRequest>> canonical_requests;
		for (const auto& request : plan.requests) {
			canonical_requests[request.shard_id].push_back({request.global_image_index, {}, false, {}, {}});
		}
		for (const auto& [shard_id, shard_requests] : canonical_requests) {
			const auto stats = planner.WriteCanonicalShardPlanTemplate(
			    shard_id, transform, options.canonical_plan_directory);
			++canonical_template_shard_count;
			canonical_template_reused_shard_count += stats.reused_existing ? 1U : 0U;
			canonical_template_sidecar_bytes += stats.sidecar_bytes;
			canonical_template_request_count += stats.request_count;
			canonical_template_rowgroup_count += stats.rowgroup_count;
			canonical_template_vector_run_count += stats.vector_run_count;
			const auto hit_plan = planner.Plan(shard_requests, transform);
			canonical_template_runtime_hit_count += hit_plan.stats.canonical_template_hit_count;
		}
	}
	size_t recipe_shard_count            = 0U;
	size_t recipe_reused_shard_count     = 0U;
	size_t recipe_rowgroup_count         = 0U;
	size_t recipe_selected_vector_count  = 0U;
	size_t recipe_exact_range_count      = 0U;
	size_t recipe_exact_storage_bytes    = 0U;
	size_t recipe_sidecar_bytes          = 0U;
	GapReplayAudit gap_replay;
	if (!options.sparse_recipe_directory.empty()) {
		std::map<uint32_t, std::map<uint32_t, std::vector<uint32_t>>> vectors_by_shard_rowgroup;
		for (const auto& run : plan.vector_runs) {
			auto& selected = vectors_by_shard_rowgroup[run.shard_id][run.rowgroup_index];
			for (uint32_t vector = run.first_vector; vector < run.first_vector + run.vector_count; ++vector) {
				selected.push_back(vector);
			}
		}
		for (const auto& [shard_id, rowgroups] : vectors_by_shard_rowgroup) {
			std::vector<galp::format::SparseReadRecipeSelection> selections;
			selections.reserve(rowgroups.size());
			for (const auto& [rowgroup_index, selected_vectors] : rowgroups) {
				selections.push_back({rowgroup_index, selected_vectors});
			}
			const auto source_fingerprint = planner.source_payload_crc64(shard_id);
			const auto stats = galp::format::write_sparse_read_recipe(
			    planner.source_fls_path(shard_id),
			    galp::format::sparse_read_recipe_path(options.sparse_recipe_directory, shard_id),
			    source_fingerprint,
			    selections);
			++recipe_shard_count;
			recipe_reused_shard_count += stats.reused_existing ? 1U : 0U;
			recipe_rowgroup_count += stats.rowgroup_count;
			recipe_selected_vector_count += stats.selected_vector_count;
			recipe_exact_range_count += stats.exact_range_count;
			recipe_exact_storage_bytes += stats.exact_storage_bytes;
			recipe_sidecar_bytes += stats.sidecar_bytes;
			if (!options.gap_replay_json.empty()) {
				for (const auto& rowgroup : stats.rowgroups) {
					gap_replay.add(shard_id, rowgroup);
				}
			}
		}
		constexpr size_t kRecipePersistentHardCap = 16U * 1024U * 1024U;
		if (recipe_sidecar_bytes > kRecipePersistentHardCap) {
			throw std::runtime_error("sparse recipe total exceeds the 16 MiB persistent-storage gate");
		}
		if (recipe_selected_vector_count != compact_selected.size()) {
			throw std::runtime_error("sparse recipe selected-vector total does not match the canonical compact plan");
		}
		if (!options.gap_replay_json.empty()) {
			if (gap_replay.rowgroup_count() != recipe_rowgroup_count ||
			    gap_replay.exact_range_count() != recipe_exact_range_count ||
			    gap_replay.exact_bytes() != recipe_exact_storage_bytes) {
				throw std::runtime_error("gap replay totals do not match sparse recipe totals");
			}
			gap_replay.write_json(options.gap_replay_json);
		}
	}
	if (canonical_template_sidecar_bytes > std::numeric_limits<size_t>::max() - recipe_sidecar_bytes ||
	    canonical_template_sidecar_bytes + recipe_sidecar_bytes > 16U * 1024U * 1024U) {
		throw std::runtime_error("canonical plan plus sparse recipe exceeds the 16 MiB persistent-storage gate");
	}
	const bool grayscale_requirement_met =
	    !options.require_grayscale ||
	    std::any_of(plan.requests.begin(), plan.requests.end(), [](const auto& request) {
		    return request.components[0].present && !request.components[1].present && !request.components[2].present;
	    });
	std::set<uint32_t> requested_shards;
	for (const auto& request : plan.requests) {
		requested_shards.insert(request.shard_id);
	}
	const bool cross_shard_requirement_met = !options.require_cross_shard || requested_shards.size() > 1U;
	if (!grayscale_requirement_met) {
		throw std::runtime_error("--require-grayscale request set contains no Y-only image");
	}
	if (!cross_shard_requirement_met) {
		throw std::runtime_error("--require-cross-shard request set touches fewer than two shards");
	}

	double legacy_ms = 0.0;
	bool vectors_equal = true;
	size_t legacy_selected_vectors = 0U;
	size_t legacy_expanded_items = 0U;
	size_t legacy_sort_items = 0U;
	if (options.compare_legacy) {
		galp::jpeg::JpegDctShardDatasetReader legacy(options.manifest);
		galp::jpeg::JpegDctDeviceBatchOptions legacy_options;
		legacy_options.layout = galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid;
		legacy_options.grid_transform = transform;
		legacy_options.enable_planless_execution = false;
		legacy_options.crop_execution_mode = galp::jpeg::JpegDctCropExecutionMode::kRowgroupReadSelectedDecode;
		const auto legacy_begin = std::chrono::steady_clock::now();
		const auto preview = legacy.PlanDeviceDctBatch(requests, legacy_options);
		const auto legacy_end = std::chrono::steady_clock::now();
		legacy_ms = std::chrono::duration<double, std::milli>(legacy_end - legacy_begin).count();
		const auto legacy_selected = legacy_vectors(preview);
		vectors_equal = legacy_selected == compact_selected;
		legacy_selected_vectors = legacy_selected.size();
		legacy_expanded_items = preview.host_expanded_transform_items_created;
		legacy_sort_items = preview.host_global_transform_sort_items;
	}

	bool execution_selected = false;
	bool execution_deterministic = false;
	bool execution_within_tolerance = false;
	bool execution_zero_expansion = false;
	bool execution_kernel_launched = false;
	bool execution_strategy_accounting_valid = false;
	bool execution_descriptor_bounds = false;
	bool execution_workset_bounds = false;
	bool execution_io_bounds = false;
	bool execution_resource_bounds = false;
	bool execution_active_output_schedule_valid = false;
	bool execution_cache_contract_valid = false;
	double planless_wall_ms = 0.0;
	double repeat_wall_ms = 0.0;
	double legacy_wall_ms = 0.0;
	std::string planless_hash;
	std::string repeat_hash;
	std::string legacy_hash;
	Difference planless_legacy_difference;
	galp::jpeg::JpegDctDeviceExecutionStats planless_execution_stats;
	galp::jpeg::JpegDctDeviceExecutionStats legacy_execution_stats;
	galp::jpeg::JpegDctDeviceCacheStats     planless_cache_stats;
	if (options.execute) {
		int device_count = 0;
		if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
			throw std::runtime_error("--execute requested but CUDA device is unavailable");
		}
		const auto descriptor_path = std::filesystem::absolute(options.descriptor_directory).string();
		if (setenv("GALP_BLOCK_MAJOR_ACCESS_DIR", descriptor_path.c_str(), 1) != 0) {
			throw std::runtime_error("failed to set GALP_BLOCK_MAJOR_ACCESS_DIR");
		}
		galp::jpeg::JpegDctDeviceBatchOptions device_options;
		device_options.layout                    = galp::jpeg::JpegDctDeviceLayout::kTransformedDctGrid;
		device_options.grid_transform            = transform;
		device_options.cache_capacity_bytes      = 0U;
		device_options.plan_cache_capacity       = 0U;
		device_options.decode_batch_rowgroups    = options.decode_batch_rowgroups;
		device_options.enable_rowgroup_prefetch  = options.prefetch_workers > 1U;
		device_options.rowgroup_prefetch_workers = std::max<uint32_t>(1U, options.prefetch_workers);
		device_options.rowgroup_prefetch_depth   = 1U;
		device_options.decode_workset_capacity_bytes =
		    static_cast<size_t>(options.workset_capacity_mib) * 1024U * 1024U;
		device_options.enable_planless_execution = true;
		device_options.crop_execution_mode =
		    galp::jpeg::JpegDctCropExecutionMode::kRowgroupReadSelectedDecode;
		DeviceRun planless;
		DeviceRun repeat;
		{
			galp::jpeg::JpegDctShardDatasetReader planless_reader(options.manifest);
			const auto preview = planless_reader.PlanDeviceDctBatch(requests, device_options);
			execution_selected = preview.uses_planless_fixed_transform;
			execution_zero_expansion = preview.host_expanded_transform_items_created == 0U &&
			                           preview.host_output_block_source_lists_created == 0U &&
			                           preview.host_global_transform_sort_items == 0U;
			planless = execute_device_batch(planless_reader, requests, device_options);
			repeat   = execute_device_batch(planless_reader, requests, device_options);
		}
		if (unsetenv("GALP_BLOCK_MAJOR_ACCESS_DIR") != 0) {
			throw std::runtime_error("failed to clear GALP_BLOCK_MAJOR_ACCESS_DIR before legacy control");
		}
		auto legacy_options                      = device_options;
		legacy_options.enable_planless_execution = false;
		galp::jpeg::JpegDctShardDatasetReader legacy_reader(options.manifest);
		auto legacy = execute_device_batch(legacy_reader, requests, legacy_options);

		planless_wall_ms = planless.wall_ms;
		repeat_wall_ms   = repeat.wall_ms;
		legacy_wall_ms   = legacy.wall_ms;
		planless_hash    = grid_hash(planless.grid);
		repeat_hash      = grid_hash(repeat.grid);
		legacy_hash      = grid_hash(legacy.grid);
		execution_deterministic = planless.grid.y == repeat.grid.y && planless.grid.cbcr == repeat.grid.cbcr;
		planless_legacy_difference = difference(planless.grid, legacy.grid);
		execution_within_tolerance = planless_legacy_difference.max_abs <= 1;
		planless_execution_stats = planless.stats;
		legacy_execution_stats   = legacy.stats;
		planless_cache_stats      = planless.cache_stats;
		execution_zero_expansion = execution_zero_expansion &&
		                           planless.stats.host_expanded_transform_items_created == 0U &&
		                           planless.stats.host_output_block_source_lists_created == 0U &&
		                           planless.stats.host_global_transform_sort_items == 0U;
		execution_kernel_launched = planless.stats.planless_transform_kernel_launch_count > 0U;
		execution_strategy_accounting_valid =
		    planless.stats.rowgroup_count > 0U &&
		    planless.stats.run_interval_exact_rowgroup_count + planless.stats.bitmap_exact_rowgroup_count +
		            planless.stats.full_rowgroup_strategy_count ==
		        planless.stats.rowgroup_count;
		const auto expected_workset_capacity =
		    static_cast<size_t>(options.workset_capacity_mib) * 1024U * 1024U;
		execution_descriptor_bounds =
		    planless.stats.compact_plan_bytes > 0U &&
		    planless.stats.compact_plan_bytes <= planless.stats.compact_plan_peak_bytes &&
		    planless.stats.planless_image_descriptor_count == requests.size();
		execution_active_output_schedule_valid =
		    planless.stats.planless_transform_full_scan_output_block_count > 0U &&
		    planless.stats.planless_transform_output_block_count > 0U &&
		    planless.stats.planless_transform_output_block_count +
		            planless.stats.planless_transform_skipped_output_block_count ==
		        planless.stats.planless_transform_full_scan_output_block_count &&
		    planless.stats.planless_transform_active_output_index_bytes ==
		        planless.stats.planless_transform_output_block_count * sizeof(uint32_t) &&
		    planless.stats.planless_transform_active_output_schedule_build_count == 1U &&
		    planless.stats.planless_transform_active_output_workset_count == planless.stats.workset_count &&
		    planless.stats.planless_transform_active_output_offset_bytes ==
		        (planless.stats.workset_count + 1U) * sizeof(uint64_t) &&
		    planless.stats.planless_transform_active_output_offsets_valid;
		execution_workset_bounds =
		    planless.stats.decode_workset_capacity_bytes == expected_workset_capacity &&
		    planless.stats.max_estimated_decode_workset_bytes <= expected_workset_capacity &&
		    planless.stats.bounded_double_buffer_peak_estimated_bytes <= expected_workset_capacity &&
		    planless.stats.oversized_decode_rowgroup_count == 0U && planless.stats.workset_count > 0U;
		execution_io_bounds =
		    planless.stats.host_io_staged_rowgroups == 0U &&
		    planless.stats.actual_vector_count <= planless.stats.full_vector_count &&
		    planless.stats.compressed_payload_bytes_read <= planless.stats.full_compressed_payload_bytes;
		execution_resource_bounds = execution_descriptor_bounds && execution_workset_bounds && execution_io_bounds &&
		                            execution_active_output_schedule_valid;
		execution_cache_contract_valid =
		    !planless.stats.cache_enabled && !planless.stats.exact_batch_plan_cache_enabled &&
		    planless.stats.plan_cache_hits == 0U && planless.stats.plan_cache_misses == 0U &&
		    planless.stats.plan_cache_evictions == 0U && planless.stats.sparse_vector_cache_hits == 0U &&
		    planless.stats.sparse_vector_cache_misses == 0U && planless.stats.dct_resize_weight_cache_hits == 0U &&
		    planless.stats.dct_resize_weight_cache_misses == 0U &&
		    planless.stats.dct_conversion_matrix_cache_hits == 0U &&
		    planless.stats.dct_conversion_matrix_cache_misses == 0U && planless.cache_stats.capacity_bytes == 0U &&
		    planless.cache_stats.resident_bytes == 0U && planless.cache_stats.peak_resident_bytes == 0U &&
		    planless.cache_stats.resident_rowgroups == 0U && planless.cache_stats.peak_resident_rowgroups == 0U &&
		    planless.cache_stats.hits == 0U && planless.cache_stats.misses == 0U &&
		    planless.cache_stats.inserts == 0U && planless.cache_stats.evictions == 0U;
	}

	const auto load_ms = std::chrono::duration<double, std::milli>(load_end - load_begin).count();
	const auto compact_ms = std::chrono::duration<double, std::milli>(plan_end - plan_begin).count();
	std::ostringstream json;
	json << std::setprecision(12)
	     << "{\"schema_version\":\"galp_block_major_plan_audit_v1\",\"pattern\":\"" << options.pattern
	     << "\",\"request_source\":\"" << (options.image_ids.empty() ? "pattern" : "explicit-image-ids")
	     << "\",\"explicit_crops\":" << (options.explicit_crops ? "true" : "false")
	     << ",\"require_grayscale\":" << (options.require_grayscale ? "true" : "false")
	     << ",\"grayscale_requirement_met\":" << (grayscale_requirement_met ? "true" : "false")
	     << ",\"require_cross_shard\":" << (options.require_cross_shard ? "true" : "false")
	     << ",\"cross_shard_requirement_met\":" << (cross_shard_requirement_met ? "true" : "false")
	     << ",\"requested_shard_count\":" << requested_shards.size() << ",\"request_count\":" << requests.size()
	     << ",\"decode_batch_rowgroups\":" << options.decode_batch_rowgroups
	     << ",\"prefetch_workers\":" << options.prefetch_workers
	     << ",\"workset_capacity_mib\":" << options.workset_capacity_mib
	     << ",\"crop_reference_blocks\":" << options.crop_reference_blocks
	     << ",\"canonical_transform_profile\":\""
	     << (options.rgbnomore_production_fp32 ? "rgbnomore-production-fp32" : "rgbnomore-int16") << "\""
	     << ",\"descriptor_load_ms\":" << load_ms
	     << ",\"compact_planning_ms\":" << compact_ms << ",\"compact_plan_bytes\":"
	     << plan.stats.compact_plan_bytes << ",\"compact_plan_peak_bytes\":" << plan.stats.compact_plan_peak_bytes
	     << ",\"unique_image_count\":" << plan.stats.unique_image_count << ",\"duplicate_output_count\":"
	     << plan.stats.duplicate_output_count << ",\"request_sort_items\":" << plan.stats.request_sort_items
	     << ",\"touched_block_groups\":" << plan.stats.touched_block_groups << ",\"group_rank_runs\":"
	     << plan.stats.group_rank_runs << ",\"selected_rowgroups\":" << plan.stats.selected_rowgroups
	     << ",\"selected_vector_runs\":" << plan.stats.selected_vector_runs << ",\"selected_vectors\":"
	     << compact_selected.size() << ",\"sparse_recipe_requested\":"
	     << (!options.sparse_recipe_directory.empty() ? "true" : "false")
	     << ",\"gap_replay_requested\":" << (!options.gap_replay_json.empty() ? "true" : "false")
	     << ",\"canonical_template_requested\":"
	     << (!options.canonical_plan_directory.empty() ? "true" : "false")
	     << ",\"canonical_template_shard_count\":" << canonical_template_shard_count
	     << ",\"canonical_template_reused_shard_count\":" << canonical_template_reused_shard_count
	     << ",\"canonical_template_runtime_hit_count\":" << canonical_template_runtime_hit_count
	     << ",\"canonical_template_request_count\":" << canonical_template_request_count
	     << ",\"canonical_template_rowgroup_count\":" << canonical_template_rowgroup_count
	     << ",\"canonical_template_vector_run_count\":" << canonical_template_vector_run_count
	     << ",\"canonical_template_sidecar_bytes\":" << canonical_template_sidecar_bytes
	     << ",\"sparse_recipe_shard_count\":" << recipe_shard_count
	     << ",\"sparse_recipe_reused_shard_count\":" << recipe_reused_shard_count
	     << ",\"sparse_recipe_rowgroup_count\":" << recipe_rowgroup_count
	     << ",\"sparse_recipe_selected_vector_count\":" << recipe_selected_vector_count
	     << ",\"sparse_recipe_exact_range_count\":" << recipe_exact_range_count
	     << ",\"sparse_recipe_exact_storage_bytes\":" << recipe_exact_storage_bytes
	     << ",\"sparse_recipe_sidecar_bytes\":" << recipe_sidecar_bytes
	     << ",\"p1_persistent_sidecar_bytes\":"
	     << canonical_template_sidecar_bytes + recipe_sidecar_bytes
	     << ",\"sparse_recipe_persistent_hard_cap_bytes\":" << 16U * 1024U * 1024U
	     << ",\"sparse_recipe_storage_gate_passed\":"
	     << (recipe_sidecar_bytes <= 16U * 1024U * 1024U ? "true" : "false")
	     << ",\"duplicate_physical_read_count\":"
	     << plan.stats.duplicate_physical_read_count << ",\"rowgroup_revisit_count\":"
	     << plan.stats.rowgroup_revisit_count << ",\"vector_run_revisit_count\":"
	     << plan.stats.vector_run_revisit_count << ",\"physical_read_order_inversions\":"
	     << plan.stats.physical_read_order_inversions << ",\"touched_rank_cells\":" << plan.stats.touched_rank_cells
		     << ",\"rank_payload_bytes\":" << plan.stats.rank_payload_bytes
		     << ",\"touched_quant_tables\":" << plan.stats.touched_quant_tables
	     << ",\"expanded_transform_items\":" << plan.stats.expanded_transform_items
	     << ",\"global_transform_sort_items\":" << plan.stats.global_transform_sort_items
	     << ",\"legacy_compared\":" << (options.compare_legacy ? "true" : "false")
	     << ",\"legacy_planning_ms\":" << legacy_ms << ",\"legacy_selected_vectors\":"
	     << legacy_selected_vectors << ",\"legacy_expanded_transform_items\":" << legacy_expanded_items
	     << ",\"legacy_sort_items\":" << legacy_sort_items << ",\"selected_vectors_equal\":"
	     << (vectors_equal ? "true" : "false") << ",\"execution_requested\":"
	     << (options.execute ? "true" : "false") << ",\"planless_execution_selected\":"
	     << (execution_selected ? "true" : "false") << ",\"planless_zero_expansion\":"
	     << (execution_zero_expansion ? "true" : "false") << ",\"planless_repeat_deterministic\":"
	     << (execution_deterministic ? "true" : "false") << ",\"planless_legacy_within_tolerance\":"
	     << (execution_within_tolerance ? "true" : "false") << ",\"planless_legacy_mismatch_count\":"
	     << planless_legacy_difference.mismatches << ",\"planless_legacy_max_abs_difference\":"
	     << planless_legacy_difference.max_abs << ",\"execution_kernel_launched\":"
	     << (execution_kernel_launched ? "true" : "false")
	     << ",\"execution_strategy_accounting_valid\":"
	     << (execution_strategy_accounting_valid ? "true" : "false")
	     << ",\"execution_descriptor_bounds\":" << (execution_descriptor_bounds ? "true" : "false")
	     << ",\"execution_workset_bounds\":" << (execution_workset_bounds ? "true" : "false")
	     << ",\"execution_io_bounds\":" << (execution_io_bounds ? "true" : "false")
	     << ",\"execution_resource_bounds\":" << (execution_resource_bounds ? "true" : "false")
	     << ",\"execution_active_output_schedule_valid\":"
	     << (execution_active_output_schedule_valid ? "true" : "false")
	     << ",\"execution_cache_contract_valid\":"
	     << (execution_cache_contract_valid ? "true" : "false") << ",\"planless_wall_ms\":" << planless_wall_ms
	     << ",\"planless_repeat_wall_ms\":" << repeat_wall_ms << ",\"legacy_wall_ms\":" << legacy_wall_ms
	     << ",\"planless_hash\":\"" << planless_hash << "\",\"planless_repeat_hash\":\"" << repeat_hash
	     << "\",\"legacy_hash\":\"" << legacy_hash << "\",\"execution_rowgroups\":"
	     << planless_execution_stats.rowgroup_count << ",\"execution_worksets\":"
	     << planless_execution_stats.workset_count
	     << ",\"execution_planless_image_descriptor_count\":"
	     << planless_execution_stats.planless_image_descriptor_count
	     << ",\"execution_compact_plan_bytes\":" << planless_execution_stats.compact_plan_bytes
	     << ",\"execution_compact_plan_peak_bytes\":" << planless_execution_stats.compact_plan_peak_bytes
	     << ",\"execution_coordinate_group_lookup_count\":"
	     << planless_execution_stats.coordinate_group_lookup_count
	     << ",\"execution_coordinate_group_index_entries\":"
	     << planless_execution_stats.coordinate_group_index_entries
	     << ",\"execution_coordinate_group_index_populated\":"
	     << planless_execution_stats.coordinate_group_index_populated
	     << ",\"execution_coordinate_group_index_holes\":"
	     << planless_execution_stats.coordinate_group_index_holes
	     << ",\"execution_coordinate_group_index_bytes\":"
	     << planless_execution_stats.coordinate_group_index_bytes
	     << ",\"execution_coordinate_group_index_density\":"
	     << planless_execution_stats.coordinate_group_index_density
	     << ",\"execution_host_io_staged_rowgroups\":" << planless_execution_stats.host_io_staged_rowgroups
	     << ",\"execution_actual_vectors\":"
	     << planless_execution_stats.actual_vector_count << ",\"execution_full_vectors\":"
	     << planless_execution_stats.full_vector_count << ",\"execution_compressed_bytes_read\":"
	     << planless_execution_stats.compressed_payload_bytes_read
	     << ",\"execution_selected_compressed_bytes\":"
	     << planless_execution_stats.selected_compressed_payload_bytes
	     << ",\"execution_full_compressed_bytes\":" << planless_execution_stats.full_compressed_payload_bytes
	     << ",\"execution_read_amplification\":" << planless_execution_stats.read_amplification
	     << ",\"execution_duplicate_physical_read_count\":"
	     << planless_execution_stats.duplicate_physical_read_count
	     << ",\"execution_rowgroup_revisit_count\":"
	     << planless_execution_stats.rowgroup_revisit_count
	     << ",\"execution_vector_run_revisit_count\":"
	     << planless_execution_stats.vector_run_revisit_count
	     << ",\"execution_physical_read_order_inversions\":"
	     << planless_execution_stats.physical_read_order_inversions
	     << ",\"execution_gpu_peak_bytes\":" << planless_execution_stats.galp_native_device_peak_in_use_bytes
	     << ",\"execution_pinned_peak_bytes\":" << planless_execution_stats.galp_native_pinned_peak_in_use_bytes
	     << ",\"legacy_execution_gpu_peak_bytes\":" << legacy_execution_stats.galp_native_device_peak_in_use_bytes
	     << ",\"legacy_execution_pinned_peak_bytes\":"
	     << legacy_execution_stats.galp_native_pinned_peak_in_use_bytes
	     << ",\"execution_decode_workset_capacity_bytes\":"
	     << planless_execution_stats.decode_workset_capacity_bytes
	     << ",\"execution_max_estimated_decode_workset_bytes\":"
	     << planless_execution_stats.max_estimated_decode_workset_bytes
	     << ",\"execution_oversized_decode_rowgroups\":"
	     << planless_execution_stats.oversized_decode_rowgroup_count
	     << ",\"execution_run_interval_exact_rowgroups\":"
	     << planless_execution_stats.run_interval_exact_rowgroup_count
	     << ",\"execution_bitmap_exact_rowgroups\":"
	     << planless_execution_stats.bitmap_exact_rowgroup_count
	     << ",\"execution_full_rowgroup_strategy_count\":"
	     << planless_execution_stats.full_rowgroup_strategy_count
	     << ",\"execution_bounded_double_buffer_enabled\":"
	     << (planless_execution_stats.bounded_double_buffer_enabled ? "true" : "false")
	     << ",\"execution_bounded_double_buffer_policy\":\""
	     << planless_execution_stats.bounded_double_buffer_policy << "\""
	     << ",\"execution_bounded_double_buffer_candidate\":"
	     << (planless_execution_stats.bounded_double_buffer_candidate ? "true" : "false")
	     << ",\"execution_planless_transform_full_scan_output_blocks\":"
	     << planless_execution_stats.planless_transform_full_scan_output_block_count
	     << ",\"execution_planless_transform_active_output_blocks\":"
	     << planless_execution_stats.planless_transform_output_block_count
	     << ",\"execution_planless_transform_skipped_output_blocks\":"
	     << planless_execution_stats.planless_transform_skipped_output_block_count
	     << ",\"execution_planless_transform_active_output_index_bytes\":"
	     << planless_execution_stats.planless_transform_active_output_index_bytes
	     << ",\"execution_planless_transform_active_output_offset_bytes\":"
	     << planless_execution_stats.planless_transform_active_output_offset_bytes
	     << ",\"execution_planless_transform_active_output_schedule_peak_bytes\":"
	     << planless_execution_stats.planless_transform_active_output_schedule_peak_bytes
	     << ",\"execution_planless_transform_source_contribution_count\":"
	     << planless_execution_stats.planless_transform_source_contribution_count
	     << ",\"execution_planless_transform_source_contribution_visit_count\":"
	     << planless_execution_stats.planless_transform_source_contribution_visit_count
	     << ",\"execution_planless_transform_output_workset_ownership_count\":"
	     << planless_execution_stats.planless_transform_output_workset_ownership_count
	     << ",\"execution_planless_transform_active_output_workset_count\":"
	     << planless_execution_stats.planless_transform_active_output_workset_count
	     << ",\"execution_planless_transform_active_output_schedule_build_count\":"
	     << planless_execution_stats.planless_transform_active_output_schedule_build_count
	     << ",\"execution_planless_transform_active_output_offsets_valid\":"
	     << (planless_execution_stats.planless_transform_active_output_offsets_valid ? "true" : "false")
	     << ",\"execution_planless_transform_active_output_planning_ms\":"
	     << planless_execution_stats.planless_transform_active_output_planning_ms
	     << ",\"execution_planless_transform_group_workset_build_ms\":"
	     << planless_execution_stats.planless_transform_group_workset_build_ms
	     << ",\"execution_planless_transform_active_output_count_ms\":"
	     << planless_execution_stats.planless_transform_active_output_count_ms
	     << ",\"execution_planless_transform_active_output_prefix_ms\":"
	     << planless_execution_stats.planless_transform_active_output_prefix_ms
	     << ",\"execution_planless_transform_active_output_fill_ms\":"
	     << planless_execution_stats.planless_transform_active_output_fill_ms
	     << ",\"execution_planless_transform_gpu_kernel_ms\":"
	     << planless_execution_stats.planless_transform_gpu_kernel_ms
	     << ",\"execution_bounded_double_buffer_worksets\":"
	     << planless_execution_stats.bounded_double_buffer_workset_count
	     << ",\"execution_bounded_double_buffer_peak_estimated_bytes\":"
	     << planless_execution_stats.bounded_double_buffer_peak_estimated_bytes
	     << ",\"execution_decoded_cache_capacity_bytes\":" << planless_cache_stats.capacity_bytes
	     << ",\"execution_decoded_cache_current_bytes\":" << planless_cache_stats.resident_bytes
	     << ",\"execution_decoded_cache_peak_bytes\":" << planless_cache_stats.peak_resident_bytes
	     << ",\"execution_decoded_cache_current_entries\":" << planless_cache_stats.resident_rowgroups
	     << ",\"execution_decoded_cache_peak_entries\":" << planless_cache_stats.peak_resident_rowgroups
	     << ",\"execution_decoded_cache_hits\":" << planless_cache_stats.hits
	     << ",\"execution_decoded_cache_misses\":" << planless_cache_stats.misses
	     << ",\"execution_decoded_cache_inserts\":" << planless_cache_stats.inserts
	     << ",\"execution_decoded_cache_evictions\":" << planless_cache_stats.evictions
	     << ",\"execution_plan_cache_hits\":" << planless_execution_stats.plan_cache_hits
	     << ",\"execution_plan_cache_misses\":" << planless_execution_stats.plan_cache_misses
	     << ",\"execution_plan_cache_evictions\":" << planless_execution_stats.plan_cache_evictions
	     << ",\"execution_resize_cache_hits\":" << planless_execution_stats.dct_resize_weight_cache_hits
	     << ",\"execution_resize_cache_misses\":" << planless_execution_stats.dct_resize_weight_cache_misses
	     << ",\"execution_conversion_cache_hits\":"
	     << planless_execution_stats.dct_conversion_matrix_cache_hits
	     << ",\"execution_conversion_cache_misses\":"
	     << planless_execution_stats.dct_conversion_matrix_cache_misses
	     << "}\n";
	std::cout << json.str();
	if (!options.output_json.empty()) {
		std::ofstream output(options.output_json, std::ios::trunc);
		output << json.str();
		if (!output) {
			throw std::runtime_error("failed to write output JSON");
		}
	}
	const bool execution_pass =
	    !options.execute ||
	    (execution_selected && execution_zero_expansion && execution_deterministic && execution_within_tolerance &&
	     execution_kernel_launched && execution_strategy_accounting_valid && execution_resource_bounds &&
	     execution_cache_contract_valid);
	const bool physical_order_pass = plan.stats.duplicate_physical_read_count == 0 &&
	                                 plan.stats.rowgroup_revisit_count == 0 &&
	                                 plan.stats.vector_run_revisit_count == 0 &&
	                                 plan.stats.physical_read_order_inversions == 0;
	return vectors_equal && physical_order_pass && execution_pass ? 0 : 3;
} catch (const std::exception& error) {
	std::cerr << "block-major plan audit failed: " << error.what() << '\n';
	return 1;
}
