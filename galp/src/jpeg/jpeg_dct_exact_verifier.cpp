#include "jpeg_dct_exact_verifier.hpp"

#if GALP_WITH_JPEG_DCT

#include "galp/jpeg_dct_storage.hpp"
#include "jpeg_dct_metadata.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>

namespace galp::jpeg {

namespace {

using BlockKey = std::tuple<uint32_t, uint32_t, uint32_t>;
using BlockMap = std::map<BlockKey, JpegDctCoefficientRow>;

struct ShardTask {
	JpegDctShardManifestEntry entry;
};

auto mismatch_order_key(const JpegDctExactMismatch& mismatch) {
	return std::tuple {mismatch.global_image_index,
	                   mismatch.semantic_slot_id,
	                   mismatch.block_y,
	                   mismatch.block_x,
	                   mismatch.coefficient,
	                   static_cast<uint8_t>(mismatch.kind)};
}

void consider_mismatch(JpegDctExactVerificationResult& result, const JpegDctExactMismatch& candidate) {
	if (!candidate.present) {
		return;
	}
	if (!result.first_mismatch.present || mismatch_order_key(candidate) < mismatch_order_key(result.first_mismatch)) {
		result.first_mismatch = candidate;
	}
}

JpegDctExactMismatch make_mismatch(const uint32_t                 image_index,
                                   const BlockKey&                key,
                                   const uint32_t                 coefficient,
                                   const int16_t                  expected,
                                   const int16_t                  actual,
                                   const JpegDctExactMismatchKind kind) {
	const auto [semantic_slot_id, block_y, block_x] = key;
	return {true, image_index, semantic_slot_id, block_y, block_x, coefficient, expected, actual, kind};
}

BlockMap source_blocks(const std::filesystem::path& source_path) {
	const auto source_table = read_jpeg_dct_file(source_path);
	BlockMap   expected;
	for (const auto& group : source_table.metadata.block_group_index) {
		for (uint32_t row_offset = 0; row_offset < group.row_count; ++row_offset) {
			const auto row = static_cast<size_t>(group.row_start + row_offset);
			if (row >= source_table.row_count) {
				throw std::runtime_error("source JPEG block-group row is outside the coefficient table");
			}
			JpegDctCoefficientRow coefficients {};
			for (size_t coefficient = 0; coefficient < coefficients.size(); ++coefficient) {
				coefficients[coefficient] = source_table.columns[coefficient].at(row);
			}
			const BlockKey key {group.semantic_slot_id, group.block_y, group.block_x};
			if (!expected.emplace(key, coefficients).second) {
				throw std::runtime_error("source JPEG contains a duplicate component/block coordinate");
			}
		}
	}
	return expected;
}

BlockMap manifest_blocks(JpegDctShardDatasetReader& reader, const uint32_t image_index) {
	const auto materialized = reader.MaterializeImageDct(image_index);
	BlockMap   actual;
	for (const auto& block : materialized.blocks) {
		const BlockKey key {block.semantic_slot_id, block.block_y, block.block_x};
		if (!actual.emplace(key, block.coefficients).second) {
			throw std::runtime_error("manifest image contains a duplicate component/block coordinate");
		}
	}
	return actual;
}

JpegDctExactVerificationResult
verify_image(JpegDctShardDatasetReader& reader, const uint32_t image_index, const std::filesystem::path& source_path) {
	const auto expected = source_blocks(source_path);
	const auto actual   = manifest_blocks(reader, image_index);

	JpegDctExactVerificationResult result;
	result.source_images   = 1U;
	result.expected_blocks = expected.size();
	result.actual_blocks   = actual.size();

	for (const auto& [key, expected_coefficients] : expected) {
		const auto actual_it = actual.find(key);
		if (actual_it == actual.end()) {
			++result.missing_blocks;
			consider_mismatch(result,
			                  make_mismatch(image_index, key, 0U, 0, 0, JpegDctExactMismatchKind::kMissingBlock));
			continue;
		}
		for (size_t coefficient = 0; coefficient < expected_coefficients.size(); ++coefficient) {
			const auto expected_value = expected_coefficients[coefficient];
			const auto actual_value   = actual_it->second[coefficient];
			if (expected_value == actual_value) {
				continue;
			}
			++result.coefficient_mismatches;
			result.max_abs_difference = std::max(
			    result.max_abs_difference, std::abs(static_cast<int>(expected_value) - static_cast<int>(actual_value)));
			consider_mismatch(result,
			                  make_mismatch(image_index,
			                                key,
			                                static_cast<uint32_t>(coefficient),
			                                expected_value,
			                                actual_value,
			                                JpegDctExactMismatchKind::kCoefficient));
		}
	}
	for (const auto& [key, coefficients] : actual) {
		(void)coefficients;
		if (!expected.contains(key)) {
			++result.extra_blocks;
			consider_mismatch(result, make_mismatch(image_index, key, 0U, 0, 0, JpegDctExactMismatchKind::kExtraBlock));
		}
	}
	return result;
}

void merge_result(JpegDctExactVerificationResult& total, const JpegDctExactVerificationResult& local) {
	total.source_images += local.source_images;
	total.expected_blocks += local.expected_blocks;
	total.actual_blocks += local.actual_blocks;
	total.missing_blocks += local.missing_blocks;
	total.extra_blocks += local.extra_blocks;
	total.coefficient_mismatches += local.coefficient_mismatches;
	total.max_abs_difference = std::max(total.max_abs_difference, local.max_abs_difference);
	consider_mismatch(total, local.first_mismatch);
}

std::vector<ShardTask> make_shard_tasks(const JpegDctShardManifest& manifest) {
	std::vector<ShardTask> tasks;
	tasks.reserve(manifest.shards.size());
	for (const auto& entry : manifest.shards) {
		if (entry.first_global_image_index > manifest.image_count ||
		    entry.image_count > manifest.image_count - entry.first_global_image_index) {
			throw std::runtime_error("manifest shard image range is outside the manifest image count");
		}
		tasks.push_back({entry});
	}

	std::sort(tasks.begin(), tasks.end(), [](const ShardTask& left, const ShardTask& right) {
		return std::tie(left.entry.first_global_image_index, left.entry.shard_id) <
		       std::tie(right.entry.first_global_image_index, right.entry.shard_id);
	});
	uint64_t expected_first_image = 0U;
	for (const auto& task : tasks) {
		if (task.entry.first_global_image_index != expected_first_image) {
			throw std::runtime_error("manifest shard image ranges contain a gap or overlap");
		}
		expected_first_image += task.entry.image_count;
	}
	if (expected_first_image != manifest.image_count) {
		throw std::runtime_error("manifest shard image ranges do not cover the manifest image count");
	}

	std::sort(tasks.begin(), tasks.end(), [](const ShardTask& left, const ShardTask& right) {
		return left.entry.shard_id < right.entry.shard_id;
	});
	for (size_t idx = 1U; idx < tasks.size(); ++idx) {
		if (tasks[idx - 1U].entry.shard_id == tasks[idx].entry.shard_id) {
			throw std::runtime_error("manifest contains duplicate shard ids");
		}
	}
	return tasks;
}

} // namespace

JpegDctExactVerificationResult verify_jpeg_dct_manifest_exact(const std::filesystem::path&              manifest_path,
                                                              const std::vector<std::filesystem::path>& source_paths,
                                                              const size_t verify_workers) {
	if (verify_workers == 0U || verify_workers > kMaxJpegDctExactVerificationWorkers) {
		throw std::invalid_argument("verify_workers must be in [1, 16]");
	}

	const auto manifest = detail::read_jpeg_dct_shard_manifest_file(manifest_path);
	if (source_paths.size() != manifest.image_count) {
		throw std::runtime_error("source JPEG count does not match the manifest image count");
	}
	const auto tasks = make_shard_tasks(manifest);
	if (tasks.empty()) {
		return {};
	}

	const auto                                  worker_count = std::min(verify_workers, tasks.size());
	std::vector<JpegDctExactVerificationResult> shard_results(tasks.size());
	std::atomic<size_t>                         next_task {0U};
	std::atomic<bool>                           cancelled {false};
	std::mutex                                  exception_mutex;
	std::exception_ptr                          first_exception;

	const auto worker = [&]() {
		try {
			JpegDctShardDatasetReader reader(manifest_path);
			while (!cancelled.load(std::memory_order_relaxed)) {
				const auto task_index = next_task.fetch_add(1U, std::memory_order_relaxed);
				if (task_index >= tasks.size()) {
					return;
				}
				const auto&                    entry = tasks[task_index].entry;
				JpegDctExactVerificationResult shard_result;
				for (uint32_t local_image = 0U; local_image < entry.image_count; ++local_image) {
					const auto global_image = entry.first_global_image_index + local_image;
					if (global_image > std::numeric_limits<uint32_t>::max()) {
						throw std::runtime_error("manifest global image index exceeds the reader index type");
					}
					merge_result(shard_result,
					             verify_image(reader,
					                          static_cast<uint32_t>(global_image),
					                          source_paths[static_cast<size_t>(global_image)]));
				}
				shard_results[task_index] = std::move(shard_result);
			}
		} catch (...) {
			{
				std::lock_guard<std::mutex> lock(exception_mutex);
				if (!first_exception) {
					first_exception = std::current_exception();
				}
			}
			cancelled.store(true, std::memory_order_relaxed);
		}
	};

	std::vector<std::thread> workers;
	workers.reserve(worker_count);
	try {
		for (size_t worker_index = 0U; worker_index < worker_count; ++worker_index) {
			workers.emplace_back(worker);
		}
	} catch (...) {
		cancelled.store(true, std::memory_order_relaxed);
		for (auto& thread : workers) {
			if (thread.joinable()) {
				thread.join();
			}
		}
		throw;
	}
	for (auto& thread : workers) {
		thread.join();
	}
	if (first_exception) {
		std::rethrow_exception(first_exception);
	}

	JpegDctExactVerificationResult result;
	for (const auto& shard_result : shard_results) {
		merge_result(result, shard_result);
	}
	return result;
}

} // namespace galp::jpeg

#endif // GALP_WITH_JPEG_DCT
