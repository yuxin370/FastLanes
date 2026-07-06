#include "galp/direct_dct.hpp"
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <torch/extension.h>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

galp::jpeg::JpegDctCropBox parse_crop(const py::object& crop) {
	galp::jpeg::JpegDctCropBox parsed {};
	if (crop.is_none()) {
		return parsed;
	}
	const auto seq = py::reinterpret_borrow<py::sequence>(crop);
	if (seq.size() != 4) {
		throw std::invalid_argument("crop must be None or a 4-item sequence: (x, y, width, height)");
	}
	parsed.x      = seq[0].cast<uint32_t>();
	parsed.y      = seq[1].cast<uint32_t>();
	parsed.width  = seq[2].cast<uint32_t>();
	parsed.height = seq[3].cast<uint32_t>();
	return parsed;
}

galp::jpeg::JpegDctCoefficientSelection parse_coefficients(const std::string& spec) {
	galp::jpeg::JpegDctCoefficientSelection selection;
	if (!galp::jpeg::parse_jpeg_dct_coefficient_selection(spec, selection)) {
		throw std::invalid_argument("invalid DCT coefficient selection; expected all, first:N, or list:0,1,...");
	}
	return selection;
}

py::dict image_layout_to_dict(const galp::jpeg::JpegDctDeviceImageLayout& layout) {
	py::dict out;
	out["global_image_index"] = layout.global_image_index;
	out["block_offset"]       = layout.block_offset;
	out["block_count"]        = layout.block_count;
	return out;
}

py::dict block_metadata_to_dict(const galp::jpeg::JpegDctDeviceBlockMetadata& block) {
	py::dict out;
	out["request_index"]      = block.request_index;
	out["global_image_index"] = block.global_image_index;
	out["semantic_slot_id"]   = block.semantic_slot_id;
	out["block_x"]            = block.block_x;
	out["block_y"]            = block.block_y;
	return out;
}

py::dict rowgroup_metadata_to_dict(const galp::jpeg::JpegDctDeviceRowgroupMetadata& rowgroup) {
	py::dict out;
	out["shard_id"]       = rowgroup.shard_id;
	out["rowgroup_index"] = rowgroup.rowgroup_index;
	return out;
}

py::dict cache_stats_to_dict(const galp::jpeg::JpegDctDeviceCacheStats& stats) {
	py::dict out;
	out["capacity_bytes"]     = stats.capacity_bytes;
	out["resident_bytes"]     = stats.resident_bytes;
	out["resident_rowgroups"] = stats.resident_rowgroups;
	out["hits"]               = stats.hits;
	out["misses"]             = stats.misses;
	out["inserts"]            = stats.inserts;
	out["evictions"]          = stats.evictions;
	return out;
}

py::dict execution_stats_to_dict(const galp::jpeg::JpegDctDeviceExecutionStats& stats) {
	py::dict out;
	out["planned_selected_vector_count"]                 = stats.planned_selected_vector_count;
	out["selected_vector_count"]                         = stats.selected_vector_count;
	out["full_vector_count"]                             = stats.full_vector_count;
	out["planned_saved_vector_count"]                    = stats.planned_saved_vector_count;
	out["actual_saved_vector_count"]                     = stats.actual_saved_vector_count;
	out["rowgroup_count"]                                = stats.rowgroup_count;
	out["workset_count"]                                 = stats.workset_count;
	out["decode_kernel_launch_count"]                    = stats.decode_kernel_launch_count;
	out["gather_kernel_launch_count"]                    = stats.gather_kernel_launch_count;
	out["prefix_gather_kernel_launch_count"]             = stats.prefix_gather_kernel_launch_count;
	out["cached_gather_kernel_launch_count"]             = stats.cached_gather_kernel_launch_count;
	out["materialize_kernel_launch_count"]               = stats.materialize_kernel_launch_count;
	out["gather_item_count"]                             = stats.gather_item_count;
	out["decoded_gather_item_count"]                     = stats.decoded_gather_item_count;
	out["cached_gather_item_count"]                      = stats.cached_gather_item_count;
	out["projection_item_count"]                         = stats.projection_item_count;
	out["decoded_projection_item_count"]                 = stats.decoded_projection_item_count;
	out["workset_upload_count"]                          = stats.workset_upload_count;
	out["scratch_upload_count"]                          = stats.scratch_upload_count;
	out["scratch_allocation_count"]                      = stats.scratch_allocation_count;
	out["internal_sync_count"]                           = stats.internal_sync_count;
	out["cached_gather_sync_count"]                      = stats.cached_gather_sync_count;
	out["decoded_batch_sync_count"]                      = stats.decoded_batch_sync_count;
	out["cached_gather_event_handoff_count"]             = stats.cached_gather_event_handoff_count;
	out["sparse_vector_cache_hits"]                      = stats.sparse_vector_cache_hits;
	out["sparse_vector_cache_misses"]                    = stats.sparse_vector_cache_misses;
	out["runtime_policy_selected_rowgroups"]             = stats.runtime_policy_selected_rowgroups;
	out["runtime_policy_full_rowgroups"]                 = stats.runtime_policy_full_rowgroups;
	out["runtime_policy_tail_full_rowgroups"]            = stats.runtime_policy_tail_full_rowgroups;
	out["runtime_policy_ratio_full_rowgroups"]           = stats.runtime_policy_ratio_full_rowgroups;
	out["runtime_policy_low_saving_full_rowgroups"]      = stats.runtime_policy_low_saving_full_rowgroups;
	out["prefetch_initial_cache_hit_rowgroup_count"]     = stats.prefetch_initial_cache_hit_rowgroup_count;
	out["prefetch_candidate_rowgroup_count"]             = stats.prefetch_candidate_rowgroup_count;
	out["prefetch_active_shard_count"]                   = stats.prefetch_active_shard_count;
	out["prefetch_config_disabled_shard_count"]          = stats.prefetch_config_disabled_shard_count;
	out["prefetch_all_hit_shard_count"]                  = stats.prefetch_all_hit_shard_count;
	out["prefetch_small_batch_disabled_shard_count"]     = stats.prefetch_small_batch_disabled_shard_count;
	out["prefetch_selected_vector_disabled_shard_count"] = stats.prefetch_selected_vector_disabled_shard_count;
	out["prefetch_selected_vector_miss_rowgroup_count"]  = stats.prefetch_selected_vector_miss_rowgroup_count;
	out["prefetch_initial_hit_runtime_miss_count"]       = stats.prefetch_initial_hit_runtime_miss_count;
	out["prefetch_skipped_repeated_runtime_miss_count"]  = stats.prefetch_skipped_repeated_runtime_miss_count;
	out["prefetched_rowgroup_count"]                     = stats.prefetched_rowgroup_count;
	out["prefetch_consumed_as_hit_count"]                = stats.prefetch_consumed_as_hit_count;
	out["prefetch_skipped_repeated_rowgroup_count"]      = stats.prefetch_skipped_repeated_rowgroup_count;
	out["prefetch_consumed_as_hit_read_ms"]              = stats.prefetch_consumed_as_hit_read_ms;
	out["prefetch_consumed_as_hit_wait_ms"]              = stats.prefetch_consumed_as_hit_wait_ms;
	out["planning_ms"]                                   = stats.planning_ms;
	out["workset_build_ms"]                              = stats.workset_build_ms;
	out["workset_upload_ms"]                             = stats.workset_upload_ms;
	out["decode_ms"]                                     = stats.decode_ms;
	out["gather_ms"]                                     = stats.gather_ms;
	out["decoded_gather_ms"]                             = stats.decoded_gather_ms;
	out["cached_gather_ms"]                              = stats.cached_gather_ms;
	out["projection_ms"]                                 = stats.projection_ms;
	out["decoded_projection_ms"]                         = stats.decoded_projection_ms;
	out["prefetch_wait_ms"]                              = stats.prefetch_wait_ms;
	out["prefetch_depth_block_ms"]                       = stats.prefetch_depth_block_ms;
	out["prefetch_queue_start_ms"]                       = stats.prefetch_queue_start_ms;
	out["prefetch_rowgroup_read_ms"]                     = stats.prefetch_rowgroup_read_ms;
	out["prefetch_ready_ahead_ms"]                       = stats.prefetch_ready_ahead_ms;
	out["sync_rowgroup_read_ms"]                         = stats.sync_rowgroup_read_ms;
	out["runtime_policy_decision"]                       = stats.runtime_policy_decision;
	out["runtime_policy_reason"]                         = stats.runtime_policy_reason;
	return out;
}

py::list image_layouts_to_list(const std::vector<galp::jpeg::JpegDctDeviceImageLayout>& layouts) {
	py::list out;
	for (const auto& layout : layouts) {
		out.append(image_layout_to_dict(layout));
	}
	return out;
}

py::list block_metadata_to_list(const std::vector<galp::jpeg::JpegDctDeviceBlockMetadata>& blocks) {
	py::list out;
	for (const auto& block : blocks) {
		out.append(block_metadata_to_dict(block));
	}
	return out;
}

py::list rowgroups_to_list(const std::vector<galp::jpeg::JpegDctDeviceRowgroupMetadata>& rowgroups) {
	py::list out;
	for (const auto& rowgroup : rowgroups) {
		out.append(rowgroup_metadata_to_dict(rowgroup));
	}
	return out;
}

class DeferredDirectDctBatchReleaseQueue {
public:
	static DeferredDirectDctBatchReleaseQueue& instance() {
		static DeferredDirectDctBatchReleaseQueue queue;
		return queue;
	}

	void defer(std::shared_ptr<galp::jpeg::DirectDctBatch> owner, const c10::DeviceIndex device_index) noexcept {
		if (!owner) {
			return;
		}
		try {
			const auto          stream = c10::cuda::getCurrentCUDAStream(device_index);
			at::cuda::CUDAEvent ready(cudaEventDisableTiming);
			ready.record(stream);
			{
				std::lock_guard<std::mutex> lock(mutex_);
				pending_.emplace_back();
				pending_.back().ready = std::move(ready);
				pending_.back().owner = std::move(owner);
			}
			reclaim_finished();
		} catch (const std::exception& e) {
			std::fprintf(stderr,
			             "GALP direct-DCT PyTorch tensor deleter: failed to defer CUDA release; "
			             "waiting for the current stream before releasing the batch: %s\n",
			             e.what());
			synchronize_current_stream(device_index);
			owner.reset();
		} catch (...) {
			std::fprintf(stderr,
			             "GALP direct-DCT PyTorch tensor deleter: failed to defer CUDA release; "
			             "waiting for the current stream before releasing the batch.\n");
			synchronize_current_stream(device_index);
			owner.reset();
		}
	}

	void reclaim_finished() noexcept {
		std::vector<std::shared_ptr<galp::jpeg::DirectDctBatch>> ready;
		try {
			std::lock_guard<std::mutex> lock(mutex_);
			for (auto it = pending_.begin(); it != pending_.end();) {
				if (it->ready.query()) {
					ready.push_back(std::move(it->owner));
					it = pending_.erase(it);
				} else {
					++it;
				}
			}
		} catch (const std::exception& e) {
			std::fprintf(
			    stderr, "GALP direct-DCT PyTorch tensor deleter: deferred release query failed: %s\n", e.what());
		} catch (...) {
			std::fprintf(stderr, "GALP direct-DCT PyTorch tensor deleter: deferred release query failed.\n");
		}
	}

	~DeferredDirectDctBatchReleaseQueue() {
		std::deque<PendingRelease> pending;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			pending.swap(pending_);
		}
		for (auto& item : pending) {
			try {
				item.ready.synchronize();
			} catch (const std::exception& e) {
				std::fprintf(stderr,
				             "GALP direct-DCT PyTorch tensor deleter: deferred release shutdown wait failed: %s\n",
				             e.what());
			} catch (...) {
				std::fprintf(stderr,
				             "GALP direct-DCT PyTorch tensor deleter: deferred release shutdown wait failed.\n");
			}
			item.owner.reset();
		}
	}

private:
	struct PendingRelease {
		at::cuda::CUDAEvent                         ready;
		std::shared_ptr<galp::jpeg::DirectDctBatch> owner;
	};

	static void synchronize_current_stream(const c10::DeviceIndex device_index) noexcept {
		try {
			c10::cuda::getCurrentCUDAStream(device_index).synchronize();
		} catch (const std::exception& e) {
			std::fprintf(stderr, "GALP direct-DCT PyTorch tensor deleter: current stream wait failed: %s\n", e.what());
		} catch (...) { std::fprintf(stderr, "GALP direct-DCT PyTorch tensor deleter: current stream wait failed.\n"); }
	}

	std::mutex                 mutex_;
	std::deque<PendingRelease> pending_;
};

struct TorchDirectDctBatch {
	explicit TorchDirectDctBatch(galp::jpeg::DirectDctBatch batch_in)
	    : batch(std::make_shared<galp::jpeg::DirectDctBatch>(std::move(batch_in))) {
	}

	torch::Tensor coefficients() {
		DeferredDirectDctBatchReleaseQueue::instance().reclaim_finished();
		if (tensor.defined()) {
			return tensor;
		}
		const auto desc         = batch->tensor();
		const auto device_index = static_cast<c10::DeviceIndex>(desc.cuda_device < 0 ? 0 : desc.cuda_device);
		auto options = torch::TensorOptions().dtype(torch::kInt16).device(torch::Device(torch::kCUDA, device_index));
		if (desc.data == nullptr || desc.empty()) {
			tensor = torch::empty({static_cast<int64_t>(desc.rows()), static_cast<int64_t>(desc.columns())}, options);
			return tensor;
		}
		auto owner = batch;
		tensor     = torch::from_blob(
            const_cast<int16_t*>(desc.data),
            {static_cast<int64_t>(desc.rows()), static_cast<int64_t>(desc.columns())},
            {static_cast<int64_t>(desc.strides[0]), static_cast<int64_t>(desc.strides[1])},
            [owner = std::move(owner), device_index](void*) mutable {
                DeferredDirectDctBatchReleaseQueue::instance().defer(std::move(owner), device_index);
            },
            options);
		return tensor;
	}

	[[nodiscard]] py::list image_layouts() const {
		return image_layouts_to_list(batch->image_layouts());
	}

	[[nodiscard]] py::list block_metadata() const {
		return block_metadata_to_list(batch->block_metadata());
	}

	[[nodiscard]] py::list rowgroups() const {
		return rowgroups_to_list(batch->rowgroups());
	}

	[[nodiscard]] py::dict cache_stats() const {
		return cache_stats_to_dict(batch->cache_stats());
	}

	[[nodiscard]] py::dict execution_stats() const {
		return execution_stats_to_dict(batch->execution_stats());
	}

	[[nodiscard]] uintptr_t device_data_ptr() const noexcept {
		return reinterpret_cast<uintptr_t>(batch->device_data());
	}

	std::shared_ptr<galp::jpeg::DirectDctBatch> batch;
	torch::Tensor                               tensor;
};

class TorchDirectDctReader {
public:
	explicit TorchDirectDctReader(const std::string& manifest_path)
	    : runtime(manifest_path) {
	}

	TorchDirectDctBatch read_batch(const std::vector<uint32_t>& image_ids,
	                               const py::object&            crop,
	                               const std::string&           dct_coeffs,
	                               const size_t                 cache_capacity_mib,
	                               const size_t                 decode_batch_rowgroups,
	                               const bool                   enable_rowgroup_prefetch) {
		galp::jpeg::JpegDctDeviceBatchOptions options;
		options.coefficient_selection    = parse_coefficients(dct_coeffs);
		options.cache_capacity_bytes     = cache_capacity_mib * size_t {1024} * size_t {1024};
		options.decode_batch_rowgroups   = decode_batch_rowgroups;
		options.enable_rowgroup_prefetch = enable_rowgroup_prefetch;
		DeferredDirectDctBatchReleaseQueue::instance().reclaim_finished();
		return TorchDirectDctBatch(runtime.ReadBatch(image_ids, parse_crop(crop), options));
	}

	[[nodiscard]] uint64_t image_count() const noexcept {
		return runtime.image_count();
	}

private:
	galp::jpeg::DirectDctRuntime runtime;
};

} // namespace

PYBIND11_MODULE(_galp_direct_dct, m) {
	m.doc() = "Stay-on-GPU GALP JPEG DCT runtime for PyTorch direct-DCT workloads";

	py::class_<TorchDirectDctBatch>(m, "DirectDctBatch")
	    .def_property_readonly("coefficients", &TorchDirectDctBatch::coefficients)
	    .def_property_readonly("image_layouts", &TorchDirectDctBatch::image_layouts)
	    .def_property_readonly("block_metadata", &TorchDirectDctBatch::block_metadata)
	    .def_property_readonly("rowgroups", &TorchDirectDctBatch::rowgroups)
	    .def_property_readonly("cache_stats", &TorchDirectDctBatch::cache_stats)
	    .def_property_readonly("execution_stats", &TorchDirectDctBatch::execution_stats)
	    .def_property_readonly("device_data_ptr", &TorchDirectDctBatch::device_data_ptr)
	    .def_property_readonly("global_image_ids",
	                           [](const TorchDirectDctBatch& batch) { return batch.batch->global_image_ids(); })
	    .def_property_readonly("selected_coefficients",
	                           [](const TorchDirectDctBatch& batch) { return batch.batch->selected_coefficients(); })
	    .def_property_readonly("cuda_device",
	                           [](const TorchDirectDctBatch& batch) { return batch.batch->cuda_device(); })
	    .def_property_readonly("block_count",
	                           [](const TorchDirectDctBatch& batch) { return batch.batch->block_count(); })
	    .def_property_readonly("coefficients_per_block",
	                           [](const TorchDirectDctBatch& batch) { return batch.batch->coefficients_per_block(); })
	    .def_property_readonly("coefficient_count",
	                           [](const TorchDirectDctBatch& batch) { return batch.batch->coefficient_count(); })
	    .def_property_readonly("coefficient_bytes",
	                           [](const TorchDirectDctBatch& batch) { return batch.batch->coefficient_bytes(); });

	py::class_<TorchDirectDctReader>(m, "DirectDctReader")
	    .def(py::init<const std::string&>(), py::arg("manifest_path"))
	    .def_property_readonly("image_count", &TorchDirectDctReader::image_count)
	    .def("read_batch",
	         &TorchDirectDctReader::read_batch,
	         py::arg("image_ids"),
	         py::arg("crop")                     = py::none(),
	         py::arg("dct_coeffs")               = "all",
	         py::arg("cache_capacity_mib")       = 0,
	         py::arg("decode_batch_rowgroups")   = galp::jpeg::kDefaultJpegDctDecodeBatchRowgroups,
	         py::arg("enable_rowgroup_prefetch") = true);
}
