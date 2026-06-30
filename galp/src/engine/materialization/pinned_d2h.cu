// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/materialization/pinned_d2h.cu
// ────────────────────────────────────────────────────────
#include "cuda/memory/device_pool.cuh"
#include "engine/materialization/pinned_d2h.cuh"
#include <cstdio>
#include <exception>
#include <limits>

namespace galp::runtime {

void validate_output_view_bounds(
    const size_t offset, const size_t n_values, const size_t elem_size, const size_t total_bytes, const char* context) {
	if (n_values == 0) {
		return;
	}
	if (elem_size == 0 || n_values > (std::numeric_limits<size_t>::max() / elem_size)) {
		throw std::out_of_range(context);
	}
	const size_t bytes = n_values * elem_size;
	if (offset > total_bytes || bytes > total_bytes - offset) {
		throw std::out_of_range(context);
	}
}

void clear_materialize_workset_state(ExecutionWorkset& workset) {
	galp::execution::for_each_type(galp::execution::SupportedTypes {}, [&](auto tag) {
		using T     = typename decltype(tag)::type;
		auto& batch = workset.buffers.host_batches.template get<T>();
		batch.device_exprs.clear();
		batch.output_offsets.clear();
		batch.work_items.clear();
		batch.work_items_explicit = false;
		batch.expr_indices.clear();

		auto& dev = workset.buffers.device_batches.template get<T>();
		dev.owned_exprs.reset();
		dev.owned_items.reset();
		dev.d_exprs = nullptr;
		dev.d_items = nullptr;
		dev.n_items = 0;
	});
	workset.slots.owned.reset();
	workset.slots.d = nullptr;
	workset.slots.mixed.clear();
}

void snapshot_materialize_entries(ExecutionWorkset& workset, PendingMaterialize& pending) {
	galp::execution::for_each_type(galp::execution::SupportedTypes {}, [&](auto tag) {
		using T     = typename decltype(tag)::type;
		auto& batch = workset.buffers.host_batches.template get<T>();
		pending.entries.reserve(pending.entries.size() + batch.device_exprs.size());
		for (size_t idx = 0; idx < batch.device_exprs.size(); ++idx) {
			const auto&               dev_expr = batch.device_exprs[idx];
			PendingMaterialize::Entry entry {};
			entry.global_expr_index = batch.expr_indices[idx];
			entry.n_values          = dev_expr.output_n_values;
			entry.output_offset     = batch.output_offsets[idx];
			entry.elem_size         = sizeof(T);
			entry.value_type        = galp::format::ToDataType<T>::value;
			pending.entries.push_back(entry);
		}
	});
}

PendingMaterialize kick_pinned_d2h_materialize(ExecutionWorkset& workset, const bool clear_workset_state) {
	PendingMaterialize pending {};
	const size_t       total_bytes = workset.outputs.used_bytes;
	pending.total_bytes            = total_bytes;
	snapshot_materialize_entries(workset, pending);

	if (pending.entries.empty()) {
		if (clear_workset_state) {
			clear_materialize_workset_state(workset);
		}
		return pending;
	}

	if (total_bytes == 0) {
		pending.active = true;
		if (clear_workset_state) {
			clear_materialize_workset_state(workset);
		}
		return pending;
	}

	if (!workset.outputs.arena.has_value()) {
		if (clear_workset_state) {
			clear_materialize_workset_state(workset);
		}
		throw std::runtime_error("kick_pinned_d2h_materialize: output arena not allocated");
	}

	auto* device_base = workset.outputs.arena->get();
	if (device_base == nullptr) {
		if (clear_workset_state) {
			clear_materialize_workset_state(workset);
		}
		throw std::runtime_error("kick_pinned_d2h_materialize: output arena has null device pointer");
	}

	constexpr size_t kPinnedBucket = 2U * 1024U * 1024U;
	const size_t     alloc_bytes   = ((total_bytes + kPinnedBucket - 1U) / kPinnedBucket) * kPinnedBucket;
	void*            pinned_raw    = galp::memory::DevicePool::instance().alloc_pinned(alloc_bytes);
	if (pinned_raw == nullptr) {
		throw std::runtime_error("kick_pinned_d2h_materialize: pinned alloc returned null");
	}
	pending.pinned_owner = std::shared_ptr<void>(pinned_raw, [](void* p) {
		if (p != nullptr) {
			try {
				galp::memory::DevicePool::instance().release_pinned(p);
			} catch (const std::exception& e) {
				std::fprintf(stderr, "pinned-D2H slot release failed: %s\n", e.what());
			}
		}
	});

	pending.d2h_stream = ensure_workset_d2h_stream(workset);

	if (workset.transfer.compute_stream) {
		galp::memory::CudaEvent kernel_done(cudaEventDisableTiming);
		kernel_done.record(workset.transfer.compute_stream.get());
		CUDA_SAFE_CALL(cudaStreamWaitEvent(pending.d2h_stream, kernel_done.get(), 0));
	}

	CUDA_SAFE_CALL(cudaMemcpyAsync(pinned_raw, device_base, total_bytes, cudaMemcpyDeviceToHost, pending.d2h_stream));
	pending.d2h_event.create_with_flags(cudaEventDisableTiming);
	pending.d2h_event.record(pending.d2h_stream);
	pending.active = true;

	if (clear_workset_state) {
		clear_materialize_workset_state(workset);
	}
	return pending;
}

void discard_pinned_d2h_materialize(PendingMaterialize& pending) noexcept {
	try {
		if (pending.d2h_event) {
			pending.d2h_event.synchronize();
			pending.d2h_event.reset();
		}
	} catch (const std::exception& e) { std::fprintf(stderr, "discard_pinned_d2h_materialize: %s\n", e.what()); }
	pending.entries.clear();
	pending.pinned_owner.reset();
	pending.total_bytes = 0;
	pending.d2h_stream  = nullptr;
	pending.active      = false;
}

} // namespace galp::runtime
