// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/execution/internal/materialize.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_INTERNAL_MATERIALIZE_CUH
#define ENGINE_EXECUTION_INTERNAL_MATERIALIZE_CUH

#include "engine/execution/internal/launch.cuh"
#include "flsgpu/memory/device_pool.cuh"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace dispatch::runtime {

// Single-DMA, pinned-buffer D2H materialization.
//
// The whole chunk's outputs live contiguously inside `workset.outputs.arena`
// (see ensure_workset_output_arena + bind_workset_output_pointers). Instead of
// issuing one synchronous cudaMemcpy(D2H) per expression into a fresh pageable
// `new T[]`, we:
//   1. allocate one pinned host buffer sized to outputs.used_bytes,
//   2. issue a single cudaMemcpyAsync on a dedicated d2h_stream,
//   3. cudaStreamSynchronize that stream once,
//   4. hand each MaterializedColumn an aliased shared_ptr<T[]> pointing into
//      the pinned buffer (lifetime is shared via the aliasing constructor).
//
// Wins: pinned D2H runs at ~26 GB/s vs ~20 GB/s pageable; one DMA replaces N
// per-column DMAs; D2H runs on its own stream so it can overlap the *next*
// chunk's H2D + kernel work in the streaming pipeline.
//
// One pinned-D2H job in flight for a given chunk. Built by
// kick_pinned_d2h_materialize() and finalized by finalize_pinned_d2h_materialize().
// Holds enough information to walk per-T host_batches snapshots and rebuild
// aliased shared_ptr<T[]> views into pinned_owner once the d2h_event fires.
struct PendingMaterialize {
	struct Entry {
		size_t              global_expr_index = 0;
		size_t              n_values          = 0;
		size_t              output_offset     = 0;
		size_t              elem_size         = 0; // sizeof(T)
		types::DataType     value_type        = types::DataType::I8;
	};

	std::shared_ptr<void>            pinned_owner;
	size_t                           total_bytes = 0;
	cudaStream_t                     d2h_stream  = nullptr;
	cudaEvent_t                      d2h_event   = nullptr;
	std::vector<Entry>               entries;
	bool                             active      = false;
};

inline void validate_output_view_bounds(const size_t offset,
                                        const size_t n_values,
                                        const size_t elem_size,
                                        const size_t total_bytes,
                                        const char*  context) {
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

inline void clear_materialize_workset_state(ExecutionWorkset& workset) {
	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T     = typename decltype(tag)::type;
		auto& batch = workset.buffers.host_batches.template get<T>();
		batch.device_exprs.clear();
		batch.output_offsets.clear();
		batch.work_items.clear();
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

inline void snapshot_materialize_entries(ExecutionWorkset& workset, PendingMaterialize& pending) {
	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T     = typename decltype(tag)::type;
		auto& batch = workset.buffers.host_batches.template get<T>();
		pending.entries.reserve(pending.entries.size() + batch.device_exprs.size());
		for (size_t idx = 0; idx < batch.device_exprs.size(); ++idx) {
			const auto& dev_expr = batch.device_exprs[idx];
			PendingMaterialize::Entry entry {};
			entry.global_expr_index = batch.expr_indices[idx];
			entry.n_values          = dev_expr.n_values;
			entry.output_offset     = batch.output_offsets[idx];
			entry.elem_size         = sizeof(T);
			entry.value_type        = types::ToDataType<T>::value;
			pending.entries.push_back(entry);
		}
	});
}

inline PendingMaterialize kick_pinned_d2h_materialize(ExecutionWorkset& workset) {
	PendingMaterialize pending {};
	const size_t total_bytes = workset.outputs.used_bytes;
	pending.total_bytes       = total_bytes;
	snapshot_materialize_entries(workset, pending);

	if (pending.entries.empty()) {
		clear_materialize_workset_state(workset);
		return pending;
	}

	if (total_bytes == 0) {
		pending.active = true;
		clear_materialize_workset_state(workset);
		return pending;
	}

	if (!workset.outputs.arena.has_value()) {
		clear_materialize_workset_state(workset);
		throw std::runtime_error("kick_pinned_d2h_materialize: output arena not allocated");
	}

	auto* device_base = workset.outputs.arena->get();
	if (device_base == nullptr) {
		clear_materialize_workset_state(workset);
		throw std::runtime_error("kick_pinned_d2h_materialize: output arena has null device pointer");
	}

	// Round up to a 2 MiB bucket so chunks of slightly different sizes reuse
	// the same pinned slot in PinnedHostPool. Without this, every distinct
	// used_bytes value forces a fresh cudaMallocHost (O(N) allocations across
	// the run); with it, the pool converges to O(few) buckets after warm-up.
	constexpr size_t kPinnedBucket = 2U * 1024U * 1024U;
	const size_t     alloc_bytes   = ((total_bytes + kPinnedBucket - 1U) / kPinnedBucket) * kPinnedBucket;
	void* pinned_raw = flsgpu::memory::DevicePool::instance().alloc_pinned(alloc_bytes);
	if (pinned_raw == nullptr) {
		throw std::runtime_error("kick_pinned_d2h_materialize: pinned alloc returned null");
	}
	pending.pinned_owner = std::shared_ptr<void>(pinned_raw, [](void* p) {
		if (p != nullptr) {
			try {
				flsgpu::memory::DevicePool::instance().release_pinned(p);
			} catch (const std::exception& e) {
				std::fprintf(stderr, "pinned-D2H slot release failed: %s\n", e.what());
			}
		}
	});

	pending.d2h_stream = ensure_workset_d2h_stream(workset);

	// d2h_stream must wait for the chunk's kernel to finish before starting
	// the copy. The kernel was launched on compute_stream; we publish a
	// stop event there and have d2h_stream wait on it.
	if (workset.transfer.compute_stream != nullptr) {
		cudaEvent_t kernel_done {};
		CUDA_SAFE_CALL(cudaEventCreateWithFlags(&kernel_done, cudaEventDisableTiming));
		CUDA_SAFE_CALL(cudaEventRecord(kernel_done, workset.transfer.compute_stream));
		CUDA_SAFE_CALL(cudaStreamWaitEvent(pending.d2h_stream, kernel_done, 0));
		CUDA_SAFE_CALL(cudaEventDestroy(kernel_done));
	}

	CUDA_SAFE_CALL(cudaMemcpyAsync(pinned_raw,
	                               device_base,
	                               total_bytes,
	                               cudaMemcpyDeviceToHost,
	                               pending.d2h_stream));
	CUDA_SAFE_CALL(cudaEventCreateWithFlags(&pending.d2h_event, cudaEventDisableTiming));
	CUDA_SAFE_CALL(cudaEventRecord(pending.d2h_event, pending.d2h_stream));
	pending.active = true;

	clear_materialize_workset_state(workset);
	return pending;
}

template <typename TargetResolver>
inline void finalize_pinned_d2h_materialize(PendingMaterialize& pending, TargetResolver&& resolve_target) {
	if (!pending.active) {
		return;
	}

	if (pending.d2h_event != nullptr) {
		CUDA_SAFE_CALL(cudaEventSynchronize(pending.d2h_event));
		CUDA_SAFE_CALL(cudaEventDestroy(pending.d2h_event));
		pending.d2h_event = nullptr;
	}

	auto* pinned_bytes = reinterpret_cast<std::byte*>(pending.pinned_owner.get());

	for (const auto& entry : pending.entries) {
		MaterializedColumn* target = resolve_target(entry.global_expr_index);
		if (target == nullptr) {
			continue;
		}

		auto build_view = [&](auto type_tag) -> bool {
			using T = typename decltype(type_tag)::type;
			if (entry.value_type != types::ToDataType<T>::value) {
				return false;
			}
			std::shared_ptr<T[]> column_view {};
			if (entry.n_values > 0) {
				if (pinned_bytes == nullptr) {
					throw std::runtime_error("finalize_pinned_d2h_materialize: missing pinned buffer");
				}
				validate_output_view_bounds(entry.output_offset,
				                            entry.n_values,
				                            entry.elem_size,
				                            pending.total_bytes,
				                            "finalize_pinned_d2h_materialize: output view exceeds arena");
				T* aliased_ptr = reinterpret_cast<T*>(pinned_bytes + entry.output_offset);
				column_view    = std::shared_ptr<T[]> {pending.pinned_owner, aliased_ptr};
			} else {
				column_view = std::shared_ptr<T[]>(new T[0], std::default_delete<T[]>());
			}
			target->values                = ValueStore {std::move(column_view)};
			target->meta.column_index     = entry.global_expr_index;
			target->meta.value_count      = entry.n_values;
			target->meta.value_type       = entry.value_type;
			target->meta.values_per_step  = 1;
			return true;
		};

		bool matched = false;
		dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
			if (!matched) {
				matched = build_view(tag);
			}
		});
		if (!matched) {
			throw std::runtime_error("finalize_pinned_d2h_materialize: unsupported value type in entry");
		}
	}

	pending.entries.clear();
	pending.pinned_owner.reset();
	pending.total_bytes = 0;
	pending.active = false;
}

// resolve_target(global_expr_index) -> MaterializedColumn* selects the slot to
// fill; returning nullptr skips that expression (used by the table path when
// expr_locations does not list a slot for skip_decompress columns).
template <typename TargetResolver>
inline void materialize_outputs_via_pinned_d2h(ExecutionWorkset& workset,
                                               TargetResolver&&  resolve_target) {
	PendingMaterialize pending {};
	const size_t total_bytes = workset.outputs.used_bytes;
	pending.total_bytes       = total_bytes;
	snapshot_materialize_entries(workset, pending);

	if (pending.entries.empty()) {
		clear_materialize_workset_state(workset);
		return;
	}

	if (total_bytes == 0) {
		pending.active = true;
		finalize_pinned_d2h_materialize(pending, std::forward<TargetResolver>(resolve_target));
		clear_materialize_workset_state(workset);
		return;
	}

	if (!workset.outputs.arena.has_value()) {
		clear_materialize_workset_state(workset);
		throw std::runtime_error("materialize_outputs_via_pinned_d2h: output arena not allocated");
	}

	auto* device_base = workset.outputs.arena->get();
	if (device_base == nullptr) {
		clear_materialize_workset_state(workset);
		throw std::runtime_error("materialize_outputs_via_pinned_d2h: output arena has null device pointer");
	}

	constexpr size_t kPinnedBucket = 2U * 1024U * 1024U;
	const size_t     alloc_bytes   = ((total_bytes + kPinnedBucket - 1U) / kPinnedBucket) * kPinnedBucket;
	void*            pinned_raw    = flsgpu::memory::DevicePool::instance().alloc_pinned(alloc_bytes);
	if (pinned_raw == nullptr) {
		throw std::runtime_error("materialize_outputs_via_pinned_d2h: pinned alloc returned null");
	}
	auto pinned_owner = std::shared_ptr<void>(pinned_raw, [](void* p) {
		if (p != nullptr) {
			try {
				flsgpu::memory::DevicePool::instance().release_pinned(p);
			} catch (const std::exception& e) {
				std::fprintf(stderr, "pinned-D2H slot release failed: %s\n", e.what());
			}
		}
	});
	pending.pinned_owner = pinned_owner;

	const cudaStream_t d2h_stream = ensure_workset_d2h_stream(workset);
	CUDA_SAFE_CALL(cudaMemcpyAsync(pinned_raw,
	                               device_base,
	                               total_bytes,
	                               cudaMemcpyDeviceToHost,
	                               d2h_stream));
	CUDA_SAFE_CALL(cudaStreamSynchronize(d2h_stream));
	pending.active = true;

	finalize_pinned_d2h_materialize(pending, std::forward<TargetResolver>(resolve_target));
	clear_materialize_workset_state(workset);
}

inline size_t column_n_values(const expr::Expression& expression) {
	if (!expression.column) {
		throw std::runtime_error("null expression column");
	}
	return std::visit([](auto&& host_col) -> size_t { return host_col.get_n_values(); }, expression.column->host);
}

inline size_t resolve_alias(const std::vector<expr::Expression>& expressions, const size_t idx) {
	if (idx >= expressions.size()) {
		throw std::out_of_range("alias index out of range");
	}

	std::vector<uint8_t> visited(expressions.size(), 0);
	size_t               cur = idx;
	for (;;) {
		if (cur >= expressions.size()) {
			throw std::out_of_range("alias target out of range");
		}
		if (visited[cur]) {
			throw std::runtime_error("alias cycle detected");
		}
		visited[cur] = 1;

		const auto* col = expressions[cur].column;
		if (!col || !col->alias_of.has_value()) {
			return cur;
		}

		const size_t next = *col->alias_of;
		if (next >= expressions.size()) {
			throw std::out_of_range("alias target out of range");
		}
		cur = next;
	}
}

inline void
apply_aliases(RowgroupData& data, const std::vector<expr::Expression>& expressions, const ExecutionConfig& cfg) {
	for (size_t i = 0; i < expressions.size(); ++i) {
		const auto* col = expressions[i].column;
		if (!col || !col->alias_of.has_value()) {
			continue;
		}

		const size_t src_idx = resolve_alias(expressions, i);
		if (src_idx >= data.columns.size() || !data.columns[src_idx].has_value()) {
			throw std::runtime_error("EXP_EQUAL: source column not decompressed");
		}
		if (data.columns[i].has_value()) {
			continue;
		}

		const size_t alias_n_values = column_n_values(expressions[i]);
		const size_t src_n_values   = column_n_values(expressions[src_idx]);
		if (alias_n_values != src_n_values) {
			throw std::runtime_error("alias/source value count mismatch");
		}

		data.columns[i]                       = data.columns[src_idx];
		data.columns[i]->meta.column_index    = i;
		data.columns[i]->meta.column_name     = col->name;
		data.columns[i]->meta.value_count     = alias_n_values;
		data.columns[i]->meta.values_per_step = cfg.chunk().values_per_step();
	}
}

inline void populate_materialized_metadata(RowgroupData&                        data,
                                           const std::vector<expr::Expression>& expressions,
                                           const ExecutionConfig&               cfg) {
	for (size_t i = 0; i < expressions.size(); ++i) {
		const auto* col = expressions[i].column;
		if (!col || !data.columns[i].has_value()) {
			continue;
		}
		data.columns[i]->meta.column_index    = i;
		data.columns[i]->meta.column_name     = col->name;
		data.columns[i]->meta.values_per_step = cfg.chunk().values_per_step();
	}
}

inline RowgroupData materialize_workset(ExecutionWorkset&                    workset,
                                        const std::vector<expr::Expression>& expressions,
                                        const ExecutionConfig&               cfg) {
	RowgroupData result {};
	result.columns.resize(expressions.size());

	materialize_outputs_via_pinned_d2h(
	    workset, [&](size_t global_expr_index) -> MaterializedColumn* {
		    if (global_expr_index >= result.columns.size()) {
			    throw std::out_of_range("materialize_workset: expression index out of range");
		    }
		    auto& slot = result.columns[global_expr_index];
		    if (!slot.has_value()) {
			    slot.emplace();
		    }
		    return &(*slot);
	    });

	apply_aliases(result, expressions, cfg);
	populate_materialized_metadata(result, expressions, cfg);
	return result;
}

inline void release_workset(ExecutionWorkset& workset, const bool preserve_resources = false) {
	// Arena mode: all device column pointers are interior offsets into
	// workset.buffers.chunk_arena->device_base_, freed wholesale by chunk_arena.reset()
	// below. Per-expression free_device_expr() would attempt cudaFree on
	// interior pointers (invalid) and walk a DevicePool mutex for every column.
	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T          = typename decltype(tag)::type;
		auto& host_batch = workset.buffers.host_batches.template get<T>();
		host_batch.device_exprs.clear();
		host_batch.output_offsets.clear();
		host_batch.work_items.clear();
		host_batch.expr_indices.clear();

		auto& device_batch = workset.buffers.device_batches.template get<T>();
		device_batch.owned_exprs.reset();
		device_batch.owned_items.reset();
		device_batch.d_exprs = nullptr;
		device_batch.d_items = nullptr;
		device_batch.n_items = 0;
	});
	workset.slots.owned.reset();
	workset.slots.d = nullptr;
	workset.slots.mixed.clear();
	workset.outputs.used_bytes = 0;
	workset.outputs.required = false;
	if (workset.transfer.h2d_stream != nullptr) {
		flsgpu::memory::sync_h2d(workset.transfer.h2d_stream);
	}
	if (workset.buffers.chunk_arena != nullptr) {
		if (preserve_resources) {
			workset.buffers.chunk_arena->reset(/*preserve_capacity=*/true);
		} else {
			workset.buffers.chunk_arena.reset();
		}
	}
	if (!preserve_resources) {
		workset.outputs.arena.reset();
		workset.outputs.capacity_bytes = 0;
	}
	if (!preserve_resources && workset.transfer.h2d_stream != nullptr) {
		if (workset.transfer.h2d_ready_event != nullptr) {
			CUDA_LOG_CALL(cudaEventDestroy(workset.transfer.h2d_ready_event));
			workset.transfer.h2d_ready_event = nullptr;
		}
		CUDA_LOG_CALL(cudaStreamDestroy(workset.transfer.h2d_stream));
		workset.transfer.h2d_stream = nullptr;
	} else if (!preserve_resources && workset.transfer.h2d_ready_event != nullptr) {
		CUDA_LOG_CALL(cudaEventDestroy(workset.transfer.h2d_ready_event));
		workset.transfer.h2d_ready_event = nullptr;
	}
	if (!preserve_resources && workset.transfer.compute_stream != nullptr) {
		CUDA_LOG_CALL(cudaStreamDestroy(workset.transfer.compute_stream));
		workset.transfer.compute_stream = nullptr;
	}
	if (!preserve_resources && workset.transfer.d2h_stream != nullptr) {
		CUDA_LOG_CALL(cudaStreamDestroy(workset.transfer.d2h_stream));
		workset.transfer.d2h_stream = nullptr;
	}
}

struct ExecutionWorksetGuard {
	ExecutionWorkset* workset = nullptr;
	bool              active  = true;

	explicit ExecutionWorksetGuard(ExecutionWorkset& ws)
	    : workset(&ws) {
	}

	~ExecutionWorksetGuard() noexcept {
		if (active && workset) {
			try {
				release_workset(*workset);
			} catch (const std::exception& ex) {
				std::fprintf(stderr, "ExecutionWorksetGuard cleanup suppressed exception: %s\n", ex.what());
			} catch (...) {
				std::fprintf(stderr, "ExecutionWorksetGuard cleanup suppressed unknown exception\n");
			}
		}
	}

	void dismiss() {
		active = false;
	}
};

} // namespace dispatch::runtime

#endif // ENGINE_EXECUTION_INTERNAL_MATERIALIZE_CUH
