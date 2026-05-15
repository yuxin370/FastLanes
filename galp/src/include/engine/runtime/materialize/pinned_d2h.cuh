// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/runtime/materialize/pinned_d2h.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_RUNTIME_MATERIALIZE_PINNED_D2H_CUH
#define ENGINE_RUNTIME_MATERIALIZE_PINNED_D2H_CUH

#include "engine/runtime/workset/streams.cuh"
#include "engine/runtime/materialize/types.cuh"
#include "memory/device_pool.cuh"
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace galp::runtime {

void validate_output_view_bounds(
    size_t offset, size_t n_values, size_t elem_size, size_t total_bytes, const char* context);
void               clear_materialize_workset_state(ExecutionWorkset& workset);
void               snapshot_materialize_entries(ExecutionWorkset& workset, PendingMaterialize& pending);
PendingMaterialize kick_pinned_d2h_materialize(ExecutionWorkset& workset);
void               discard_pinned_d2h_materialize(PendingMaterialize& pending) noexcept;

template <typename TargetResolver>
inline void finalize_pinned_d2h_materialize(PendingMaterialize& pending, TargetResolver&& resolve_target) {
	if (!pending.active) {
		return;
	}

	if (pending.d2h_event) {
		pending.d2h_event.synchronize();
		pending.d2h_event.reset();
	}

	auto* pinned_bytes = reinterpret_cast<std::byte*>(pending.pinned_owner.get());

	for (const auto& entry : pending.entries) {
		MaterializedColumn* target = resolve_target(entry.global_expr_index);
		if (target == nullptr) {
			continue;
		}

		auto build_view = [&](auto type_tag) -> bool {
			using T = typename decltype(type_tag)::type;
			if (entry.value_type != galp::format::ToDataType<T>::value) {
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
			target->values               = ValueStore {std::move(column_view)};
			target->meta.column_index    = entry.global_expr_index;
			target->meta.value_count     = entry.n_values;
			target->meta.value_type      = entry.value_type;
			target->meta.values_per_step = 1;
			return true;
		};

		bool matched = false;
		galp::execution::for_each_type(galp::execution::SupportedTypes {}, [&](auto tag) {
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
	pending.active      = false;
}

template <typename TargetResolver>
inline void materialize_outputs_via_pinned_d2h(ExecutionWorkset& workset, TargetResolver&& resolve_target) {
	PendingMaterialize pending {};
	const size_t       total_bytes = workset.outputs.used_bytes;
	pending.total_bytes            = total_bytes;
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
	void*            pinned_raw    = galp::memory::DevicePool::instance().alloc_pinned(alloc_bytes);
	if (pinned_raw == nullptr) {
		throw std::runtime_error("materialize_outputs_via_pinned_d2h: pinned alloc returned null");
	}
	auto pinned_owner    = std::shared_ptr<void>(pinned_raw, [](void* p) {
        if (p != nullptr) {
            try {
                galp::memory::DevicePool::instance().release_pinned(p);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "pinned-D2H slot release failed: %s\n", e.what());
            }
        }
    });
	pending.pinned_owner = pinned_owner;

	const cudaStream_t d2h_stream = ensure_workset_d2h_stream(workset);
	CUDA_SAFE_CALL(cudaMemcpyAsync(pinned_raw, device_base, total_bytes, cudaMemcpyDeviceToHost, d2h_stream));
	CUDA_SAFE_CALL(cudaStreamSynchronize(d2h_stream));
	pending.active = true;

	finalize_pinned_d2h_materialize(pending, std::forward<TargetResolver>(resolve_target));
	clear_materialize_workset_state(workset);
}

} // namespace galp::runtime

#endif // ENGINE_RUNTIME_MATERIALIZE_PINNED_D2H_CUH
