// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/workset/upload.cu
// ────────────────────────────────────────────────────────
#include "engine/workset/append.cuh"
#include "engine/workset/streams.cuh"
#include "engine/workset/upload.cuh"
#include "cuda/cuda_macros.cuh"
#include <chrono>
#include <stdexcept>

namespace galp::runtime {
namespace {

size_t round_up_capacity_bytes(const size_t bytes, const size_t alignment = 65536U) {
	if (bytes == 0) {
		return 0;
	}
	return ((bytes + alignment - 1U) / alignment) * alignment;
}

void ensure_workset_output_arena(ExecutionWorkset& workset) {
	if (workset.outputs.used_bytes == 0) {
		workset.outputs.arena.reset();
		workset.outputs.capacity_bytes = 0;
		return;
	}
	if (workset.outputs.arena.has_value() && workset.outputs.capacity_bytes >= workset.outputs.used_bytes) {
		return;
	}
	const size_t alloc_bytes = round_up_capacity_bytes(workset.outputs.used_bytes);
	workset.outputs.arena.emplace(alloc_bytes, ensure_workset_h2d_stream(workset));
	workset.outputs.capacity_bytes = alloc_bytes;
}

template <typename T>
void bind_workset_output_pointers(ExecutionWorkset& workset) {
	auto& batch = workset.buffers.host_batches.template get<T>();
	if (batch.device_exprs.empty()) {
		return;
	}
	auto* base = workset.outputs.arena.has_value() ? workset.outputs.arena->get() : nullptr;
	if (base == nullptr) {
		if (workset.outputs.required) {
			for (const auto& expr : batch.device_exprs) {
				if (expr.n_values > 0) {
					throw std::runtime_error("output arena not allocated for write_out=true");
				}
			}
		}
		for (auto& expr : batch.device_exprs) {
			expr.out = nullptr;
		}
		return;
	}
	for (size_t idx = 0; idx < batch.device_exprs.size(); ++idx) {
		if (batch.device_exprs[idx].n_values == 0) {
			batch.device_exprs[idx].out = nullptr;
			continue;
		}
		batch.device_exprs[idx].out = reinterpret_cast<T*>(base + batch.output_offsets[idx]);
	}
}

void build_mixed_slots(ExecutionWorkset& workset) {
	clear_mixed_slots(workset.slots);

	const size_t total_items = count_expr_work_items(workset);
	if (total_items == 0) {
		return;
	}
	workset.slots.mixed.reserve((total_items + 1U) / 2U);

	galp::execution::WorkItemAny pending_half {};
	bool                         has_pending_half = false;

	const auto append_work = [&](const galp::execution::WorkItemAny& work, const uint32_t semantic_lanes) {
		if (semantic_lanes == galp::execution::lane_count_for_type(galp::execution::TypeTag::I8)) {
			if (has_pending_half) {
				workset.slots.mixed.push_back(
				    galp::execution::MixedWorkSlot {pending_half, galp::execution::invalid_work_item()});
				has_pending_half = false;
			}
			workset.slots.mixed.push_back(galp::execution::MixedWorkSlot {work, galp::execution::invalid_work_item()});
			return;
		}
		if (has_pending_half) {
			workset.slots.mixed.push_back(galp::execution::MixedWorkSlot {pending_half, work});
			has_pending_half = false;
		} else {
			pending_half     = work;
			has_pending_half = true;
		}
	};

	galp::execution::for_each_type(galp::execution::SupportedTypes {}, [&](auto tag) {
		using T              = typename decltype(tag)::type;
		constexpr auto type  = galp::execution::type_tag_for<T>();
		const auto&    batch = workset.buffers.host_batches.template get<T>();
		for (uint32_t expr_idx = 0; expr_idx < batch.device_exprs.size(); ++expr_idx) {
			const auto& expr           = batch.device_exprs[expr_idx];
			const auto  semantic_lanes = galp::execution::semantic_lane_count(type, expr.plan);
			const auto  n_vecs         = galp::codec::utils::get_n_vecs_from_size(expr.n_values);
			for (uint32_t vec = 0; vec < n_vecs; ++vec) {
				append_work(galp::execution::WorkItemAny {expr_idx, vec, type}, semantic_lanes);
			}
		}
	});

	if (has_pending_half) {
		workset.slots.mixed.push_back(
		    galp::execution::MixedWorkSlot {pending_half, galp::execution::invalid_work_item()});
	}
}

} // namespace

void clear_mixed_slots(WorksetSlots& slots) {
	slots.mixed.clear();
	slots.owned.reset();
	slots.d = nullptr;
}

UploadBreakdown upload_workset(ExecutionWorkset& workset, const ExecutionConfig& cfg) {
	using clock   = std::chrono::steady_clock;
	const auto ms = [](auto a, auto b) {
		return std::chrono::duration<double, std::milli>(b - a).count();
	};
	UploadBreakdown breakdown {};

	const auto t0         = clock::now();
	const auto h2d_stream = ensure_workset_h2d_stream(workset);
	workset.slots.owned.reset();
	workset.slots.d           = nullptr;
	const bool mixed_dispatch = cfg.launch_strategy == galp::execution::LaunchStrategy::MixedDispatch;
	if (!mixed_dispatch) {
		clear_mixed_slots(workset.slots);
	}
	workset.buffers.payload_arena_bytes = 0;

	galp::execution::for_each_type(galp::execution::SupportedTypes {}, [&](auto tag) {
		using T         = typename decltype(tag)::type;
		auto& dev_batch = workset.buffers.device_batches.template get<T>();
		dev_batch.owned_exprs.reset();
		dev_batch.owned_items.reset();
		dev_batch.d_exprs = nullptr;
		dev_batch.d_items = nullptr;
		dev_batch.n_items = 0;
	});
	const auto t0a          = clock::now();
	breakdown.prep_reset_ms = ms(t0, t0a);

	ensure_workset_output_arena(workset);
	const auto t0b                 = clock::now();
	breakdown.prep_output_arena_ms = ms(t0a, t0b);
	galp::execution::for_each_type(galp::execution::SupportedTypes {}, [&](auto tag) {
		using T = typename decltype(tag)::type;
		bind_workset_output_pointers<T>(workset);
	});
	const auto t0c         = clock::now();
	breakdown.prep_bind_ms = ms(t0b, t0c);

	if (mixed_dispatch) {
		build_mixed_slots(workset);
	}
	const auto t1           = clock::now();
	breakdown.prep_slots_ms = ms(t0c, t1);
	breakdown.prep_ms       = ms(t0, t1);

	if (workset.buffers.chunk_arena) {
		workset.buffers.chunk_arena->coalesce_backing_regions();
		workset.buffers.payload_arena_bytes = workset.buffers.chunk_arena->total_bytes();
		galp::execution::for_each_type(galp::execution::SupportedTypes {}, [&](auto tag) {
			using T          = typename decltype(tag)::type;
			auto& host_batch = workset.buffers.host_batches.template get<T>();
			auto& dev_batch  = workset.buffers.device_batches.template get<T>();
			auto& arena      = *workset.buffers.chunk_arena;
			if (!host_batch.device_exprs.empty()) {
				const auto expr_idx = arena.template add<galp::execution::DeviceExpression<T>>(
				    host_batch.device_exprs.size(), host_batch.device_exprs.data());
				arena.resolve_to(reinterpret_cast<void**>(&dev_batch.d_exprs), expr_idx);
			}
			if (!mixed_dispatch && !host_batch.work_items.empty()) {
				const auto item_idx = arena.template add<galp::execution::WorkItemAny>(host_batch.work_items.size(),
				                                                                       host_batch.work_items.data());
				dev_batch.n_items   = host_batch.work_items.size();
				arena.resolve_to(reinterpret_cast<void**>(&dev_batch.d_items), item_idx);
			}
		});
		if (!workset.slots.mixed.empty()) {
			auto&      arena    = *workset.buffers.chunk_arena;
			const auto slot_idx = arena.template add<galp::execution::MixedWorkSlot>(workset.slots.mixed.size(),
			                                                                         workset.slots.mixed.data());
			arena.resolve_to(reinterpret_cast<void**>(&workset.slots.d), slot_idx);
		}
		const auto t2             = clock::now();
		breakdown.arena_pack_ms   = ms(t1, t2);
		breakdown.arena           = workset.buffers.chunk_arena->upload(/*resolve_before_pack=*/true,
                                                              /*backing_regions_coalesced=*/true);
		const auto t3             = clock::now();
		breakdown.arena_upload_ms = ms(t2, t3);
	}

	const auto t4 = clock::now();
	if (h2d_stream != nullptr) {
		CUDA_SAFE_CALL(cudaEventRecord(ensure_workset_h2d_ready_event(workset), h2d_stream));
	}
	const auto t5             = clock::now();
	breakdown.event_record_ms = ms(t4, t5);
	return breakdown;
}

} // namespace galp::runtime
