// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/workset/upload.cu
// ────────────────────────────────────────────────────────
#include "cuda/cuda_macros.cuh"
#include "engine/workset/append.cuh"
#include "engine/workset/streams.cuh"
#include "engine/workset/upload.cuh"
#include <chrono>
#include <limits>
#include <stdexcept>

namespace galp::runtime {
namespace {

__global__ void scatter_packed_rowgroup_ranges_kernel(const DeviceScatterCopy* copies, const size_t copy_count) {
	const size_t copy_index = static_cast<size_t>(blockIdx.x);
	if (copy_index >= copy_count) {
		return;
	}
	const auto copy = copies[copy_index];
	for (size_t offset = threadIdx.x; offset < copy.size; offset += blockDim.x) {
		copy.destination[offset] = copy.source[offset];
	}
}

size_t round_up_capacity_bytes(const size_t bytes, const size_t minimum_capacity = 65536U) {
	if (bytes == 0) {
		return 0;
	}
	size_t capacity = minimum_capacity;
	while (capacity < bytes) {
		if (capacity > std::numeric_limits<size_t>::max() / 2U) {
			throw std::overflow_error("workset output arena capacity overflow");
		}
		capacity *= 2U;
	}
	return capacity;
}

galp::memory::ArenaCapacityMetrics ensure_workset_output_arena(ExecutionWorkset& workset) {
	galp::memory::ArenaCapacityMetrics metrics {};
	metrics.requested_bytes        = workset.outputs.used_bytes;
	metrics.minimum_capacity_bytes = workset.outputs.minimum_capacity_bytes;
	metrics.capacity_before_bytes  = workset.outputs.capacity_bytes;
	if (workset.outputs.used_bytes == 0) {
		workset.outputs.arena.reset();
		workset.outputs.capacity_bytes = 0;
		return metrics;
	}
	const size_t required_bytes = std::max(workset.outputs.used_bytes, workset.outputs.minimum_capacity_bytes);
	if (workset.outputs.arena.has_value() && workset.outputs.capacity_bytes >= required_bytes) {
		metrics.capacity_bytes = workset.outputs.capacity_bytes;
		return metrics;
	}
	const size_t alloc_bytes = round_up_capacity_bytes(required_bytes);
	workset.outputs.arena.emplace(alloc_bytes, ensure_workset_h2d_stream(workset));
	workset.outputs.capacity_bytes = alloc_bytes;
	metrics.capacity_bytes = alloc_bytes;
	metrics.growth_count   = 1U;
	metrics.growth_bytes   = alloc_bytes - std::min(alloc_bytes, metrics.capacity_before_bytes);
	return metrics;
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

void build_mixed_slots(ExecutionWorkset& workset, const ExecutionConfig& cfg) {
	clear_mixed_slots(workset.slots);

	const size_t total_items = count_expr_work_items(workset);
	if (total_items == 0) {
		return;
	}
	// Reserve the conservative one-slot-per-item ceiling for both vectors so a
	// later type mix or scalar tail does not repeatedly grow and copy storage.
	workset.slots.reserve_host_slots(total_items, total_items);

	// Multi-vector unpack decodes `decode_vector_width` vectors per work item, so synthesized
	// fallback items must step by that width — matching the chunked items add_expression_to_batch
	// builds for typed dispatch. Per-vector items here would make each slot decode
	// `unpack_n_vectors` vectors into overlapping (and tail out-of-bounds) outputs.
	const uint32_t decode_vector_width = std::max(1U, cfg.unpack_n_vectors);

	const auto append_to_slots = [](std::vector<galp::execution::MixedWorkSlot>& slots,
	                                galp::execution::WorkItemAny&               pending_half,
	                                bool&                                       has_pending_half,
	                                const galp::execution::WorkItemAny&         work,
	                                const uint32_t                              semantic_lanes) {
		if (semantic_lanes == galp::execution::lane_count_for_type(galp::execution::TypeTag::I8)) {
			if (has_pending_half) {
				slots.push_back(galp::execution::MixedWorkSlot {pending_half, galp::execution::invalid_work_item()});
				has_pending_half = false;
			}
			slots.push_back(galp::execution::MixedWorkSlot {work, galp::execution::invalid_work_item()});
			return;
		}
		if (has_pending_half) {
			slots.push_back(galp::execution::MixedWorkSlot {pending_half, work});
			has_pending_half = false;
		} else {
			pending_half     = work;
			has_pending_half = true;
		}
	};

	galp::execution::WorkItemAny pending_half {};
	bool                         has_pending_half = false;
	galp::execution::WorkItemAny pending_scalar_tail_half {};
	bool                         has_pending_scalar_tail_half = false;

	galp::execution::for_each_type(galp::execution::SupportedTypes {}, [&](auto tag) {
		using T              = typename decltype(tag)::type;
		constexpr auto type  = galp::execution::type_tag_for<T>();
		const auto&    batch = workset.buffers.host_batches.template get<T>();
		if (batch.work_items_explicit) {
			std::vector<bool> has_explicit_items(batch.device_exprs.size(), false);
			for (const auto& work : batch.work_items) {
				if (work.expr_index >= batch.device_exprs.size()) {
					throw std::out_of_range("mixed work item expression index out of range");
				}
				has_explicit_items[work.expr_index] = true;
				const auto semantic_lanes =
				    galp::execution::semantic_lane_count(type, batch.device_exprs[work.expr_index].plan);
				append_to_slots(workset.slots.mixed, pending_half, has_pending_half, work, semantic_lanes);
			}
			for (const auto& work : batch.scalar_tail_work_items) {
				if (work.expr_index >= batch.device_exprs.size()) {
					throw std::out_of_range("mixed scalar-tail work item expression index out of range");
				}
				has_explicit_items[work.expr_index] = true;
				const auto semantic_lanes =
				    galp::execution::semantic_lane_count(type, batch.device_exprs[work.expr_index].plan);
				append_to_slots(workset.slots.scalar_tail_mixed,
				                pending_scalar_tail_half,
				                has_pending_scalar_tail_half,
				                work,
				                semantic_lanes);
			}
			for (uint32_t expr_idx = 0; expr_idx < batch.device_exprs.size(); ++expr_idx) {
				if (has_explicit_items[expr_idx]) {
					continue;
				}
				const auto& expr           = batch.device_exprs[expr_idx];
				const auto  semantic_lanes = galp::execution::semantic_lane_count(type, expr.plan);
				const auto  n_vecs         = galp::codec::utils::get_n_vecs_from_size(expr.n_values);
				const uint32_t full_n_vecs =
				    decode_vector_width <= 1U
				        ? static_cast<uint32_t>(n_vecs)
				        : static_cast<uint32_t>((n_vecs / decode_vector_width) * decode_vector_width);
				for (uint32_t vec = 0; vec < full_n_vecs; vec += decode_vector_width) {
					append_to_slots(workset.slots.mixed,
					                pending_half,
					                has_pending_half,
					                galp::execution::WorkItemAny {expr_idx, vec, type, vec},
					                semantic_lanes);
				}
				for (uint32_t vec = full_n_vecs; vec < n_vecs; ++vec) {
					append_to_slots(workset.slots.scalar_tail_mixed,
					                pending_scalar_tail_half,
					                has_pending_scalar_tail_half,
					                galp::execution::WorkItemAny {expr_idx, vec, type, vec},
					                semantic_lanes);
				}
			}
			return;
		}
		for (uint32_t expr_idx = 0; expr_idx < batch.device_exprs.size(); ++expr_idx) {
			const auto& expr           = batch.device_exprs[expr_idx];
			const auto  semantic_lanes = galp::execution::semantic_lane_count(type, expr.plan);
			const auto  n_vecs         = galp::codec::utils::get_n_vecs_from_size(expr.n_values);
			const uint32_t full_n_vecs =
			    decode_vector_width <= 1U ? static_cast<uint32_t>(n_vecs)
			                              : static_cast<uint32_t>((n_vecs / decode_vector_width) * decode_vector_width);
			for (uint32_t vec = 0; vec < full_n_vecs; vec += decode_vector_width) {
				append_to_slots(workset.slots.mixed,
				                pending_half,
				                has_pending_half,
				                galp::execution::WorkItemAny {expr_idx, vec, type, vec},
				                semantic_lanes);
			}
			for (uint32_t vec = full_n_vecs; vec < n_vecs; ++vec) {
				append_to_slots(workset.slots.scalar_tail_mixed,
				                pending_scalar_tail_half,
				                has_pending_scalar_tail_half,
				                galp::execution::WorkItemAny {expr_idx, vec, type, vec},
				                semantic_lanes);
			}
		}
	});

	if (has_pending_half) {
		workset.slots.mixed.push_back(
		    galp::execution::MixedWorkSlot {pending_half, galp::execution::invalid_work_item()});
	}
	if (has_pending_scalar_tail_half) {
		workset.slots.scalar_tail_mixed.push_back(
		    galp::execution::MixedWorkSlot {pending_scalar_tail_half, galp::execution::invalid_work_item()});
	}
}

} // namespace

void clear_mixed_slots(WorksetSlots& slots) {
	slots.mixed.clear();
	slots.scalar_tail_mixed.clear();
	slots.owned.reset();
	slots.owned_scalar_tail.reset();
	slots.d = nullptr;
	slots.d_scalar_tail = nullptr;
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
	workset.slots.owned_scalar_tail.reset();
	workset.slots.d             = nullptr;
	workset.slots.d_scalar_tail = nullptr;
	const bool mixed_dispatch = cfg.launch_strategy == galp::execution::LaunchStrategy::MixedDispatch;
	if (!mixed_dispatch) {
		clear_mixed_slots(workset.slots);
	}
	workset.buffers.payload_arena_bytes = 0;
	workset.buffers.d_device_scatter_copies = nullptr;

	galp::execution::for_each_type(galp::execution::SupportedTypes {}, [&](auto tag) {
		using T         = typename decltype(tag)::type;
		auto& dev_batch = workset.buffers.device_batches.template get<T>();
		dev_batch.owned_exprs.reset();
		dev_batch.owned_items.reset();
		dev_batch.owned_scalar_tail_items.reset();
		dev_batch.d_exprs = nullptr;
		dev_batch.d_items = nullptr;
		dev_batch.d_scalar_tail_items = nullptr;
		dev_batch.n_items = 0;
		dev_batch.n_scalar_tail_items = 0;
	});
	const auto t0a          = clock::now();
	breakdown.prep_reset_ms = ms(t0, t0a);

	breakdown.output_arena = ensure_workset_output_arena(workset);
	const auto t0b                 = clock::now();
	breakdown.prep_output_arena_ms = ms(t0a, t0b);
	galp::execution::for_each_type(galp::execution::SupportedTypes {}, [&](auto tag) {
		using T = typename decltype(tag)::type;
		bind_workset_output_pointers<T>(workset);
	});
	const auto t0c         = clock::now();
	breakdown.prep_bind_ms = ms(t0b, t0c);

	if (mixed_dispatch) {
		build_mixed_slots(workset, cfg);
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
			if (!mixed_dispatch && !host_batch.scalar_tail_work_items.empty()) {
				const auto item_idx = arena.template add<galp::execution::WorkItemAny>(
				    host_batch.scalar_tail_work_items.size(), host_batch.scalar_tail_work_items.data());
				dev_batch.n_scalar_tail_items = host_batch.scalar_tail_work_items.size();
				arena.resolve_to(reinterpret_cast<void**>(&dev_batch.d_scalar_tail_items), item_idx);
			}
		});
		if (!workset.slots.mixed.empty()) {
			auto&      arena    = *workset.buffers.chunk_arena;
			const auto slot_idx = arena.template add<galp::execution::MixedWorkSlot>(workset.slots.mixed.size(),
			                                                                         workset.slots.mixed.data());
			arena.resolve_to(reinterpret_cast<void**>(&workset.slots.d), slot_idx);
		}
		if (!workset.slots.scalar_tail_mixed.empty()) {
			auto&      arena    = *workset.buffers.chunk_arena;
			const auto slot_idx = arena.template add<galp::execution::MixedWorkSlot>(
			    workset.slots.scalar_tail_mixed.size(), workset.slots.scalar_tail_mixed.data());
			arena.resolve_to(reinterpret_cast<void**>(&workset.slots.d_scalar_tail), slot_idx);
		}
		if (!workset.buffers.device_scatter_copies.empty()) {
			auto& arena = *workset.buffers.chunk_arena;
			const auto scatter_idx = arena.template add<DeviceScatterCopy>(
			    workset.buffers.device_scatter_copies.size(), workset.buffers.device_scatter_copies.data());
			arena.resolve_to(reinterpret_cast<void**>(&workset.buffers.d_device_scatter_copies), scatter_idx);
		}
		const auto t2             = clock::now();
		breakdown.arena_pack_ms   = ms(t1, t2);
		breakdown.arena = workset.buffers.chunk_arena->upload(/*resolve_before_pack=*/true,
		                                                       /*backing_regions_coalesced=*/true);
		const auto t3             = clock::now();
		breakdown.arena_upload_ms = ms(t2, t3);
		if (!workset.buffers.device_scatter_copies.empty()) {
			if (workset.buffers.device_scatter_copies.size() > std::numeric_limits<unsigned>::max()) {
				throw std::runtime_error("packed rowgroup device scatter count exceeds CUDA grid range");
			}
			constexpr unsigned kScatterThreads = 128U;
			scatter_packed_rowgroup_ranges_kernel<<<
			    static_cast<unsigned>(workset.buffers.device_scatter_copies.size()), kScatterThreads, 0, h2d_stream>>>(
			    workset.buffers.d_device_scatter_copies, workset.buffers.device_scatter_copies.size());
			CUDA_SAFE_CALL(cudaGetLastError());
		}
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
