// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/execution/internal/prepare.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_INTERNAL_PREPARE_CUH
#define ENGINE_EXECUTION_INTERNAL_PREPARE_CUH

#include "engine/execution/common.cuh"
#include "engine/execution/dict_ref_resolver.cuh"
#include <chrono>
#include <cstdlib>
#include <memory>
#include <optional>
#include <unordered_set>

namespace dispatch::runtime {

// Per-call breakdown of upload_workset sub-stages. Populated by upload_workset()
// and accumulated by the benchmark loop to surface which phase dominates the
// upload_workset_ms metric (see table.cu consumer site).
struct UploadBreakdown {
	double prep_ms               = 0.0; // output arena, bind pointers, build mixed slots
	double prep_reset_ms         = 0.0; // workset + device-batch reset loop
	double prep_output_arena_ms  = 0.0; // ensure_workset_output_arena
	double prep_bind_ms          = 0.0; // bind_workset_output_pointers per-type loop
	double prep_slots_ms         = 0.0; // build_mixed_slots
	double arena_pack_ms         = 0.0; // arena.add + resolve_to for metadata
	double arena_upload_ms       = 0.0; // DeviceArena::upload (capacity + pack + DMA issue)
	double event_record_ms       = 0.0; // cudaEventRecord for h2d ready
};

struct ExecutionWorkset {
	using HostBatches   = typename dispatch::BatchSetFromList<dispatch::SupportedTypes>::type;
	using DeviceBatches = typename dispatch::DeviceBatchSetFromList<dispatch::SupportedTypes>::type;

	HostBatches                                      host_batches;
	DeviceBatches                                    device_batches;
	std::vector<dispatch::MixedWorkSlot>             mixed_slots;
	std::unique_ptr<flsgpu::memory::DeviceArena>     chunk_arena;
	std::optional<GPUArray<uint8_t>>                 output_arena;
	std::optional<GPUArray<dispatch::MixedWorkSlot>> owned_slots;
	dispatch::MixedWorkSlot*                         d_slots = nullptr;
	size_t                                           payload_arena_bytes = 0;
	size_t                                           output_arena_capacity_bytes = 0;
	size_t                                           output_arena_used_bytes = 0;
	cudaStream_t                                     h2d_stream = nullptr;
	cudaStream_t                                     compute_stream = nullptr;
	cudaEvent_t                                      h2d_ready_event = nullptr;
	UploadBreakdown                                  last_upload_breakdown {};
};

inline bool use_async_h2d() {
	static const bool enabled = (std::getenv("GALP_DISABLE_ASYNC_H2D") == nullptr);
	return enabled;
}

inline bool force_h2d_stream_for_sync() {
	static const bool enabled = (std::getenv("GALP_FORCE_H2D_STREAM") != nullptr);
	return enabled;
}

inline cudaStream_t ensure_workset_h2d_stream(ExecutionWorkset& workset) {
	if (!use_async_h2d() && !force_h2d_stream_for_sync()) {
		return nullptr;
	}
	if (workset.h2d_stream == nullptr) {
		CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&workset.h2d_stream, cudaStreamNonBlocking));
	}
	return workset.h2d_stream;
}

inline cudaStream_t ensure_workset_compute_stream(ExecutionWorkset& workset) {
	if (workset.compute_stream == nullptr) {
		CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&workset.compute_stream, cudaStreamNonBlocking));
	}
	return workset.compute_stream;
}

inline cudaEvent_t ensure_workset_h2d_ready_event(ExecutionWorkset& workset) {
	if (workset.h2d_ready_event == nullptr) {
		CUDA_SAFE_CALL(cudaEventCreateWithFlags(&workset.h2d_ready_event, cudaEventDisableTiming));
	}
	return workset.h2d_ready_event;
}

inline void reserve_batch_expr_storage(ExecutionWorkset& workset, const size_t additional_exprs) {
	if (additional_exprs == 0) {
		return;
	}
	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T = typename decltype(tag)::type;
		auto& batch = workset.host_batches.template get<T>();
		batch.device_exprs.reserve(batch.device_exprs.size() + additional_exprs);
		batch.output_offsets.reserve(batch.output_offsets.size() + additional_exprs);
		batch.expr_indices.reserve(batch.expr_indices.size() + additional_exprs);
	});
}

inline bool any_batch_has_device_exprs(const ExecutionWorkset& workset) {
	bool has = false;
	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T = typename decltype(tag)::type;
		if (!workset.host_batches.template get<T>().device_exprs.empty()) {
			has = true;
		}
	});
	return has;
}

inline size_t round_up_capacity_bytes(const size_t bytes, const size_t alignment = 65536U) {
	if (bytes == 0) {
		return 0;
	}
	return ((bytes + alignment - 1U) / alignment) * alignment;
}

inline bool begin_workset_chunk_arena(ExecutionWorkset& workset, const size_t additional_exprs) {
	if (workset.chunk_arena) {
		// Mid-chunk reentry. Two shapes reach here:
		//   1. Streaming/table path: an outer begin_workset_chunk_arena already
		//      reserved cumulative capacity for this chunk and prior rowgroups'
		//      add_expression_to_batch have registered resolver targets like
		//      arena.resolve_to(&device_exprs[i].field). Calling reserve() now
		//      can reallocate device_exprs and dangle those targets, so
		//      DeviceArena::upload(resolve_before_pack=true) would write resolved
		//      pointers into freed memory.
		//   2. preserve_resources=true reuse: release_workset cleared the batches
		//      (size==0) but kept the arena. Here reserving is safe and may be
		//      needed to avoid a mid-build realloc inside this very append.
		if (!any_batch_has_device_exprs(workset)) {
			reserve_batch_expr_storage(workset, additional_exprs);
		}
		return true;
	}
	// Resolver callbacks created while appending expressions may capture references
	// into batch.device_exprs. Reserve before append_expressions() starts for each
	// new chunk/workset so vector growth cannot relocate those references mid-build.
	reserve_batch_expr_storage(workset, additional_exprs);
	workset.chunk_arena = std::make_unique<flsgpu::memory::DeviceArena>(ensure_workset_h2d_stream(workset));
	return true;
}

template <typename T>
inline size_t reserve_workset_output_bytes(ExecutionWorkset& workset, const size_t n_values) {
	if (n_values == 0) {
		return workset.output_arena_used_bytes;
	}
	constexpr size_t kAlign = alignof(T) > 16 ? alignof(T) : 16U;
	workset.output_arena_used_bytes = (workset.output_arena_used_bytes + (kAlign - 1U)) & ~(kAlign - 1U);
	const size_t offset = workset.output_arena_used_bytes;
	workset.output_arena_used_bytes += n_values * sizeof(T);
	return offset;
}

inline void ensure_workset_output_arena(ExecutionWorkset& workset) {
	if (workset.output_arena_used_bytes == 0) {
		workset.output_arena.reset();
		workset.output_arena_capacity_bytes = 0;
		return;
	}
	if (workset.output_arena.has_value() && workset.output_arena_capacity_bytes >= workset.output_arena_used_bytes) {
		return;
	}
	const size_t alloc_bytes = round_up_capacity_bytes(workset.output_arena_used_bytes);
	workset.output_arena.emplace(alloc_bytes, ensure_workset_h2d_stream(workset));
	workset.output_arena_capacity_bytes = alloc_bytes;
}

template <typename T>
inline void bind_workset_output_pointers(ExecutionWorkset& workset) {
	auto& batch = workset.host_batches.template get<T>();
	if (batch.device_exprs.empty()) {
		return;
	}
	// If every active expression had n_values == 0 the arena was never
	// allocated (nothing to write). Leave each device_expr.out null — the
	// kernel dispatch has no work to issue for a zero-length expression, so
	// the pointer is never dereferenced.
	auto* base = workset.output_arena.has_value() ? workset.output_arena->get() : nullptr;
	for (size_t idx = 0; idx < batch.device_exprs.size(); ++idx) {
		if (batch.device_exprs[idx].n_values == 0) {
			batch.device_exprs[idx].out = nullptr;
			continue;
		}
		if (base == nullptr) {
			throw std::runtime_error("output arena not allocated");
		}
		batch.device_exprs[idx].out = reinterpret_cast<T*>(base + batch.output_offsets[idx]);
	}
}

inline size_t count_work_items(const ExecutionWorkset& workset) {
	size_t total = 0;
	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T = typename decltype(tag)::type;
		total += workset.host_batches.template get<T>().work_items.size();
	});
	return total;
}

inline size_t count_active_columns(const std::vector<expr::Expression>& expressions) {
	size_t total = 0;
	for (const auto& expr : expressions) {
		if (expr.column && !expr.column->skip_decompress) {
			++total;
		}
	}
	return total;
}

#ifndef NDEBUG
inline void validate_unique_work_item(std::unordered_set<uint64_t>& seen, const dispatch::WorkItemAny& work) {
	const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(work.type)) << 56) |
	                     (static_cast<uint64_t>(work.expr_index) << 28) | static_cast<uint64_t>(work.vector_index);
	if (!seen.insert(key).second) {
		throw std::runtime_error("duplicate work item detected (type,expr_index,vector_index)");
	}
}
#endif

inline dispatch::PlanKind plan_for_work_item(const ExecutionWorkset& workset, const dispatch::WorkItemAny& work) {
	switch (work.type) {
	case dispatch::TypeTag::I8:
		return workset.host_batches.template get<int8_t>().device_exprs[work.expr_index].plan;
	case dispatch::TypeTag::I16:
		return workset.host_batches.template get<int16_t>().device_exprs[work.expr_index].plan;
	default:
		throw std::runtime_error("unsupported work item type");
	}
}

inline uint32_t semantic_lanes_for_work_item(const ExecutionWorkset& workset, const dispatch::WorkItemAny& work) {
	return dispatch::semantic_lane_count(work.type, plan_for_work_item(workset, work));
}

inline void build_mixed_slots(ExecutionWorkset& workset) {
	workset.mixed_slots.clear();
	workset.owned_slots.reset();
	workset.d_slots = nullptr;

	const size_t total_items = count_work_items(workset);
	if (total_items == 0) {
		return;
	}
	workset.mixed_slots.reserve((total_items + 1U) / 2U);

	std::optional<dispatch::WorkItemAny> pending_half;
#ifndef NDEBUG
	std::unordered_set<uint64_t> seen;
	seen.reserve(total_items * 2 + 1);
#endif

	const auto append_work = [&](const dispatch::WorkItemAny& work) {
#ifndef NDEBUG
		validate_unique_work_item(seen, work);
#endif
		const uint32_t semantic_lanes = semantic_lanes_for_work_item(workset, work);
		if (semantic_lanes == dispatch::lane_count_for_type(dispatch::TypeTag::I8)) {
			if (pending_half.has_value()) {
				workset.mixed_slots.push_back(dispatch::MixedWorkSlot {*pending_half, dispatch::invalid_work_item()});
				pending_half.reset();
			}
			workset.mixed_slots.push_back(dispatch::MixedWorkSlot {work, dispatch::invalid_work_item()});
			return;
		}
		if (pending_half.has_value()) {
			workset.mixed_slots.push_back(dispatch::MixedWorkSlot {*pending_half, work});
			pending_half.reset();
		} else {
			pending_half = work;
		}
	};

	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T = typename decltype(tag)::type;
		for (const auto& work : workset.host_batches.template get<T>().work_items) {
			append_work(work);
		}
	});

	if (pending_half.has_value()) {
		workset.mixed_slots.push_back(dispatch::MixedWorkSlot {*pending_half, dispatch::invalid_work_item()});
	}
}

inline void append_expressions(ExecutionWorkset&              workset,
                               std::vector<expr::Expression>& expressions,
                               const ExecutionConfig&         cfg,
                               size_t*                        out_total_bytes       = nullptr,
                               size_t*                        out_n_exprs           = nullptr,
                               const size_t                   expr_index_base       = 0,
                               const bool                     use_global_expr_index = false) {
	using namespace dispatch;
	using namespace dispatch::detail;

	dispatch::resolve_dict_refs(expressions);
	begin_workset_chunk_arena(workset, expressions.size());
	flsgpu::memory::DeviceArena* active_chunk_arena = workset.chunk_arena.get();

	size_t active_expr_count = 0;

	for (size_t i = 0; i < expressions.size(); ++i) {
		const auto& expr = expressions[i];
		if (!expr.column || expr.column->skip_decompress) {
			continue;
		}
		// Direct-DMA backing is only valid from CUDA-pinned memory; a pageable
		// source would force cudaMemcpyAsync into synchronous internal staging,
		// erasing the zero-copy win and stalling the h2d stream.
		if (expr.column->host_owned_by_backing && expr.column->backing_is_pinned &&
		    expr.column->backing_base != nullptr && expr.column->backing_bytes > 0) {
			active_chunk_arena->register_backing(
			    expr.column->backing_base, expr.column->backing_bytes);
		}
		++active_expr_count;
		if (out_n_exprs) {
			++(*out_n_exprs);
		}

		std::visit(
		    [&](auto&& host_col) {
			    using HostColT      = std::decay_t<decltype(host_col)>;
			    using T             = typename dispatch::host_value_type<HostColT>::type;
			    constexpr auto plan = dispatch::detail::plan_for_host_col<HostColT>();
			    if constexpr (dispatch::is_supported_type_v<T>) {
				    if (out_total_bytes) {
					    *out_total_bytes += host_col.get_n_values() * sizeof(T);
				    }
				    const size_t materialize_expr_index =
				        use_global_expr_index ? (expr_index_base + active_expr_count - 1U) : i;
				    const size_t output_offset = reserve_workset_output_bytes<T>(workset, host_col.get_n_values());
				    add_expression_to_batch<T>(materialize_expr_index,
				                               host_col,
				                               plan,
				                               workset.host_batches.template get<T>(),
				                               output_offset,
				                               cfg.freq_prefetch_all_branchless,
				                               cfg.freq_hybrid_patcher,
				                               cfg.freq_branchless_threshold,
				                               *active_chunk_arena);
			    }
		    },
		    expr.column->host);
	}

}

inline void upload_workset(ExecutionWorkset& workset) {
	using clock   = std::chrono::steady_clock;
	const auto ms = [](auto a, auto b) {
		return std::chrono::duration<double, std::milli>(b - a).count();
	};
	workset.last_upload_breakdown = {};

	const auto t0 = clock::now();
	const auto h2d_stream = ensure_workset_h2d_stream(workset);
	workset.owned_slots.reset();
	workset.d_slots = nullptr;
	workset.mixed_slots.clear();
	workset.payload_arena_bytes = 0;

	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T          = typename decltype(tag)::type;
		auto& dev_batch  = workset.device_batches.template get<T>();
		dev_batch.owned_exprs.reset();
		dev_batch.owned_items.reset();
		dev_batch.d_exprs = nullptr;
		dev_batch.d_items = nullptr;
		dev_batch.n_items = 0;
	});
	const auto t0a = clock::now();
	workset.last_upload_breakdown.prep_reset_ms = ms(t0, t0a);

	ensure_workset_output_arena(workset);
	const auto t0b = clock::now();
	workset.last_upload_breakdown.prep_output_arena_ms = ms(t0a, t0b);
	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T = typename decltype(tag)::type;
		bind_workset_output_pointers<T>(workset);
	});
	const auto t0c = clock::now();
	workset.last_upload_breakdown.prep_bind_ms = ms(t0b, t0c);

	build_mixed_slots(workset);
	const auto t1 = clock::now();
	workset.last_upload_breakdown.prep_slots_ms = ms(t0c, t1);
	workset.last_upload_breakdown.prep_ms = ms(t0, t1);

	// Unified arena: pack payload + metadata into the single chunk_arena,
	// then upload with resolve_before_pack=true so device addresses are embedded
	// in the metadata before the pinned H2D copy.
	if (workset.chunk_arena) {
		// append_expressions() has already appended all column payload buffers to
		// the arena. Snapshot the size before metadata packing so the benchmark
		// metric continues to report payload-only bytes.
		workset.payload_arena_bytes = workset.chunk_arena->total_bytes();
		dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
			using T          = typename decltype(tag)::type;
			auto& host_batch = workset.host_batches.template get<T>();
			auto& dev_batch  = workset.device_batches.template get<T>();
			auto& arena      = *workset.chunk_arena;
			if (!host_batch.device_exprs.empty()) {
				const auto expr_idx = arena.template add<dispatch::DeviceExpression<T>>(
				    host_batch.device_exprs.size(), host_batch.device_exprs.data());
				arena.resolve_to(reinterpret_cast<void**>(&dev_batch.d_exprs), expr_idx);
			}
			if (!host_batch.work_items.empty()) {
				const auto item_idx = arena.template add<dispatch::WorkItemAny>(
				    host_batch.work_items.size(), host_batch.work_items.data());
				dev_batch.n_items = host_batch.work_items.size();
				arena.resolve_to(reinterpret_cast<void**>(&dev_batch.d_items), item_idx);
			}
		});
		if (!workset.mixed_slots.empty()) {
			auto& arena         = *workset.chunk_arena;
			const auto slot_idx = arena.template add<dispatch::MixedWorkSlot>(
			    workset.mixed_slots.size(), workset.mixed_slots.data());
			arena.resolve_to(reinterpret_cast<void**>(&workset.d_slots), slot_idx);
		}
		const auto t2 = clock::now();
		workset.last_upload_breakdown.arena_pack_ms = ms(t1, t2);
		workset.chunk_arena->upload(/*resolve_before_pack=*/true);
		const auto t3 = clock::now();
		workset.last_upload_breakdown.arena_upload_ms = ms(t2, t3);
	}

	const auto t4 = clock::now();
	if (h2d_stream != nullptr) {
		CUDA_SAFE_CALL(cudaEventRecord(ensure_workset_h2d_ready_event(workset), h2d_stream));
	}
	const auto t5 = clock::now();
	workset.last_upload_breakdown.event_record_ms = ms(t4, t5);
}

} // namespace dispatch::runtime

#endif // ENGINE_EXECUTION_INTERNAL_PREPARE_CUH
