// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/execution/internal/prepare.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_INTERNAL_PREPARE_CUH
#define ENGINE_EXECUTION_INTERNAL_PREPARE_CUH

#include "engine/execution/common.cuh"
#include "engine/execution/dict_ref_resolver.cuh"
#include <cstdlib>
#include <memory>
#include <optional>
#include <unordered_set>

namespace dispatch::runtime {

struct ExecutionWorkset {
	using HostBatches   = typename dispatch::BatchSetFromList<dispatch::SupportedTypes>::type;
	using DeviceBatches = typename dispatch::DeviceBatchSetFromList<dispatch::SupportedTypes>::type;

	HostBatches                                      host_batches;
	DeviceBatches                                    device_batches;
	std::vector<dispatch::MixedWorkSlot>             mixed_slots;
	std::unique_ptr<flsgpu::memory::DeviceArena>     chunk_arena;
	std::optional<GPUArray<dispatch::MixedWorkSlot>> owned_slots;
	dispatch::MixedWorkSlot*                         d_slots = nullptr;
	size_t                                           payload_arena_bytes = 0;
	cudaStream_t                                     h2d_stream = nullptr;
	cudaStream_t                                     compute_stream = nullptr;
	cudaEvent_t                                      h2d_ready_event = nullptr;
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
		batch.device_outputs.reserve(batch.device_outputs.size() + additional_exprs);
	});
}

inline bool begin_workset_chunk_arena(ExecutionWorkset& workset, const size_t additional_exprs) {
	if (workset.chunk_arena) {
		return true;
	}
	// Resolver callbacks created while appending expressions may capture references
	// into batch.device_exprs. Reserve only before the arena exists so later
	// append_expressions() calls cannot relocate those vectors out from under
	// already-registered callbacks.
	reserve_batch_expr_storage(workset, additional_exprs);
	workset.chunk_arena = std::make_unique<flsgpu::memory::DeviceArena>(ensure_workset_h2d_stream(workset));
	return true;
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

inline void validate_unique_work_item(std::unordered_set<uint64_t>& seen, const dispatch::WorkItemAny& work) {
	const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(work.type)) << 56) |
	                     (static_cast<uint64_t>(work.expr_index) << 28) | static_cast<uint64_t>(work.vector_index);
	if (!seen.insert(key).second) {
		throw std::runtime_error("duplicate work item detected (type,expr_index,vector_index)");
	}
}

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

	std::unordered_set<uint64_t>         seen;
	std::optional<dispatch::WorkItemAny> pending_half;
	seen.reserve(total_items * 2 + 1);

	const auto append_work = [&](const dispatch::WorkItemAny& work) {
		validate_unique_work_item(seen, work);
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
	const auto h2d_stream = ensure_workset_h2d_stream(workset);
	begin_workset_chunk_arena(workset, expressions.size());
	flsgpu::memory::DeviceArena* active_chunk_arena = workset.chunk_arena.get();

	size_t active_expr_count = 0;

	for (size_t i = 0; i < expressions.size(); ++i) {
		const auto& expr = expressions[i];
		if (!expr.column || expr.column->skip_decompress) {
			continue;
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
				    add_expression_to_batch<T>(materialize_expr_index,
				                               host_col,
				                               plan,
				                               workset.host_batches.template get<T>(),
				                               cfg.freq_prefetch_all_branchless,
				                               cfg.freq_hybrid_patcher,
				                               cfg.freq_branchless_threshold,
				                               h2d_stream,
				                               *active_chunk_arena);
			    }
		    },
		    expr.column->host);
	}

}

inline void upload_workset(ExecutionWorkset& workset) {
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

	build_mixed_slots(workset);

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
				arena.add_resolver([&arena, &dev_batch, expr_idx]() {
					dev_batch.d_exprs = arena.template get<dispatch::DeviceExpression<T>>(expr_idx);
				});
			}
			if (!host_batch.work_items.empty()) {
				const auto item_idx = arena.template add<dispatch::WorkItemAny>(
				    host_batch.work_items.size(), host_batch.work_items.data());
				dev_batch.n_items = host_batch.work_items.size();
				arena.add_resolver([&arena, &dev_batch, item_idx]() {
					dev_batch.d_items = arena.template get<dispatch::WorkItemAny>(item_idx);
				});
			}
		});
		if (!workset.mixed_slots.empty()) {
			auto& arena         = *workset.chunk_arena;
			const auto slot_idx = arena.template add<dispatch::MixedWorkSlot>(
			    workset.mixed_slots.size(), workset.mixed_slots.data());
			arena.add_resolver([&arena, &workset, slot_idx]() {
				workset.d_slots = arena.template get<dispatch::MixedWorkSlot>(slot_idx);
			});
		}
		workset.chunk_arena->upload(/*resolve_before_pack=*/true);
	}

	if (h2d_stream != nullptr) {
		CUDA_SAFE_CALL(cudaEventRecord(ensure_workset_h2d_ready_event(workset), h2d_stream));
	}
}

} // namespace dispatch::runtime

#endif // ENGINE_EXECUTION_INTERNAL_PREPARE_CUH
