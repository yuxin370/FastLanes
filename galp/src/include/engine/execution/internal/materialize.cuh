// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/execution/internal/materialize.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_INTERNAL_MATERIALIZE_CUH
#define ENGINE_EXECUTION_INTERNAL_MATERIALIZE_CUH

#include "engine/execution/internal/launch.cuh"

namespace dispatch::runtime {

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
	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T     = typename decltype(tag)::type;
		auto& batch = workset.host_batches.template get<T>();
		dispatch::detail::finalize_batch(batch, result);
		auto& device_batch = workset.device_batches.template get<T>();
		device_batch.owned_exprs.reset();
		device_batch.owned_items.reset();
		device_batch.d_exprs = nullptr;
		device_batch.d_items = nullptr;
		device_batch.n_items = 0;
	});
	workset.owned_slots.reset();
	workset.d_slots = nullptr;
	workset.mixed_slots.clear();

	apply_aliases(result, expressions, cfg);
	populate_materialized_metadata(result, expressions, cfg);
	return result;
}

inline void release_workset(ExecutionWorkset& workset, const bool preserve_resources = false) {
	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T          = typename decltype(tag)::type;
		auto& host_batch = workset.host_batches.template get<T>();
		for (auto& expr : host_batch.device_exprs) {
			dispatch::detail::free_device_expr(expr);
		}
		host_batch.device_exprs.clear();
		host_batch.device_outputs.clear();
		host_batch.work_items.clear();
		host_batch.expr_indices.clear();

		auto& device_batch = workset.device_batches.template get<T>();
		device_batch.owned_exprs.reset();
		device_batch.owned_items.reset();
		device_batch.d_exprs = nullptr;
		device_batch.d_items = nullptr;
		device_batch.n_items = 0;
	});
	workset.owned_slots.reset();
	workset.d_slots = nullptr;
	workset.mixed_slots.clear();
	if (workset.h2d_stream != nullptr) {
		flsgpu::memory::sync_h2d(workset.h2d_stream);
	}
	if (workset.chunk_arena != nullptr) {
		if (preserve_resources) {
			workset.chunk_arena->reset();
		} else {
			workset.chunk_arena.reset();
		}
	}
	if (!preserve_resources && workset.h2d_stream != nullptr) {
		if (workset.h2d_ready_event != nullptr) {
			CUDA_SAFE_CALL(cudaEventDestroy(workset.h2d_ready_event));
			workset.h2d_ready_event = nullptr;
		}
		CUDA_SAFE_CALL(cudaStreamDestroy(workset.h2d_stream));
		workset.h2d_stream = nullptr;
	} else if (!preserve_resources && workset.h2d_ready_event != nullptr) {
		CUDA_SAFE_CALL(cudaEventDestroy(workset.h2d_ready_event));
		workset.h2d_ready_event = nullptr;
	}
	if (!preserve_resources && workset.compute_stream != nullptr) {
		CUDA_SAFE_CALL(cudaStreamDestroy(workset.compute_stream));
		workset.compute_stream = nullptr;
	}
}

struct ExecutionWorksetGuard {
	ExecutionWorkset* workset = nullptr;
	bool              active  = true;

	explicit ExecutionWorksetGuard(ExecutionWorkset& ws)
	    : workset(&ws) {
	}

	~ExecutionWorksetGuard() {
		if (active && workset) {
			release_workset(*workset);
		}
	}

	void dismiss() {
		active = false;
	}
};

} // namespace dispatch::runtime

#endif // ENGINE_EXECUTION_INTERNAL_MATERIALIZE_CUH
