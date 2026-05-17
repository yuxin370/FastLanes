// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/runtime/materialize/metadata.cu
// ────────────────────────────────────────────────────────
#include "runtime/materialize/metadata.cuh"
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <utility>

namespace galp::runtime {

size_t column_n_values(const galp::expression::Expression& expression) {
	if (!expression.column) {
		throw std::runtime_error("null expression column");
	}
	return std::visit([](auto&& host_col) -> size_t { return host_col.get_n_values(); }, expression.column->host);
}

size_t resolve_alias(const std::vector<galp::expression::Expression>& expressions, const size_t idx) {
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

void apply_aliases(RowgroupData&                                    data,
                   const std::vector<galp::expression::Expression>& expressions,
                   const ExecutionConfig&                           cfg) {
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

void populate_materialized_metadata(RowgroupData&                                    data,
                                    const std::vector<galp::expression::Expression>& expressions,
                                    const ExecutionConfig&                           cfg) {
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

RowgroupData materialize_workset(ExecutionWorkset&                                workset,
                                 const std::vector<galp::expression::Expression>& expressions,
                                 const ExecutionConfig&                           cfg) {
	RowgroupData result {};
	result.columns.resize(expressions.size());

	materialize_outputs_via_pinned_d2h(workset, [&](size_t global_expr_index) -> MaterializedColumn* {
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

void release_workset(ExecutionWorkset& workset, const bool preserve_resources) {
	galp::execution::for_each_type(galp::execution::SupportedTypes {}, [&](auto tag) {
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
	workset.outputs.required   = false;
	if (workset.transfer.h2d_stream) {
		galp::memory::sync_h2d(workset.transfer.h2d_stream.get());
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
	if (!preserve_resources) {
		workset.transfer.h2d_ready_event.reset();
		workset.transfer.h2d_stream.reset();
		workset.transfer.compute_stream.reset();
		workset.transfer.d2h_stream.reset();
	}
}

ExecutionWorksetGuard::ExecutionWorksetGuard(ExecutionWorkset& ws)
    : workset(&ws) {
}

ExecutionWorksetGuard::~ExecutionWorksetGuard() noexcept {
	if (active && workset) {
		try {
			release_workset(*workset);
		} catch (const std::exception& ex) {
			std::fprintf(stderr, "ExecutionWorksetGuard cleanup suppressed exception: %s\n", ex.what());
		} catch (...) { std::fprintf(stderr, "ExecutionWorksetGuard cleanup suppressed unknown exception\n"); }
	}
}

void ExecutionWorksetGuard::dismiss() {
	active = false;
}

} // namespace galp::runtime
