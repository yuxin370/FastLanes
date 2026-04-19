// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/execution/table.cu
// ────────────────────────────────────────────────────────
#include "engine/execution/internal/materialize.cuh"
#include "engine/execution/internal/streaming_pipeline.cuh"
#include "engine/execution/rowgroup.cuh"
#include "engine/execution/table.cuh"
#include <array>
#include <memory>

namespace dispatch {
namespace {

struct GlobalExprLocation {
	size_t rowgroup_slot;
	size_t local_expr_index;
};

struct PendingRowgroup {
	size_t                        rowgroup_index = 0;
	reader::Rowgroup              rowgroup {};
	std::vector<expr::Expression> expressions;
	RowgroupData                  materialized;
};

struct StreamingChunkState {
	runtime::ExecutionWorkset       workset {};
	runtime::AsyncWorksetRun        run {};
	std::vector<PendingRowgroup>    rowgroups;
	std::vector<GlobalExprLocation> expr_locations;
	size_t                          global_expr_base = 0;
	size_t                          work_items       = 0;
	bool                            submitted        = false;
};

void reset_streaming_chunk(StreamingChunkState& chunk) {
	chunk.rowgroups.clear();
	chunk.expr_locations.clear();
	chunk.global_expr_base = 0;
	chunk.work_items       = 0;
	chunk.submitted        = false;
}

void submit_streaming_chunk(StreamingChunkState& chunk, const TableDecompressionConfig& cfg) {
	if (chunk.rowgroups.empty()) {
		return;
	}
	runtime::begin_workset_chunk_arena(chunk.workset, chunk.global_expr_base);
	size_t expr_index_base = 0;
	for (auto& pending : chunk.rowgroups) {
		runtime::append_expressions(
		    chunk.workset, pending.expressions, cfg.execution, nullptr, nullptr, expr_index_base, true);
		expr_index_base += runtime::count_active_columns(pending.expressions);
	}
	runtime::upload_workset(chunk.workset);
	chunk.run = runtime::run_workset_async(chunk.workset, 1, cfg.execution);
	chunk.submitted = true;
}

template <typename T>
void materialize_table_batch(Batch<T>&                              batch,
                             std::vector<PendingRowgroup>&          rowgroups,
                             const std::vector<GlobalExprLocation>& expr_locations) {
	for (size_t idx = 0; idx < batch.device_exprs.size(); ++idx) {
		auto& expr = batch.device_exprs[idx];
		auto  host = std::shared_ptr<T[]>(new T[expr.n_values], std::default_delete<T[]>());
		if (expr.n_values > 0) {
			if (expr.out == nullptr) {
				throw std::runtime_error("table materialization device output pointer not initialized");
			}
			CUDA_SAFE_CALL(cudaMemcpy(host.get(), expr.out, expr.n_values * sizeof(T), cudaMemcpyDeviceToHost));
		}

		const size_t global_expr_index = batch.expr_indices[idx];
		if (global_expr_index >= expr_locations.size()) {
			throw std::out_of_range("table materialization expr index out of range");
		}
		const auto& location = expr_locations[global_expr_index];
		if (location.rowgroup_slot >= rowgroups.size()) {
			throw std::out_of_range("table materialization rowgroup slot out of range");
		}
		auto& rowgroup = rowgroups[location.rowgroup_slot];
		if (location.local_expr_index >= rowgroup.materialized.columns.size()) {
			throw std::out_of_range("table materialization local expr index out of range");
		}

		MaterializedColumn out {};
		out.values                                               = ValueStore {std::move(host)};
		out.meta.column_index                                    = location.local_expr_index;
		out.meta.value_count                                     = expr.n_values;
		out.meta.value_type                                      = types::ToDataType<T>::value;
		out.meta.values_per_step                                 = 1;
		rowgroup.materialized.columns[location.local_expr_index] = std::move(out);
		// Arena mode: column pointers live in chunk_arena device_base_; per-expr
		// free_device_expr would cudaFree interior offsets. Arena teardown frees them.
	}

	batch.device_exprs.clear();
	batch.output_offsets.clear();
	batch.work_items.clear();
	batch.expr_indices.clear();
}

void materialize_table_workset(runtime::ExecutionWorkset&             workset,
                               std::vector<PendingRowgroup>&          rowgroups,
                               const std::vector<GlobalExprLocation>& expr_locations,
                               const ExecutionConfig&                 cfg) {
	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T = typename decltype(tag)::type;
		materialize_table_batch(workset.host_batches.template get<T>(), rowgroups, expr_locations);
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

	for (auto& rowgroup : rowgroups) {
		runtime::apply_aliases(rowgroup.materialized, rowgroup.expressions, cfg);
		runtime::populate_materialized_metadata(rowgroup.materialized, rowgroup.expressions, cfg);
	}
}

TableData decompress_table_per_rowgroup(const std::filesystem::path&    fls_path,
                                        const TableDecompressionConfig& cfg,
                                        const TableRowgroupPredicate&   should_decompress,
                                        const TableRowgroupCallback&    on_rowgroup) {
	reader::reader rdr(fls_path);
	TableData      out {};

	for (size_t rg_idx = 0; rg_idx < rdr.rowgroup_count(); ++rg_idx) {
		if (!should_decompress(rg_idx)) {
			continue;
		}

		auto rowgroup    = cfg.use_zero_copy_parse ? rdr.read_rowgroup_zero_copy_materialized(rg_idx)
		                                           : rdr.read_rowgroup(rg_idx);
		auto expressions = expr::assemble(rowgroup);
		auto result      = decompress_rowgroup(expressions, cfg.execution);

		++out.rowgroups;
		out.total_columns += rowgroup.columns.size();
		on_rowgroup(rg_idx, rowgroup, expressions, result);
		free_rowgroup(rowgroup);
	}

	return out;
}

TableData decompress_table_whole_table(const std::filesystem::path&    fls_path,
                                       const TableDecompressionConfig& cfg,
                                       const TableRowgroupPredicate&   should_decompress,
                                       const TableRowgroupCallback&    on_rowgroup) {
	reader::reader                       rdr(fls_path);
	TableData                            out {};
	runtime::StreamingDoubleBuffer<StreamingChunkState> pipeline {};

	const auto consume_chunk = [&](StreamingChunkState& chunk) {
		if (!chunk.submitted || chunk.rowgroups.empty()) {
			return;
		}
		runtime::wait_workset_async(chunk.run);
		materialize_table_workset(chunk.workset, chunk.rowgroups, chunk.expr_locations, cfg.execution);
		runtime::release_workset(chunk.workset);

		for (auto& pending : chunk.rowgroups) {
			on_rowgroup(pending.rowgroup_index, pending.rowgroup, pending.expressions, pending.materialized);
			free_rowgroup(pending.rowgroup);
		}
		reset_streaming_chunk(chunk);
	};

	for (size_t rg_idx = 0; rg_idx < rdr.rowgroup_count(); ++rg_idx) {
		if (!should_decompress(rg_idx)) {
			continue;
		}

		auto& chunk = pipeline.build_chunk();
		chunk.rowgroups.emplace_back();
		auto& pending          = chunk.rowgroups.back();
		pending.rowgroup_index = rg_idx;
		pending.rowgroup       = cfg.use_zero_copy_parse ? rdr.read_rowgroup_zero_copy_materialized(rg_idx)
		                                                 : rdr.read_rowgroup(rg_idx);
		pending.expressions    = expr::assemble(pending.rowgroup);
		pending.materialized.columns.resize(pending.expressions.size());

		const size_t active_columns = runtime::count_active_columns(pending.expressions);
		chunk.expr_locations.reserve(chunk.expr_locations.size() + active_columns);
		for (size_t expr_idx = 0; expr_idx < pending.expressions.size(); ++expr_idx) {
			const auto& expr = pending.expressions[expr_idx];
			if (expr.column && !expr.column->skip_decompress) {
				chunk.expr_locations.push_back(GlobalExprLocation {chunk.rowgroups.size() - 1, expr_idx});
			}
		}

		chunk.global_expr_base += active_columns;
		chunk.work_items += active_columns * pending.rowgroup.n_vecs;

		++out.rowgroups;
		out.total_columns += pending.rowgroup.columns.size();

		if (chunk.work_items >= runtime::kStreamingTargetWorkItems) {
			pipeline.submit_build_and_rotate(
			    [&](StreamingChunkState& target) { submit_streaming_chunk(target, cfg); }, consume_chunk);
		}
	}

	pipeline.flush([&](StreamingChunkState& chunk) { submit_streaming_chunk(chunk, cfg); }, consume_chunk);

	return out;
}

} // namespace

TableData decompress_table(const std::filesystem::path& fls_path, const TableDecompressionConfig& cfg) {
	return decompress_table(
	    fls_path,
	    cfg,
	    [](size_t) { return true; },
	    [](size_t, reader::Rowgroup&, const std::vector<expr::Expression>&, const RowgroupData&) {});
}

TableData decompress_table(const std::filesystem::path&    fls_path,
                           const TableDecompressionConfig& cfg,
                           const std::optional<size_t>&    rowgroup) {
	if (rowgroup.has_value()) {
		reader::reader rdr(fls_path);
		if (*rowgroup >= rdr.rowgroup_count()) {
			throw std::out_of_range("rowgroup index out of range");
		}
	}
	return decompress_table(
	    fls_path,
	    cfg,
	    [rowgroup](const size_t rg_idx) { return !rowgroup.has_value() || *rowgroup == rg_idx; },
	    [](size_t, reader::Rowgroup&, const std::vector<expr::Expression>&, const RowgroupData&) {});
}

TableData decompress_table(const std::filesystem::path&    fls_path,
                           const TableDecompressionConfig& cfg,
                           const TableRowgroupPredicate&   should_decompress,
                           const TableRowgroupCallback&    on_rowgroup) {
	if (cfg.scope == TableDecompressionScope::WholeTable) {
		return decompress_table_whole_table(fls_path, cfg, should_decompress, on_rowgroup);
	}
	return decompress_table_per_rowgroup(fls_path, cfg, should_decompress, on_rowgroup);
}

} // namespace dispatch
