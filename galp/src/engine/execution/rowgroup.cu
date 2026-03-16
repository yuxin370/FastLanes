// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/execution/rowgroup.cu
// ────────────────────────────────────────────────────────
#include "engine/data/value-store.cuh"
#include "engine/execution/dict_ref_resolver.cuh"
#include "engine/execution/rowgroup.cuh"
#include "engine/execution/table.cuh"

namespace dispatch {
namespace {

using HostBatchSet = typename BatchSetFromList<SupportedTypes>::type;

struct PreparedBatches {
	HostBatchSet host_batches;
	size_t       n_expressions = 0;
	size_t       total_bytes   = 0;
	size_t       n_work_items  = 0;
};

inline size_t column_n_values(const expr::Expression& expression) {
	if (!expression.column) {
		throw std::runtime_error("null expression column");
	}
	return std::visit([](auto&& host_col) -> size_t { return host_col.get_n_values(); }, expression.column->host);
}

struct WorksetCleanupGuard {
	BenchmarkWorkset* workset = nullptr;
	bool              active  = true;

	explicit WorksetCleanupGuard(BenchmarkWorkset& ws)
	    : workset(&ws) {
	}
	~WorksetCleanupGuard() {
		if (active && workset) {
			free_batches(*workset);
		}
	}
	void dismiss() {
		active = false;
	}
};

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

inline PreparedBatches prepare_batches(const std::vector<expr::Expression>& expressions) {
	PreparedBatches prepared;

	for (size_t i = 0; i < expressions.size(); ++i) {
		const auto& expr = expressions[i];
		if (!expr.column || expr.column->skip_decompress) {
			continue;
		}
		++prepared.n_expressions;

		std::visit(
		    [&](auto&& host_col) {
			    using HostColT      = std::decay_t<decltype(host_col)>;
			    using T             = typename host_value_type<HostColT>::type;
			    constexpr auto plan = detail::plan_for_host_col<HostColT>();

			    static_assert(is_supported_type_v<T>, "dispatch rowgroup only supports int8_t/int16_t");
			    prepared.total_bytes += host_col.get_n_values() * sizeof(T);
			    add_expression_to_batch<T>(i, host_col, plan, prepared.host_batches.template get<T>());
		    },
		    expr.column->host);
	}

	dispatch::for_each_type(SupportedTypes {}, [&](auto tag) {
		using T           = typename decltype(tag)::type;
		auto& typed_batch = prepared.host_batches.template get<T>();
		prepared.n_work_items += typed_batch.work_items.size();
	});

	return prepared;
}

inline RowgroupData
run_materialize(PreparedBatches& prepared, const std::vector<expr::Expression>& expressions, const Config& cfg) {
	RowgroupData materialized;
	materialized.columns.resize(expressions.size());
	BenchmarkWorkset workset {};
	workset.host_batches = std::move(prepared.host_batches);
	WorksetCleanupGuard guard(workset);
	prepare_dispatch_buffers(workset);
	run_kernel(workset, 1, cfg.gpu_dispatch_kernel, true);
	dispatch::for_each_type(SupportedTypes {}, [&](auto tag) {
		using T     = typename decltype(tag)::type;
		auto& batch = workset.host_batches.template get<T>();
		detail::finalize_batch(batch, materialized);
	});

	for (size_t i = 0; i < expressions.size(); ++i) {
		const auto* col = expressions[i].column;
		if (!col || !col->alias_of.has_value()) {
			continue;
		}

		const size_t src_idx = resolve_alias(expressions, i);
		if (src_idx >= materialized.columns.size() || !materialized.columns[src_idx].has_value()) {
			throw std::runtime_error("EXP_EQUAL: source column not decompressed");
		}
		if (materialized.columns[i].has_value()) {
			continue;
		}

		const size_t alias_n_values = column_n_values(expressions[i]);
		const size_t src_n_values   = column_n_values(expressions[src_idx]);
		if (alias_n_values != src_n_values) {
			throw std::runtime_error("alias/source value count mismatch");
		}

		// Alias columns share the same host result buffer; avoid deep-copy.
		materialized.columns[i]                       = materialized.columns[src_idx];
		materialized.columns[i]->meta.column_index    = i;
		materialized.columns[i]->meta.column_name     = col->name;
		materialized.columns[i]->meta.value_count     = alias_n_values;
		materialized.columns[i]->meta.values_per_step = cfg.chunk().values_per_step();
	}

	for (size_t i = 0; i < expressions.size(); ++i) {
		const auto* col = expressions[i].column;
		if (!col || !materialized.columns[i].has_value()) {
			continue;
		}
		materialized.columns[i]->meta.column_index    = i;
		materialized.columns[i]->meta.column_name     = col->name;
		materialized.columns[i]->meta.values_per_step = cfg.chunk().values_per_step();
	}

	guard.dismiss();
	free_batches(workset);
	return materialized;
}

inline BenchmarkResult run_benchmark(PreparedBatches&& prepared, const Config& cfg) {
	BenchmarkResult bench {};
	bench.n_samples     = cfg.n_samples;
	bench.n_expressions = prepared.n_expressions;
	bench.total_bytes   = prepared.total_bytes;
	bench.n_work_items  = prepared.n_work_items;

	BenchmarkWorkset workset {};
	workset.host_batches = std::move(prepared.host_batches);
	WorksetCleanupGuard guard(workset);

	prepare_dispatch_buffers(workset);
	bench.n_work_items = workset.work_items.size();
	bench.total_ms     = run_kernel(workset, cfg.n_samples, false, false);
	bench.avg_us       = (cfg.n_samples > 0) ? (bench.total_ms * 1000.0 / static_cast<double>(cfg.n_samples)) : 0.0;

	guard.dismiss();
	free_batches(workset);
	return bench;
}

} // namespace

RowgroupExecuteResult
execute_rowgroup(std::vector<expr::Expression>& expressions, const Config& cfg, const ExecuteMode mode) {
	dispatch::resolve_dict_refs(expressions);
	// dispatch::sync_expression_ops_after_resolve(expressions); // only for validation

	auto                  prepared = prepare_batches(expressions);
	RowgroupExecuteResult out;
	if (mode == ExecuteMode::Materialize) {
		out.materialized = run_materialize(prepared, expressions, cfg);
		return out;
	}

	out.benchmark = run_benchmark(std::move(prepared), cfg);
	return out;
}

RowgroupExecuteResult
execute_rowgroup(const std::vector<expr::Expression>& expressions, const Config& cfg, const ExecuteMode mode) {
	auto mutable_expressions = expressions;
	return execute_rowgroup(mutable_expressions, cfg, mode);
}

RowgroupData decompress_rowgroup(std::vector<expr::Expression>& expressions, const Config& cfg) {
	auto exec = execute_rowgroup(expressions, cfg, ExecuteMode::Materialize);
	return std::move(exec.materialized.value());
}

RowgroupData decompress_rowgroup(const std::vector<expr::Expression>& expressions, const Config& cfg) {
	auto mutable_expressions = expressions;
	return decompress_rowgroup(mutable_expressions, cfg);
}

BenchmarkResult benchmark_rowgroup(std::vector<expr::Expression>& expressions, const Config& cfg) {
	auto exec = execute_rowgroup(expressions, cfg, ExecuteMode::BenchmarkOnly);
	return std::move(exec.benchmark.value());
}

BenchmarkResult benchmark_rowgroup(const std::vector<expr::Expression>& expressions, const Config& cfg) {
	auto mutable_expressions = expressions;
	return benchmark_rowgroup(mutable_expressions, cfg);
}

} // namespace dispatch
