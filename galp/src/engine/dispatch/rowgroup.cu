// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/dispatch/rowgroup.cu
// ────────────────────────────────────────────────────────
#include "engine/dispatch/rowgroup.cuh"
#include "engine/reader.cuh"
#include <cstring>

namespace dispatch {
namespace detail {

template <typename T>
void release_batch(Batch<T>& batch) {
	for (auto& expr : batch.device_exprs) {
		free_device_expr(expr);
	}
}

template <typename T>
void launch_batch_no_sync(const Batch<T>&                batch,
                          GPUArray<DeviceExpression<T>>& d_exprs,
                          GPUArray<WorkItemAny>&         d_items) {
	if (batch.device_exprs.empty() || batch.work_items.empty()) {
		return;
	}
	constexpr unsigned UNPACK_N_VECTORS = 1;
	constexpr unsigned UNPACK_N_VALUES  = 1;
	const int          threads          = utils::get_n_lanes<T>();
	const dim3         block(static_cast<unsigned>(threads));
	const dim3         grid(static_cast<unsigned>(batch.work_items.size()));

	kernels::device::decompress_rowgroup<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>
	    <<<grid, block>>>(d_exprs.get(), d_items.get(), batch.work_items.size());
	CUDA_SAFE_CALL(cudaGetLastError());
}

} // namespace detail

namespace {

inline size_t resolve_alias(const std::vector<expr::Expression>& expressions, size_t idx) {
	size_t cur = idx;
	for (;;) {
		const auto* col = expressions[cur].column;
		if (!col || !col->alias_of.has_value()) {
			return cur;
		}
		cur = *col->alias_of;
	}
}

inline DecompressResult clone_result(const DecompressResult& src, const size_t n_values) {
	return std::visit(
	    [&](auto&& ptr) -> DecompressResult {
		    using T  = std::remove_pointer_t<decltype(ptr.get())>;
		    auto out = std::make_unique<T[]>(n_values);
		    std::memcpy(out.get(), ptr.get(), n_values * sizeof(T));
		    return DecompressResult {std::move(out)};
	    },
	    src);
}

} // namespace

RowgroupDecompressResult decompress_rowgroup(const std::vector<expr::Expression>& expressions, const Config& cfg) {
	(void)cfg;
	RowgroupDecompressResult result;
	result.columns.resize(expressions.size());

	typename BatchSetFromList<SupportedTypes>::type batches;

	for (size_t i = 0; i < expressions.size(); ++i) {
		const auto& expr = expressions[i];
		if (!expr.column || expr.column->skip_decompress) {
			continue;
		}
		const auto plan = plan_for_ops(expr.ops);

		std::visit(
		    [&](auto&& host_col) {
			    using HostColT = std::decay_t<decltype(host_col)>;
			    using T        = typename host_value_type<HostColT>::type;

			    static_assert(is_supported_type_v<T>, "dispatch rowgroup only supports int8_t/int16_t");
			    add_expression_to_batch<T>(i, host_col, plan, batches.template get<T>());
		    },
		    expr.column->host);
	}

	dispatch::for_each_type(SupportedTypes {}, [&](auto tag) {
		using T     = typename decltype(tag)::type;
		auto& batch = batches.template get<T>();
		detail::launch_batch(batch);
		detail::finalize_batch(batch, result);
	});

	for (size_t i = 0; i < expressions.size(); ++i) {
		const auto* col = expressions[i].column;
		if (!col || !col->alias_of.has_value()) {
			continue;
		}
		const size_t src_idx = resolve_alias(expressions, i);
		if (src_idx >= result.columns.size() || !result.columns[src_idx].has_value()) {
			throw std::runtime_error("EXP_EQUAL: source column not decompressed");
		}
		if (result.columns[i].has_value()) {
			continue;
		}
		const size_t n_values =
		    std::visit([](auto&& host_col) -> size_t { return host_col.get_n_values(); }, col->host);
		result.columns[i] = clone_result(*result.columns[src_idx], n_values);
	}

	return result;
}

BenchmarkResult benchmark_rowgroup(const std::vector<expr::Expression>& expressions, const Config& cfg) {
	BenchmarkResult result {};
	result.n_samples = cfg.n_samples;

	typename BatchSetFromList<SupportedTypes>::type                 batches;
	typename dispatch::DeviceBatchSetFromList<SupportedTypes>::type device_batches;

	for (size_t i = 0; i < expressions.size(); ++i) {
		const auto& expr = expressions[i];
		if (!expr.column || expr.column->skip_decompress) {
			continue;
		}
		const auto plan = plan_for_ops(expr.ops);
		++result.n_expressions;

		std::visit(
		    [&](auto&& host_col) {
			    using HostColT = std::decay_t<decltype(host_col)>;
			    using T        = typename host_value_type<HostColT>::type;

			    static_assert(is_supported_type_v<T>, "dispatch rowgroup only supports int8_t/int16_t");
			    result.total_bytes += host_col.get_n_values() * sizeof(T);
			    add_expression_to_batch<T>(i, host_col, plan, batches.template get<T>());
		    },
		    expr.column->host);
	}

	dispatch::for_each_type(SupportedTypes {}, [&](auto tag) {
		using T     = typename decltype(tag)::type;
		auto& batch = batches.template get<T>();
		result.n_work_items += batch.work_items.size();
		if (batch.device_exprs.empty() || batch.work_items.empty()) {
			return;
		}
		auto& dev = device_batches.template get<T>();
		dev.d_exprs.emplace(batch.device_exprs.size(), batch.device_exprs.data());
		dev.d_items.emplace(batch.work_items.size(), batch.work_items.data());
	});

	struct EventPair {
		cudaEvent_t start {};
		cudaEvent_t stop {};
	};

	EventPair ev;
	CUDA_SAFE_CALL(cudaEventCreate(&ev.start));
	CUDA_SAFE_CALL(cudaEventCreate(&ev.stop));

	CUDA_SAFE_CALL(cudaEventRecord(ev.start, 0));

	for (uint32_t sample = 0; sample < cfg.n_samples; ++sample) {
		dispatch::for_each_type(SupportedTypes {}, [&](auto tag) {
			using T     = typename decltype(tag)::type;
			auto& batch = batches.template get<T>();
			auto& dev   = device_batches.template get<T>();
			if (dev.empty()) {
				return;
			}
			detail::launch_batch_no_sync<T>(batch, *dev.d_exprs, *dev.d_items);
		});
	}

	CUDA_SAFE_CALL(cudaEventRecord(ev.stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(ev.stop));

	float ms = 0.0f;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, ev.start, ev.stop));
	CUDA_SAFE_CALL(cudaEventDestroy(ev.start));
	CUDA_SAFE_CALL(cudaEventDestroy(ev.stop));

	result.total_ms = static_cast<double>(ms);
	result.avg_us   = (cfg.n_samples > 0) ? (result.total_ms * 1000.0 / static_cast<double>(cfg.n_samples)) : 0.0;

	dispatch::for_each_type(SupportedTypes {}, [&](auto tag) {
		using T     = typename decltype(tag)::type;
		auto& batch = batches.template get<T>();
		detail::release_batch(batch);
	});

	return result;
}

} // namespace dispatch
