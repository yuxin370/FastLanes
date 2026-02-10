// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/dispatch/table.cu
// ────────────────────────────────────────────────────────
#include "engine/dispatch/table.cuh"
#include "engine/reader.cuh"
#include <chrono>

namespace dispatch::table {

double append_expressions(TableBatches& table_batches, const std::vector<expr::Expression>& expressions) {
	using namespace dispatch;
	using namespace dispatch::detail;

	const auto start = std::chrono::steady_clock::now();

	for (size_t i = 0; i < expressions.size(); ++i) {
		const auto& expr = expressions[i];
		if (!expr.column || expr.column->skip_decompress) {
			continue;
		}
		const auto plan = plan_for_ops(expr.ops);

		std::visit(
		    [&](auto&& host_col) {
			    using HostColT = std::decay_t<decltype(host_col)>;
			    using T        = typename dispatch::host_value_type<HostColT>::type;
			    if constexpr (dispatch::is_supported_type_v<T>) {
				    add_expression_to_batch<T>(i, host_col, plan, table_batches.host_batches.template get<T>());
			    }
		    },
		    expr.column->host);
	}

	const auto end = std::chrono::steady_clock::now();
	return std::chrono::duration<double, std::milli>(end - start).count();
}

double finalize_batches(TableBatches& table_batches) {
	const auto start = std::chrono::steady_clock::now();

	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T          = typename decltype(tag)::type;
		auto& host_batch = table_batches.host_batches.template get<T>();
		if (!host_batch.device_exprs.empty() && !host_batch.work_items.empty()) {
			auto& dev_batch = table_batches.device_batches.template get<T>();
			dev_batch.d_exprs.emplace(host_batch.device_exprs.size(), host_batch.device_exprs.data());
			table_batches.work_items.reserve(table_batches.work_items.size() + host_batch.work_items.size());
			table_batches.work_items.insert(
			    table_batches.work_items.end(), host_batch.work_items.begin(), host_batch.work_items.end());
		}
	});

	if (!table_batches.work_items.empty()) {
		table_batches.d_items.emplace(table_batches.work_items.size(), table_batches.work_items.data());
	}

	const auto end = std::chrono::steady_clock::now();
	return std::chrono::duration<double, std::milli>(end - start).count();
}

double run_kernel(TableBatches& table_batches, uint32_t samples, size_t* out_grid) {
	const size_t n_items = table_batches.work_items.size();
	if (n_items == 0 || !table_batches.d_items.has_value()) {
		return 0.0;
	}

	auto* d_exprs_i8  = table_batches.device_batches.template get<int8_t>().d_exprs
	                        ? table_batches.device_batches.template get<int8_t>().d_exprs->get()
	                        : nullptr;
	auto* d_exprs_i16 = table_batches.device_batches.template get<int16_t>().d_exprs
	                        ? table_batches.device_batches.template get<int16_t>().d_exprs->get()
	                        : nullptr;
	if (!d_exprs_i8 && !d_exprs_i16) {
		return 0.0;
	}

	constexpr unsigned UNPACK_N_VECTORS = 1;
	constexpr unsigned UNPACK_N_VALUES  = 1;
	const auto         launch           = make_table_launch_config(n_items);
	if (out_grid) {
		*out_grid = launch.grid.x;
	}

	flsgpu::memory::sync_h2d();

	cudaEvent_t start {};
	cudaEvent_t stop {};
	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));

	CUDA_SAFE_CALL(cudaEventRecord(start, 0));

	for (uint32_t sample = 0; sample < samples; ++sample) {
		kernels::device::decompress_table<UNPACK_N_VECTORS, UNPACK_N_VALUES>
		    <<<launch.grid, launch.block>>>(d_exprs_i8, d_exprs_i16, table_batches.d_items->get(), n_items);
		CUDA_SAFE_CALL(cudaGetLastError());
	}

	CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));

	float ms = 0.0f;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, start, stop));
	CUDA_SAFE_CALL(cudaEventDestroy(start));
	CUDA_SAFE_CALL(cudaEventDestroy(stop));

	return static_cast<double>(ms);
}

void free_batches(TableBatches& table_batches) {
	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T          = typename decltype(tag)::type;
		auto& host_batch = table_batches.host_batches.template get<T>();
		for (auto& expr : host_batch.device_exprs) {
			dispatch::detail::free_device_expr(expr);
		}
	});
}

} // namespace dispatch::table
