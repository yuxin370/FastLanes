// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/operators/batch_kernel.cuh
// ────────────────────────────────────────────────────────
// Low-level per-Batch kernel launching + finalization. Higher-level workset
// orchestration (`runtime::launch_typed_batches` etc.) lives in
// cuda/launch/launch.cuh and builds on these primitives.
#ifndef ENGINE_EXECUTION_INTERNAL_BATCH_KERNEL_CUH
#define ENGINE_EXECUTION_INTERNAL_BATCH_KERNEL_CUH

#include "core/data/model.cuh"
#include "core/data/value_store.cuh"
#include "engine/operators/batch.cuh"
#include "engine/config.cuh"
#include "engine/unpack_dispatch.cuh"
#include "core/expression.cuh"
#include "cuda/launch/dispatch.cuh"
#include "core/lane_policy.cuh"
#include "core/types.cuh"
#include "cuda/cuda_macros.cuh"
#include "cuda/memory/device_pool.cuh"
#include "cuda/memory/gpu_array.cuh"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace galp::execution::detail {

template <typename T, bool WRITE_OUT = true>
void launch_batch_no_sync(const galp::execution::Batch<T>&  batch,
                          const DeviceExpression<T>* d_exprs,
                          const WorkItemAny*         d_items,
                          const size_t               n_items,
                          const ExecutionConfig&     cfg,
                          cudaStream_t               stream = 0) {
	if (batch.device_exprs.empty() || !d_exprs || !d_items || n_items == 0) {
		return;
	}
	uint32_t           threads          = static_cast<uint32_t>(galp::codec::utils::get_n_lanes<T>());
	for (const auto& work : batch.work_items) {
		threads = std::max(threads,
		                   galp::execution::semantic_lane_count(type_tag_for<T>(), batch.device_exprs[work.expr_index].plan));
	}
	const dim3 block(static_cast<unsigned>(threads));
	const dim3 grid(static_cast<unsigned>(n_items));

	runtime::with_unpack_config(cfg, [&](auto unpack_n_vectors, auto unpack_n_values) {
		constexpr unsigned UNPACK_N_VECTORS = decltype(unpack_n_vectors)::value;
		constexpr unsigned UNPACK_N_VALUES  = decltype(unpack_n_values)::value;
		galp::kernels::device::decompress_dispatch_typed<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>
		    <<<grid, block, 0, stream>>>(d_exprs, d_items, n_items);
		CUDA_SAFE_CALL(cudaGetLastError());
	});
}

template <typename T>
void launch_batch(const Batch<T>& batch, const ExecutionConfig& cfg = {}) {
	if (batch.device_exprs.empty() || batch.work_items.empty()) {
		return;
	}
	GPUArray<DeviceExpression<T>> d_exprs(batch.device_exprs.size(), batch.device_exprs.data());
	GPUArray<WorkItemAny>         d_items(batch.work_items.size(), batch.work_items.data());
	galp::memory::sync_h2d();
	launch_batch_no_sync<T>(batch, d_exprs.get(), d_items.get(), batch.work_items.size(), cfg);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

template <typename T>
void finalize_batch(Batch<T>& batch, RowgroupData& result) {
	for (size_t idx = 0; idx < batch.device_exprs.size(); ++idx) {
		auto& expr = batch.device_exprs[idx];
		auto  host = std::shared_ptr<T[]>(new T[expr.n_values], std::default_delete<T[]>());
		if (expr.n_values > 0) {
			if (expr.out == nullptr) {
				throw std::runtime_error("device output pointer not initialized");
			}
			CUDA_SAFE_CALL(cudaMemcpy(host.get(), expr.out, expr.n_values * sizeof(T), cudaMemcpyDeviceToHost));
		}
		MaterializedColumn out {};
		out.values                              = ValueStore {std::move(host)};
		out.meta.column_index                   = batch.expr_indices[idx];
		out.meta.value_count                    = expr.n_values;
		out.meta.value_type                     = galp::format::ToDataType<T>::value;
		out.meta.values_per_step                = 1;
		result.columns[batch.expr_indices[idx]] = std::move(out);
		// Arena mode: column pointers live in chunk_arena device_base_; per-expr
		// free_device_expr would cudaFree interior offsets. Arena teardown frees them.
	}
	batch.device_exprs.clear();
	batch.output_offsets.clear();
	batch.work_items.clear();
	batch.expr_indices.clear();
}

} // namespace galp::execution::detail

#endif // ENGINE_EXECUTION_INTERNAL_BATCH_KERNEL_CUH
