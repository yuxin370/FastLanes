// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/operators/batch.cuh
// ────────────────────────────────────────────────────────
// Host- and device-side batch containers plus their TypeList-driven
// parameter-pack wrappers. No dispatch logic — just the data shapes that
// fill_device_expr / launch_batch / finalize_batch operate on.
#ifndef ENGINE_EXECUTION_BATCH_CUH
#define ENGINE_EXECUTION_BATCH_CUH

#include "core/expression.cuh"
#include "core/lane_policy.cuh"
#include "cuda/memory/gpu_array.cuh"
#include <cstddef>
#include <optional>
#include <tuple>
#include <vector>

namespace galp::execution {

template <typename T>
struct Batch {
	std::vector<size_t>              expr_indices;
	std::vector<DeviceExpression<T>> device_exprs;
	std::vector<size_t>              output_offsets;
	std::vector<WorkItemAny>         work_items;
};

template <typename... Ts>
struct BatchSet {
	std::tuple<Batch<Ts>...> batches;

	template <typename T>
	Batch<T>& get() {
		return std::get<Batch<T>>(batches);
	}

	template <typename T>
	const Batch<T>& get() const {
		return std::get<Batch<T>>(batches);
	}
};

template <typename List>
struct BatchSetFromList;

template <typename... Ts>
struct BatchSetFromList<galp::execution::TypeList<Ts...>> {
	using type = BatchSet<Ts...>;
};

template <typename T>
struct DeviceBatch {
	std::optional<GPUArray<DeviceExpression<T>>> owned_exprs;
	std::optional<GPUArray<WorkItemAny>>         owned_items;
	DeviceExpression<T>* d_exprs = nullptr;
	WorkItemAny*         d_items = nullptr;
	size_t               n_items = 0;
};

template <typename... Ts>
struct DeviceBatchSet {
	std::tuple<DeviceBatch<Ts>...> batches;

	template <typename T>
	DeviceBatch<T>& get() {
		return std::get<DeviceBatch<T>>(batches);
	}

	template <typename T>
	const DeviceBatch<T>& get() const {
		return std::get<DeviceBatch<T>>(batches);
	}
};

template <typename List>
struct DeviceBatchSetFromList;

template <typename... Ts>
struct DeviceBatchSetFromList<galp::execution::TypeList<Ts...>> {
	using type = DeviceBatchSet<Ts...>;
};

} // namespace galp::execution

#endif // ENGINE_EXECUTION_BATCH_CUH
