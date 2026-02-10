// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/dispatch.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_DISPATCH_CUH
#define ENGINE_DISPATCH_CUH

#include "engine/device-utils.cuh"
#include "engine/expression.cuh"
#include "engine/kernels.cuh"
#include "engine/types.cuh"
#include "flsgpu/host-utils.cuh"
#include "flsgpu/structs.cuh"
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

namespace reader {
struct Rowgroup;
} // namespace reader

namespace dispatch {

template <typename>
inline constexpr bool always_false_v = false;

struct Config {
	unsigned unpack_n_vectors = 1;
	unsigned unpack_n_values  = 1;
	uint32_t n_samples        = 1;
};

struct RowgroupDecompressResult {
	std::vector<std::optional<DecompressResult>> columns;
};

struct BenchmarkResult {
	double   total_ms;
	double   avg_us;
	uint32_t n_samples;
	size_t   n_expressions;
	size_t   n_work_items;
	size_t   total_bytes;
};

template <typename ColumnT>
struct column_value_type;
template <typename T>
struct column_value_type<flsgpu::device::BPColumn<T>> {
	using type = T;
};
template <typename T>
struct column_value_type<flsgpu::device::CONSTANTColumn<T>> {
	using type = T;
};
template <typename T>
struct column_value_type<flsgpu::device::FFORColumn<T>> {
	using type = T;
};
template <typename T, typename IndexT>
struct column_value_type<flsgpu::device::DICTFFORColumn<T, IndexT>> {
	using type = T;
};
template <typename T, typename IndexT>
struct column_value_type<flsgpu::device::DICTSLPATCHColumn<T, IndexT>> {
	using type = T;
};
template <typename T>
struct column_value_type<flsgpu::device::FREQColumn<T>> {
	using type = T;
};
template <typename T>
struct column_value_type<flsgpu::device::CROSSRLEColumn<T>> {
	using type = T;
};
template <typename T>
struct column_value_type<flsgpu::device::SLPATCHColumn<T>> {
	using type = T;
};
template <typename T, typename IndexT>
struct column_value_type<flsgpu::device::RLEColumn<T, IndexT>> {
	using type = T;
};

template <typename ColumnT>
struct host_value_type;

template <typename T>
struct host_value_type<flsgpu::host::BPColumn<T>> {
	using type = T;
};
template <typename T>
struct host_value_type<flsgpu::host::CONSTANTColumn<T>> {
	using type = T;
};
template <typename T>
struct host_value_type<flsgpu::host::FFORColumn<T>> {
	using type = T;
};
template <typename T, typename IndexT>
struct host_value_type<flsgpu::host::DICTFFORColumn<T, IndexT>> {
	using type = T;
};
template <typename T, typename IndexT>
struct host_value_type<flsgpu::host::DICTSLPATCHColumn<T, IndexT>> {
	using type = T;
};
template <typename T>
struct host_value_type<flsgpu::host::FREQColumn<T>> {
	using type = T;
};
template <typename T>
struct host_value_type<flsgpu::host::CROSSRLEColumn<T>> {
	using type = T;
};
template <typename T>
struct host_value_type<flsgpu::host::SLPATCHColumn<T>> {
	using type = T;
};
template <typename T, typename IndexT>
struct host_value_type<flsgpu::host::RLEColumn<T, IndexT>> {
	using type = T;
};

namespace detail {

template <typename HostColT>
DecompressResult decompress_host(const HostColT& host_col, const PlanKind plan, const Config& cfg);

template <typename HostColT>
struct host_plan_kind;

template <typename T>
struct host_plan_kind<flsgpu::host::BPColumn<T>> {
	static constexpr PlanKind value = PlanKind::UNCOMPRESSED;
};
template <typename T>
struct host_plan_kind<flsgpu::host::CONSTANTColumn<T>> {
	static constexpr PlanKind value = PlanKind::CONSTANT;
};
template <typename T>
struct host_plan_kind<flsgpu::host::FFORColumn<T>> {
	static constexpr PlanKind value = PlanKind::UNFFOR;
};
template <typename T>
struct host_plan_kind<flsgpu::host::SLPATCHColumn<T>> {
	static constexpr PlanKind value = PlanKind::UNFFOR_SLPATCH;
};
template <typename T, typename IndexT>
struct host_plan_kind<flsgpu::host::DICTFFORColumn<T, IndexT>> {
	static constexpr PlanKind value = PlanKind::DICT_FFOR;
};
template <typename T, typename IndexT>
struct host_plan_kind<flsgpu::host::DICTSLPATCHColumn<T, IndexT>> {
	static constexpr PlanKind value = PlanKind::DICT_FFOR_SLPATCH;
};
template <typename T>
struct host_plan_kind<flsgpu::host::FREQColumn<T>> {
	static constexpr PlanKind value = PlanKind::FREQUENCY;
};
template <typename T>
struct host_plan_kind<flsgpu::host::CROSSRLEColumn<T>> {
	static constexpr PlanKind value = PlanKind::CROSS_RLE;
};
template <typename T, typename IndexT>
struct host_plan_kind<flsgpu::host::RLEColumn<T, IndexT>> {
	static constexpr PlanKind value = PlanKind::RLE;
};

template <typename HostColT>
constexpr PlanKind plan_for_host_col() {
	return host_plan_kind<HostColT>::value;
}

template <typename T, typename HostColT>
void fill_device_expr(DeviceExpression<T>& expr, const HostColT& host_col, const PlanKind plan) {
	switch (plan) {
	case PlanKind::UNCOMPRESSED:
		if constexpr (std::is_same_v<HostColT, flsgpu::host::BPColumn<T>>) {
			expr.col.bp = host_col.copy_to_device();
			return;
		}
		break;
	case PlanKind::CONSTANT:
		if constexpr (std::is_same_v<HostColT, flsgpu::host::CONSTANTColumn<T>>) {
			expr.col.constant = host_col.copy_to_device();
			return;
		}
		break;
	case PlanKind::FREQUENCY:
		if constexpr (std::is_same_v<HostColT, flsgpu::host::FREQColumn<T>>) {
			expr.col.freq = host_col.copy_to_device();
			return;
		}
		break;
	case PlanKind::UNFFOR:
		if constexpr (std::is_same_v<HostColT, flsgpu::host::FFORColumn<T>>) {
			expr.col.ffor = host_col.copy_to_device();
			return;
		}
		break;
	case PlanKind::UNFFOR_SLPATCH:
		if constexpr (std::is_same_v<HostColT, flsgpu::host::SLPATCHColumn<T>>) {
			expr.col.slpatch = host_col.copy_to_device();
			return;
		}
		break;
	case PlanKind::DICT_FFOR:
		if constexpr (std::is_same_v<HostColT, flsgpu::host::DICTFFORColumn<T, uint8_t>>) {
			expr.dict_index_bits = 8;
			expr.col.dictffor_u8 = host_col.copy_to_device();
			return;
		}
		if constexpr (std::is_same_v<HostColT, flsgpu::host::DICTFFORColumn<T, uint16_t>>) {
			expr.dict_index_bits = 16;
			expr.col.dictffor    = host_col.copy_to_device();
			return;
		}
		break;
	case PlanKind::DICT_FFOR_SLPATCH:
		if constexpr (std::is_same_v<HostColT, flsgpu::host::DICTSLPATCHColumn<T, uint8_t>>) {
			expr.dict_index_bits    = 8;
			expr.col.dictslpatch_u8 = host_col.copy_to_device();
			return;
		}
		if constexpr (std::is_same_v<HostColT, flsgpu::host::DICTSLPATCHColumn<T, uint16_t>>) {
			expr.dict_index_bits = 16;
			expr.col.dictslpatch = host_col.copy_to_device();
			return;
		}
		break;
	case PlanKind::CROSS_RLE:
		if constexpr (std::is_same_v<HostColT, flsgpu::host::CROSSRLEColumn<T>>) {
			expr.col.crossrle = host_col.copy_to_device();
			return;
		}
		break;
	case PlanKind::RLE:
		if constexpr (std::is_same_v<HostColT, flsgpu::host::RLEColumn<T, uint16_t>>) {
			expr.col.rle = host_col.copy_to_device();
			return;
		}
		break;
	default:
		break;
	}
	throw std::runtime_error("dispatch rowgroup plan/column mismatch");
}

template <typename T>
void free_device_expr(const DeviceExpression<T>& expr) {
	switch (expr.plan) {
	case PlanKind::UNCOMPRESSED:
		flsgpu::host::free_column(expr.col.bp);
		break;
	case PlanKind::CONSTANT:
		flsgpu::host::free_column(expr.col.constant);
		break;
	case PlanKind::FREQUENCY:
		flsgpu::host::free_column(expr.col.freq);
		break;
	case PlanKind::UNFFOR:
		flsgpu::host::free_column(expr.col.ffor);
		break;
	case PlanKind::UNFFOR_SLPATCH:
		flsgpu::host::free_column(expr.col.slpatch);
		break;
	case PlanKind::DICT_FFOR:
		if (expr.dict_index_bits == 8) {
			flsgpu::host::free_column(expr.col.dictffor_u8);
		} else {
			flsgpu::host::free_column(expr.col.dictffor);
		}
		break;
	case PlanKind::DICT_FFOR_SLPATCH:
		if (expr.dict_index_bits == 8) {
			flsgpu::host::free_column(expr.col.dictslpatch_u8);
		} else {
			flsgpu::host::free_column(expr.col.dictslpatch);
		}
		break;
	case PlanKind::CROSS_RLE:
		flsgpu::host::free_column(expr.col.crossrle);
		break;
	case PlanKind::RLE:
		flsgpu::host::free_column(expr.col.rle);
		break;
	default:
		break;
	}
}

template <typename T>
struct Batch {
	std::vector<size_t>              expr_indices;
	std::vector<DeviceExpression<T>> device_exprs;
	std::vector<GPUArray<T>>         device_outputs;
	std::vector<WorkItem>            work_items;
};

template <typename... Ts>
struct BatchSet {
	std::tuple<Batch<Ts>...> batches;

	template <typename T>
	Batch<T>& get() {
		return std::get<Batch<T>>(batches);
	}
};

template <typename List>
struct BatchSetFromList;

template <typename... Ts>
struct BatchSetFromList<dispatch::TypeList<Ts...>> {
	using type = BatchSet<Ts...>;
};

template <typename T, typename HostColT>
void add_expression_to_batch(const size_t expr_index, const HostColT& host_col, const PlanKind plan, Batch<T>& batch) {
	DeviceExpression<T> expr {};
	expr.plan     = plan;
	expr.n_values = host_col.get_n_values();
	batch.device_outputs.emplace_back(expr.n_values);
	expr.out = batch.device_outputs.back().get();

	fill_device_expr(expr, host_col, plan);

	const auto device_idx = static_cast<uint32_t>(batch.device_exprs.size());
	batch.device_exprs.push_back(expr);
	batch.expr_indices.push_back(expr_index);

	const size_t n_vecs = utils::get_n_vecs_from_size(expr.n_values);
	for (size_t vec = 0; vec < n_vecs; ++vec) {
		batch.work_items.push_back(WorkItem {device_idx, static_cast<uint32_t>(vec)});
	}
}

template <typename T>
void launch_batch(const Batch<T>& batch) {
	if (batch.device_exprs.empty() || batch.work_items.empty()) {
		return;
	}
	GPUArray<DeviceExpression<T>> d_exprs(batch.device_exprs.size(), batch.device_exprs.data());
	GPUArray<WorkItem>            d_items(batch.work_items.size(), batch.work_items.data());
	flsgpu::memory::sync_h2d();

	constexpr unsigned UNPACK_N_VECTORS = 1;
	constexpr unsigned UNPACK_N_VALUES  = 1;
	const auto         launch           = make_workitem_launch_config<T, UNPACK_N_VECTORS>(batch.work_items.size());

	kernels::device::decompress_rowgroup<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>
	    <<<launch.grid, launch.block>>>(d_exprs.get(), d_items.get(), batch.work_items.size());
	CUDA_SAFE_CALL(cudaGetLastError());
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

template <typename T>
void finalize_batch(Batch<T>& batch, RowgroupDecompressResult& result) {
	for (size_t idx = 0; idx < batch.device_exprs.size(); ++idx) {
		auto& expr = batch.device_exprs[idx];
		auto  host = std::make_unique<T[]>(expr.n_values);
		batch.device_outputs[idx].copy_to_host(host.get());
		result.columns[batch.expr_indices[idx]] = DecompressResult {std::move(host)};
		free_device_expr(expr);
	}
}

} // namespace detail

DecompressResult         decompress(const expr::Expression& expression, const Config& cfg = {});
RowgroupDecompressResult decompress_rowgroup(const std::vector<expr::Expression>& expressions, const Config& cfg = {});
BenchmarkResult          benchmark_rowgroup(const std::vector<expr::Expression>& expressions, const Config& cfg = {});

} // namespace dispatch

#endif // ENGINE_DISPATCH_CUH
