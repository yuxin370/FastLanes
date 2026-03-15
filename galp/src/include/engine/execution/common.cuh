// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/execution/common.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_COMMON_CUH
#define ENGINE_EXECUTION_COMMON_CUH

#include "engine/data/model.cuh"
#include "engine/data/value-store.cuh"
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

namespace dispatch {

template <typename>
inline constexpr bool always_false_v = false;

struct Config {
	unsigned unpack_n_vectors = 1;
	unsigned unpack_n_values  = 1;
	uint32_t n_samples        = 1;
	bool     gpu_dispatch_kernel = true;

	constexpr DecodeChunk chunk() const {
		return DecodeChunk {unpack_n_vectors, unpack_n_values};
	}
};

enum class ExecuteMode {
	Materialize,
	BenchmarkOnly,
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
struct host_value_type<flsgpu::host::DICTREFColumn<T, IndexT>> {
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
ValueStore decompress_host(const HostColT& host_col, const PlanKind plan, const Config& cfg);

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
	static constexpr PlanKind value =
	    std::is_same_v<IndexT, uint8_t> ? PlanKind::DICT_FFOR_U8 : PlanKind::DICT_FFOR_U16;
};
template <typename T, typename IndexT>
struct host_plan_kind<flsgpu::host::DICTREFColumn<T, IndexT>> {
	static constexpr PlanKind value =
	    std::is_same_v<IndexT, uint8_t> ? PlanKind::DICT_FFOR_U8 : PlanKind::DICT_FFOR_U16;
};
template <typename T, typename IndexT>
struct host_plan_kind<flsgpu::host::DICTSLPATCHColumn<T, IndexT>> {
	static constexpr PlanKind value =
	    std::is_same_v<IndexT, uint8_t> ? PlanKind::DICT_FFOR_SLPATCH_U8 : PlanKind::DICT_FFOR_SLPATCH_U16;
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

template <typename T>
bool should_use_freq_extended(const flsgpu::host::FREQColumn<T>& host_col,
                              const bool                         freq_prefetch_all_branchless,
                              const bool                         freq_hybrid_patcher,
                              const float                        freq_branchless_threshold) {
	if (!freq_prefetch_all_branchless) {
		return false;
	}
	if (!freq_hybrid_patcher) {
		return true;
	}
	if (host_col.get_n_vecs() == 0) {
		return false;
	}
	const double exc_per_vec =
	    static_cast<double>(host_col.n_exceptions) / static_cast<double>(host_col.get_n_vecs());
	return exc_per_vec >= static_cast<double>(freq_branchless_threshold);
}

template <typename T, typename HostColT>
void fill_device_expr(DeviceExpression<T>& expr,
                      const HostColT&      host_col,
                      const PlanKind       plan,
                      const bool           freq_use_extended            = false) {
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
			if (freq_use_extended) {
				auto extended          = host_col.create_extended_column();
				expr.col.freq_extended = extended.copy_to_device();
				expr.freq_use_extended = true;
				flsgpu::host::free_column(extended);
				return;
			}
			expr.col.freq          = host_col.copy_to_device();
			expr.freq_use_extended = false;
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
	case PlanKind::DICT_FFOR_U8:
		if constexpr (std::is_same_v<HostColT, flsgpu::host::DICTFFORColumn<T, uint8_t>>) {
			expr.col.dictffor_u8 = host_col.copy_to_device();
			return;
		}
		break;
	case PlanKind::DICT_FFOR_U16:
		if constexpr (std::is_same_v<HostColT, flsgpu::host::DICTFFORColumn<T, uint16_t>>) {
			expr.col.dictffor_u16 = host_col.copy_to_device();
			return;
		}
		break;
	case PlanKind::DICT_FFOR_SLPATCH_U8:
		if constexpr (std::is_same_v<HostColT, flsgpu::host::DICTSLPATCHColumn<T, uint8_t>>) {
			expr.col.dictslpatch_u8 = host_col.copy_to_device();
			return;
		}
		break;
	case PlanKind::DICT_FFOR_SLPATCH_U16:
		if constexpr (std::is_same_v<HostColT, flsgpu::host::DICTSLPATCHColumn<T, uint16_t>>) {
			expr.col.dictslpatch_u16 = host_col.copy_to_device();
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
		if (expr.freq_use_extended) {
			flsgpu::host::free_column(expr.col.freq_extended);
		} else {
			flsgpu::host::free_column(expr.col.freq);
		}
		break;
	case PlanKind::UNFFOR:
		flsgpu::host::free_column(expr.col.ffor);
		break;
	case PlanKind::UNFFOR_SLPATCH:
		flsgpu::host::free_column(expr.col.slpatch);
		break;
	case PlanKind::DICT_FFOR_U8:
		flsgpu::host::free_column(expr.col.dictffor_u8);
		break;
	case PlanKind::DICT_FFOR_U16:
		flsgpu::host::free_column(expr.col.dictffor_u16);
		break;
	case PlanKind::DICT_FFOR_SLPATCH_U8:
		flsgpu::host::free_column(expr.col.dictslpatch_u8);
		break;
	case PlanKind::DICT_FFOR_SLPATCH_U16:
		flsgpu::host::free_column(expr.col.dictslpatch_u16);
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

} // namespace detail

template <typename T>
struct Batch {
	std::vector<size_t>              expr_indices;
	std::vector<DeviceExpression<T>> device_exprs;
	std::vector<GPUArray<T>>         device_outputs;
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
struct BatchSetFromList<dispatch::TypeList<Ts...>> {
	using type = BatchSet<Ts...>;
};

template <typename T>
struct DeviceBatch {
	std::optional<GPUArray<DeviceExpression<T>>> d_exprs;
	std::optional<GPUArray<WorkItemAny>>         d_items;
	size_t                                       n_items = 0;
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
struct DeviceBatchSetFromList<dispatch::TypeList<Ts...>> {
	using type = DeviceBatchSet<Ts...>;
};

template <typename T, typename HostColT>
void add_expression_to_batch(const size_t    expr_index,
                             const HostColT& host_col,
                             const PlanKind  plan,
                             Batch<T>&       batch,
                             const bool      freq_prefetch_all_branchless = false,
                             const bool      freq_hybrid_patcher          = false,
                             const float     freq_branchless_threshold    = 6.0f) {
	DeviceExpression<T> expr {};
	expr.plan     = plan;
	expr.n_values = host_col.get_n_values();
	batch.device_outputs.emplace_back(expr.n_values);
	expr.out = batch.device_outputs.back().get();

	bool use_freq_extended = false;
	if constexpr (std::is_same_v<HostColT, flsgpu::host::FREQColumn<T>>) {
		use_freq_extended = detail::should_use_freq_extended(
		    host_col, freq_prefetch_all_branchless, freq_hybrid_patcher, freq_branchless_threshold);
	}
	detail::fill_device_expr(expr, host_col, plan, use_freq_extended);

	const auto device_idx = static_cast<uint32_t>(batch.device_exprs.size());
	batch.device_exprs.push_back(expr);
	batch.expr_indices.push_back(expr_index);

	const size_t n_vecs = utils::get_n_vecs_from_size(expr.n_values);
	for (size_t vec = 0; vec < n_vecs; ++vec) {
		batch.work_items.push_back(WorkItemAny {device_idx, static_cast<uint32_t>(vec), type_tag_for<T>()});
	}
}

namespace detail {

template <typename T, bool WRITE_OUT = true>
void launch_batch_no_sync(const dispatch::Batch<T>&  batch,
                          const DeviceExpression<T>* d_exprs,
                          const WorkItemAny*         d_items,
                          const size_t               n_items,
                          cudaStream_t               stream = 0) {
	if (batch.device_exprs.empty() || !d_exprs || !d_items || n_items == 0) {
		return;
	}
	constexpr unsigned UNPACK_N_VECTORS = 1;
	constexpr unsigned UNPACK_N_VALUES  = 1;
	const int          threads          = utils::get_n_lanes<T>();
	const dim3         block(static_cast<unsigned>(threads));
	const dim3         grid(static_cast<unsigned>(n_items));

	kernels::device::decompress_dispatch_typed<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>
	    <<<grid, block, 0, stream>>>(d_exprs, d_items, n_items);
	CUDA_SAFE_CALL(cudaGetLastError());
}

template <typename T>
void launch_batch(const Batch<T>& batch) {
	if (batch.device_exprs.empty() || batch.work_items.empty()) {
		return;
	}
	GPUArray<DeviceExpression<T>> d_exprs(batch.device_exprs.size(), batch.device_exprs.data());
	GPUArray<WorkItemAny>         d_items(batch.work_items.size(), batch.work_items.data());
	flsgpu::memory::sync_h2d();
	launch_batch_no_sync<T>(batch, d_exprs.get(), d_items.get(), batch.work_items.size());
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

template <typename T>
void finalize_batch(Batch<T>& batch, RowgroupData& result) {
	for (size_t idx = 0; idx < batch.device_exprs.size(); ++idx) {
		auto& expr = batch.device_exprs[idx];
		auto  host = std::shared_ptr<T[]>(new T[expr.n_values], std::default_delete<T[]>());
		batch.device_outputs[idx].copy_to_host(host.get());
		MaterializedColumn out {};
		out.values                              = ValueStore {std::move(host)};
		out.meta.column_index                   = batch.expr_indices[idx];
		out.meta.value_count                    = expr.n_values;
		out.meta.value_type                     = types::ToDataType<T>::value;
		out.meta.values_per_step                = 1;
		result.columns[batch.expr_indices[idx]] = std::move(out);
		free_device_expr(expr);
	}
	batch.device_exprs.clear();
	batch.device_outputs.clear();
	batch.work_items.clear();
	batch.expr_indices.clear();
}

} // namespace detail

} // namespace dispatch

#endif // ENGINE_EXECUTION_COMMON_CUH
