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
#include "engine/lane-policy.cuh"
#include "engine/types.cuh"
#include "flsgpu/host-utils.cuh"
#include "flsgpu/structs.cuh"
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

namespace dispatch {

template <typename>
inline constexpr bool always_false_v = false;

enum class LaunchStrategy {
	TypedBatches,
	MixedDispatch,
};

struct ExecutionConfig {
	unsigned       unpack_n_vectors             = 1;
	unsigned       unpack_n_values              = 1;
	LaunchStrategy launch_strategy              = LaunchStrategy::MixedDispatch;
	bool           write_out                    = true;
	bool           freq_prefetch_all_branchless = false;
	bool           freq_hybrid_patcher          = false;
	float          freq_branchless_threshold    = 6.0f;

	constexpr DecodeChunk chunk() const {
		return DecodeChunk {unpack_n_vectors, unpack_n_values};
	}
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
ValueStore decompress_host(const HostColT& host_col, const PlanKind plan, const ExecutionConfig& cfg);

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
	static constexpr PlanKind value = std::is_same_v<IndexT, uint8_t> ? PlanKind::RLE_U8 : PlanKind::RLE_U16;
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
	const double exc_per_vec = static_cast<double>(host_col.n_exceptions) / static_cast<double>(host_col.get_n_vecs());
	return exc_per_vec >= static_cast<double>(freq_branchless_threshold);
}

// Arena-based overload: packs column data into shared arena; pointers resolved on arena.upload().
template <typename T, typename HostColT>
void fill_device_expr(DeviceExpression<T>& expr,
                      const HostColT&      host_col,
                      const PlanKind       plan,
                      const bool           freq_use_extended,
                      flsgpu::memory::DeviceArena& arena) {
	switch (plan) {
	case PlanKind::UNCOMPRESSED:
		if constexpr (std::is_same_v<HostColT, flsgpu::host::BPColumn<T>>) {
			host_col.copy_to_device(arena, expr.col.bp);
			return;
		}
		break;
	case PlanKind::CONSTANT:
		if constexpr (std::is_same_v<HostColT, flsgpu::host::CONSTANTColumn<T>>) {
			host_col.copy_to_device(arena, expr.col.constant);
			return;
		}
		break;
	case PlanKind::FREQUENCY:
		if constexpr (std::is_same_v<HostColT, flsgpu::host::FREQColumn<T>>) {
			if (freq_use_extended) {
				// Extended column is a temporary — capture by value in the defer_free
				// lambda to keep its arrays alive until arena.upload() packs them.
				// Value-capture avoids a separate heap allocation for the wrapper struct.
				auto extended = host_col.create_extended_column();
				extended.copy_to_device(arena, expr.col.freq_extended);
				expr.freq_use_extended = true;
				arena.defer_free([ext = extended]() {
					flsgpu::host::free_column(ext);
				});
				return;
			}
			host_col.copy_to_device(arena, expr.col.freq);
			expr.freq_use_extended = false;
			return;
		}
		break;
	case PlanKind::UNFFOR:
		if constexpr (std::is_same_v<HostColT, flsgpu::host::FFORColumn<T>>) {
			host_col.copy_to_device(arena, expr.col.ffor);
			return;
		}
		break;
	case PlanKind::UNFFOR_SLPATCH:
		if constexpr (std::is_same_v<HostColT, flsgpu::host::SLPATCHColumn<T>>) {
			host_col.copy_to_device(arena, expr.col.slpatch);
			return;
		}
		break;
	case PlanKind::DICT_FFOR_U8:
		if constexpr (std::is_same_v<HostColT, flsgpu::host::DICTFFORColumn<T, uint8_t>>) {
			host_col.copy_to_device(arena, expr.col.dictffor_u8);
			return;
		}
		break;
	case PlanKind::DICT_FFOR_U16:
		if constexpr (std::is_same_v<HostColT, flsgpu::host::DICTFFORColumn<T, uint16_t>>) {
			host_col.copy_to_device(arena, expr.col.dictffor_u16);
			return;
		}
		break;
	case PlanKind::DICT_FFOR_SLPATCH_U8:
		if constexpr (std::is_same_v<HostColT, flsgpu::host::DICTSLPATCHColumn<T, uint8_t>>) {
			host_col.copy_to_device(arena, expr.col.dictslpatch_u8);
			return;
		}
		break;
	case PlanKind::DICT_FFOR_SLPATCH_U16:
		if constexpr (std::is_same_v<HostColT, flsgpu::host::DICTSLPATCHColumn<T, uint16_t>>) {
			host_col.copy_to_device(arena, expr.col.dictslpatch_u16);
			return;
		}
		break;
	case PlanKind::CROSS_RLE:
		if constexpr (std::is_same_v<HostColT, flsgpu::host::CROSSRLEColumn<T>>) {
			host_col.copy_to_device(arena, expr.col.crossrle);
			return;
		}
		break;
	case PlanKind::RLE_U8:
		if constexpr (std::is_same_v<HostColT, flsgpu::host::RLEColumn<T, uint8_t>>) {
			host_col.copy_to_device(arena, expr.col.rle_u8);
			return;
		}
		break;
	case PlanKind::RLE_U16:
		if constexpr (std::is_same_v<HostColT, flsgpu::host::RLEColumn<T, uint16_t>>) {
			host_col.copy_to_device(arena, expr.col.rle_u16);
			return;
		}
		break;
	default:
		break;
	}
	throw std::runtime_error("fill_device_expr(arena): plan/column type mismatch, plan=" +
	                         std::to_string(static_cast<int>(plan)));
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
	case PlanKind::RLE_U8:
		flsgpu::host::free_column(expr.col.rle_u8);
		break;
	case PlanKind::RLE_U16:
		flsgpu::host::free_column(expr.col.rle_u16);
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
struct BatchSetFromList<dispatch::TypeList<Ts...>> {
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
struct DeviceBatchSetFromList<dispatch::TypeList<Ts...>> {
	using type = DeviceBatchSet<Ts...>;
};

// Arena-based overload: emplaces expr into pre-reserved batch, packs column data into shared arena.
// batch.device_exprs MUST be pre-reserved to avoid reallocation (resolver callbacks capture &out).
template <typename T, typename HostColT>
void add_expression_to_batch(const size_t    expr_index,
                             const HostColT& host_col,
                             const PlanKind  plan,
                             Batch<T>&       batch,
                             const size_t    output_offset,
                             const bool      freq_prefetch_all_branchless,
                             const bool      freq_hybrid_patcher,
                             const float     freq_branchless_threshold,
                             flsgpu::memory::DeviceArena& arena) {
	// Emplace into pre-reserved vector — address is stable.
	batch.device_exprs.emplace_back();
	auto& expr    = batch.device_exprs.back();
	expr.plan     = plan;
	expr.n_values = host_col.get_n_values();
	expr.out      = nullptr;
	batch.output_offsets.push_back(output_offset);

	bool use_freq_extended = false;
	if constexpr (std::is_same_v<HostColT, flsgpu::host::FREQColumn<T>>) {
		use_freq_extended = detail::should_use_freq_extended(
		    host_col, freq_prefetch_all_branchless, freq_hybrid_patcher, freq_branchless_threshold);
	}
	detail::fill_device_expr(expr, host_col, plan, use_freq_extended, arena);

	const auto device_idx = static_cast<uint32_t>(batch.device_exprs.size() - 1);
	batch.expr_indices.push_back(expr_index);

	const size_t n_vecs = utils::get_n_vecs_from_size(expr.n_values);
	batch.work_items.reserve(batch.work_items.size() + n_vecs);
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
	uint32_t           threads          = static_cast<uint32_t>(utils::get_n_lanes<T>());
	for (const auto& work : batch.work_items) {
		threads = std::max(threads,
		                   dispatch::semantic_lane_count(type_tag_for<T>(), batch.device_exprs[work.expr_index].plan));
	}
	const dim3 block(static_cast<unsigned>(threads));
	const dim3 grid(static_cast<unsigned>(n_items));

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
		out.meta.value_type                     = types::ToDataType<T>::value;
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

} // namespace detail

} // namespace dispatch

#endif // ENGINE_EXECUTION_COMMON_CUH
