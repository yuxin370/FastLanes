// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/execution/dispatch.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_DISPATCH_CUH
#define ENGINE_EXECUTION_DISPATCH_CUH

#include "engine/device-utils.cuh"
#include "engine/expression.cuh"
#include "flsgpu/consts.cuh"
#include "flsgpu/fls.cuh"

namespace device_exec {

template <typename T,
          int  UNPACK_N_VECTORS,
          int  UNPACK_N_VALUES,
          bool WRITE_OUT    = true,
          typename MappingT = T,
          typename DecompressorT>
__device__ __forceinline__ void run_decompressor(DecompressorT&& iterator, const lane_t lane, T* __restrict out) {
	const auto mapping = VectorToWarpMapping<MappingT, UNPACK_N_VECTORS>();
	T          registers[UNPACK_N_VALUES * UNPACK_N_VECTORS];
	uint32_t   acc = 2166136261u;
	for (si_t i = 0; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
		iterator.unpack_next_into(registers);
		if constexpr (WRITE_OUT) {
			write_registers_to_global<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, mapping.N_LANES>(lane, i, registers, out);
		} else {
#pragma unroll
			for (int k = 0; k < UNPACK_N_VALUES * UNPACK_N_VECTORS; ++k) {
				acc ^= static_cast<uint32_t>(registers[k]);
			}
		}
	}
	if constexpr (!WRITE_OUT) {
		if (acc == 0u) {
			out[0] = static_cast<T>(acc);
		}
	}
}

template <typename T,
          int  UNPACK_N_VECTORS,
          int  UNPACK_N_VALUES,
          bool WRITE_OUT = true,
          typename CodeT,
          typename DecompressorT>
__device__ __forceinline__ void run_rle_decompressor(DecompressorT&& iterator, const lane_t lane, T* __restrict out) {
	const auto mapping = VectorToWarpMapping<CodeT, UNPACK_N_VECTORS>();
	uint32_t   acc     = 2166136261u;
	(void)lane;
	for (si_t i = 0; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
		if constexpr (WRITE_OUT) {
			iterator.unpack_next_untransposed_into(out, i);
		} else {
			T registers[UNPACK_N_VALUES * UNPACK_N_VECTORS];
			iterator.unpack_next_into(registers);
#pragma unroll
			for (int k = 0; k < UNPACK_N_VALUES * UNPACK_N_VECTORS; ++k) {
				acc ^= static_cast<uint32_t>(registers[k]);
			}
		}
	}
	if constexpr (!WRITE_OUT) {
		if (acc == 0u) {
			out[0] = static_cast<T>(acc);
		}
	}
}

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES, bool WRITE_OUT = true>
__device__ __forceinline__ void
execute_plan(const dispatch::DeviceExpression<T>& expr, const vi_t vector_index, const lane_t lane, T* __restrict out) {
	switch (expr.plan) {
	case dispatch::PlanKind::UNCOMPRESSED: {
		using UnpackerT = flsgpu::device::
		    BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, flsgpu::device::BPFunctor<T>>;
		using DecompressorT =
		    flsgpu::device::BPDecompressor<T, UNPACK_N_VECTORS, UnpackerT, flsgpu::device::BPColumn<T>>;
		auto iterator = DecompressorT(expr.col.bp, vector_index, lane);
		run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(iterator, lane, out);
		break;
	}
	case dispatch::PlanKind::CONSTANT: {
		using DecompressorT = flsgpu::device::
		    CONSTANTDecompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, flsgpu::device::CONSTANTColumn<T>>;
		auto iterator = DecompressorT(expr.col.constant, vector_index, lane);
		run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(iterator, lane, out);
		break;
	}
	case dispatch::PlanKind::UNFFOR: {
		using UnpackerT =
		    flsgpu::device::BitUnpackerStatefulBranchless<T,
		                                                  UNPACK_N_VECTORS,
		                                                  UNPACK_N_VALUES,
		                                                  flsgpu::device::FFORFunctor<T, UNPACK_N_VECTORS>>;
		using DecompressorT =
		    flsgpu::device::FFORDecompressor<T, UNPACK_N_VECTORS, UnpackerT, flsgpu::device::FFORColumn<T>>;
		auto iterator = DecompressorT(expr.col.ffor, vector_index, lane);
		run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(iterator, lane, out);
		break;
	}
	case dispatch::PlanKind::UNFFOR_SLPATCH: {
		using UnpackerT =
		    flsgpu::device::BitUnpackerStatefulBranchless<T,
		                                                  UNPACK_N_VECTORS,
		                                                  UNPACK_N_VALUES,
		                                                  flsgpu::device::FFORFunctor<T, UNPACK_N_VECTORS>>;
		using PatcherT      = flsgpu::device::StatefulSLPATCHExceptionPatcher<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
		using DecompressorT = flsgpu::device::
		    SLPATCHDecompressor<T, UNPACK_N_VECTORS, UnpackerT, PatcherT, flsgpu::device::SLPATCHColumn<T>>;
		auto iterator = DecompressorT(expr.col.slpatch, vector_index, lane);
		run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(iterator, lane, out);
		break;
	}
	case dispatch::PlanKind::DICT_FFOR_U8: {
		using ColumnT    = flsgpu::device::DICTFFORColumn<T, uint8_t>;
		using IndexT     = typename ColumnT::INDEX_T;
		using ProcessorT = flsgpu::device::DICTFunctor<T, UNPACK_N_VECTORS, IndexT>;
		using UnpackerT =
		    flsgpu::device::BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, ProcessorT, IndexT>;
		using DecompressorT = flsgpu::device::DICTDecompressor<T, UNPACK_N_VECTORS, UnpackerT, ColumnT, ProcessorT>;
		auto iterator       = DecompressorT(expr.col.dictffor_u8, vector_index, lane);
		run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT, IndexT>(iterator, lane, out);
		break;
	}
	case dispatch::PlanKind::DICT_FFOR_U16: {
		using ColumnT    = flsgpu::device::DICTFFORColumn<T, uint16_t>;
		using IndexT     = typename ColumnT::INDEX_T;
		using ProcessorT = flsgpu::device::DICTFunctor<T, UNPACK_N_VECTORS, IndexT>;
		using UnpackerT =
		    flsgpu::device::BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, ProcessorT, IndexT>;
		using DecompressorT = flsgpu::device::DICTDecompressor<T, UNPACK_N_VECTORS, UnpackerT, ColumnT, ProcessorT>;
		auto iterator       = DecompressorT(expr.col.dictffor_u16, vector_index, lane);
		run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(iterator, lane, out);
		break;
	}
	case dispatch::PlanKind::DICT_FFOR_SLPATCH_U8: {
		using ColumnT    = flsgpu::device::DICTSLPATCHColumn<T, uint8_t>;
		using IndexT     = typename ColumnT::INDEX_T;
		using ProcessorT = flsgpu::device::DICTFunctor<T, UNPACK_N_VECTORS, IndexT>;
		using UnpackerT =
		    flsgpu::device::BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, ProcessorT, IndexT>;
		using PatcherT =
		    flsgpu::device::StatefulSLPATCHDictExceptionPatcher<T, IndexT, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
		using DecompressorT = flsgpu::device::
		    DICTSLPATCHDecompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, UnpackerT, PatcherT, ColumnT, ProcessorT>;
		auto iterator = DecompressorT(expr.col.dictslpatch_u8, vector_index, lane);
		run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT, IndexT>(iterator, lane, out);
		break;
	}
	case dispatch::PlanKind::DICT_FFOR_SLPATCH_U16: {
		using ColumnT    = flsgpu::device::DICTSLPATCHColumn<T, uint16_t>;
		using IndexT     = typename ColumnT::INDEX_T;
		using ProcessorT = flsgpu::device::DICTFunctor<T, UNPACK_N_VECTORS, IndexT>;
		using UnpackerT =
		    flsgpu::device::BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, ProcessorT, IndexT>;
		using PatcherT =
		    flsgpu::device::StatefulSLPATCHDictExceptionPatcher<T, IndexT, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
		using DecompressorT = flsgpu::device::
		    DICTSLPATCHDecompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, UnpackerT, PatcherT, ColumnT, ProcessorT>;
		auto iterator = DecompressorT(expr.col.dictslpatch_u16, vector_index, lane);
		run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(iterator, lane, out);
		break;
	}
	case dispatch::PlanKind::FREQUENCY: {
		if (expr.freq_use_extended) {
			using PatcherT =
			    flsgpu::device::PrefetchAllBranchlessFREQExceptionPatcher<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
			using DecompressorT =
			    flsgpu::device::FREQDecompressor<T, UNPACK_N_VECTORS, PatcherT, flsgpu::device::FREQExtendedColumn<T>>;
			auto iterator = DecompressorT(expr.col.freq_extended, vector_index, lane);
			run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(iterator, lane, out);
		} else {
			using PatcherT = flsgpu::device::StatefulFREQExceptionPatcher<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
			using DecompressorT =
			    flsgpu::device::FREQDecompressor<T, UNPACK_N_VECTORS, PatcherT, flsgpu::device::FREQColumn<T>>;
			auto iterator = DecompressorT(expr.col.freq, vector_index, lane);
			run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(iterator, lane, out);
		}
		break;
	}
	case dispatch::PlanKind::CROSS_RLE: {
		using ExpanderT = flsgpu::device::StatefulCROSSRLEExpander<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
		using DecompressorT =
		    flsgpu::device::CROSSRLEDecompressor<T, UNPACK_N_VECTORS, ExpanderT, flsgpu::device::CROSSRLEColumn<T>>;
		auto iterator = DecompressorT(expr.col.crossrle, vector_index, lane);
		run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(iterator, lane, out);
		break;
	}
	case dispatch::PlanKind::RLE_U8: {
		using CodeT                       = uint8_t;
		constexpr int RLE_UNPACK_N_VALUES = utils::get_values_per_lane<CodeT>();
		if (lane >= static_cast<lane_t>(utils::get_n_lanes<CodeT>())) {
			return;
		}
		using UnpackerT =
		    flsgpu::device::BitUnpackerStatefulBranchless<CodeT,
		                                                  UNPACK_N_VECTORS,
		                                                  RLE_UNPACK_N_VALUES,
		                                                  flsgpu::device::FFORFunctor<CodeT, UNPACK_N_VECTORS>>;
		using ExpanderT     = flsgpu::device::DummyRLEExpander<T,
		                                                       CodeT,
		                                                       UNPACK_N_VECTORS,
		                                                       RLE_UNPACK_N_VALUES,
		                                                       flsgpu::device::FastLanes1024Untransposer>;
		using DecompressorT = flsgpu::device::RLEDecompressor<T,
		                                                      CodeT,
		                                                      UNPACK_N_VECTORS,
		                                                      RLE_UNPACK_N_VALUES,
		                                                      UnpackerT,
		                                                      ExpanderT,
		                                                      flsgpu::device::RLEColumn<T, CodeT>>;
		auto iterator       = DecompressorT(expr.col.rle_u8, vector_index, lane);
		run_rle_decompressor<T, UNPACK_N_VECTORS, RLE_UNPACK_N_VALUES, WRITE_OUT, CodeT>(iterator, lane, out);
		break;
	}
	case dispatch::PlanKind::RLE_U16: {
		using CodeT                       = uint16_t;
		constexpr int RLE_UNPACK_N_VALUES = utils::get_values_per_lane<CodeT>();
		if (lane >= static_cast<lane_t>(utils::get_n_lanes<CodeT>())) {
			return;
		}
		using UnpackerT =
		    flsgpu::device::BitUnpackerStatefulBranchless<CodeT,
		                                                  UNPACK_N_VECTORS,
		                                                  RLE_UNPACK_N_VALUES,
		                                                  flsgpu::device::FFORFunctor<CodeT, UNPACK_N_VECTORS>>;
		using ExpanderT     = flsgpu::device::DummyRLEExpander<T,
		                                                       CodeT,
		                                                       UNPACK_N_VECTORS,
		                                                       RLE_UNPACK_N_VALUES,
		                                                       flsgpu::device::FastLanes1024Untransposer>;
		using DecompressorT = flsgpu::device::RLEDecompressor<T,
		                                                      CodeT,
		                                                      UNPACK_N_VECTORS,
		                                                      RLE_UNPACK_N_VALUES,
		                                                      UnpackerT,
		                                                      ExpanderT,
		                                                      flsgpu::device::RLEColumn<T, CodeT>>;
		auto iterator       = DecompressorT(expr.col.rle_u16, vector_index, lane);
		run_rle_decompressor<T, UNPACK_N_VECTORS, RLE_UNPACK_N_VALUES, WRITE_OUT, CodeT>(iterator, lane, out);
		break;
	}
	default:
		break;
	}
}

} // namespace device_exec

namespace kernels { namespace device {

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES, bool WRITE_OUT = true>
__global__ void decompress_dispatch_typed(const dispatch::DeviceExpression<T>* exprs,
                                          const dispatch::WorkItemAny*         work_items,
                                          const size_t                         n_items) {
	const auto     mapping  = VectorToWarpMapping<T, UNPACK_N_VECTORS>();
	const lane_t   lane     = mapping.get_lane();
	const uint32_t item_idx = static_cast<uint32_t>(mapping.get_vector_index());
	if (item_idx >= n_items) {
		return;
	}

	const auto work = work_items[item_idx];
	if constexpr (std::is_same_v<T, int8_t>) {
		if (work.type != dispatch::TypeTag::I8) {
			return;
		}
	} else if constexpr (std::is_same_v<T, int16_t>) {
		if (work.type != dispatch::TypeTag::I16) {
			return;
		}
	}

	const auto*  expr         = exprs + work.expr_index;
	const vi_t   vector_index = static_cast<vi_t>(work.vector_index);
	const size_t n_vecs       = utils::get_n_vecs_from_size(expr->n_values);
	if (static_cast<size_t>(vector_index) >= n_vecs) {
		return;
	}

	T* out = expr->out + vector_index * consts::VALUES_PER_VECTOR;
	device_exec::execute_plan<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(*expr, vector_index, lane, out);
}

template <int UNPACK_N_VECTORS, int UNPACK_N_VALUES, bool WRITE_OUT = true>
__global__ void decompress_dispatch_mixed(const dispatch::DeviceExpression<int8_t>*  exprs_i8,
                                          const dispatch::DeviceExpression<int16_t>* exprs_i16,
                                          const dispatch::WorkItemAny*               work_items,
                                          const size_t                               n_items,
                                          const uint32_t                             i16_start_index) {
	constexpr uint32_t lanes_i8   = utils::get_n_lanes<int8_t>();
	constexpr uint32_t lanes_i16  = utils::get_n_lanes<int16_t>();
	constexpr uint32_t warp_lanes = (lanes_i8 > lanes_i16) ? lanes_i8 : lanes_i16;

	const uint32_t global_thread = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t item_idx      = global_thread / warp_lanes;
	const uint32_t lane          = global_thread - item_idx * warp_lanes;
	if (item_idx >= n_items || work_items == nullptr) {
		return;
	}

	const auto work = work_items[item_idx];
	switch (work.type) {
	case dispatch::TypeTag::I8: {
		if (exprs_i8 == nullptr || lane >= lanes_i8) {
			return;
		}
		const auto*  expr         = exprs_i8 + work.expr_index;
		const vi_t   vector_index = static_cast<vi_t>(work.vector_index);
		const size_t n_vecs       = utils::get_n_vecs_from_size(expr->n_values);
		if (static_cast<size_t>(vector_index) >= n_vecs) {
			return;
		}
		int8_t* out = expr->out + vector_index * consts::VALUES_PER_VECTOR;
		device_exec::execute_plan<int8_t, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(*expr, vector_index, lane, out);
		break;
	}
	case dispatch::TypeTag::I16: {
		if (exprs_i16 == nullptr) {
			return;
		}
		const auto* expr0              = exprs_i16 + work.expr_index;
		const bool  full_lanes_dict_u8 = (expr0->plan == dispatch::PlanKind::DICT_FFOR_U8 ||
                                         expr0->plan == dispatch::PlanKind::DICT_FFOR_SLPATCH_U8);
		if (full_lanes_dict_u8) {
			const vi_t   vector_index = static_cast<vi_t>(work.vector_index);
			const size_t n_vecs       = utils::get_n_vecs_from_size(expr0->n_values);
			if (static_cast<size_t>(vector_index) >= n_vecs) {
				return;
			}
			int16_t* out = expr0->out + vector_index * consts::VALUES_PER_VECTOR;
			device_exec::execute_plan<int16_t, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(
			    *expr0, vector_index, lane, out);
			break;
		}

		// Pair contiguous i16 items in the i16 range [i16_start_index, n_items).
		if (item_idx < i16_start_index) {
			return;
		}
		const uint32_t i16_local = item_idx - i16_start_index;
		if ((i16_local & 1u) != 0u) {
			return; // odd local index is handled by previous even item
		}
		const bool has_second_i16 =
		    (item_idx + 1 < n_items) && (work_items[item_idx + 1].type == dispatch::TypeTag::I16);
		const bool run_second = has_second_i16 && (lane >= lanes_i16);
		auto       work_i16   = run_second ? work_items[item_idx + 1] : work;
		auto       lane_i16   = run_second ? (lane - lanes_i16) : lane;
		if (lane_i16 >= lanes_i16) {
			return;
		}

		const auto*  expr         = exprs_i16 + work_i16.expr_index;
		const vi_t   vector_index = static_cast<vi_t>(work_i16.vector_index);
		const size_t n_vecs       = utils::get_n_vecs_from_size(expr->n_values);
		if (static_cast<size_t>(vector_index) >= n_vecs) {
			return;
		}
		int16_t* out = expr->out + vector_index * consts::VALUES_PER_VECTOR;
		device_exec::execute_plan<int16_t, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(
		    *expr, vector_index, lane_i16, out);
		break;
	}
	default:
		break;
	}
}

}} // namespace kernels::device

#endif // ENGINE_EXECUTION_DISPATCH_CUH
