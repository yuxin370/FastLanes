// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/execution/dispatch.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_DISPATCH_CUH
#define ENGINE_EXECUTION_DISPATCH_CUH

#include "engine/device-utils.cuh"
#include "engine/expression.cuh"
#include "engine/lane-policy.cuh"
#include "flsgpu/consts.cuh"
#include "flsgpu/fls.cuh"

namespace device_exec {

template <typename T,
          int  UNPACK_N_VECTORS,
          int  UNPACK_N_VALUES,
          bool WRITE_OUT    = true,
          typename MappingT = T,
          typename UntransposerT = flsgpu::device::IdentityUntransposer,
          typename DecompressorT>
__device__ __forceinline__ void run_decompressor(DecompressorT&& iterator, const lane_t lane, T* __restrict out) {
	const auto mapping = VectorToWarpMapping<MappingT, UNPACK_N_VECTORS>();
	T          registers[UNPACK_N_VALUES * UNPACK_N_VECTORS];
	uint32_t   acc = 2166136261u;
	for (si_t i = 0; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
		iterator.unpack_next_into(registers);
		if constexpr (WRITE_OUT) {
			write_registers_to_global<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, mapping.N_LANES, UntransposerT>(
			    lane, i, registers, out);
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
		using IndexT                       = uint8_t;
		constexpr int RLE_UNPACK_N_VALUES = utils::get_values_per_lane<IndexT>();
		using UnpackerT =
		    flsgpu::device::BitUnpackerStatefulBranchless<IndexT,
		                                                  UNPACK_N_VECTORS,
		                                                  RLE_UNPACK_N_VALUES,
		                                                  flsgpu::device::FFORFunctor<IndexT, UNPACK_N_VECTORS>>;
		using ExpanderT     = flsgpu::device::DummyRLEExpander<T, IndexT, UNPACK_N_VECTORS, RLE_UNPACK_N_VALUES>;
		using DecompressorT = flsgpu::device::RLEDecompressor<T,
		                                                      IndexT,
		                                                      UNPACK_N_VECTORS,
		                                                      RLE_UNPACK_N_VALUES,
		                                                      UnpackerT,
		                                                      ExpanderT,
		                                                      flsgpu::device::RLEColumn<T, IndexT>>;
		auto iterator       = DecompressorT(expr.col.rle_u8, vector_index, lane);
		run_decompressor<T,
		                 UNPACK_N_VECTORS,
		                 RLE_UNPACK_N_VALUES,
		                 WRITE_OUT,
		                 IndexT,
		                 flsgpu::device::FastLanes1024InputUntransposer>(iterator, lane, out);
		break;
	}
	case dispatch::PlanKind::RLE_U16: {
		using IndexT                       = uint16_t;
		constexpr int RLE_UNPACK_N_VALUES = utils::get_values_per_lane<IndexT>();
		using UnpackerT =
		    flsgpu::device::BitUnpackerStatefulBranchless<IndexT,
		                                                  UNPACK_N_VECTORS,
		                                                  RLE_UNPACK_N_VALUES,
		                                                  flsgpu::device::FFORFunctor<IndexT, UNPACK_N_VECTORS>>;
		using ExpanderT     = flsgpu::device::DummyRLEExpander<T, IndexT, UNPACK_N_VECTORS, RLE_UNPACK_N_VALUES>;
		using DecompressorT = flsgpu::device::RLEDecompressor<T,
		                                                      IndexT,
		                                                      UNPACK_N_VECTORS,
		                                                      RLE_UNPACK_N_VALUES,
		                                                      UnpackerT,
		                                                      ExpanderT,
		                                                      flsgpu::device::RLEColumn<T, IndexT>>;
		auto iterator       = DecompressorT(expr.col.rle_u16, vector_index, lane);
		run_decompressor<T,
		                 UNPACK_N_VECTORS,
		                 RLE_UNPACK_N_VALUES,
		                 WRITE_OUT,
		                 IndexT,
		                 flsgpu::device::FastLanes1024InputUntransposer>(iterator, lane, out);
		break;
	}
	default:
		break;
	}
}

} // namespace device_exec

namespace kernels { namespace device {

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES, bool WRITE_OUT = true>
__device__ __forceinline__ void execute_typed_work_item(const dispatch::DeviceExpression<T>* exprs,
                                                        const dispatch::WorkItemAny          work,
                                                        const lane_t                         lane) {
	if (exprs == nullptr) {
		return;
	}
	const auto*  expr         = exprs + work.expr_index;
	const vi_t   vector_index = static_cast<vi_t>(work.vector_index);
	const size_t n_vecs       = utils::get_n_vecs_from_size(expr->n_values);
	if (static_cast<size_t>(vector_index) >= n_vecs) {
		return;
	}
	if (static_cast<uint32_t>(lane) >= dispatch::semantic_lane_count(dispatch::type_tag_for<T>(), expr->plan)) {
		return;
	}

	T* out = expr->out + vector_index * consts::VALUES_PER_VECTOR;
	device_exec::execute_plan<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(*expr, vector_index, lane, out);
}

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES, bool WRITE_OUT = true>
__global__ void decompress_dispatch_typed(const dispatch::DeviceExpression<T>* exprs,
                                          const dispatch::WorkItemAny*         work_items,
                                          const size_t                         n_items) {
	const lane_t   lane     = static_cast<lane_t>(threadIdx.x);
	const uint32_t item_idx = static_cast<uint32_t>(blockIdx.x);
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
	execute_typed_work_item<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(exprs, work, lane);
}

template <int UNPACK_N_VECTORS, int UNPACK_N_VALUES, bool WRITE_OUT = true>
__global__ void decompress_dispatch_mixed(const dispatch::DeviceExpression<int8_t>*  exprs_i8,
                                          const dispatch::DeviceExpression<int16_t>* exprs_i16,
                                          const dispatch::MixedWorkSlot*             slots,
                                          const size_t                               n_slots) {
	constexpr uint32_t warp_lanes = dispatch::lane_count_for_type(dispatch::TypeTag::I8);
	constexpr uint32_t half_lanes = dispatch::lane_count_for_type(dispatch::TypeTag::I16);

	const uint32_t global_thread = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t slot_idx      = global_thread / warp_lanes;
	const uint32_t lane          = global_thread - slot_idx * warp_lanes;
	if (slot_idx >= n_slots || slots == nullptr) {
		return;
	}

	const auto slot = slots[slot_idx];
	const auto run_work_item = [&](const dispatch::WorkItemAny work, const lane_t work_lane) {
		if (!dispatch::is_valid_work_item(work)) {
			return;
		}
		switch (work.type) {
		case dispatch::TypeTag::I8:
			execute_typed_work_item<int8_t, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(exprs_i8, work, work_lane);
			break;
		case dispatch::TypeTag::I16:
			execute_typed_work_item<int16_t, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(exprs_i16, work, work_lane);
			break;
		default:
			break;
		}
	};

	if (dispatch::is_valid_work_item(slot.second)) {
		if (lane < half_lanes) {
			run_work_item(slot.first, static_cast<lane_t>(lane));
		} else {
			run_work_item(slot.second, static_cast<lane_t>(lane - half_lanes));
		}
		return;
	}

	run_work_item(slot.first, static_cast<lane_t>(lane));
}

}} // namespace kernels::device

#endif // ENGINE_EXECUTION_DISPATCH_CUH
