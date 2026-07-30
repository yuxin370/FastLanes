// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/dispatch.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_ENGINE_DISPATCH_CUH
#define GALP_ENGINE_DISPATCH_CUH

#include "codecs/consts.cuh"
#include "codecs/device_ops.cuh"
#include "core/expression.cuh"
#include "core/lane_policy.cuh"
#include "cuda/device_utils.cuh"
#include "engine/config.cuh"

namespace galp::kernels::detail {

template <typename T,
          int  UNPACK_N_VECTORS,
          int  UNPACK_N_VALUES,
          bool WRITE_OUT         = true,
          typename MappingT      = T,
          typename UntransposerT = galp::codec::device::IdentityUntransposer,
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
		(void)out;
		asm volatile("" : : "r"(acc) : "memory");
	}
}

template <typename T, int UNPACK_N_VECTORS, bool WRITE_OUT = true, typename DecompressorT>
__device__ __forceinline__ void
run_delta_register_decompressor(DecompressorT&& iterator, const lane_t lane, T* __restrict out) {
	using UIntT = typename galp::codec::utils::same_width_uint<T>::type;
	constexpr int N_VALUES = galp::codec::utils::get_values_per_lane<T>();
	constexpr int N_LANES  = galp::codec::utils::get_n_lanes<T>();
	UIntT        registers[N_VALUES * UNPACK_N_VECTORS];

	iterator.decode_lane_into(registers);
	if constexpr (WRITE_OUT) {
		write_registers_to_global<T,
		                          UNPACK_N_VECTORS,
		                          N_VALUES,
		                          N_LANES,
		                          galp::codec::device::FastLanes1024InputUntransposer>(lane, 0, registers, out);
	} else {
		uint32_t acc = 2166136261u;
#pragma unroll
		for (int index = 0; index < N_VALUES * UNPACK_N_VECTORS; ++index) {
			acc ^= static_cast<uint32_t>(registers[index]);
		}
		(void)out;
		asm volatile("" : : "r"(acc) : "memory");
	}
}

template <typename T,
          int                           UNPACK_N_VECTORS,
          int                           UNPACK_N_VALUES,
          bool                          WRITE_OUT     = true,
	          galp::execution::DeltaDecoder DELTA_DECODER = galp::execution::DeltaDecoder::Register>
__device__ __forceinline__ void execute_plan(const galp::execution::DeviceExpression<T>& expr,
                                             const vi_t                                  vector_index,
                                             const lane_t                                lane,
                                             T* __restrict out) {
	switch (expr.plan) {
	case galp::execution::PlanKind::UNCOMPRESSED: {
		using UnpackerT = galp::codec::device::
		    BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, galp::codec::device::BPFunctor<T>>;
		using DecompressorT =
		    galp::codec::device::BPDecompressor<T, UNPACK_N_VECTORS, UnpackerT, galp::codec::device::BPColumn<T>>;
		auto iterator = DecompressorT(expr.col.bp, vector_index, lane);
		run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(iterator, lane, out);
		break;
	}
	case galp::execution::PlanKind::CONSTANT: {
		using DecompressorT = galp::codec::device::
		    CONSTANTDecompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, galp::codec::device::CONSTANTColumn<T>>;
		auto iterator = DecompressorT(expr.col.constant, vector_index, lane);
		run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(iterator, lane, out);
		break;
	}
	case galp::execution::PlanKind::DELTA: {
		using UIntT   = typename galp::codec::utils::same_width_uint<T>::type;
		using ColumnT = galp::codec::device::DELTAColumn<T>;
		if constexpr (DELTA_DECODER == galp::execution::DeltaDecoder::Register) {
			constexpr int REGISTER_N_VALUES = galp::codec::utils::get_values_per_lane<T>();
			using UnpackerT                 = galp::codec::device::BitUnpackerLaneTile<
			                    UIntT,
			                    UNPACK_N_VECTORS,
			                    REGISTER_N_VALUES,
			                    galp::codec::device::FFORFunctor<UIntT, UNPACK_N_VECTORS>>;
			using DecompressorT =
			    galp::codec::device::DELTARegisterDecompressor<T, UNPACK_N_VECTORS, UnpackerT, ColumnT>;
			auto iterator = DecompressorT(expr.col.delta, vector_index, lane);
			run_delta_register_decompressor<T, UNPACK_N_VECTORS, WRITE_OUT>(iterator, lane, out);
		} else {
			using UnpackerT = galp::codec::device::BitUnpackerStatefulBranchless<
			    UIntT,
			    UNPACK_N_VECTORS,
			    UNPACK_N_VALUES,
			    galp::codec::device::FFORFunctor<UIntT, UNPACK_N_VECTORS>>;
			using DecompressorT = galp::codec::device::DELTADecompressor<T, UNPACK_N_VECTORS, UnpackerT, ColumnT>;
			auto iterator       = DecompressorT(expr.col.delta, vector_index, lane);
			run_decompressor<T,
			                 UNPACK_N_VECTORS,
			                 UNPACK_N_VALUES,
			                 WRITE_OUT,
			                 T,
			                 galp::codec::device::FastLanes1024InputUntransposer>(iterator, lane, out);
		}
		break;
	}
	case galp::execution::PlanKind::UNFFOR: {
		using UnpackerT =
		    galp::codec::device::BitUnpackerStatefulBranchless<T,
		                                                       UNPACK_N_VECTORS,
		                                                       UNPACK_N_VALUES,
		                                                       galp::codec::device::FFORFunctor<T, UNPACK_N_VECTORS>>;
		using DecompressorT =
		    galp::codec::device::FFORDecompressor<T, UNPACK_N_VECTORS, UnpackerT, galp::codec::device::FFORColumn<T>>;
		auto iterator = DecompressorT(expr.col.ffor, vector_index, lane);
		run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(iterator, lane, out);
		break;
	}
	case galp::execution::PlanKind::UNFFOR_SLPATCH: {
		using UnpackerT =
		    galp::codec::device::BitUnpackerStatefulBranchless<T,
		                                                       UNPACK_N_VECTORS,
		                                                       UNPACK_N_VALUES,
		                                                       galp::codec::device::FFORFunctor<T, UNPACK_N_VECTORS>>;
		using PatcherT = galp::codec::device::StatefulSLPATCHExceptionPatcher<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
		using DecompressorT = galp::codec::device::
		    SLPATCHDecompressor<T, UNPACK_N_VECTORS, UnpackerT, PatcherT, galp::codec::device::SLPATCHColumn<T>>;
		auto iterator = DecompressorT(expr.col.slpatch, vector_index, lane);
		run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(iterator, lane, out);
		break;
	}
	case galp::execution::PlanKind::DICT_FFOR_U8: {
		using ColumnT    = galp::codec::device::DICTFFORColumn<T, uint8_t>;
		using IndexT     = typename ColumnT::INDEX_T;
		using ProcessorT = galp::codec::device::DICTFunctor<T, UNPACK_N_VECTORS, IndexT>;
		using UnpackerT  = galp::codec::device::
		    BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, ProcessorT, IndexT>;
		using DecompressorT =
		    galp::codec::device::DICTDecompressor<T, UNPACK_N_VECTORS, UnpackerT, ColumnT, ProcessorT>;
		auto iterator = DecompressorT(expr.col.dictffor_u8, vector_index, lane);
		run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT, IndexT>(iterator, lane, out);
		break;
	}
	case galp::execution::PlanKind::DICT_FFOR_U16: {
		using ColumnT    = galp::codec::device::DICTFFORColumn<T, uint16_t>;
		using IndexT     = typename ColumnT::INDEX_T;
		using ProcessorT = galp::codec::device::DICTFunctor<T, UNPACK_N_VECTORS, IndexT>;
		using UnpackerT  = galp::codec::device::
		    BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, ProcessorT, IndexT>;
		using DecompressorT =
		    galp::codec::device::DICTDecompressor<T, UNPACK_N_VECTORS, UnpackerT, ColumnT, ProcessorT>;
		auto iterator = DecompressorT(expr.col.dictffor_u16, vector_index, lane);
		run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(iterator, lane, out);
		break;
	}
	case galp::execution::PlanKind::DICT_FFOR_SLPATCH_U8: {
		using ColumnT    = galp::codec::device::DICTSLPATCHColumn<T, uint8_t>;
		using IndexT     = typename ColumnT::INDEX_T;
		using ProcessorT = galp::codec::device::DICTFunctor<T, UNPACK_N_VECTORS, IndexT>;
		using UnpackerT  = galp::codec::device::
		    BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, ProcessorT, IndexT>;
		using PatcherT =
		    galp::codec::device::StatefulSLPATCHDictExceptionPatcher<T, IndexT, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
		using DecompressorT = galp::codec::device::
		    DICTSLPATCHDecompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, UnpackerT, PatcherT, ColumnT, ProcessorT>;
		auto iterator = DecompressorT(expr.col.dictslpatch_u8, vector_index, lane);
		run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT, IndexT>(iterator, lane, out);
		break;
	}
	case galp::execution::PlanKind::DICT_FFOR_SLPATCH_U16: {
		using ColumnT    = galp::codec::device::DICTSLPATCHColumn<T, uint16_t>;
		using IndexT     = typename ColumnT::INDEX_T;
		using ProcessorT = galp::codec::device::DICTFunctor<T, UNPACK_N_VECTORS, IndexT>;
		using UnpackerT  = galp::codec::device::
		    BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, ProcessorT, IndexT>;
		using PatcherT =
		    galp::codec::device::StatefulSLPATCHDictExceptionPatcher<T, IndexT, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
		using DecompressorT = galp::codec::device::
		    DICTSLPATCHDecompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, UnpackerT, PatcherT, ColumnT, ProcessorT>;
		auto iterator = DecompressorT(expr.col.dictslpatch_u16, vector_index, lane);
		run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(iterator, lane, out);
		break;
	}
	case galp::execution::PlanKind::FREQUENCY: {
		if (expr.freq_use_extended) {
			using PatcherT =
			    galp::codec::device::PrefetchAllBranchlessFREQExceptionPatcher<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
			using DecompressorT = galp::codec::device::
			    FREQDecompressor<T, UNPACK_N_VECTORS, PatcherT, galp::codec::device::FREQExtendedColumn<T>>;
			auto iterator = DecompressorT(expr.col.freq_extended, vector_index, lane);
			run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(iterator, lane, out);
		} else {
			using PatcherT = galp::codec::device::StatefulFREQExceptionPatcher<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
			using DecompressorT = galp::codec::device::
			    FREQDecompressor<T, UNPACK_N_VECTORS, PatcherT, galp::codec::device::FREQColumn<T>>;
			auto iterator = DecompressorT(expr.col.freq, vector_index, lane);
			run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(iterator, lane, out);
		}
		break;
	}
	case galp::execution::PlanKind::CROSS_RLE: {
		using ExpanderT     = galp::codec::device::StatefulCROSSRLEExpander<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
		using DecompressorT = galp::codec::device::
		    CROSSRLEDecompressor<T, UNPACK_N_VECTORS, ExpanderT, galp::codec::device::CROSSRLEColumn<T>>;
		auto iterator = DecompressorT(expr.col.crossrle, vector_index, lane);
		run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>(iterator, lane, out);
		break;
	}
	case galp::execution::PlanKind::RLE_U8: {
		using IndexT                      = uint8_t;
		constexpr int RLE_UNPACK_N_VALUES = galp::codec::utils::get_values_per_lane<IndexT>();
		using UnpackerT                   = galp::codec::device::BitUnpackerStatefulBranchless<
		                      IndexT,
		                      UNPACK_N_VECTORS,
		                      RLE_UNPACK_N_VALUES,
		                      galp::codec::device::FFORFunctor<IndexT, UNPACK_N_VECTORS>>;
		using ExpanderT     = galp::codec::device::DummyRLEExpander<T, IndexT, UNPACK_N_VECTORS, RLE_UNPACK_N_VALUES>;
		using DecompressorT = galp::codec::device::RLEDecompressor<T,
		                                                           IndexT,
		                                                           UNPACK_N_VECTORS,
		                                                           RLE_UNPACK_N_VALUES,
		                                                           UnpackerT,
		                                                           ExpanderT,
		                                                           galp::codec::device::RLEColumn<T, IndexT>>;
		auto iterator       = DecompressorT(expr.col.rle_u8, vector_index, lane);
		run_decompressor<T,
		                 UNPACK_N_VECTORS,
		                 RLE_UNPACK_N_VALUES,
		                 WRITE_OUT,
		                 IndexT,
		                 galp::codec::device::FastLanes1024InputUntransposer>(iterator, lane, out);
		break;
	}
	case galp::execution::PlanKind::RLE_U16: {
		using IndexT                      = uint16_t;
		constexpr int RLE_UNPACK_N_VALUES = galp::codec::utils::get_values_per_lane<IndexT>();
		using UnpackerT                   = galp::codec::device::BitUnpackerStatefulBranchless<
		                      IndexT,
		                      UNPACK_N_VECTORS,
		                      RLE_UNPACK_N_VALUES,
		                      galp::codec::device::FFORFunctor<IndexT, UNPACK_N_VECTORS>>;
		using ExpanderT     = galp::codec::device::DummyRLEExpander<T, IndexT, UNPACK_N_VECTORS, RLE_UNPACK_N_VALUES>;
		using DecompressorT = galp::codec::device::RLEDecompressor<T,
		                                                           IndexT,
		                                                           UNPACK_N_VECTORS,
		                                                           RLE_UNPACK_N_VALUES,
		                                                           UnpackerT,
		                                                           ExpanderT,
		                                                           galp::codec::device::RLEColumn<T, IndexT>>;
		auto iterator       = DecompressorT(expr.col.rle_u16, vector_index, lane);
		run_decompressor<T,
		                 UNPACK_N_VECTORS,
		                 RLE_UNPACK_N_VALUES,
		                 WRITE_OUT,
		                 IndexT,
		                 galp::codec::device::FastLanes1024InputUntransposer>(iterator, lane, out);
		break;
	}
	case galp::execution::PlanKind::RLE_SLPATCH_U16: {
		using IndexT                      = uint16_t;
		constexpr int RLE_UNPACK_N_VALUES = galp::codec::utils::get_values_per_lane<IndexT>();
		using UnpackerT                   = galp::codec::device::BitUnpackerStatefulBranchless<
		                      IndexT,
		                      UNPACK_N_VECTORS,
		                      RLE_UNPACK_N_VALUES,
		                      galp::codec::device::FFORFunctor<IndexT, UNPACK_N_VECTORS>>;
		using PatcherT =
		    galp::codec::device::StatefulSLPATCHExceptionPatcher<IndexT, UNPACK_N_VECTORS, RLE_UNPACK_N_VALUES>;
		using ExpanderT = galp::codec::device::DummyRLEExpander<T, IndexT, UNPACK_N_VECTORS, RLE_UNPACK_N_VALUES>;
		using DecompressorT =
		    galp::codec::device::RLESLPATCHDecompressor<T,
		                                                IndexT,
		                                                UNPACK_N_VECTORS,
		                                                RLE_UNPACK_N_VALUES,
		                                                UnpackerT,
		                                                PatcherT,
		                                                ExpanderT,
		                                                galp::codec::device::RLESLPATCHColumn<T, IndexT>>;
		auto iterator = DecompressorT(expr.col.rle_slpatch_u16, vector_index, lane);
		run_decompressor<T,
		                 UNPACK_N_VECTORS,
		                 RLE_UNPACK_N_VALUES,
		                 WRITE_OUT,
		                 IndexT,
		                 galp::codec::device::FastLanes1024InputUntransposer>(iterator, lane, out);
		break;
	}
	default:
		break;
	}
}

} // namespace galp::kernels::detail

namespace galp::kernels { namespace device {

template <typename T,
          int                           UNPACK_N_VECTORS,
          int                           UNPACK_N_VALUES,
          bool                          WRITE_OUT     = true,
	          galp::execution::DeltaDecoder DELTA_DECODER = galp::execution::DeltaDecoder::Register>
__device__ __forceinline__ void execute_typed_work_item(const galp::execution::DeviceExpression<T>* exprs,
                                                        const galp::execution::WorkItemAny          work,
                                                        const lane_t                                lane) {
	if (exprs == nullptr) {
		return;
	}
	const auto*  expr         = exprs + work.expr_index;
	const vi_t   vector_index = static_cast<vi_t>(work.vector_index);
	const size_t n_vecs       = galp::codec::utils::get_n_vecs_from_size(expr->n_values);
	// A multi-vector work item decodes UNPACK_N_VECTORS consecutive vectors starting at
	// vector_index, so every vector in the chunk must be in range; guarding only the first
	// lets the unpacker read vector_offsets/bit_widths past the column for a tail chunk.
	if (static_cast<size_t>(vector_index) + static_cast<size_t>(UNPACK_N_VECTORS) > n_vecs) {
		return;
	}
	if (static_cast<uint32_t>(lane) >=
	    galp::execution::semantic_lane_count(galp::execution::type_tag_for<T>(), expr->plan)) {
		return;
	}

	T* out = nullptr;
	if constexpr (WRITE_OUT) {
		out = expr->out + static_cast<size_t>(work.output_vector_index) * galp::codec::consts::VALUES_PER_VECTOR;
	}
	galp::kernels::detail::execute_plan<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT, DELTA_DECODER>(
	    *expr, vector_index, lane, out);
}

template <typename T,
          int                           UNPACK_N_VECTORS,
          int                           UNPACK_N_VALUES,
          bool                          WRITE_OUT     = true,
	          galp::execution::DeltaDecoder DELTA_DECODER = galp::execution::DeltaDecoder::Register>
__global__ void decompress_dispatch_typed(const galp::execution::DeviceExpression<T>* exprs,
                                          const galp::execution::WorkItemAny*         work_items,
                                          const size_t                                n_items) {
	const lane_t   lane     = static_cast<lane_t>(threadIdx.x);
	const uint32_t item_idx = static_cast<uint32_t>(blockIdx.x);
	if (item_idx >= n_items) {
		return;
	}

	const auto work = work_items[item_idx];
	if constexpr (std::is_same_v<T, int8_t>) {
		if (work.type != galp::execution::TypeTag::I8) {
			return;
		}
	} else if constexpr (std::is_same_v<T, int16_t>) {
		if (work.type != galp::execution::TypeTag::I16) {
			return;
		}
	}
	execute_typed_work_item<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT, DELTA_DECODER>(exprs, work, lane);
}

template <int                           UNPACK_N_VECTORS,
          int                           UNPACK_N_VALUES,
          bool                          WRITE_OUT     = true,
	          galp::execution::DeltaDecoder DELTA_DECODER = galp::execution::DeltaDecoder::Register>
__global__ void decompress_dispatch_mixed(const galp::execution::DeviceExpression<int8_t>*  exprs_i8,
                                          const galp::execution::DeviceExpression<int16_t>* exprs_i16,
                                          const galp::execution::MixedWorkSlot*             slots,
                                          const size_t                                      n_slots) {
	const MixedSlotMapping mapping(n_slots);
	const uint32_t         slot_idx = mapping.slot_index();
	const uint32_t         lane     = mapping.slot_lane();
	if (slot_idx >= n_slots || slots == nullptr) {
		return;
	}

	const auto slot          = slots[slot_idx];
	const auto run_work_item = [&](const galp::execution::WorkItemAny work, const lane_t work_lane) {
		if (!galp::execution::is_valid_work_item(work)) {
			return;
		}
		switch (work.type) {
		case galp::execution::TypeTag::I8:
			execute_typed_work_item<int8_t, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT, DELTA_DECODER>(
			    exprs_i8, work, work_lane);
			break;
		case galp::execution::TypeTag::I16:
			execute_typed_work_item<int16_t, UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT, DELTA_DECODER>(
			    exprs_i16, work, work_lane);
			break;
		default:
			break;
		}
	};

	if (galp::execution::is_valid_work_item(slot.second)) {
		if (mapping.is_first_half(lane)) {
			run_work_item(slot.first, static_cast<lane_t>(lane));
		} else {
			run_work_item(slot.second, static_cast<lane_t>(lane - MixedSlotMapping::HALF_SLOT_LANES));
		}
		return;
	}

	run_work_item(slot.first, static_cast<lane_t>(lane));
}

}} // namespace galp::kernels::device

#endif // GALP_ENGINE_DISPATCH_CUH
