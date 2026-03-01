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

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES, typename DecompressorT>
__device__ __forceinline__ void run_decompressor(DecompressorT&& iterator, const lane_t lane, T* __restrict out) {
	const auto mapping = VectorToWarpMapping<T, UNPACK_N_VECTORS>();
	T          registers[UNPACK_N_VALUES * UNPACK_N_VECTORS];
	for (si_t i = 0; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
		iterator.unpack_next_into(registers);
		write_registers_to_global<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, mapping.N_LANES>(lane, i, registers, out);
	}
}

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES>
__device__ __forceinline__ void
execute_plan(const dispatch::DeviceExpression<T>& expr, const vi_t vector_index, const lane_t lane, T* __restrict out) {
	switch (expr.plan) {
	case dispatch::PlanKind::UNCOMPRESSED: {
		using UnpackerT = flsgpu::device::
		    BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, flsgpu::device::BPFunctor<T>>;
		using DecompressorT =
		    flsgpu::device::BPDecompressor<T, UNPACK_N_VECTORS, UnpackerT, flsgpu::device::BPColumn<T>>;
		auto iterator = DecompressorT(expr.col.bp, vector_index, lane);
		run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(iterator, lane, out);
		break;
	}
	case dispatch::PlanKind::CONSTANT: {
		using DecompressorT = flsgpu::device::
		    CONSTANTDecompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, flsgpu::device::CONSTANTColumn<T>>;
		auto iterator = DecompressorT(expr.col.constant, vector_index, lane);
		run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(iterator, lane, out);
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
		run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(iterator, lane, out);
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
		run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(iterator, lane, out);
		break;
	}
	case dispatch::PlanKind::DICT_FFOR: {
		if (expr.dict_index_bits == 8) {
			using IndexT     = uint8_t;
			using ProcessorT = flsgpu::device::DICTFunctorIdx<T, IndexT, UNPACK_N_VECTORS>;
			using UnpackerT  = flsgpu::device::
			    BitUnpackerStatefulBranchlessIdx<T, IndexT, UNPACK_N_VECTORS, UNPACK_N_VALUES, ProcessorT>;
			using DecompressorT = flsgpu::device::
			    DICTDecompressor<T, UNPACK_N_VECTORS, UnpackerT, flsgpu::device::DICTFFORColumn<T, IndexT>, ProcessorT>;
			auto iterator = DecompressorT(expr.col.dictffor_u8, vector_index, lane);
			run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(iterator, lane, out);
		} else {
			using ProcessorT = flsgpu::device::DICTFunctor<T, UNPACK_N_VECTORS>;
			using UnpackerT =
			    flsgpu::device::BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, ProcessorT>;
			using DecompressorT =
			    flsgpu::device::DICTDecompressor<T, UNPACK_N_VECTORS, UnpackerT, flsgpu::device::DICTFFORColumn<T>, ProcessorT>;
			auto iterator = DecompressorT(expr.col.dictffor, vector_index, lane);
			run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(iterator, lane, out);
		}
		break;
	}
	case dispatch::PlanKind::DICT_FFOR_SLPATCH: {
		if (expr.dict_index_bits == 8) {
			using IndexT     = uint8_t;
			using ProcessorT = flsgpu::device::DICTFunctorIdx<T, IndexT, UNPACK_N_VECTORS>;
			using UnpackerT  = flsgpu::device::
			    BitUnpackerStatefulBranchlessIdx<T, IndexT, UNPACK_N_VECTORS, UNPACK_N_VALUES, ProcessorT>;
			using PatcherT =
			    flsgpu::device::StatefulSLPATCHDictExceptionPatcher<T, IndexT, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
			using DecompressorT = flsgpu::device::DICTSLPATCHDecompressor<T,
			                                                              UNPACK_N_VECTORS,
			                                                              UNPACK_N_VALUES,
			                                                              UnpackerT,
			                                                              PatcherT,
			                                                              flsgpu::device::DICTSLPATCHColumn<T, IndexT>,
			                                                              ProcessorT>;
			auto iterator = DecompressorT(expr.col.dictslpatch_u8, vector_index, lane);
			run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(iterator, lane, out);
		} else {
			using IndexT     = typename flsgpu::device::DICTSLPATCHColumn<T>::INDEX_T;
			using ProcessorT = flsgpu::device::DICTFunctor<T, UNPACK_N_VECTORS>;
			using UnpackerT =
			    flsgpu::device::BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, ProcessorT>;
			using PatcherT =
			    flsgpu::device::StatefulSLPATCHDictExceptionPatcher<T, IndexT, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
			using DecompressorT = flsgpu::device::DICTSLPATCHDecompressor<T,
			                                                              UNPACK_N_VECTORS,
			                                                              UNPACK_N_VALUES,
			                                                              UnpackerT,
			                                                              PatcherT,
			                                                              flsgpu::device::DICTSLPATCHColumn<T>,
			                                                              ProcessorT>;
			auto iterator = DecompressorT(expr.col.dictslpatch, vector_index, lane);
			run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(iterator, lane, out);
		}
		break;
	}
	case dispatch::PlanKind::FREQUENCY: {
		using PatcherT = flsgpu::device::StatefulFREQExceptionPatcher<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
		using DecompressorT =
		    flsgpu::device::FREQDecompressor<T, UNPACK_N_VECTORS, PatcherT, flsgpu::device::FREQColumn<T>>;
		auto iterator = DecompressorT(expr.col.freq, vector_index, lane);
		run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(iterator, lane, out);
		break;
	}
	case dispatch::PlanKind::CROSS_RLE: {
		using ExpanderT = flsgpu::device::StatefulCROSSRLEExpander<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
		using DecompressorT =
		    flsgpu::device::CROSSRLEDecompressor<T, UNPACK_N_VECTORS, ExpanderT, flsgpu::device::CROSSRLEColumn<T>>;
		auto iterator = DecompressorT(expr.col.crossrle, vector_index, lane);
		run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(iterator, lane, out);
		break;
	}
	case dispatch::PlanKind::RLE: {
		using CodeT = uint16_t;
		using UnpackerT = flsgpu::device::BitUnpackerStatefulBranchless<
		    CodeT,
		    UNPACK_N_VECTORS,
		    UNPACK_N_VALUES,
		    flsgpu::device::FFORFunctor<CodeT, UNPACK_N_VECTORS>>;
		using ExpanderT = flsgpu::device::DummyRLEExpander<T, CodeT, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
		using DecompressorT = flsgpu::device::RLEDecompressor<T,
		                                                      CodeT,
		                                                      UNPACK_N_VECTORS,
		                                                      UNPACK_N_VALUES,
		                                                      UnpackerT,
		                                                      ExpanderT,
		                                                      flsgpu::device::RLEColumn<T, CodeT>>;
		auto iterator = DecompressorT(expr.col.rle, vector_index, lane);
		run_decompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(iterator, lane, out);
		break;
	}
	default:
		break;
	}
}

} // namespace device_exec

namespace kernels {
namespace device {

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES>
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

	const auto expr         = exprs[work.expr_index];
	const vi_t vector_index = static_cast<vi_t>(work.vector_index);
	const size_t n_vecs     = utils::get_n_vecs_from_size(expr.n_values);
	if (static_cast<size_t>(vector_index) >= n_vecs) {
		return;
	}

	T* out = expr.out + vector_index * consts::VALUES_PER_VECTOR;
	device_exec::execute_plan<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(expr, vector_index, lane, out);
}

template <int UNPACK_N_VECTORS, int UNPACK_N_VALUES>
__global__ void decompress_dispatch_mixed(const dispatch::DeviceExpression<int8_t>*  exprs_i8,
                                          const dispatch::DeviceExpression<int16_t>* exprs_i16,
                                          const dispatch::WorkItemAny*               work_items,
                                          const size_t                               n_items) {
	constexpr uint32_t lanes_i8    = utils::get_n_lanes<int8_t>();
	constexpr uint32_t lanes_i16   = utils::get_n_lanes<int16_t>();
	constexpr uint32_t group_lanes = (lanes_i8 > lanes_i16) ? lanes_i8 : lanes_i16;

	const uint32_t global_thread = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t item_idx      = global_thread / group_lanes;
	if (item_idx >= n_items) {
		return;
	}
	const uint32_t lane = global_thread - item_idx * group_lanes;

	const auto work = work_items[item_idx];
	switch (work.type) {
	case dispatch::TypeTag::I8: {
		if (lane >= lanes_i8 || exprs_i8 == nullptr) {
			return;
		}
		const auto   expr         = exprs_i8[work.expr_index];
		const vi_t   vector_index = static_cast<vi_t>(work.vector_index);
		const size_t n_vecs       = utils::get_n_vecs_from_size(expr.n_values);
		if (static_cast<size_t>(vector_index) >= n_vecs) {
			return;
		}
		int8_t* out = expr.out + vector_index * consts::VALUES_PER_VECTOR;
		device_exec::execute_plan<int8_t, UNPACK_N_VECTORS, UNPACK_N_VALUES>(expr, vector_index, lane, out);
		break;
	}
	case dispatch::TypeTag::I16: {
		if (lane >= lanes_i16 || exprs_i16 == nullptr) {
			return;
		}
		const auto   expr         = exprs_i16[work.expr_index];
		const vi_t   vector_index = static_cast<vi_t>(work.vector_index);
		const size_t n_vecs       = utils::get_n_vecs_from_size(expr.n_values);
		if (static_cast<size_t>(vector_index) >= n_vecs) {
			return;
		}
		int16_t* out = expr.out + vector_index * consts::VALUES_PER_VECTOR;
		device_exec::execute_plan<int16_t, UNPACK_N_VECTORS, UNPACK_N_VALUES>(expr, vector_index, lane, out);
		break;
	}
	default:
		break;
	}
}

} // namespace device
} // namespace kernels

#endif // ENGINE_EXECUTION_DISPATCH_CUH
