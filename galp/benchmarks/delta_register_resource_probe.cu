// PTXAS-only resource probe for the production whole-lane DELTA decode path.
// This file is compiled directly by run_delta_microbenchmark.py and is not
// linked into the benchmark binary.
#include "engine/dispatch.cuh"

namespace {

template <typename T, unsigned UNPACK_N_VECTORS>
__device__ __forceinline__ void probe_delta_register(const galp::codec::device::DELTAColumn<T> column,
	                                                  T* __restrict out) {
	using UIntT = typename galp::codec::utils::same_width_uint<T>::type;
	constexpr unsigned N_VALUES = galp::codec::utils::get_values_per_lane<T>();
	constexpr unsigned N_LANES  = galp::codec::utils::get_n_lanes<T>();
	using UnpackerT = galp::codec::device::BitUnpackerLaneTile<
	    UIntT,
	    UNPACK_N_VECTORS,
	    N_VALUES,
	    galp::codec::device::FFORFunctor<UIntT, UNPACK_N_VECTORS>>;
	using DecompressorT =
	    galp::codec::device::DELTARegisterDecompressor<T, UNPACK_N_VECTORS, UnpackerT, decltype(column)>;

	const auto global_thread = blockIdx.x * blockDim.x + threadIdx.x;
	const auto lane          = static_cast<lane_t>(global_thread % N_LANES);
	const auto vector_index  = static_cast<vi_t>((global_thread / N_LANES) * UNPACK_N_VECTORS);
	auto       decoder       = DecompressorT(column, vector_index, lane);
	galp::kernels::detail::run_delta_register_decompressor<T, UNPACK_N_VECTORS>(decoder, lane, out);
}

} // namespace

#define GALP_DELTA_REGISTER_PROBE(NAME, TYPE, UNPACK)                                                                  \
	extern "C" __global__ void NAME(const galp::codec::device::DELTAColumn<TYPE> column, TYPE* out) {                  \
		probe_delta_register<TYPE, UNPACK>(column, out);                                                               \
	}

GALP_DELTA_REGISTER_PROBE(delta_register_i8_u1, int8_t, 1)
GALP_DELTA_REGISTER_PROBE(delta_register_i8_u2, int8_t, 2)
GALP_DELTA_REGISTER_PROBE(delta_register_i8_u4, int8_t, 4)
GALP_DELTA_REGISTER_PROBE(delta_register_i16_u1, int16_t, 1)
GALP_DELTA_REGISTER_PROBE(delta_register_i16_u2, int16_t, 2)
GALP_DELTA_REGISTER_PROBE(delta_register_i16_u4, int16_t, 4)

#undef GALP_DELTA_REGISTER_PROBE
