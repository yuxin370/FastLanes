// PTXAS-only resource probe for the production whole-lane DELTA decode path.
// This file is compiled directly by run_delta_microbenchmark.py and is not
// linked into the benchmark binary.
#include "engine/dispatch.cuh"

namespace {

template <typename T, unsigned UNPACK_N_VECTORS>
__device__ __forceinline__ void probe_delta_register(const fastlens::codec::device::DELTAColumn<T> column,
	                                                  T* __restrict out) {
	using UIntT = typename fastlens::codec::utils::same_width_uint<T>::type;
	constexpr unsigned N_VALUES = fastlens::codec::utils::get_values_per_lane<T>();
	constexpr unsigned N_LANES  = fastlens::codec::utils::get_n_lanes<T>();
	using UnpackerT = fastlens::codec::device::BitUnpackerLaneTile<
	    UIntT,
	    UNPACK_N_VECTORS,
	    N_VALUES,
	    fastlens::codec::device::FFORFunctor<UIntT, UNPACK_N_VECTORS>>;
	using DecompressorT =
	    fastlens::codec::device::DELTARegisterDecompressor<T, UNPACK_N_VECTORS, UnpackerT, decltype(column)>;

	const auto global_thread = blockIdx.x * blockDim.x + threadIdx.x;
	const auto lane          = static_cast<lane_t>(global_thread % N_LANES);
	const auto vector_index  = static_cast<vi_t>((global_thread / N_LANES) * UNPACK_N_VECTORS);
	auto       decoder       = DecompressorT(column, vector_index, lane);
	fastlens::kernels::detail::run_delta_register_decompressor<T, UNPACK_N_VECTORS>(decoder, lane, out);
}

} // namespace

#define FASTLENS_DELTA_REGISTER_PROBE(NAME, TYPE, UNPACK)                                                                  \
	extern "C" __global__ void NAME(const fastlens::codec::device::DELTAColumn<TYPE> column, TYPE* out) {                  \
		probe_delta_register<TYPE, UNPACK>(column, out);                                                               \
	}

FASTLENS_DELTA_REGISTER_PROBE(delta_register_i8_u1, int8_t, 1)
FASTLENS_DELTA_REGISTER_PROBE(delta_register_i8_u2, int8_t, 2)
FASTLENS_DELTA_REGISTER_PROBE(delta_register_i8_u4, int8_t, 4)
FASTLENS_DELTA_REGISTER_PROBE(delta_register_i16_u1, int16_t, 1)
FASTLENS_DELTA_REGISTER_PROBE(delta_register_i16_u2, int16_t, 2)
FASTLENS_DELTA_REGISTER_PROBE(delta_register_i16_u4, int16_t, 4)

#undef FASTLENS_DELTA_REGISTER_PROBE
