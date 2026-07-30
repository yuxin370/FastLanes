// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/cuda/device_kernels.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_CUDA_DEVICE_KERNELS_CUH
#define GALP_CUDA_DEVICE_KERNELS_CUH

#include "cuda/device_utils.cuh"
#include "codecs/consts.cuh"
#include "codecs/decode/alp.cuh"
#include <cstddef>
#include <cstdint>

namespace galp::kernels::device {

template <typename T,
	      int UNPACK_N_VECTORS,
	      int UNPACK_N_VALUES,
	      typename DecompressorT,
	      typename ColumnT,
	      typename UntransposerT = galp::codec::device::IdentityUntransposer>
__global__ void
decompress_column(const ColumnT column, T* out, const size_t scheduled_n_vecs = 0, const size_t vector_offset = 0) {
	constexpr uint32_t N_VALUES           = UNPACK_N_VALUES * UNPACK_N_VECTORS;
	const auto         mapping            = VectorToWarpMapping<T, UNPACK_N_VECTORS>();
	const lane_t       lane               = mapping.get_lane();
	const size_t       local_vector_index = static_cast<size_t>(mapping.get_vector_index());

	size_t       n_vecs = galp::codec::utils::get_n_vecs_from_size(column.n_values);
	const size_t default_active_n_vecs =
	    UNPACK_N_VECTORS <= 1 ? n_vecs : (n_vecs / static_cast<size_t>(UNPACK_N_VECTORS)) * UNPACK_N_VECTORS;
	const size_t active_n_vecs = scheduled_n_vecs != 0 ? scheduled_n_vecs : default_active_n_vecs;
	if (local_vector_index >= active_n_vecs) {
		return;
	}

	const size_t vector_index_size = vector_offset + local_vector_index;
	if (vector_index_size >= n_vecs) {
		return;
	}
	const auto vector_index = static_cast<vi_t>(vector_index_size);

	out += vector_index * galp::codec::consts::VALUES_PER_VECTOR;

	T    registers[N_VALUES];
	auto iterator = DecompressorT(column, vector_index, lane);

	const size_t vector_base = static_cast<size_t>(vector_index) * galp::codec::consts::VALUES_PER_VECTOR;
	for (si_t i = 0; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
		iterator.unpack_next_into(registers);

#pragma unroll
		for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
#pragma unroll
			for (int w {0}; w < UNPACK_N_VALUES; ++w) {
				const uint32_t in_idx  = static_cast<uint32_t>(lane) + static_cast<uint32_t>(i + w) * mapping.N_LANES;
				const size_t out_idx = static_cast<size_t>(v) * galp::codec::consts::VALUES_PER_VECTOR +
				                       UntransposerT::map_index(in_idx);
				if (vector_base + out_idx < column.n_values) {
					out[out_idx] = registers[w + v * UNPACK_N_VALUES];
				}
			}
		}
	}
}

// Standalone whole-lane DELTA kernel. The generic iterator kernel uses a
// signed output tile in addition to DELTARegisterDecompressor's unsigned
// transform tile. This specialization keeps unpack, FFOR restoration, prefix,
// untranspose and bounded tail stores on a single unsigned register tile.
template <typename T,
          int UNPACK_N_VECTORS,
          typename DecompressorT,
          typename ColumnT,
          typename UntransposerT = galp::codec::device::IdentityUntransposer>
__global__ void decompress_delta_register_column(const ColumnT column,
	                                                T* __restrict out,
	                                                const size_t scheduled_n_vecs = 0,
	                                                const size_t vector_offset    = 0) {
	using UIntT = typename galp::codec::utils::same_width_uint<T>::type;
	constexpr int N_VALUES = galp::codec::utils::get_values_per_lane<T>();
	const auto    mapping  = VectorToWarpMapping<T, UNPACK_N_VECTORS>();
	const lane_t  lane     = mapping.get_lane();
	const size_t  local_vector_index = static_cast<size_t>(mapping.get_vector_index());

	const size_t n_vecs = galp::codec::utils::get_n_vecs_from_size(column.n_values);
	const size_t default_active_n_vecs =
	    UNPACK_N_VECTORS <= 1 ? n_vecs : (n_vecs / static_cast<size_t>(UNPACK_N_VECTORS)) * UNPACK_N_VECTORS;
	const size_t active_n_vecs = scheduled_n_vecs != 0 ? scheduled_n_vecs : default_active_n_vecs;
	if (local_vector_index >= active_n_vecs) {
		return;
	}

	const size_t vector_index_size = vector_offset + local_vector_index;
	if (vector_index_size >= n_vecs) {
		return;
	}
	const auto vector_index = static_cast<vi_t>(vector_index_size);
	UIntT     registers[N_VALUES * UNPACK_N_VECTORS];
	auto      decoder = DecompressorT(column, vector_index, lane);
	decoder.decode_lane_into(registers);

#pragma unroll
	for (int vector = 0; vector < UNPACK_N_VECTORS; ++vector) {
#pragma unroll
		for (int position = 0; position < N_VALUES; ++position) {
			const uint32_t in_idx = static_cast<uint32_t>(lane) + static_cast<uint32_t>(position) * mapping.N_LANES;
			const size_t output_index =
			    (vector_index_size + static_cast<size_t>(vector)) * galp::codec::consts::VALUES_PER_VECTOR +
			    UntransposerT::map_index(in_idx);
			if (output_index < column.n_values) {
				out[output_index] = static_cast<T>(registers[position + vector * N_VALUES]);
			}
		}
	}
}

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES, typename DecompressorT, typename ColumnT>
__global__ void query_column(const ColumnT column,
                             bool*         out,
                             const T       magic_value,
                             const size_t  scheduled_n_vecs = 0,
                             const size_t  vector_offset    = 0) {
	constexpr uint32_t N_VALUES           = UNPACK_N_VALUES * UNPACK_N_VECTORS;
	const auto         mapping            = VectorToWarpMapping<T, UNPACK_N_VECTORS>();
	const lane_t       lane               = mapping.get_lane();
	const size_t       local_vector_index = static_cast<size_t>(mapping.get_vector_index());
	const size_t       n_vecs             = galp::codec::utils::get_n_vecs_from_size(column.n_values);
	const size_t       default_active_n_vecs =
        UNPACK_N_VECTORS <= 1 ? n_vecs : (n_vecs / static_cast<size_t>(UNPACK_N_VECTORS)) * UNPACK_N_VECTORS;
	const size_t active_n_vecs = scheduled_n_vecs != 0 ? scheduled_n_vecs : default_active_n_vecs;
	if (local_vector_index >= active_n_vecs) {
		return;
	}
	const size_t vector_index_size = vector_offset + local_vector_index;
	if (vector_index_size >= n_vecs) {
		return;
	}
	const auto vector_index = static_cast<vi_t>(vector_index_size);
	T          registers[N_VALUES];
	auto       checker = MagicChecker<T, N_VALUES>(magic_value);

	DecompressorT unpacker = DecompressorT(column, vector_index, lane);
	for (si_t i = 0; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
		unpacker.unpack_next_into(registers);
		checker.check(registers);
	}
	checker.write_result(out);
}

template <typename T,
          int UNPACK_N_VECTORS,
          int UNPACK_N_VALUES,
          typename DecompressorT,
          typename ColumnT,
          int N_REPETITIONS = 10>
__global__ void compute_column(const ColumnT column,
                               bool* __restrict out,
                               const T      runtime_zero,
                               const size_t scheduled_n_vecs = 0,
                               const size_t vector_offset    = 0) {
	constexpr T        RANDOM_VALUE       = 3;
	constexpr uint32_t N_VALUES           = UNPACK_N_VALUES * UNPACK_N_VECTORS;
	const auto         mapping            = VectorToWarpMapping<T, UNPACK_N_VECTORS>();
	const lane_t       lane               = mapping.get_lane();
	const size_t       local_vector_index = static_cast<size_t>(mapping.get_vector_index());
	const size_t       n_vecs             = galp::codec::utils::get_n_vecs_from_size(column.n_values);
	const size_t       default_active_n_vecs =
        UNPACK_N_VECTORS <= 1 ? n_vecs : (n_vecs / static_cast<size_t>(UNPACK_N_VECTORS)) * UNPACK_N_VECTORS;
	const size_t active_n_vecs = scheduled_n_vecs != 0 ? scheduled_n_vecs : default_active_n_vecs;
	if (local_vector_index >= active_n_vecs) {
		return;
	}
	const size_t vector_index_size = vector_offset + local_vector_index;
	if (vector_index_size >= n_vecs) {
		return;
	}
	const auto    vector_index = static_cast<vi_t>(vector_index_size);
	T             registers[N_VALUES];
	auto          checker      = MagicChecker<T, N_VALUES>(1);
	DecompressorT decompressor = DecompressorT(column, vector_index, lane);

	for (si_t i = 0; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
		decompressor.unpack_next_into(registers);

#pragma unroll
		for (int32_t j {0}; j < N_VALUES; ++j) {
#pragma unroll
			for (int32_t k {0}; k < N_REPETITIONS; ++k) {
				registers[j] *= RANDOM_VALUE;
				registers[j] <<= RANDOM_VALUE;
				registers[j] += RANDOM_VALUE;
				registers[j] ^= runtime_zero;
			}
		}
		checker.check(registers);
	}
	checker.write_result(out);
}

} // namespace galp::kernels::device

#endif // GALP_CUDA_DEVICE_KERNELS_CUH
