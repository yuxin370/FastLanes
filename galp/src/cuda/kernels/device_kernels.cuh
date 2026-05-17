// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/kernels/device_kernels.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_ENGINE_KERNELS_DEVICE_KERNELS_CUH
#define GALP_ENGINE_KERNELS_DEVICE_KERNELS_CUH

#include "cuda/device_utils.cuh"
#include "codecs/consts.cuh"
#include "codecs/decode/alp.cuh"
#include <cstddef>
#include <cstdint>

namespace galp::kernels::device {

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES, typename DecompressorT, typename ColumnT>
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
				const size_t   out_idx = static_cast<size_t>(v) * galp::codec::consts::VALUES_PER_VECTOR + in_idx;
				if (vector_base + out_idx < column.n_values) {
					out[out_idx] = registers[w + v * UNPACK_N_VALUES];
				}
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

#endif // GALP_ENGINE_KERNELS_DEVICE_KERNELS_CUH
