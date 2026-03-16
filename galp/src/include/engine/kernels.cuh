// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/kernels.cuh
// ────────────────────────────────────────────────────────
#ifndef FLS_GLOBAL_CUH
#define FLS_GLOBAL_CUH

#include "engine/device-utils.cuh"
#include "engine/execution/dispatch.cuh"
#include "flsgpu/consts.cuh"
#include "flsgpu/flsgpu-api.cuh"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <type_traits>

namespace kernels {
namespace device {

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES, typename DecompressorT, typename ColumnT>
__global__ void decompress_column(const ColumnT column, T* out) {
	constexpr uint32_t N_VALUES     = UNPACK_N_VALUES * UNPACK_N_VECTORS;
	const auto         mapping      = VectorToWarpMapping<T, UNPACK_N_VECTORS>();
	const lane_t       lane         = mapping.get_lane();
	const int32_t      vector_index = mapping.get_vector_index();

	size_t n_vecs = utils::get_n_vecs_from_size(column.n_values);
	if ((size_t)vector_index >= n_vecs) {
		return;
	}

	out += vector_index * consts::VALUES_PER_VECTOR;

	T    registers[N_VALUES];
	auto iterator = DecompressorT(column, vector_index, lane);

	uint32_t acc = 2166136261u;
	for (si_t i = 0; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
		iterator.unpack_next_into(registers);

#pragma unroll
		for (int k = 0; k < N_VALUES; ++k) {
			acc ^= (uint32_t)registers[k];
		}
		// write_registers_to_global<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, mapping.N_LANES>(lane, i, registers, out);
	}
	if ((acc & 0xFFFFFFFF) == 0) {
		out[0] = (T)acc;
	}
}

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES, typename DecompressorT, typename ColumnT>
__global__ void query_column(const ColumnT column, bool* out, const T magic_value) {
	constexpr uint32_t N_VALUES     = UNPACK_N_VALUES * UNPACK_N_VECTORS;
	const auto         mapping      = VectorToWarpMapping<T, UNPACK_N_VECTORS>();
	const lane_t       lane         = mapping.get_lane();
	const int32_t      vector_index = mapping.get_vector_index();
	T                  registers[N_VALUES];
	auto               checker = MagicChecker<T, N_VALUES>(magic_value);

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
__global__ void compute_column(const ColumnT column, bool* __restrict out, const T runtime_zero) {
	constexpr T        RANDOM_VALUE = 3;
	constexpr uint32_t N_VALUES     = UNPACK_N_VALUES * UNPACK_N_VECTORS;
	const auto         mapping      = VectorToWarpMapping<T, UNPACK_N_VECTORS>();
	const lane_t       lane         = mapping.get_lane();
	const vi_t         vector_index = mapping.get_vector_index();
	T                  registers[N_VALUES];
	auto               checker      = MagicChecker<T, N_VALUES>(1);
	DecompressorT      decompressor = DecompressorT(column, vector_index, lane);

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

} // namespace device

namespace host {

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES, typename DecompressorT, typename ColumnT>
__host__ T* decompress_column(const ColumnT column, const uint32_t n_samples) {
	size_t                      n_vecs = utils::get_n_vecs_from_size(column.n_values);
	const ThreadblockMapping<T> mapping(UNPACK_N_VECTORS, n_vecs);
	GPUArray<T>                 device_out(column.n_values);

	cudaEvent_t ev_start {}, ev_stop {};
	CUDA_SAFE_CALL(cudaEventCreate(&ev_start));
	CUDA_SAFE_CALL(cudaEventCreate(&ev_stop));
	CUDA_SAFE_CALL(cudaEventRecord(ev_start, 0));

	size_t shmem_bytes = 0;
	for (uint32_t i {0}; i < n_samples; ++i) {
		device::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>
		    <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK, shmem_bytes>>>(column, device_out.get());
		CUDA_SAFE_CALL(cudaGetLastError());
	}

	CUDA_SAFE_CALL(cudaEventRecord(ev_stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(ev_stop));

	float ms = 0.0f;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, ev_start, ev_stop));
	CUDA_SAFE_CALL(cudaEventDestroy(ev_start));
	CUDA_SAFE_CALL(cudaEventDestroy(ev_stop));

	const double avg_us = (n_samples > 0) ? (ms * 1000.0 / (double)n_samples) : 0.0;
	printf("[Decompress KERNEL TIME] unpack_vecs=%u unpack_vals=%u total=%.3f ms n_samples=%u avg=%.3f us\n",
	       (unsigned)UNPACK_N_VECTORS,
	       UNPACK_N_VALUES,
	       (double)ms,
	       n_samples,
	       avg_us);

	T* out = new T[column.n_values];
	device_out.copy_to_host(out);
	return out;
}

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES, typename DecompressorT, typename ColumnT>
__host__ bool query_column(const ColumnT column, const T magic_value, const uint32_t n_samples) {
	size_t                      n_vecs = utils::get_n_vecs_from_size(column.n_values);
	const ThreadblockMapping<T> mapping(UNPACK_N_VECTORS, n_vecs);
	bool                        result = false;
	GPUArray<bool>              device_out(1, &result);

	for (uint32_t i {0}; i < n_samples; ++i) {
		device::query_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>
		    <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column, device_out.get(), magic_value);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
	}

	device_out.copy_to_host(&result);
	return result;
}

template <typename T,
          unsigned UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES,
          typename DecompressorT,
          typename ColumnT,
          unsigned N_REPETITIONS>
__host__ bool compute_column(const ColumnT column, const uint32_t n_samples) {
	size_t                      n_vecs = utils::get_n_vecs_from_size(column.n_values);
	const ThreadblockMapping<T> mapping(UNPACK_N_VECTORS, n_vecs);
	GPUArray<bool>              device_out(1);
	bool                        result;

	for (uint32_t i {0}; i < n_samples; ++i) {
		device::compute_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT, N_REPETITIONS>
		    <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column, device_out.get(), 0);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
	}
	device_out.copy_to_host(&result);
	return result;
}

} // namespace host
} // namespace kernels

#endif // FLS_GLOBAL_CUH
