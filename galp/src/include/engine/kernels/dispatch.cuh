// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/kernels/dispatch.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_ENGINE_KERNELS_DISPATCH_CUH
#define GALP_ENGINE_KERNELS_DISPATCH_CUH

#include "engine/device-utils.cuh"
#include "engine/execution/dispatch.cuh"
#include "engine/kernels/device_kernels.cuh"
#include "engine/kernels/rebind.cuh"
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace galp::kernels::detail {

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES, typename DecompressorT, typename ColumnT>
__host__ void
launch_decompress_column_sample(const ColumnT column, T* out, const size_t n_vecs, const size_t shmem_bytes) {
	if constexpr (UNPACK_N_VECTORS == 1) {
		const ThreadblockMapping<T> mapping(UNPACK_N_VECTORS, n_vecs);
		device::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>
		    <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK, shmem_bytes>>>(column, out, n_vecs, 0);
	} else {
		const size_t full_n_vecs = full_vector_count(n_vecs, UNPACK_N_VECTORS);
		if (full_n_vecs != 0) {
			const ThreadblockMapping<T> mapping(UNPACK_N_VECTORS, full_n_vecs);
			device::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>
			    <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK, shmem_bytes>>>(column, out, full_n_vecs, 0);
		}
		if (full_n_vecs != n_vecs) {
			using TailDecompressorT                 = ScalarTailDecompressorT<DecompressorT>;
			const size_t                tail_n_vecs = n_vecs - full_n_vecs;
			const ThreadblockMapping<T> mapping(1, tail_n_vecs);
			device::decompress_column<T, 1, UNPACK_N_VALUES, TailDecompressorT, ColumnT>
			    <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK, shmem_bytes>>>(column, out, tail_n_vecs, full_n_vecs);
		}
	}
}

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES, typename DecompressorT, typename ColumnT>
__host__ void launch_query_column_sample(const ColumnT column, bool* out, const T magic_value, const size_t n_vecs) {
	if constexpr (UNPACK_N_VECTORS == 1) {
		const ThreadblockMapping<T> mapping(UNPACK_N_VECTORS, n_vecs);
		device::query_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>
		    <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column, out, magic_value, n_vecs, 0);
	} else {
		const size_t full_n_vecs = full_vector_count(n_vecs, UNPACK_N_VECTORS);
		if (full_n_vecs != 0) {
			const ThreadblockMapping<T> mapping(UNPACK_N_VECTORS, full_n_vecs);
			device::query_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>
			    <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column, out, magic_value, full_n_vecs, 0);
		}
		if (full_n_vecs != n_vecs) {
			using TailDecompressorT                 = ScalarTailDecompressorT<DecompressorT>;
			const size_t                tail_n_vecs = n_vecs - full_n_vecs;
			const ThreadblockMapping<T> mapping(1, tail_n_vecs);
			device::query_column<T, 1, UNPACK_N_VALUES, TailDecompressorT, ColumnT>
			    <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column, out, magic_value, tail_n_vecs, full_n_vecs);
		}
	}
}

template <typename T,
          unsigned UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES,
          typename DecompressorT,
          typename ColumnT,
          unsigned N_REPETITIONS>
__host__ void launch_compute_column_sample(const ColumnT column, bool* out, const T runtime_zero, const size_t n_vecs) {
	if constexpr (UNPACK_N_VECTORS == 1) {
		const ThreadblockMapping<T> mapping(UNPACK_N_VECTORS, n_vecs);
		device::compute_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT, N_REPETITIONS>
		    <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column, out, runtime_zero, n_vecs, 0);
	} else {
		const size_t full_n_vecs = full_vector_count(n_vecs, UNPACK_N_VECTORS);
		if (full_n_vecs != 0) {
			const ThreadblockMapping<T> mapping(UNPACK_N_VECTORS, full_n_vecs);
			device::compute_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT, N_REPETITIONS>
			    <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(column, out, runtime_zero, full_n_vecs, 0);
		}
		if (full_n_vecs != n_vecs) {
			using TailDecompressorT                 = ScalarTailDecompressorT<DecompressorT>;
			const size_t                tail_n_vecs = n_vecs - full_n_vecs;
			const ThreadblockMapping<T> mapping(1, tail_n_vecs);
			device::compute_column<T, 1, UNPACK_N_VALUES, TailDecompressorT, ColumnT, N_REPETITIONS>
			    <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(
			        column, out, runtime_zero, tail_n_vecs, full_n_vecs);
		}
	}
}

} // namespace galp::kernels::detail

namespace galp::kernels::host {

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES, typename DecompressorT, typename ColumnT>
__host__ T* decompress_column(const ColumnT column, const uint32_t n_samples) {
	size_t      n_vecs = galp::codec::utils::get_n_vecs_from_size(column.n_values);
	GPUArray<T> device_out(column.n_values);

	cudaEvent_t ev_start {}, ev_stop {};
	CUDA_SAFE_CALL(cudaEventCreate(&ev_start));
	CUDA_SAFE_CALL(cudaEventCreate(&ev_stop));
	CUDA_SAFE_CALL(cudaEventRecord(ev_start, 0));

	size_t shmem_bytes = 0;
	for (uint32_t i {0}; i < n_samples; ++i) {
		detail::launch_decompress_column_sample<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, device_out.get(), n_vecs, shmem_bytes);
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
	size_t         n_vecs = galp::codec::utils::get_n_vecs_from_size(column.n_values);
	bool           result = false;
	GPUArray<bool> device_out(1, &result);

	for (uint32_t i {0}; i < n_samples; ++i) {
		detail::launch_query_column_sample<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, device_out.get(), magic_value, n_vecs);
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
	size_t         n_vecs = galp::codec::utils::get_n_vecs_from_size(column.n_values);
	bool           result = false;
	GPUArray<bool> device_out(1, &result);

	for (uint32_t i {0}; i < n_samples; ++i) {
		detail::
		    launch_compute_column_sample<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT, N_REPETITIONS>(
		        column, device_out.get(), 0, n_vecs);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
	}
	device_out.copy_to_host(&result);
	return result;
}

} // namespace galp::kernels::host

#endif // GALP_ENGINE_KERNELS_DISPATCH_CUH
