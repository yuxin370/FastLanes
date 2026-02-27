// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/kernels/host.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_KERNELS_HOST_CUH
#define ENGINE_KERNELS_HOST_CUH

#include "engine/device-utils.cuh"
#include "engine/kernels/device.cuh"
#include "flsgpu/flsgpu-api.cuh"
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace kernels {
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

#endif // ENGINE_KERNELS_HOST_CUH
