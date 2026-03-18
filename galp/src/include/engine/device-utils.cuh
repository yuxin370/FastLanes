// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/device-utils.cuh
// ────────────────────────────────────────────────────────
#include "flsgpu/fls/untransposers.cuh"
#include "flsgpu/flsgpu-api.cuh"
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>

#ifndef GPU_DEVICE_UTILS_CUH
#define GPU_DEVICE_UTILS_CUH

namespace lane_policy {

// Value-lane policy keeps FastLanes semantics:
// logical lanes are derived from the encoded value type width.
template <typename T>
struct ValueLanePolicy {
	static constexpr uint32_t semantic_lanes = static_cast<uint32_t>(utils::get_n_lanes<T>());
	static constexpr uint32_t scheduling_lanes =
	    (semantic_lanes < uint32_t {consts::THREADS_PER_WARP}) ? uint32_t {consts::THREADS_PER_WARP} : semantic_lanes;
	static constexpr uint32_t values_per_lane = static_cast<uint32_t>(utils::get_values_per_lane<T>());
};

} // namespace lane_policy

template <typename T>
struct SingleVectorPerWarpThreadblockMapping {
	using Policy = lane_policy::ValueLanePolicy<T>;

	static constexpr unsigned N_WARPS_PER_BLOCK =
	    std::max(Policy::semantic_lanes / uint32_t {consts::THREADS_PER_WARP},
	             8u); // at least 8 warps per block to ensure enough parallelism for latency hiding
	static constexpr unsigned N_THREADS_PER_BLOCK = N_WARPS_PER_BLOCK * consts::THREADS_PER_WARP;
	static constexpr unsigned N_CONCURRENT_VECTORS_PER_BLOCK =
	    N_THREADS_PER_BLOCK / std::max(Policy::semantic_lanes, uint32_t {consts::THREADS_PER_WARP});

	const unsigned n_blocks;

	// to cover all the vectors, we need at least n_vecs / (unpack_n_vecs * N_CONCURRENT_VECTORS_PER_BLOCK) blocks
	SingleVectorPerWarpThreadblockMapping(const size_t unpack_n_vecs, const size_t n_vecs)
	    : n_blocks(std::max(static_cast<unsigned long>(1), n_vecs / (unpack_n_vecs * N_CONCURRENT_VECTORS_PER_BLOCK))) {
	}
};
template <typename T>
struct FillWarpThreadblockMapping {
	using Policy = lane_policy::ValueLanePolicy<T>;

	static constexpr unsigned N_WARPS_PER_BLOCK =
	    std::max(Policy::semantic_lanes / uint32_t {consts::THREADS_PER_WARP}, 8u);
	static constexpr unsigned N_THREADS_PER_BLOCK            = N_WARPS_PER_BLOCK * consts::THREADS_PER_WARP;
	static constexpr unsigned N_CONCURRENT_VECTORS_PER_BLOCK = N_THREADS_PER_BLOCK / Policy::semantic_lanes;

	const unsigned n_blocks;

	FillWarpThreadblockMapping(const size_t unpack_n_vecs, const size_t n_vecs)
	    : n_blocks(
	          (std::max(static_cast<unsigned long>(1), n_vecs / (unpack_n_vecs * N_CONCURRENT_VECTORS_PER_BLOCK)))) {
	}
};

#ifdef SingleVectorMapping
template <typename T>
using ThreadblockMapping = SingleVectorPerWarpThreadblockMapping<T>;
#else
template <typename T>
using ThreadblockMapping = FillWarpThreadblockMapping<T>;
#endif

template <typename T, unsigned UNPACK_N_VECTORS>
struct SingleVectorPerWarpMapping {
	using Policy = lane_policy::ValueLanePolicy<T>;

	static constexpr uint32_t N_LANES          = Policy::semantic_lanes;
	static constexpr uint32_t N_VALUES_IN_LANE = Policy::values_per_lane;

	__device__ __forceinline__ lane_t get_lane() const {
		return threadIdx.x % N_LANES;
	}

	__device__ __forceinline__ vi_t get_vector_index() const {
		// Concurrent vectors per block: how many vectors can be processed
		// by the block simultaneously, assuming that each thread is 1 lane

		const int32_t concurrent_vectors_per_block =
		    blockDim.x / std::max(N_LANES, uint32_t {consts::THREADS_PER_WARP});
		const int32_t vectors_per_block = concurrent_vectors_per_block * UNPACK_N_VECTORS;

		const int32_t concurrent_vector_index = threadIdx.x / N_LANES;
		const int32_t block_index             = blockIdx.x;

		return vectors_per_block * block_index + concurrent_vector_index * UNPACK_N_VECTORS;
	}
};

template <typename T, unsigned UNPACK_N_VECTORS>
struct FillWarpMapping {
	using Policy = lane_policy::ValueLanePolicy<T>;

	static constexpr uint32_t N_LANES          = Policy::semantic_lanes;
	static constexpr uint32_t N_VALUES_IN_LANE = Policy::values_per_lane;

	__device__ __forceinline__ lane_t get_lane() const {
		return threadIdx.x % N_LANES;
	}

	__device__ __forceinline__ vi_t get_vector_index() const {
		// Concurrent vectors per block: how many vectors can be processed
		// by the block simultaneously, assuming that each thread is 1 lane

		const int32_t concurrent_vectors_per_block = blockDim.x / N_LANES;
		const int32_t vectors_per_block            = concurrent_vectors_per_block * UNPACK_N_VECTORS;

		const int32_t concurrent_vector_index = threadIdx.x / N_LANES;
		const int32_t block_index             = blockIdx.x;

		return vectors_per_block * block_index + concurrent_vector_index * UNPACK_N_VECTORS;
	}
};

#ifdef SingleVectorMapping
template <typename T, unsigned UNPACK_N_VECTORS>
using VectorToWarpMapping = SingleVectorPerWarpMapping<T, UNPACK_N_VECTORS>;
#else
template <typename T, unsigned UNPACK_N_VECTORS>
using VectorToWarpMapping = FillWarpMapping<T, UNPACK_N_VECTORS>;
#endif

struct MixedSlotMapping {
	static constexpr uint32_t SLOT_LANES          = static_cast<uint32_t>(utils::get_n_lanes<int8_t>());
	static constexpr uint32_t HALF_SLOT_LANES     = static_cast<uint32_t>(utils::get_n_lanes<int16_t>());
	static constexpr uint32_t N_THREADS_PER_BLOCK = 128;

	size_t n_slots = 0;

	__host__ __device__ explicit constexpr MixedSlotMapping(const size_t slot_count = 0)
	    : n_slots(slot_count) {
	}

	__host__ __device__ constexpr size_t n_threads() const {
		return n_slots * static_cast<size_t>(SLOT_LANES);
	}

	__host__ __device__ constexpr unsigned n_blocks() const {
		return static_cast<unsigned>((n_threads() + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK);
	}

	__device__ __forceinline__ uint32_t global_thread() const {
		return blockIdx.x * blockDim.x + threadIdx.x;
	}

	__device__ __forceinline__ uint32_t slot_index() const {
		return global_thread() / SLOT_LANES;
	}

	__device__ __forceinline__ uint32_t slot_lane() const {
		return global_thread() % SLOT_LANES;
	}

	__host__ __device__ constexpr bool is_first_half(const uint32_t lane) const {
		return lane < HALF_SLOT_LANES;
	}
};

template <typename T,
          unsigned UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES,
          unsigned N_LANES,
          typename UntransposerT = flsgpu::device::IdentityUntransposer>
__device__ __forceinline__ void write_registers_to_global(const lane_t lane,
                                                          const si_t   index_offset,
                                                          const T* __restrict registers,
                                                          T* __restrict out) {
	for (int v {0}; v < UNPACK_N_VECTORS; ++v) {
		for (int w {0}; w < UNPACK_N_VALUES; ++w) {
			const uint32_t in_idx = static_cast<uint32_t>(lane) + static_cast<uint32_t>(index_offset + w) * N_LANES;
			out[v * consts::VALUES_PER_VECTOR + UntransposerT::map_index(in_idx)] = registers[w + v * UNPACK_N_VALUES];
		}
	}
}

template <typename T, unsigned N_VALUES>
struct MagicChecker {
	const T magic_value;
	bool    no_magic_found = true;

	__device__ __forceinline__ MagicChecker(const T magic_value)
	    : magic_value(magic_value) {
	}

	__device__ __forceinline__ void check(const T* __restrict registers) {
#pragma unroll
		for (int i = 0; i < N_VALUES; ++i) {
			no_magic_found &= registers[i] != magic_value;
		}
	}

	__device__ __forceinline__ void write_result(bool* __restrict out) {
		// This is a branch, as we do not want to write 0s, only emit a write
		// if we found a magic value
		if (!no_magic_found) {
			*out = true;
		}
	}
};

#endif // GPU_DEVICE_UTILS_CUH
