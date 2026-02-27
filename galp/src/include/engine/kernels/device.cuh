// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/kernels/device.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_KERNELS_DEVICE_CUH
#define ENGINE_KERNELS_DEVICE_CUH

#include "engine/device-utils.cuh"
#include "engine/kernels/execute_plan.cuh"
#include "engine/kernels/traits.cuh"
#include "flsgpu/consts.cuh"
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace kernels {
namespace device {

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES>
__global__ void decompress_rowgroup(const dispatch::DeviceExpression<T>* exprs,
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

    const size_t n_vecs = utils::get_n_vecs_from_size(expr.n_values);
    if (static_cast<size_t>(vector_index) >= n_vecs) {
        return;
    }

    T* out = expr.out + vector_index * consts::VALUES_PER_VECTOR;

    device_exec::execute_plan<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>(expr, vector_index, lane, out);
}

template <int UNPACK_N_VECTORS, int UNPACK_N_VALUES>
__global__ void decompress_table(const dispatch::DeviceExpression<int8_t>*  exprs_i8,
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

    T registers[N_VALUES];
    auto iterator = DecompressorT(column, vector_index, lane);

	// uint32_t acc = 2166136261u;
    for (si_t i = 0; i < mapping.N_VALUES_IN_LANE; i += UNPACK_N_VALUES) {
        iterator.unpack_next_into(registers);

        // #pragma unroll
		// for (int k = 0; k < N_VALUES; ++k) {
		// 	acc ^= (uint32_t)registers[k];
		// }
        write_registers_to_global<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, mapping.N_LANES>(lane, i, registers, out);
    }
    // if ( (acc & 0xFFFFFFFF) == 0 ) {
    //     out[0] = (T)acc;
    // }
}

template <typename T, int UNPACK_N_VECTORS, int UNPACK_N_VALUES, typename DecompressorT, typename ColumnT>
__global__ void query_column(const ColumnT column, bool* out, const T magic_value) {
    constexpr uint32_t N_VALUES     = UNPACK_N_VALUES * UNPACK_N_VECTORS;
    const auto         mapping      = VectorToWarpMapping<T, UNPACK_N_VECTORS>();
    const lane_t       lane         = mapping.get_lane();
    const int32_t      vector_index = mapping.get_vector_index();
    T    registers[N_VALUES];
    auto checker = MagicChecker<T, N_VALUES>(magic_value);

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
                registers[j] >>= RANDOM_VALUE;
                registers[j] &= runtime_zero;
            }
        }

        checker.check(registers);
    }

    checker.write_result(out);
}

} // namespace device
} // namespace kernels

#endif // ENGINE_KERNELS_DEVICE_CUH
