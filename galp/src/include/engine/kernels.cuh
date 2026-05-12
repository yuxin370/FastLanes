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
#include "flsgpu/flsgpu.cuh"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <type_traits>

namespace galp::kernels {
namespace device {

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

} // namespace device

namespace detail {

template <typename T, unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors {
	using type = T;
};

template <typename T, unsigned OLD_UNPACK_N_VECTORS, unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<galp::codec::device::FFORFunctor<T, OLD_UNPACK_N_VECTORS>, NEW_UNPACK_N_VECTORS> {
	using type = galp::codec::device::FFORFunctor<T, NEW_UNPACK_N_VECTORS>;
};

template <typename T, unsigned OLD_UNPACK_N_VECTORS, typename IndexT, unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<galp::codec::device::DICTFunctor<T, OLD_UNPACK_N_VECTORS, IndexT>, NEW_UNPACK_N_VECTORS> {
	using type = galp::codec::device::DICTFunctor<T, NEW_UNPACK_N_VECTORS, IndexT>;
};

template <typename T, unsigned OLD_UNPACK_N_VECTORS, unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<galp::codec::device::ALPFunctor<T, OLD_UNPACK_N_VECTORS>, NEW_UNPACK_N_VECTORS> {
	using type = galp::codec::device::ALPFunctor<T, NEW_UNPACK_N_VECTORS>;
};

template <typename T,
          unsigned OLD_UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES,
          typename OutputProcessor,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<
    galp::codec::device::BitUnpackerDummy<T, OLD_UNPACK_N_VECTORS, UNPACK_N_VALUES, OutputProcessor>,
    NEW_UNPACK_N_VECTORS> {
	using type = galp::codec::device::BitUnpackerDummy<
	    T,
	    NEW_UNPACK_N_VECTORS,
	    UNPACK_N_VALUES,
	    typename RebindUnpackVectors<OutputProcessor, NEW_UNPACK_N_VECTORS>::type>;
};

template <typename T,
          unsigned OLD_UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES,
          typename OutputProcessor,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<
    galp::codec::device::BitUnpackerOldFls<T, OLD_UNPACK_N_VECTORS, UNPACK_N_VALUES, OutputProcessor>,
    NEW_UNPACK_N_VECTORS> {
	using type = galp::codec::device::BitUnpackerOldFls<
	    T,
	    NEW_UNPACK_N_VECTORS,
	    UNPACK_N_VALUES,
	    typename RebindUnpackVectors<OutputProcessor, NEW_UNPACK_N_VECTORS>::type>;
};

template <typename OutT,
          unsigned OLD_UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES,
          typename OutputProcessor,
          typename InT,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<
    galp::codec::device::
        BitUnpackerStatefulBranchless<OutT, OLD_UNPACK_N_VECTORS, UNPACK_N_VALUES, OutputProcessor, InT>,
    NEW_UNPACK_N_VECTORS> {
	using type = galp::codec::device::BitUnpackerStatefulBranchless<
	    OutT,
	    NEW_UNPACK_N_VECTORS,
	    UNPACK_N_VALUES,
	    typename RebindUnpackVectors<OutputProcessor, NEW_UNPACK_N_VECTORS>::type,
	    InT>;
};

#define GALP_REBIND_T_UV(TypeName)                                                                                     \
	template <typename T, unsigned OLD_UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES, unsigned NEW_UNPACK_N_VECTORS>      \
	struct RebindUnpackVectors<galp::codec::device::TypeName<T, OLD_UNPACK_N_VECTORS, UNPACK_N_VALUES>,                \
	                           NEW_UNPACK_N_VECTORS> {                                                                 \
		using type = galp::codec::device::TypeName<T, NEW_UNPACK_N_VECTORS, UNPACK_N_VALUES>;                          \
	}

GALP_REBIND_T_UV(DummyFREQExceptionPatcher);
GALP_REBIND_T_UV(StatelessFREQExceptionPatcher);
GALP_REBIND_T_UV(StatefulFREQExceptionPatcher);
GALP_REBIND_T_UV(NaiveFREQExceptionPatcher);
GALP_REBIND_T_UV(NaiveBranchlessFREQExceptionPatcher);
GALP_REBIND_T_UV(PrefetchPositionFREQExceptionPatcher);
GALP_REBIND_T_UV(PrefetchAllFREQExceptionPatcher);
GALP_REBIND_T_UV(PrefetchAllBranchlessFREQExceptionPatcher);
GALP_REBIND_T_UV(DummyALPExceptionPatcher);
GALP_REBIND_T_UV(StatelessALPExceptionPatcher);
GALP_REBIND_T_UV(StatefulALPExceptionPatcher);
GALP_REBIND_T_UV(NaiveALPExceptionPatcher);
GALP_REBIND_T_UV(NaiveBranchlessALPExceptionPatcher);
GALP_REBIND_T_UV(PrefetchPositionALPExceptionPatcher);
GALP_REBIND_T_UV(PrefetchAllALPExceptionPatcher);
GALP_REBIND_T_UV(PrefetchAllBranchlessALPExceptionPatcher);
GALP_REBIND_T_UV(StatefulSLPATCHExceptionPatcher);
GALP_REBIND_T_UV(StatelessSLPATCHExceptionPatcher);
GALP_REBIND_T_UV(DummyCROSSRLEExpander);
GALP_REBIND_T_UV(StatefulCROSSRLEExpander);
GALP_REBIND_T_UV(StatefulCacheCROSSRLEExpander);
GALP_REBIND_T_UV(PrefetchStatefulCROSSRLEExpander);
GALP_REBIND_T_UV(StatefulAdvanceCROSSRLEExpander);
GALP_REBIND_T_UV(StatefulShuffleCROSSRLEExpander);
GALP_REBIND_T_UV(StatefulExtendedCROSSRLEExpander);
GALP_REBIND_T_UV(BranchlessCROSSRLEExpander);
GALP_REBIND_T_UV(PrefetchBranchlessCROSSRLEExpander);

#undef GALP_REBIND_T_UV

template <typename T,
          typename IndexT,
          unsigned OLD_UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<
    galp::codec::device::StatefulSLPATCHDictExceptionPatcher<T, IndexT, OLD_UNPACK_N_VECTORS, UNPACK_N_VALUES>,
    NEW_UNPACK_N_VECTORS> {
	using type =
	    galp::codec::device::StatefulSLPATCHDictExceptionPatcher<T, IndexT, NEW_UNPACK_N_VECTORS, UNPACK_N_VALUES>;
};

template <typename T,
          typename IndexT,
          unsigned OLD_UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<
    galp::codec::device::StatelessSLPATCHDictExceptionPatcher<T, IndexT, OLD_UNPACK_N_VECTORS, UNPACK_N_VALUES>,
    NEW_UNPACK_N_VECTORS> {
	using type =
	    galp::codec::device::StatelessSLPATCHDictExceptionPatcher<T, IndexT, NEW_UNPACK_N_VECTORS, UNPACK_N_VALUES>;
};

template <typename ValueT,
          typename IndexT,
          unsigned OLD_UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<galp::codec::device::DummyRLEExpander<ValueT, IndexT, OLD_UNPACK_N_VECTORS, UNPACK_N_VALUES>,
                           NEW_UNPACK_N_VECTORS> {
	using type = galp::codec::device::DummyRLEExpander<ValueT, IndexT, NEW_UNPACK_N_VECTORS, UNPACK_N_VALUES>;
};

template <typename T,
          unsigned OLD_UNPACK_N_VECTORS,
          typename UnpackerT,
          typename ColumnT,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<galp::codec::device::BPDecompressor<T, OLD_UNPACK_N_VECTORS, UnpackerT, ColumnT>,
                           NEW_UNPACK_N_VECTORS> {
	using type =
	    galp::codec::device::BPDecompressor<T,
	                                        NEW_UNPACK_N_VECTORS,
	                                        typename RebindUnpackVectors<UnpackerT, NEW_UNPACK_N_VECTORS>::type,
	                                        ColumnT>;
};

template <typename T,
          unsigned OLD_UNPACK_N_VECTORS,
          typename UnpackerT,
          typename ColumnT,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<galp::codec::device::FFORDecompressor<T, OLD_UNPACK_N_VECTORS, UnpackerT, ColumnT>,
                           NEW_UNPACK_N_VECTORS> {
	using type =
	    galp::codec::device::FFORDecompressor<T,
	                                          NEW_UNPACK_N_VECTORS,
	                                          typename RebindUnpackVectors<UnpackerT, NEW_UNPACK_N_VECTORS>::type,
	                                          ColumnT>;
};

template <typename T,
          unsigned OLD_UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES,
          typename ColumnT,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<galp::codec::device::CONSTANTDecompressor<T, OLD_UNPACK_N_VECTORS, UNPACK_N_VALUES, ColumnT>,
                           NEW_UNPACK_N_VECTORS> {
	using type = galp::codec::device::CONSTANTDecompressor<T, NEW_UNPACK_N_VECTORS, UNPACK_N_VALUES, ColumnT>;
};

template <typename T, unsigned OLD_UNPACK_N_VECTORS, typename PatcherT, typename ColumnT, unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<galp::codec::device::FREQDecompressor<T, OLD_UNPACK_N_VECTORS, PatcherT, ColumnT>,
                           NEW_UNPACK_N_VECTORS> {
	using type =
	    galp::codec::device::FREQDecompressor<T,
	                                          NEW_UNPACK_N_VECTORS,
	                                          typename RebindUnpackVectors<PatcherT, NEW_UNPACK_N_VECTORS>::type,
	                                          ColumnT>;
};

template <typename T,
          unsigned OLD_UNPACK_N_VECTORS,
          typename UnpackerT,
          typename PatcherT,
          typename ColumnT,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<
    galp::codec::device::SLPATCHDecompressor<T, OLD_UNPACK_N_VECTORS, UnpackerT, PatcherT, ColumnT>,
    NEW_UNPACK_N_VECTORS> {
	using type =
	    galp::codec::device::SLPATCHDecompressor<T,
	                                             NEW_UNPACK_N_VECTORS,
	                                             typename RebindUnpackVectors<UnpackerT, NEW_UNPACK_N_VECTORS>::type,
	                                             typename RebindUnpackVectors<PatcherT, NEW_UNPACK_N_VECTORS>::type,
	                                             ColumnT>;
};

template <typename T,
          unsigned OLD_UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES,
          typename UnpackerT,
          typename PatcherT,
          typename ColumnT,
          typename ProcessorT,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<
    galp::codec::device::
        DICTSLPATCHDecompressor<T, OLD_UNPACK_N_VECTORS, UNPACK_N_VALUES, UnpackerT, PatcherT, ColumnT, ProcessorT>,
    NEW_UNPACK_N_VECTORS> {
	using type = galp::codec::device::DICTSLPATCHDecompressor<
	    T,
	    NEW_UNPACK_N_VECTORS,
	    UNPACK_N_VALUES,
	    typename RebindUnpackVectors<UnpackerT, NEW_UNPACK_N_VECTORS>::type,
	    typename RebindUnpackVectors<PatcherT, NEW_UNPACK_N_VECTORS>::type,
	    ColumnT,
	    typename RebindUnpackVectors<ProcessorT, NEW_UNPACK_N_VECTORS>::type>;
};

template <typename T,
          unsigned OLD_UNPACK_N_VECTORS,
          typename UnpackerT,
          typename ColumnT,
          typename ProcessorT,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<
    galp::codec::device::DICTDecompressor<T, OLD_UNPACK_N_VECTORS, UnpackerT, ColumnT, ProcessorT>,
    NEW_UNPACK_N_VECTORS> {
	using type =
	    galp::codec::device::DICTDecompressor<T,
	                                          NEW_UNPACK_N_VECTORS,
	                                          typename RebindUnpackVectors<UnpackerT, NEW_UNPACK_N_VECTORS>::type,
	                                          ColumnT,
	                                          typename RebindUnpackVectors<ProcessorT, NEW_UNPACK_N_VECTORS>::type>;
};

template <typename T,
          unsigned OLD_UNPACK_N_VECTORS,
          typename ExpanderT,
          typename ColumnT,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<galp::codec::device::CROSSRLEDecompressor<T, OLD_UNPACK_N_VECTORS, ExpanderT, ColumnT>,
                           NEW_UNPACK_N_VECTORS> {
	using type =
	    galp::codec::device::CROSSRLEDecompressor<T,
	                                              NEW_UNPACK_N_VECTORS,
	                                              typename RebindUnpackVectors<ExpanderT, NEW_UNPACK_N_VECTORS>::type,
	                                              ColumnT>;
};

template <typename ValueT,
          typename IndexT,
          unsigned OLD_UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES,
          typename UnpackerT,
          typename ExpanderT,
          typename ColumnT,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<
    galp::codec::device::
        RLEDecompressor<ValueT, IndexT, OLD_UNPACK_N_VECTORS, UNPACK_N_VALUES, UnpackerT, ExpanderT, ColumnT>,
    NEW_UNPACK_N_VECTORS> {
	using type =
	    galp::codec::device::RLEDecompressor<ValueT,
	                                         IndexT,
	                                         NEW_UNPACK_N_VECTORS,
	                                         UNPACK_N_VALUES,
	                                         typename RebindUnpackVectors<UnpackerT, NEW_UNPACK_N_VECTORS>::type,
	                                         typename RebindUnpackVectors<ExpanderT, NEW_UNPACK_N_VECTORS>::type,
	                                         ColumnT>;
};

template <typename T,
          unsigned OLD_UNPACK_N_VECTORS,
          typename UnpackerT,
          typename PatcherT,
          typename ColumnT,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<galp::codec::device::ALPDecompressor<T, OLD_UNPACK_N_VECTORS, UnpackerT, PatcherT, ColumnT>,
                           NEW_UNPACK_N_VECTORS> {
	using type =
	    galp::codec::device::ALPDecompressor<T,
	                                         NEW_UNPACK_N_VECTORS,
	                                         typename RebindUnpackVectors<UnpackerT, NEW_UNPACK_N_VECTORS>::type,
	                                         typename RebindUnpackVectors<PatcherT, NEW_UNPACK_N_VECTORS>::type,
	                                         ColumnT>;
};

template <typename DecompressorT>
using ScalarTailDecompressorT = typename RebindUnpackVectors<DecompressorT, 1>::type;

inline size_t full_vector_count(const size_t n_vecs, const size_t unpack_n_vectors) {
	return unpack_n_vectors <= 1 ? n_vecs : (n_vecs / unpack_n_vectors) * unpack_n_vectors;
}

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

} // namespace detail

namespace host {

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

} // namespace host
} // namespace galp::kernels

#endif // FLS_GLOBAL_CUH
