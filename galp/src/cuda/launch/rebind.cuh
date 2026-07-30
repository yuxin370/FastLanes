// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/kernels/rebind.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_ENGINE_KERNELS_REBIND_CUH
#define GALP_ENGINE_KERNELS_REBIND_CUH

#include "codecs/decode/alp.cuh"
#include <cstddef>

namespace galp::kernels::detail {

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

template <typename OutT,
          unsigned OLD_UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES,
          typename OutputProcessor,
          typename InT,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<
    galp::codec::device::BitUnpackerLaneTile<
        OutT, OLD_UNPACK_N_VECTORS, UNPACK_N_VALUES, OutputProcessor, InT>,
    NEW_UNPACK_N_VECTORS> {
	using type = galp::codec::device::BitUnpackerLaneTile<
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
          typename UnpackerT,
          typename ColumnT,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<galp::codec::device::DELTADecompressor<T, OLD_UNPACK_N_VECTORS, UnpackerT, ColumnT>,
                           NEW_UNPACK_N_VECTORS> {
	using type =
	    galp::codec::device::DELTADecompressor<T,
	                                           NEW_UNPACK_N_VECTORS,
	                                           typename RebindUnpackVectors<UnpackerT, NEW_UNPACK_N_VECTORS>::type,
	                                           ColumnT>;
};

template <typename T,
          unsigned OLD_UNPACK_N_VECTORS,
          typename UnpackerT,
          typename ColumnT,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<galp::codec::device::DELTARegisterDecompressor<T, OLD_UNPACK_N_VECTORS, UnpackerT, ColumnT>,
                           NEW_UNPACK_N_VECTORS> {
	using type = galp::codec::device::DELTARegisterDecompressor<
	    T,
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

template <typename ValueT,
          typename IndexT,
          unsigned OLD_UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES,
          typename UnpackerT,
          typename PatcherT,
          typename ExpanderT,
          typename ColumnT,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<galp::codec::device::RLESLPATCHDecompressor<ValueT,
                                                                       IndexT,
                                                                       OLD_UNPACK_N_VECTORS,
                                                                       UNPACK_N_VALUES,
                                                                       UnpackerT,
                                                                       PatcherT,
                                                                       ExpanderT,
                                                                       ColumnT>,
                           NEW_UNPACK_N_VECTORS> {
	using type =
	    galp::codec::device::RLESLPATCHDecompressor<ValueT,
	                                                IndexT,
	                                                NEW_UNPACK_N_VECTORS,
	                                                UNPACK_N_VALUES,
	                                                typename RebindUnpackVectors<UnpackerT, NEW_UNPACK_N_VECTORS>::type,
	                                                typename RebindUnpackVectors<PatcherT, NEW_UNPACK_N_VECTORS>::type,
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

} // namespace galp::kernels::detail

#endif // GALP_ENGINE_KERNELS_REBIND_CUH
