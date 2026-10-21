// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// fastlens/src/kernels/rebind.cuh
// ────────────────────────────────────────────────────────
#ifndef FASTLENS_ENGINE_KERNELS_REBIND_CUH
#define FASTLENS_ENGINE_KERNELS_REBIND_CUH

#include "codecs/decode/alp.cuh"
#include <cstddef>

namespace fastlens::kernels::detail {

template <typename T, unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors {
	using type = T;
};

template <typename T, unsigned OLD_UNPACK_N_VECTORS, unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<fastlens::codec::device::FFORFunctor<T, OLD_UNPACK_N_VECTORS>, NEW_UNPACK_N_VECTORS> {
	using type = fastlens::codec::device::FFORFunctor<T, NEW_UNPACK_N_VECTORS>;
};

template <typename T, unsigned OLD_UNPACK_N_VECTORS, typename IndexT, unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<fastlens::codec::device::DICTFunctor<T, OLD_UNPACK_N_VECTORS, IndexT>, NEW_UNPACK_N_VECTORS> {
	using type = fastlens::codec::device::DICTFunctor<T, NEW_UNPACK_N_VECTORS, IndexT>;
};

template <typename T, unsigned OLD_UNPACK_N_VECTORS, unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<fastlens::codec::device::ALPFunctor<T, OLD_UNPACK_N_VECTORS>, NEW_UNPACK_N_VECTORS> {
	using type = fastlens::codec::device::ALPFunctor<T, NEW_UNPACK_N_VECTORS>;
};

template <typename T,
          unsigned OLD_UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES,
          typename OutputProcessor,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<
    fastlens::codec::device::BitUnpackerDummy<T, OLD_UNPACK_N_VECTORS, UNPACK_N_VALUES, OutputProcessor>,
    NEW_UNPACK_N_VECTORS> {
	using type = fastlens::codec::device::BitUnpackerDummy<
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
    fastlens::codec::device::BitUnpackerOldFls<T, OLD_UNPACK_N_VECTORS, UNPACK_N_VALUES, OutputProcessor>,
    NEW_UNPACK_N_VECTORS> {
	using type = fastlens::codec::device::BitUnpackerOldFls<
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
    fastlens::codec::device::
        BitUnpackerStatefulBranchless<OutT, OLD_UNPACK_N_VECTORS, UNPACK_N_VALUES, OutputProcessor, InT>,
    NEW_UNPACK_N_VECTORS> {
	using type = fastlens::codec::device::BitUnpackerStatefulBranchless<
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
    fastlens::codec::device::BitUnpackerLaneTile<
        OutT, OLD_UNPACK_N_VECTORS, UNPACK_N_VALUES, OutputProcessor, InT>,
    NEW_UNPACK_N_VECTORS> {
	using type = fastlens::codec::device::BitUnpackerLaneTile<
	    OutT,
	    NEW_UNPACK_N_VECTORS,
	    UNPACK_N_VALUES,
	    typename RebindUnpackVectors<OutputProcessor, NEW_UNPACK_N_VECTORS>::type,
	    InT>;
};

#define FASTLENS_REBIND_T_UV(TypeName)                                                                                     \
	template <typename T, unsigned OLD_UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES, unsigned NEW_UNPACK_N_VECTORS>      \
	struct RebindUnpackVectors<fastlens::codec::device::TypeName<T, OLD_UNPACK_N_VECTORS, UNPACK_N_VALUES>,                \
	                           NEW_UNPACK_N_VECTORS> {                                                                 \
		using type = fastlens::codec::device::TypeName<T, NEW_UNPACK_N_VECTORS, UNPACK_N_VALUES>;                          \
	}

FASTLENS_REBIND_T_UV(DummyFREQExceptionPatcher);
FASTLENS_REBIND_T_UV(StatelessFREQExceptionPatcher);
FASTLENS_REBIND_T_UV(StatefulFREQExceptionPatcher);
FASTLENS_REBIND_T_UV(NaiveFREQExceptionPatcher);
FASTLENS_REBIND_T_UV(NaiveBranchlessFREQExceptionPatcher);
FASTLENS_REBIND_T_UV(PrefetchPositionFREQExceptionPatcher);
FASTLENS_REBIND_T_UV(PrefetchAllFREQExceptionPatcher);
FASTLENS_REBIND_T_UV(PrefetchAllBranchlessFREQExceptionPatcher);
FASTLENS_REBIND_T_UV(DummyALPExceptionPatcher);
FASTLENS_REBIND_T_UV(StatelessALPExceptionPatcher);
FASTLENS_REBIND_T_UV(StatefulALPExceptionPatcher);
FASTLENS_REBIND_T_UV(NaiveALPExceptionPatcher);
FASTLENS_REBIND_T_UV(NaiveBranchlessALPExceptionPatcher);
FASTLENS_REBIND_T_UV(PrefetchPositionALPExceptionPatcher);
FASTLENS_REBIND_T_UV(PrefetchAllALPExceptionPatcher);
FASTLENS_REBIND_T_UV(PrefetchAllBranchlessALPExceptionPatcher);
FASTLENS_REBIND_T_UV(StatefulSLPATCHExceptionPatcher);
FASTLENS_REBIND_T_UV(StatelessSLPATCHExceptionPatcher);
FASTLENS_REBIND_T_UV(DummyCROSSRLEExpander);
FASTLENS_REBIND_T_UV(StatefulCROSSRLEExpander);
FASTLENS_REBIND_T_UV(StatefulCacheCROSSRLEExpander);
FASTLENS_REBIND_T_UV(PrefetchStatefulCROSSRLEExpander);
FASTLENS_REBIND_T_UV(StatefulAdvanceCROSSRLEExpander);
FASTLENS_REBIND_T_UV(StatefulShuffleCROSSRLEExpander);
FASTLENS_REBIND_T_UV(StatefulExtendedCROSSRLEExpander);
FASTLENS_REBIND_T_UV(BranchlessCROSSRLEExpander);
FASTLENS_REBIND_T_UV(PrefetchBranchlessCROSSRLEExpander);

#undef FASTLENS_REBIND_T_UV

template <typename T,
          typename IndexT,
          unsigned OLD_UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<
    fastlens::codec::device::StatefulSLPATCHDictExceptionPatcher<T, IndexT, OLD_UNPACK_N_VECTORS, UNPACK_N_VALUES>,
    NEW_UNPACK_N_VECTORS> {
	using type =
	    fastlens::codec::device::StatefulSLPATCHDictExceptionPatcher<T, IndexT, NEW_UNPACK_N_VECTORS, UNPACK_N_VALUES>;
};

template <typename T,
          typename IndexT,
          unsigned OLD_UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<
    fastlens::codec::device::StatelessSLPATCHDictExceptionPatcher<T, IndexT, OLD_UNPACK_N_VECTORS, UNPACK_N_VALUES>,
    NEW_UNPACK_N_VECTORS> {
	using type =
	    fastlens::codec::device::StatelessSLPATCHDictExceptionPatcher<T, IndexT, NEW_UNPACK_N_VECTORS, UNPACK_N_VALUES>;
};

template <typename ValueT,
          typename IndexT,
          unsigned OLD_UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<fastlens::codec::device::DummyRLEExpander<ValueT, IndexT, OLD_UNPACK_N_VECTORS, UNPACK_N_VALUES>,
                           NEW_UNPACK_N_VECTORS> {
	using type = fastlens::codec::device::DummyRLEExpander<ValueT, IndexT, NEW_UNPACK_N_VECTORS, UNPACK_N_VALUES>;
};

template <typename T,
          unsigned OLD_UNPACK_N_VECTORS,
          typename UnpackerT,
          typename ColumnT,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<fastlens::codec::device::BPDecompressor<T, OLD_UNPACK_N_VECTORS, UnpackerT, ColumnT>,
                           NEW_UNPACK_N_VECTORS> {
	using type =
	    fastlens::codec::device::BPDecompressor<T,
	                                        NEW_UNPACK_N_VECTORS,
	                                        typename RebindUnpackVectors<UnpackerT, NEW_UNPACK_N_VECTORS>::type,
	                                        ColumnT>;
};

template <typename T,
          unsigned OLD_UNPACK_N_VECTORS,
          typename UnpackerT,
          typename ColumnT,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<fastlens::codec::device::FFORDecompressor<T, OLD_UNPACK_N_VECTORS, UnpackerT, ColumnT>,
                           NEW_UNPACK_N_VECTORS> {
	using type =
	    fastlens::codec::device::FFORDecompressor<T,
	                                          NEW_UNPACK_N_VECTORS,
	                                          typename RebindUnpackVectors<UnpackerT, NEW_UNPACK_N_VECTORS>::type,
	                                          ColumnT>;
};

template <typename T,
          unsigned OLD_UNPACK_N_VECTORS,
          typename UnpackerT,
          typename ColumnT,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<fastlens::codec::device::DELTADecompressor<T, OLD_UNPACK_N_VECTORS, UnpackerT, ColumnT>,
                           NEW_UNPACK_N_VECTORS> {
	using type =
	    fastlens::codec::device::DELTADecompressor<T,
	                                           NEW_UNPACK_N_VECTORS,
	                                           typename RebindUnpackVectors<UnpackerT, NEW_UNPACK_N_VECTORS>::type,
	                                           ColumnT>;
};

template <typename T,
          unsigned OLD_UNPACK_N_VECTORS,
          typename UnpackerT,
          typename ColumnT,
          unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<fastlens::codec::device::DELTARegisterDecompressor<T, OLD_UNPACK_N_VECTORS, UnpackerT, ColumnT>,
                           NEW_UNPACK_N_VECTORS> {
	using type = fastlens::codec::device::DELTARegisterDecompressor<
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
struct RebindUnpackVectors<fastlens::codec::device::CONSTANTDecompressor<T, OLD_UNPACK_N_VECTORS, UNPACK_N_VALUES, ColumnT>,
                           NEW_UNPACK_N_VECTORS> {
	using type = fastlens::codec::device::CONSTANTDecompressor<T, NEW_UNPACK_N_VECTORS, UNPACK_N_VALUES, ColumnT>;
};

template <typename T, unsigned OLD_UNPACK_N_VECTORS, typename PatcherT, typename ColumnT, unsigned NEW_UNPACK_N_VECTORS>
struct RebindUnpackVectors<fastlens::codec::device::FREQDecompressor<T, OLD_UNPACK_N_VECTORS, PatcherT, ColumnT>,
                           NEW_UNPACK_N_VECTORS> {
	using type =
	    fastlens::codec::device::FREQDecompressor<T,
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
    fastlens::codec::device::SLPATCHDecompressor<T, OLD_UNPACK_N_VECTORS, UnpackerT, PatcherT, ColumnT>,
    NEW_UNPACK_N_VECTORS> {
	using type =
	    fastlens::codec::device::SLPATCHDecompressor<T,
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
    fastlens::codec::device::
        DICTSLPATCHDecompressor<T, OLD_UNPACK_N_VECTORS, UNPACK_N_VALUES, UnpackerT, PatcherT, ColumnT, ProcessorT>,
    NEW_UNPACK_N_VECTORS> {
	using type = fastlens::codec::device::DICTSLPATCHDecompressor<
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
    fastlens::codec::device::DICTDecompressor<T, OLD_UNPACK_N_VECTORS, UnpackerT, ColumnT, ProcessorT>,
    NEW_UNPACK_N_VECTORS> {
	using type =
	    fastlens::codec::device::DICTDecompressor<T,
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
struct RebindUnpackVectors<fastlens::codec::device::CROSSRLEDecompressor<T, OLD_UNPACK_N_VECTORS, ExpanderT, ColumnT>,
                           NEW_UNPACK_N_VECTORS> {
	using type =
	    fastlens::codec::device::CROSSRLEDecompressor<T,
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
    fastlens::codec::device::
        RLEDecompressor<ValueT, IndexT, OLD_UNPACK_N_VECTORS, UNPACK_N_VALUES, UnpackerT, ExpanderT, ColumnT>,
    NEW_UNPACK_N_VECTORS> {
	using type =
	    fastlens::codec::device::RLEDecompressor<ValueT,
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
struct RebindUnpackVectors<fastlens::codec::device::RLESLPATCHDecompressor<ValueT,
                                                                       IndexT,
                                                                       OLD_UNPACK_N_VECTORS,
                                                                       UNPACK_N_VALUES,
                                                                       UnpackerT,
                                                                       PatcherT,
                                                                       ExpanderT,
                                                                       ColumnT>,
                           NEW_UNPACK_N_VECTORS> {
	using type =
	    fastlens::codec::device::RLESLPATCHDecompressor<ValueT,
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
struct RebindUnpackVectors<fastlens::codec::device::ALPDecompressor<T, OLD_UNPACK_N_VECTORS, UnpackerT, PatcherT, ColumnT>,
                           NEW_UNPACK_N_VECTORS> {
	using type =
	    fastlens::codec::device::ALPDecompressor<T,
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

} // namespace fastlens::kernels::detail

#endif // FASTLENS_ENGINE_KERNELS_REBIND_CUH
