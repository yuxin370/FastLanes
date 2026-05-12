// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/benchmark/generated-bindings/alpextended-float-query_multi_column-bindings.cu
// ────────────────────────────────────────────────────────
#include "engine/kernels.cuh"
#include "engine/multi-column-host-kernels.cuh"
#include "generated-bindings/kernel-bindings.cuh"
#include <stdexcept>

namespace galp::bench::bindings {

template <>
bool query_multi_column<float, galp::codec::host::ALPExtendedColumn<float>>(
    const galp::codec::host::ALPExtendedColumn<float>& column,
    const unsigned                               unpack_n_vectors,
    const unsigned                               unpack_n_values,
    const galp::format::Unpacker                        unpacker,
    const galp::format::Patcher                         patcher,
    const float                                  magic_value,
    const uint32_t                               n_samples) {
	if (unpack_n_vectors == 1 && unpack_n_values == 32 && unpacker == galp::format::Unpacker::OldFls &&
	    patcher == galp::format::Patcher::Naive) {
		return galp::kernels::host::query_multi_column<
		    float,
		    1,
		    32,
		    galp::codec::device::ALPDecompressor<
		        float,
		        1,
		        galp::codec::device::BitUnpackerOldFls<float, 1, 32, galp::codec::device::ALPFunctor<float, 1>>,
		        galp::codec::device::NaiveALPExceptionPatcher<float, 1, 32>,
		        galp::codec::device::ALPExtendedColumn<float>>,
		    galp::codec::host::ALPExtendedColumn<float>>(column, magic_value, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 32 && unpacker == galp::format::Unpacker::OldFls &&
	    patcher == galp::format::Patcher::NaiveBranchless) {
		return galp::kernels::host::query_multi_column<
		    float,
		    1,
		    32,
		    galp::codec::device::ALPDecompressor<
		        float,
		        1,
		        galp::codec::device::BitUnpackerOldFls<float, 1, 32, galp::codec::device::ALPFunctor<float, 1>>,
		        galp::codec::device::NaiveBranchlessALPExceptionPatcher<float, 1, 32>,
		        galp::codec::device::ALPExtendedColumn<float>>,
		    galp::codec::host::ALPExtendedColumn<float>>(column, magic_value, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 32 && unpacker == galp::format::Unpacker::OldFls &&
	    patcher == galp::format::Patcher::PrefetchAll) {
		return galp::kernels::host::query_multi_column<
		    float,
		    1,
		    32,
		    galp::codec::device::ALPDecompressor<
		        float,
		        1,
		        galp::codec::device::BitUnpackerOldFls<float, 1, 32, galp::codec::device::ALPFunctor<float, 1>>,
		        galp::codec::device::PrefetchAllALPExceptionPatcher<float, 1, 32>,
		        galp::codec::device::ALPExtendedColumn<float>>,
		    galp::codec::host::ALPExtendedColumn<float>>(column, magic_value, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 32 && unpacker == galp::format::Unpacker::OldFls &&
	    patcher == galp::format::Patcher::PrefetchAllBranchless) {
		return galp::kernels::host::query_multi_column<
		    float,
		    1,
		    32,
		    galp::codec::device::ALPDecompressor<
		        float,
		        1,
		        galp::codec::device::BitUnpackerOldFls<float, 1, 32, galp::codec::device::ALPFunctor<float, 1>>,
		        galp::codec::device::PrefetchAllBranchlessALPExceptionPatcher<float, 1, 32>,
		        galp::codec::device::ALPExtendedColumn<float>>,
		    galp::codec::host::ALPExtendedColumn<float>>(column, magic_value, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::Naive) {
		return galp::kernels::host::query_multi_column<
		    float,
		    1,
		    1,
		    galp::codec::device::ALPDecompressor<
		        float,
		        1,
		        galp::codec::device::BitUnpackerStatefulBranchless<float, 1, 1, galp::codec::device::ALPFunctor<float, 1>>,
		        galp::codec::device::NaiveALPExceptionPatcher<float, 1, 1>,
		        galp::codec::device::ALPExtendedColumn<float>>,
		    galp::codec::host::ALPExtendedColumn<float>>(column, magic_value, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::NaiveBranchless) {
		return galp::kernels::host::query_multi_column<
		    float,
		    1,
		    1,
		    galp::codec::device::ALPDecompressor<
		        float,
		        1,
		        galp::codec::device::BitUnpackerStatefulBranchless<float, 1, 1, galp::codec::device::ALPFunctor<float, 1>>,
		        galp::codec::device::NaiveBranchlessALPExceptionPatcher<float, 1, 1>,
		        galp::codec::device::ALPExtendedColumn<float>>,
		    galp::codec::host::ALPExtendedColumn<float>>(column, magic_value, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::PrefetchAll) {
		return galp::kernels::host::query_multi_column<
		    float,
		    1,
		    1,
		    galp::codec::device::ALPDecompressor<
		        float,
		        1,
		        galp::codec::device::BitUnpackerStatefulBranchless<float, 1, 1, galp::codec::device::ALPFunctor<float, 1>>,
		        galp::codec::device::PrefetchAllALPExceptionPatcher<float, 1, 1>,
		        galp::codec::device::ALPExtendedColumn<float>>,
		    galp::codec::host::ALPExtendedColumn<float>>(column, magic_value, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::PrefetchAllBranchless) {
		return galp::kernels::host::query_multi_column<
		    float,
		    1,
		    1,
		    galp::codec::device::ALPDecompressor<
		        float,
		        1,
		        galp::codec::device::BitUnpackerStatefulBranchless<float, 1, 1, galp::codec::device::ALPFunctor<float, 1>>,
		        galp::codec::device::PrefetchAllBranchlessALPExceptionPatcher<float, 1, 1>,
		        galp::codec::device::ALPExtendedColumn<float>>,
		    galp::codec::host::ALPExtendedColumn<float>>(column, magic_value, n_samples);
	}

	if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::Naive) {
		return galp::kernels::host::query_multi_column<
		    float,
		    4,
		    1,
		    galp::codec::device::ALPDecompressor<
		        float,
		        4,
		        galp::codec::device::BitUnpackerStatefulBranchless<float, 4, 1, galp::codec::device::ALPFunctor<float, 4>>,
		        galp::codec::device::NaiveALPExceptionPatcher<float, 4, 1>,
		        galp::codec::device::ALPExtendedColumn<float>>,
		    galp::codec::host::ALPExtendedColumn<float>>(column, magic_value, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::NaiveBranchless) {
		return galp::kernels::host::query_multi_column<
		    float,
		    4,
		    1,
		    galp::codec::device::ALPDecompressor<
		        float,
		        4,
		        galp::codec::device::BitUnpackerStatefulBranchless<float, 4, 1, galp::codec::device::ALPFunctor<float, 4>>,
		        galp::codec::device::NaiveBranchlessALPExceptionPatcher<float, 4, 1>,
		        galp::codec::device::ALPExtendedColumn<float>>,
		    galp::codec::host::ALPExtendedColumn<float>>(column, magic_value, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::PrefetchAll) {
		return galp::kernels::host::query_multi_column<
		    float,
		    4,
		    1,
		    galp::codec::device::ALPDecompressor<
		        float,
		        4,
		        galp::codec::device::BitUnpackerStatefulBranchless<float, 4, 1, galp::codec::device::ALPFunctor<float, 4>>,
		        galp::codec::device::PrefetchAllALPExceptionPatcher<float, 4, 1>,
		        galp::codec::device::ALPExtendedColumn<float>>,
		    galp::codec::host::ALPExtendedColumn<float>>(column, magic_value, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::PrefetchAllBranchless) {
		return galp::kernels::host::query_multi_column<
		    float,
		    4,
		    1,
		    galp::codec::device::ALPDecompressor<
		        float,
		        4,
		        galp::codec::device::BitUnpackerStatefulBranchless<float, 4, 1, galp::codec::device::ALPFunctor<float, 4>>,
		        galp::codec::device::PrefetchAllBranchlessALPExceptionPatcher<float, 4, 1>,
		        galp::codec::device::ALPExtendedColumn<float>>,
		    galp::codec::host::ALPExtendedColumn<float>>(column, magic_value, n_samples);
	}
	throw std::invalid_argument("Could not find correct binding in query_multi_column ALPExtended<float>");
}

} // namespace galp::bench::bindings
