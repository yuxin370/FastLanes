// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/benchmark/generated-bindings/slpatch-int16_t-decompress_column-bindings.cu
// ────────────────────────────────────────────────────────
#include "engine/kernels/dispatch.cuh"
#include "galp_bench/generated/multi_column_host_kernels.cuh"
#include "generated-bindings/kernel-bindings.cuh"
#include <stdexcept>

namespace galp::bench::bindings {

template <>
int16_t*
decompress_column<int16_t, galp::codec::device::SLPATCHColumn<int16_t>>(const galp::codec::device::SLPATCHColumn<int16_t> column,
                                                                   const unsigned        unpack_n_vectors,
                                                                   const unsigned        unpack_n_values,
                                                                   const galp::format::Unpacker unpacker,
                                                                   const galp::format::Patcher  patcher,
                                                                   const galp::format::Expander expander,
                                                                   const uint32_t        n_samples) {
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::Dummy &&
	    patcher == galp::format::Patcher::Stateful) {
		return galp::kernels::host::decompress_column<
		    int16_t,
		    1,
		    1,
		    galp::codec::device::SLPATCHDecompressor<
		        int16_t,
		        1,
		        galp::codec::device::BitUnpackerDummy<int16_t, 1, 1, galp::codec::device::FFORFunctor<int16_t, 1>>,
		        galp::codec::device::StatefulSLPATCHExceptionPatcher<int16_t, 1, 1>,
		        galp::codec::device::SLPATCHColumn<int16_t>>,
		    galp::codec::device::SLPATCHColumn<int16_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::Dummy &&
	    patcher == galp::format::Patcher::Stateless) {
		return galp::kernels::host::decompress_column<
		    int16_t,
		    1,
		    1,
		    galp::codec::device::SLPATCHDecompressor<
		        int16_t,
		        1,
		        galp::codec::device::BitUnpackerDummy<int16_t, 1, 1, galp::codec::device::FFORFunctor<int16_t, 1>>,
		        galp::codec::device::StatelessSLPATCHExceptionPatcher<int16_t, 1, 1>,
		        galp::codec::device::SLPATCHColumn<int16_t>>,
		    galp::codec::device::SLPATCHColumn<int16_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::Stateful) {
		return galp::kernels::host::decompress_column<
		    int16_t,
		    1,
		    1,
		    galp::codec::device::SLPATCHDecompressor<
		        int16_t,
		        1,
		        galp::codec::device::BitUnpackerStatefulBranchless<int16_t, 1, 1, galp::codec::device::FFORFunctor<int16_t, 1>>,
		        galp::codec::device::StatefulSLPATCHExceptionPatcher<int16_t, 1, 1>,
		        galp::codec::device::SLPATCHColumn<int16_t>>,
		    galp::codec::device::SLPATCHColumn<int16_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::Stateless) {
		return galp::kernels::host::decompress_column<
		    int16_t,
		    1,
		    1,
		    galp::codec::device::SLPATCHDecompressor<
		        int16_t,
		        1,
		        galp::codec::device::BitUnpackerStatefulBranchless<int16_t, 1, 1, galp::codec::device::FFORFunctor<int16_t, 1>>,
		        galp::codec::device::StatelessSLPATCHExceptionPatcher<int16_t, 1, 1>,
		        galp::codec::device::SLPATCHColumn<int16_t>>,
		    galp::codec::device::SLPATCHColumn<int16_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::Dummy &&
	    patcher == galp::format::Patcher::Stateful) {
		return galp::kernels::host::decompress_column<
		    int16_t,
		    4,
		    1,
		    galp::codec::device::SLPATCHDecompressor<
		        int16_t,
		        4,
		        galp::codec::device::BitUnpackerDummy<int16_t, 4, 1, galp::codec::device::FFORFunctor<int16_t, 4>>,
		        galp::codec::device::StatefulSLPATCHExceptionPatcher<int16_t, 4, 1>,
		        galp::codec::device::SLPATCHColumn<int16_t>>,
		    galp::codec::device::SLPATCHColumn<int16_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::Dummy &&
	    patcher == galp::format::Patcher::Stateless) {
		return galp::kernels::host::decompress_column<
		    int16_t,
		    4,
		    1,
		    galp::codec::device::SLPATCHDecompressor<
		        int16_t,
		        4,
		        galp::codec::device::BitUnpackerDummy<int16_t, 4, 1, galp::codec::device::FFORFunctor<int16_t, 4>>,
		        galp::codec::device::StatelessSLPATCHExceptionPatcher<int16_t, 4, 1>,
		        galp::codec::device::SLPATCHColumn<int16_t>>,
		    galp::codec::device::SLPATCHColumn<int16_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::Stateful) {
		return galp::kernels::host::decompress_column<
		    int16_t,
		    4,
		    1,
		    galp::codec::device::SLPATCHDecompressor<
		        int16_t,
		        4,
		        galp::codec::device::BitUnpackerStatefulBranchless<int16_t, 4, 1, galp::codec::device::FFORFunctor<int16_t, 4>>,
		        galp::codec::device::StatefulSLPATCHExceptionPatcher<int16_t, 4, 1>,
		        galp::codec::device::SLPATCHColumn<int16_t>>,
		    galp::codec::device::SLPATCHColumn<int16_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::Stateless) {
		return galp::kernels::host::decompress_column<
		    int16_t,
		    4,
		    1,
		    galp::codec::device::SLPATCHDecompressor<
		        int16_t,
		        4,
		        galp::codec::device::BitUnpackerStatefulBranchless<int16_t, 4, 1, galp::codec::device::FFORFunctor<int16_t, 4>>,
		        galp::codec::device::StatelessSLPATCHExceptionPatcher<int16_t, 4, 1>,
		        galp::codec::device::SLPATCHColumn<int16_t>>,
		    galp::codec::device::SLPATCHColumn<int16_t>>(column, n_samples);
	}
	throw std::invalid_argument("Could not find correct binding in decompress_column SLPATCH<int16_t>");
}

} // namespace galp::bench::bindings
