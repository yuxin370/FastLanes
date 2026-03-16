// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/benchmark/generated-bindings/slpatch-int16_t-decompress_column-bindings.cu
// ────────────────────────────────────────────────────────
#include "engine/kernels.cuh"
#include "engine/multi-column-host-kernels.cuh"
#include "generated-bindings/kernel-bindings.cuh"
#include <stdexcept>

namespace bindings {

template <>
int16_t*
decompress_column<int16_t, flsgpu::device::SLPATCHColumn<int16_t>>(const flsgpu::device::SLPATCHColumn<int16_t> column,
                                                                   const unsigned        unpack_n_vectors,
                                                                   const unsigned        unpack_n_values,
                                                                   const enums::Unpacker unpacker,
                                                                   const enums::Patcher  patcher,
                                                                   const enums::Expander expander,
                                                                   const uint32_t        n_samples) {
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == enums::Unpacker::Dummy &&
	    patcher == enums::Patcher::Stateful) {
		return kernels::host::decompress_column<
		    int16_t,
		    1,
		    1,
		    flsgpu::device::SLPATCHDecompressor<
		        int16_t,
		        1,
		        flsgpu::device::BitUnpackerDummy<int16_t, 1, 1, flsgpu::device::FFORFunctor<int16_t, 1>>,
		        flsgpu::device::StatefulSLPATCHExceptionPatcher<int16_t, 1, 1>,
		        flsgpu::device::SLPATCHColumn<int16_t>>,
		    flsgpu::device::SLPATCHColumn<int16_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == enums::Unpacker::Dummy &&
	    patcher == enums::Patcher::Stateless) {
		return kernels::host::decompress_column<
		    int16_t,
		    1,
		    1,
		    flsgpu::device::SLPATCHDecompressor<
		        int16_t,
		        1,
		        flsgpu::device::BitUnpackerDummy<int16_t, 1, 1, flsgpu::device::FFORFunctor<int16_t, 1>>,
		        flsgpu::device::StatelessSLPATCHExceptionPatcher<int16_t, 1, 1>,
		        flsgpu::device::SLPATCHColumn<int16_t>>,
		    flsgpu::device::SLPATCHColumn<int16_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == enums::Unpacker::StatefulBranchless &&
	    patcher == enums::Patcher::Stateful) {
		return kernels::host::decompress_column<
		    int16_t,
		    1,
		    1,
		    flsgpu::device::SLPATCHDecompressor<
		        int16_t,
		        1,
		        flsgpu::device::BitUnpackerStatefulBranchless<int16_t, 1, 1, flsgpu::device::FFORFunctor<int16_t, 1>>,
		        flsgpu::device::StatefulSLPATCHExceptionPatcher<int16_t, 1, 1>,
		        flsgpu::device::SLPATCHColumn<int16_t>>,
		    flsgpu::device::SLPATCHColumn<int16_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == enums::Unpacker::StatefulBranchless &&
	    patcher == enums::Patcher::Stateless) {
		return kernels::host::decompress_column<
		    int16_t,
		    1,
		    1,
		    flsgpu::device::SLPATCHDecompressor<
		        int16_t,
		        1,
		        flsgpu::device::BitUnpackerStatefulBranchless<int16_t, 1, 1, flsgpu::device::FFORFunctor<int16_t, 1>>,
		        flsgpu::device::StatelessSLPATCHExceptionPatcher<int16_t, 1, 1>,
		        flsgpu::device::SLPATCHColumn<int16_t>>,
		    flsgpu::device::SLPATCHColumn<int16_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == enums::Unpacker::Dummy &&
	    patcher == enums::Patcher::Stateful) {
		return kernels::host::decompress_column<
		    int16_t,
		    4,
		    1,
		    flsgpu::device::SLPATCHDecompressor<
		        int16_t,
		        4,
		        flsgpu::device::BitUnpackerDummy<int16_t, 4, 1, flsgpu::device::FFORFunctor<int16_t, 4>>,
		        flsgpu::device::StatefulSLPATCHExceptionPatcher<int16_t, 4, 1>,
		        flsgpu::device::SLPATCHColumn<int16_t>>,
		    flsgpu::device::SLPATCHColumn<int16_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == enums::Unpacker::Dummy &&
	    patcher == enums::Patcher::Stateless) {
		return kernels::host::decompress_column<
		    int16_t,
		    4,
		    1,
		    flsgpu::device::SLPATCHDecompressor<
		        int16_t,
		        4,
		        flsgpu::device::BitUnpackerDummy<int16_t, 4, 1, flsgpu::device::FFORFunctor<int16_t, 4>>,
		        flsgpu::device::StatelessSLPATCHExceptionPatcher<int16_t, 4, 1>,
		        flsgpu::device::SLPATCHColumn<int16_t>>,
		    flsgpu::device::SLPATCHColumn<int16_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == enums::Unpacker::StatefulBranchless &&
	    patcher == enums::Patcher::Stateful) {
		return kernels::host::decompress_column<
		    int16_t,
		    4,
		    1,
		    flsgpu::device::SLPATCHDecompressor<
		        int16_t,
		        4,
		        flsgpu::device::BitUnpackerStatefulBranchless<int16_t, 4, 1, flsgpu::device::FFORFunctor<int16_t, 4>>,
		        flsgpu::device::StatefulSLPATCHExceptionPatcher<int16_t, 4, 1>,
		        flsgpu::device::SLPATCHColumn<int16_t>>,
		    flsgpu::device::SLPATCHColumn<int16_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == enums::Unpacker::StatefulBranchless &&
	    patcher == enums::Patcher::Stateless) {
		return kernels::host::decompress_column<
		    int16_t,
		    4,
		    1,
		    flsgpu::device::SLPATCHDecompressor<
		        int16_t,
		        4,
		        flsgpu::device::BitUnpackerStatefulBranchless<int16_t, 4, 1, flsgpu::device::FFORFunctor<int16_t, 4>>,
		        flsgpu::device::StatelessSLPATCHExceptionPatcher<int16_t, 4, 1>,
		        flsgpu::device::SLPATCHColumn<int16_t>>,
		    flsgpu::device::SLPATCHColumn<int16_t>>(column, n_samples);
	}
	throw std::invalid_argument("Could not find correct binding in decompress_column SLPATCH<int16_t>");
}

} // namespace bindings
