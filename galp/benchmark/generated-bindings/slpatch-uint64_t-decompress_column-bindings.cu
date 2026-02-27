// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/benchmark/generated-bindings/slpatch-uint64_t-decompress_column-bindings.cu
// ────────────────────────────────────────────────────────
#include "engine/kernels.cuh"
#include "engine/multi-column-host-kernels.cuh"
#include "generated-bindings/kernel-bindings.cuh"
#include <stdexcept>

namespace bindings {

template <>
uint64_t* decompress_column<uint64_t, flsgpu::device::SLPATCHColumn<uint64_t>>(
    const flsgpu::device::SLPATCHColumn<uint64_t> column,
    const unsigned                                unpack_n_vectors,
    const unsigned                                unpack_n_values,
    const enums::Unpacker                         unpacker,
    const enums::Patcher                          patcher,
    const enums::Expander                         expander,
    const uint32_t                                n_samples) {
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == enums::Unpacker::Dummy &&
	    patcher == enums::Patcher::Stateful) {
		return kernels::host::decompress_column<
		    uint64_t,
		    1,
		    1,
		    flsgpu::device::SLPATCHDecompressor<
		        uint64_t,
		        1,
		        flsgpu::device::BitUnpackerDummy<uint64_t, 1, 1, flsgpu::device::FFORFunctor<uint64_t, 1>>,
		        flsgpu::device::StatefulSLPATCHExceptionPatcher<uint64_t, 1, 1>,
		        flsgpu::device::SLPATCHColumn<uint64_t>>,
		    flsgpu::device::SLPATCHColumn<uint64_t>>(column, n_samples);
	}

	if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == enums::Unpacker::StatefulBranchless &&
	    patcher == enums::Patcher::Stateful) {
		return kernels::host::decompress_column<
		    uint64_t,
		    1,
		    1,
		    flsgpu::device::SLPATCHDecompressor<
		        uint64_t,
		        1,
		        flsgpu::device::BitUnpackerStatefulBranchless<uint64_t, 1, 1, flsgpu::device::FFORFunctor<uint64_t, 1>>,
		        flsgpu::device::StatefulSLPATCHExceptionPatcher<uint64_t, 1, 1>,
		        flsgpu::device::SLPATCHColumn<uint64_t>>,
		    flsgpu::device::SLPATCHColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == enums::Unpacker::Dummy &&
	    patcher == enums::Patcher::Stateful) {
		return kernels::host::decompress_column<
		    uint64_t,
		    4,
		    1,
		    flsgpu::device::SLPATCHDecompressor<
		        uint64_t,
		        4,
		        flsgpu::device::BitUnpackerDummy<uint64_t, 4, 1, flsgpu::device::FFORFunctor<uint64_t, 4>>,
		        flsgpu::device::StatefulSLPATCHExceptionPatcher<uint64_t, 4, 1>,
		        flsgpu::device::SLPATCHColumn<uint64_t>>,
		    flsgpu::device::SLPATCHColumn<uint64_t>>(column, n_samples);
	}

	if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == enums::Unpacker::StatefulBranchless &&
	    patcher == enums::Patcher::Stateful) {
		return kernels::host::decompress_column<
		    uint64_t,
		    4,
		    1,
		    flsgpu::device::SLPATCHDecompressor<
		        uint64_t,
		        4,
		        flsgpu::device::BitUnpackerStatefulBranchless<uint64_t, 4, 1, flsgpu::device::FFORFunctor<uint64_t, 4>>,
		        flsgpu::device::StatefulSLPATCHExceptionPatcher<uint64_t, 4, 1>,
		        flsgpu::device::SLPATCHColumn<uint64_t>>,
		    flsgpu::device::SLPATCHColumn<uint64_t>>(column, n_samples);
	}
	throw std::invalid_argument("Could not find correct binding in decompress_column SLPATCH<uint64_t>");
}

} // namespace bindings
