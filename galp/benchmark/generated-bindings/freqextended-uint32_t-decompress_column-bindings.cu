// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/benchmark/generated-bindings/freqextended-uint32_t-decompress_column-bindings.cu
// ────────────────────────────────────────────────────────
#include "engine/kernels.cuh"
#include "engine/multi-column-host-kernels.cuh"
#include "generated-bindings/kernel-bindings.cuh"
#include <stdexcept>

namespace galp::bench::bindings {

template <>
uint32_t* decompress_column<uint32_t, galp::codec::device::FREQExtendedColumn<uint32_t>>(
    const galp::codec::device::FREQExtendedColumn<uint32_t> column,
    const unsigned                                     unpack_n_vectors,
    const unsigned                                     unpack_n_values,
    const galp::format::Unpacker                              unpacker,
    const galp::format::Patcher                               patcher,
    const galp::format::Expander                              expander,
    const uint32_t                                     n_samples) {
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && patcher == galp::format::Patcher::Naive) {
		return galp::kernels::host::decompress_column<
		    uint32_t,
		    1,
		    1,
		    galp::codec::device::FREQDecompressor<uint32_t,
		                                     1,
		                                     galp::codec::device::NaiveFREQExceptionPatcher<uint32_t, 1, 1>,
		                                     galp::codec::device::FREQExtendedColumn<uint32_t>>,
		    galp::codec::device::FREQExtendedColumn<uint32_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && patcher == galp::format::Patcher::NaiveBranchless) {
		return galp::kernels::host::decompress_column<
		    uint32_t,
		    1,
		    1,
		    galp::codec::device::FREQDecompressor<uint32_t,
		                                     1,
		                                     galp::codec::device::NaiveBranchlessFREQExceptionPatcher<uint32_t, 1, 1>,
		                                     galp::codec::device::FREQExtendedColumn<uint32_t>>,
		    galp::codec::device::FREQExtendedColumn<uint32_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && patcher == galp::format::Patcher::PrefetchAll) {
		return galp::kernels::host::decompress_column<
		    uint32_t,
		    1,
		    1,
		    galp::codec::device::FREQDecompressor<uint32_t,
		                                     1,
		                                     galp::codec::device::PrefetchAllFREQExceptionPatcher<uint32_t, 1, 1>,
		                                     galp::codec::device::FREQExtendedColumn<uint32_t>>,
		    galp::codec::device::FREQExtendedColumn<uint32_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && patcher == galp::format::Patcher::PrefetchAllBranchless) {
		return galp::kernels::host::decompress_column<
		    uint32_t,
		    1,
		    1,
		    galp::codec::device::FREQDecompressor<uint32_t,
		                                     1,
		                                     galp::codec::device::PrefetchAllBranchlessFREQExceptionPatcher<uint32_t, 1, 1>,
		                                     galp::codec::device::FREQExtendedColumn<uint32_t>>,
		    galp::codec::device::FREQExtendedColumn<uint32_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && patcher == galp::format::Patcher::Naive) {
		return galp::kernels::host::decompress_column<
		    uint32_t,
		    4,
		    1,
		    galp::codec::device::FREQDecompressor<uint32_t,
		                                     4,
		                                     galp::codec::device::NaiveFREQExceptionPatcher<uint32_t, 4, 1>,
		                                     galp::codec::device::FREQExtendedColumn<uint32_t>>,
		    galp::codec::device::FREQExtendedColumn<uint32_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && patcher == galp::format::Patcher::NaiveBranchless) {
		return galp::kernels::host::decompress_column<
		    uint32_t,
		    4,
		    1,
		    galp::codec::device::FREQDecompressor<uint32_t,
		                                     4,
		                                     galp::codec::device::NaiveBranchlessFREQExceptionPatcher<uint32_t, 4, 1>,
		                                     galp::codec::device::FREQExtendedColumn<uint32_t>>,
		    galp::codec::device::FREQExtendedColumn<uint32_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && patcher == galp::format::Patcher::PrefetchAll) {
		return galp::kernels::host::decompress_column<
		    uint32_t,
		    4,
		    1,
		    galp::codec::device::FREQDecompressor<uint32_t,
		                                     4,
		                                     galp::codec::device::PrefetchAllFREQExceptionPatcher<uint32_t, 4, 1>,
		                                     galp::codec::device::FREQExtendedColumn<uint32_t>>,
		    galp::codec::device::FREQExtendedColumn<uint32_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && patcher == galp::format::Patcher::PrefetchAllBranchless) {
		return galp::kernels::host::decompress_column<
		    uint32_t,
		    4,
		    1,
		    galp::codec::device::FREQDecompressor<uint32_t,
		                                     4,
		                                     galp::codec::device::PrefetchAllBranchlessFREQExceptionPatcher<uint32_t, 4, 1>,
		                                     galp::codec::device::FREQExtendedColumn<uint32_t>>,
		    galp::codec::device::FREQExtendedColumn<uint32_t>>(column, n_samples);
	}
	throw std::invalid_argument("Could not find correct binding in decompress_column FREQExtended<uint32_t>");
}

} // namespace galp::bench::bindings
