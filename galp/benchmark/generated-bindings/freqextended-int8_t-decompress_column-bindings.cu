// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/benchmark/generated-bindings/freqextended-int8_t-decompress_column-bindings.cu
// ────────────────────────────────────────────────────────
#include "engine/kernels.cuh"
#include "engine/multi-column-host-kernels.cuh"
#include "generated-bindings/kernel-bindings.cuh"
#include <stdexcept>

namespace galp::bench::bindings {

template <>
int8_t* decompress_column<int8_t, galp::codec::device::FREQExtendedColumn<int8_t>>(
    const galp::codec::device::FREQExtendedColumn<int8_t> column,
    const unsigned                                   unpack_n_vectors,
    const unsigned                                   unpack_n_values,
    const galp::format::Unpacker                            unpacker,
    const galp::format::Patcher                             patcher,
    const galp::format::Expander                            expander,
    const uint32_t                                   n_samples) {
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && patcher == galp::format::Patcher::Naive) {
		return galp::kernels::host::decompress_column<
		    int8_t,
		    1,
		    1,
		    galp::codec::device::FREQDecompressor<int8_t,
		                                     1,
		                                     galp::codec::device::NaiveFREQExceptionPatcher<int8_t, 1, 1>,
		                                     galp::codec::device::FREQExtendedColumn<int8_t>>,
		    galp::codec::device::FREQExtendedColumn<int8_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && patcher == galp::format::Patcher::NaiveBranchless) {
		return galp::kernels::host::decompress_column<
		    int8_t,
		    1,
		    1,
		    galp::codec::device::FREQDecompressor<int8_t,
		                                     1,
		                                     galp::codec::device::NaiveBranchlessFREQExceptionPatcher<int8_t, 1, 1>,
		                                     galp::codec::device::FREQExtendedColumn<int8_t>>,
		    galp::codec::device::FREQExtendedColumn<int8_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && patcher == galp::format::Patcher::PrefetchAll) {
		return galp::kernels::host::decompress_column<
		    int8_t,
		    1,
		    1,
		    galp::codec::device::FREQDecompressor<int8_t,
		                                     1,
		                                     galp::codec::device::PrefetchAllFREQExceptionPatcher<int8_t, 1, 1>,
		                                     galp::codec::device::FREQExtendedColumn<int8_t>>,
		    galp::codec::device::FREQExtendedColumn<int8_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && patcher == galp::format::Patcher::PrefetchAllBranchless) {
		return galp::kernels::host::decompress_column<
		    int8_t,
		    1,
		    1,
		    galp::codec::device::FREQDecompressor<int8_t,
		                                     1,
		                                     galp::codec::device::PrefetchAllBranchlessFREQExceptionPatcher<int8_t, 1, 1>,
		                                     galp::codec::device::FREQExtendedColumn<int8_t>>,
		    galp::codec::device::FREQExtendedColumn<int8_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && patcher == galp::format::Patcher::Naive) {
		return galp::kernels::host::decompress_column<
		    int8_t,
		    4,
		    1,
		    galp::codec::device::FREQDecompressor<int8_t,
		                                     4,
		                                     galp::codec::device::NaiveFREQExceptionPatcher<int8_t, 4, 1>,
		                                     galp::codec::device::FREQExtendedColumn<int8_t>>,
		    galp::codec::device::FREQExtendedColumn<int8_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && patcher == galp::format::Patcher::NaiveBranchless) {
		return galp::kernels::host::decompress_column<
		    int8_t,
		    4,
		    1,
		    galp::codec::device::FREQDecompressor<int8_t,
		                                     4,
		                                     galp::codec::device::NaiveBranchlessFREQExceptionPatcher<int8_t, 4, 1>,
		                                     galp::codec::device::FREQExtendedColumn<int8_t>>,
		    galp::codec::device::FREQExtendedColumn<int8_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && patcher == galp::format::Patcher::PrefetchAll) {
		return galp::kernels::host::decompress_column<
		    int8_t,
		    4,
		    1,
		    galp::codec::device::FREQDecompressor<int8_t,
		                                     4,
		                                     galp::codec::device::PrefetchAllFREQExceptionPatcher<int8_t, 4, 1>,
		                                     galp::codec::device::FREQExtendedColumn<int8_t>>,
		    galp::codec::device::FREQExtendedColumn<int8_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && patcher == galp::format::Patcher::PrefetchAllBranchless) {
		return galp::kernels::host::decompress_column<
		    int8_t,
		    4,
		    1,
		    galp::codec::device::FREQDecompressor<int8_t,
		                                     4,
		                                     galp::codec::device::PrefetchAllBranchlessFREQExceptionPatcher<int8_t, 4, 1>,
		                                     galp::codec::device::FREQExtendedColumn<int8_t>>,
		    galp::codec::device::FREQExtendedColumn<int8_t>>(column, n_samples);
	}
	throw std::invalid_argument("Could not find correct binding in decompress_column FREQExtended<int8_t>");
}

} // namespace galp::bench::bindings
