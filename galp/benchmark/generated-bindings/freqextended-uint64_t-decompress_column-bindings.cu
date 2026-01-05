
#include "engine/kernels.cuh"
#include "engine/multi-column-host-kernels.cuh"
#include "generated-bindings/kernel-bindings.cuh"
#include <stdexcept>

namespace bindings {

template <>
uint64_t* decompress_column<uint64_t, flsgpu::device::FREQExtendedColumn<uint64_t>>(
    const flsgpu::device::FREQExtendedColumn<uint64_t> column,
    const unsigned                                     unpack_n_vectors,
    const unsigned                                     unpack_n_values,
    const enums::Unpacker                              unpacker,
    const enums::Patcher                               patcher,
    const uint32_t                                     n_samples) {

	if (unpack_n_vectors == 1 && unpack_n_values == 1 && patcher == enums::Patcher::Naive) {
		return kernels::host::decompress_column<
		    uint64_t,
		    1,
		    1,
		    flsgpu::device::FREQDecompressor<uint64_t,
		                                     1,
		                                     flsgpu::device::NaiveFREQExceptionPatcher<uint64_t, 1, 1>,
		                                     flsgpu::device::FREQExtendedColumn<uint64_t>>,
		    flsgpu::device::FREQExtendedColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && patcher == enums::Patcher::NaiveBranchless) {
		return kernels::host::decompress_column<
		    uint64_t,
		    1,
		    1,
		    flsgpu::device::FREQDecompressor<uint64_t,
		                                     1,
		                                     flsgpu::device::NaiveBranchlessFREQExceptionPatcher<uint64_t, 1, 1>,
		                                     flsgpu::device::FREQExtendedColumn<uint64_t>>,
		    flsgpu::device::FREQExtendedColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && patcher == enums::Patcher::PrefetchAll) {
		return kernels::host::decompress_column<
		    uint64_t,
		    1,
		    1,
		    flsgpu::device::FREQDecompressor<uint64_t,
		                                     1,
		                                     flsgpu::device::PrefetchAllFREQExceptionPatcher<uint64_t, 1, 1>,
		                                     flsgpu::device::FREQExtendedColumn<uint64_t>>,
		    flsgpu::device::FREQExtendedColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && patcher == enums::Patcher::PrefetchAllBranchless) {
		return kernels::host::decompress_column<
		    uint64_t,
		    1,
		    1,
		    flsgpu::device::FREQDecompressor<uint64_t,
		                                     1,
		                                     flsgpu::device::PrefetchAllBranchlessFREQExceptionPatcher<uint64_t, 1, 1>,
		                                     flsgpu::device::FREQExtendedColumn<uint64_t>>,
		    flsgpu::device::FREQExtendedColumn<uint64_t>>(column, n_samples);
	}

	if (unpack_n_vectors == 4 && unpack_n_values == 1 && patcher == enums::Patcher::Naive) {
		return kernels::host::decompress_column<
		    uint64_t,
		    4,
		    1,
		    flsgpu::device::FREQDecompressor<uint64_t,
		                                     4,
		                                     flsgpu::device::NaiveFREQExceptionPatcher<uint64_t, 4, 1>,
		                                     flsgpu::device::FREQExtendedColumn<uint64_t>>,
		    flsgpu::device::FREQExtendedColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && patcher == enums::Patcher::NaiveBranchless) {
		return kernels::host::decompress_column<
		    uint64_t,
		    4,
		    1,
		    flsgpu::device::FREQDecompressor<uint64_t,
		                                     4,
		                                     flsgpu::device::NaiveBranchlessFREQExceptionPatcher<uint64_t, 4, 1>,
		                                     flsgpu::device::FREQExtendedColumn<uint64_t>>,
		    flsgpu::device::FREQExtendedColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && patcher == enums::Patcher::PrefetchAll) {
		return kernels::host::decompress_column<
		    uint64_t,
		    4,
		    1,
		    flsgpu::device::FREQDecompressor<uint64_t,
		                                     4,
		                                     flsgpu::device::PrefetchAllFREQExceptionPatcher<uint64_t, 4, 1>,
		                                     flsgpu::device::FREQExtendedColumn<uint64_t>>,
		    flsgpu::device::FREQExtendedColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && patcher == enums::Patcher::PrefetchAllBranchless) {
		return kernels::host::decompress_column<
		    uint64_t,
		    4,
		    1,
		    flsgpu::device::FREQDecompressor<uint64_t,
		                                     4,
		                                     flsgpu::device::PrefetchAllBranchlessFREQExceptionPatcher<uint64_t, 4, 1>,
		                                     flsgpu::device::FREQExtendedColumn<uint64_t>>,
		    flsgpu::device::FREQExtendedColumn<uint64_t>>(column, n_samples);
	}
	throw std::invalid_argument("Could not find correct binding in decompress_column FREQExtended<uint64_t>");
}

} // namespace bindings
