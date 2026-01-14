
#include "engine/kernels.cuh"
#include "engine/multi-column-host-kernels.cuh"
#include "generated-bindings/kernel-bindings.cuh"
#include <stdexcept>

namespace bindings {

template <>
uint64_t* decompress_column<uint64_t, flsgpu::device::CROSSRLEColumn<uint64_t>>(
    const flsgpu::device::CROSSRLEColumn<uint64_t> column,
    const unsigned                                 unpack_n_vectors,
    const unsigned                                 unpack_n_values,
    const enums::Unpacker                          unpacker,
    const enums::Patcher                           patcher,
    const uint32_t                                 n_samples) {
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && expander == enums::Expander::None) {
		return kernels::host::decompress_column<
		    uint64_t,
		    1,
		    1,
		    flsgpu::device::CROSSRLEDecompressor<uint64_t,
		                                         1,
		                                         flsgpu::device::NoneCROSSRLEExpander<uint64_t, 1, 1>,
		                                         flsgpu::device::CROSSRLEColumn<uint64_t>>,
		    flsgpu::device::CROSSRLEColumn<uint64_t>>(column, n_samples);
	}

	if (unpack_n_vectors == 1 && unpack_n_values == 1 && expander == enums::Expander::None) {
		return kernels::host::decompress_column<
		    uint64_t,
		    1,
		    1,
		    flsgpu::device::CROSSRLEDecompressor<uint64_t,
		                                         1,
		                                         flsgpu::device::NoneCROSSRLEExpander<uint64_t, 1, 1>,
		                                         flsgpu::device::CROSSRLEColumn<uint64_t>>,
		    flsgpu::device::CROSSRLEColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && expander == enums::Expander::None) {
		return kernels::host::decompress_column<
		    uint64_t,
		    4,
		    1,
		    flsgpu::device::CROSSRLEDecompressor<uint64_t,
		                                         4,
		                                         flsgpu::device::NoneCROSSRLEExpander<uint64_t, 4, 1>,
		                                         flsgpu::device::CROSSRLEColumn<uint64_t>>,
		    flsgpu::device::CROSSRLEColumn<uint64_t>>(column, n_samples);
	}

	if (unpack_n_vectors == 4 && unpack_n_values == 1 && expander == enums::Expander::None) {
		return kernels::host::decompress_column<
		    uint64_t,
		    4,
		    1,
		    flsgpu::device::CROSSRLEDecompressor<uint64_t,
		                                         4,
		                                         flsgpu::device::NoneCROSSRLEExpander<uint64_t, 4, 1>,
		                                         flsgpu::device::CROSSRLEColumn<uint64_t>>,
		    flsgpu::device::CROSSRLEColumn<uint64_t>>(column, n_samples);
	}
	throw std::invalid_argument("Could not find correct binding in decompress_column CROSSRLE<uint64_t>");
}

} // namespace bindings
