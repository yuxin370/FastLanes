// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/benchmark/generated-bindings/crossrlelanemask-uint32_t-decompress_column-bindings.cu
// ────────────────────────────────────────────────────────
#include "engine/kernels.cuh"
#include "engine/multi-column-host-kernels.cuh"
#include "generated-bindings/kernel-bindings.cuh"
#include <stdexcept>

namespace bindings {

template <>
uint32_t* decompress_column<uint32_t, flsgpu::device::CROSSRLELaneMaskColumn<uint32_t>>(
    const flsgpu::device::CROSSRLELaneMaskColumn<uint32_t> column,
    const unsigned                                         unpack_n_vectors,
    const unsigned                                         unpack_n_values,
    const enums::Unpacker                                  unpacker,
    const enums::Patcher                                   patcher,
    const enums::Expander                                  expander,
    const uint32_t                                         n_samples) {
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && expander == enums::Expander::Branchless) {
		return kernels::host::decompress_column<
		    uint32_t,
		    1,
		    1,
		    flsgpu::device::CROSSRLEDecompressor<uint32_t,
		                                         1,
		                                         flsgpu::device::BranchlessCROSSRLEExpander<uint32_t, 1, 1>,
		                                         flsgpu::device::CROSSRLELaneMaskColumn<uint32_t>>,
		    flsgpu::device::CROSSRLELaneMaskColumn<uint32_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && expander == enums::Expander::PrefetchBranchless) {
		return kernels::host::decompress_column<
		    uint32_t,
		    1,
		    1,
		    flsgpu::device::CROSSRLEDecompressor<uint32_t,
		                                         1,
		                                         flsgpu::device::PrefetchBranchlessCROSSRLEExpander<uint32_t, 1, 1>,
		                                         flsgpu::device::CROSSRLELaneMaskColumn<uint32_t>>,
		    flsgpu::device::CROSSRLELaneMaskColumn<uint32_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && expander == enums::Expander::Branchless) {
		return kernels::host::decompress_column<
		    uint32_t,
		    4,
		    1,
		    flsgpu::device::CROSSRLEDecompressor<uint32_t,
		                                         4,
		                                         flsgpu::device::BranchlessCROSSRLEExpander<uint32_t, 4, 1>,
		                                         flsgpu::device::CROSSRLELaneMaskColumn<uint32_t>>,
		    flsgpu::device::CROSSRLELaneMaskColumn<uint32_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && expander == enums::Expander::PrefetchBranchless) {
		return kernels::host::decompress_column<
		    uint32_t,
		    4,
		    1,
		    flsgpu::device::CROSSRLEDecompressor<uint32_t,
		                                         4,
		                                         flsgpu::device::PrefetchBranchlessCROSSRLEExpander<uint32_t, 4, 1>,
		                                         flsgpu::device::CROSSRLELaneMaskColumn<uint32_t>>,
		    flsgpu::device::CROSSRLELaneMaskColumn<uint32_t>>(column, n_samples);
	}
	throw std::invalid_argument("Could not find correct binding in decompress_column CROSSRLELaneMask<uint32_t>");
}

} // namespace bindings
