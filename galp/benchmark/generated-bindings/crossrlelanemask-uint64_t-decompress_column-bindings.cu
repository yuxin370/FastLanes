// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/benchmark/generated-bindings/crossrlelanemask-uint64_t-decompress_column-bindings.cu
// ────────────────────────────────────────────────────────
#include "engine/kernels.cuh"
#include "engine/multi-column-host-kernels.cuh"
#include "generated-bindings/kernel-bindings.cuh"
#include <stdexcept>

namespace galp::bench::bindings {

template <>
uint64_t* decompress_column<uint64_t, galp::codec::device::CROSSRLELaneMaskColumn<uint64_t>>(
    const galp::codec::device::CROSSRLELaneMaskColumn<uint64_t> column,
    const unsigned                                         unpack_n_vectors,
    const unsigned                                         unpack_n_values,
    const galp::format::Unpacker                                  unpacker,
    const galp::format::Patcher                                   patcher,
    const galp::format::Expander                                  expander,
    const uint32_t                                         n_samples) {
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && expander == galp::format::Expander::Branchless) {
		return galp::kernels::host::decompress_column<
		    uint64_t,
		    1,
		    1,
		    galp::codec::device::CROSSRLEDecompressor<uint64_t,
		                                         1,
		                                         galp::codec::device::BranchlessCROSSRLEExpander<uint64_t, 1, 1>,
		                                         galp::codec::device::CROSSRLELaneMaskColumn<uint64_t>>,
		    galp::codec::device::CROSSRLELaneMaskColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && expander == galp::format::Expander::PrefetchBranchless) {
		return galp::kernels::host::decompress_column<
		    uint64_t,
		    1,
		    1,
		    galp::codec::device::CROSSRLEDecompressor<uint64_t,
		                                         1,
		                                         galp::codec::device::PrefetchBranchlessCROSSRLEExpander<uint64_t, 1, 1>,
		                                         galp::codec::device::CROSSRLELaneMaskColumn<uint64_t>>,
		    galp::codec::device::CROSSRLELaneMaskColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && expander == galp::format::Expander::Branchless) {
		return galp::kernels::host::decompress_column<
		    uint64_t,
		    4,
		    1,
		    galp::codec::device::CROSSRLEDecompressor<uint64_t,
		                                         4,
		                                         galp::codec::device::BranchlessCROSSRLEExpander<uint64_t, 4, 1>,
		                                         galp::codec::device::CROSSRLELaneMaskColumn<uint64_t>>,
		    galp::codec::device::CROSSRLELaneMaskColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && expander == galp::format::Expander::PrefetchBranchless) {
		return galp::kernels::host::decompress_column<
		    uint64_t,
		    4,
		    1,
		    galp::codec::device::CROSSRLEDecompressor<uint64_t,
		                                         4,
		                                         galp::codec::device::PrefetchBranchlessCROSSRLEExpander<uint64_t, 4, 1>,
		                                         galp::codec::device::CROSSRLELaneMaskColumn<uint64_t>>,
		    galp::codec::device::CROSSRLELaneMaskColumn<uint64_t>>(column, n_samples);
	}
	throw std::invalid_argument("Could not find correct binding in decompress_column CROSSRLELaneMask<uint64_t>");
}

} // namespace galp::bench::bindings
