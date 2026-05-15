// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/benchmark/generated-bindings/crossrlelanemask-uint32_t-decompress_column-bindings.cu
// ────────────────────────────────────────────────────────
#include "engine/kernels/dispatch.cuh"
#include "galp_bench/generated/multi_column_host_kernels.cuh"
#include "generated-bindings/kernel-bindings.cuh"
#include <stdexcept>

namespace galp::bench::bindings {

template <>
uint32_t* decompress_column<uint32_t, galp::codec::device::CROSSRLELaneMaskColumn<uint32_t>>(
    const galp::codec::device::CROSSRLELaneMaskColumn<uint32_t> column,
    const unsigned                                         unpack_n_vectors,
    const unsigned                                         unpack_n_values,
    const galp::format::Unpacker                                  unpacker,
    const galp::format::Patcher                                   patcher,
    const galp::format::Expander                                  expander,
    const uint32_t                                         n_samples) {
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && expander == galp::format::Expander::Branchless) {
		return galp::kernels::host::decompress_column<
		    uint32_t,
		    1,
		    1,
		    galp::codec::device::CROSSRLEDecompressor<uint32_t,
		                                         1,
		                                         galp::codec::device::BranchlessCROSSRLEExpander<uint32_t, 1, 1>,
		                                         galp::codec::device::CROSSRLELaneMaskColumn<uint32_t>>,
		    galp::codec::device::CROSSRLELaneMaskColumn<uint32_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && expander == galp::format::Expander::PrefetchBranchless) {
		return galp::kernels::host::decompress_column<
		    uint32_t,
		    1,
		    1,
		    galp::codec::device::CROSSRLEDecompressor<uint32_t,
		                                         1,
		                                         galp::codec::device::PrefetchBranchlessCROSSRLEExpander<uint32_t, 1, 1>,
		                                         galp::codec::device::CROSSRLELaneMaskColumn<uint32_t>>,
		    galp::codec::device::CROSSRLELaneMaskColumn<uint32_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && expander == galp::format::Expander::Branchless) {
		return galp::kernels::host::decompress_column<
		    uint32_t,
		    4,
		    1,
		    galp::codec::device::CROSSRLEDecompressor<uint32_t,
		                                         4,
		                                         galp::codec::device::BranchlessCROSSRLEExpander<uint32_t, 4, 1>,
		                                         galp::codec::device::CROSSRLELaneMaskColumn<uint32_t>>,
		    galp::codec::device::CROSSRLELaneMaskColumn<uint32_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && expander == galp::format::Expander::PrefetchBranchless) {
		return galp::kernels::host::decompress_column<
		    uint32_t,
		    4,
		    1,
		    galp::codec::device::CROSSRLEDecompressor<uint32_t,
		                                         4,
		                                         galp::codec::device::PrefetchBranchlessCROSSRLEExpander<uint32_t, 4, 1>,
		                                         galp::codec::device::CROSSRLELaneMaskColumn<uint32_t>>,
		    galp::codec::device::CROSSRLELaneMaskColumn<uint32_t>>(column, n_samples);
	}
	throw std::invalid_argument("Could not find correct binding in decompress_column CROSSRLELaneMask<uint32_t>");
}

} // namespace galp::bench::bindings
