// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/benchmark/generated-bindings/crossrle-uint64_t-decompress_column-bindings.cu
// ────────────────────────────────────────────────────────
#include "engine/kernels/dispatch.cuh"
#include "galp_bench/generated/multi_column_host_kernels.cuh"
#include "generated-bindings/kernel-bindings.cuh"
#include <stdexcept>

namespace galp::bench::bindings {

template <>
uint64_t* decompress_column<uint64_t, galp::codec::device::CROSSRLEColumn<uint64_t>>(
    const galp::codec::device::CROSSRLEColumn<uint64_t> column,
    const unsigned                                 unpack_n_vectors,
    const unsigned                                 unpack_n_values,
    const galp::format::Unpacker                          unpacker,
    const galp::format::Patcher                           patcher,
    const galp::format::Expander                          expander,
    const uint32_t                                 n_samples) {
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && expander == galp::format::Expander::Dummy) {
		return galp::kernels::host::decompress_column<
		    uint64_t,
		    1,
		    1,
		    galp::codec::device::CROSSRLEDecompressor<uint64_t,
		                                         1,
		                                         galp::codec::device::DummyCROSSRLEExpander<uint64_t, 1, 1>,
		                                         galp::codec::device::CROSSRLEColumn<uint64_t>>,
		    galp::codec::device::CROSSRLEColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && expander == galp::format::Expander::Stateful) {
		return galp::kernels::host::decompress_column<
		    uint64_t,
		    1,
		    1,
		    galp::codec::device::CROSSRLEDecompressor<uint64_t,
		                                         1,
		                                         galp::codec::device::StatefulCROSSRLEExpander<uint64_t, 1, 1>,
		                                         galp::codec::device::CROSSRLEColumn<uint64_t>>,
		    galp::codec::device::CROSSRLEColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && expander == galp::format::Expander::StatefulCache) {
		return galp::kernels::host::decompress_column<
		    uint64_t,
		    1,
		    1,
		    galp::codec::device::CROSSRLEDecompressor<uint64_t,
		                                         1,
		                                         galp::codec::device::StatefulCacheCROSSRLEExpander<uint64_t, 1, 1>,
		                                         galp::codec::device::CROSSRLEColumn<uint64_t>>,
		    galp::codec::device::CROSSRLEColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && expander == galp::format::Expander::PrefetchStateful) {
		return galp::kernels::host::decompress_column<
		    uint64_t,
		    1,
		    1,
		    galp::codec::device::CROSSRLEDecompressor<uint64_t,
		                                         1,
		                                         galp::codec::device::PrefetchStatefulCROSSRLEExpander<uint64_t, 1, 1>,
		                                         galp::codec::device::CROSSRLEColumn<uint64_t>>,
		    galp::codec::device::CROSSRLEColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && expander == galp::format::Expander::StatefulShuffle) {
		return galp::kernels::host::decompress_column<
		    uint64_t,
		    1,
		    1,
		    galp::codec::device::CROSSRLEDecompressor<uint64_t,
		                                         1,
		                                         galp::codec::device::StatefulShuffleCROSSRLEExpander<uint64_t, 1, 1>,
		                                         galp::codec::device::CROSSRLEColumn<uint64_t>>,
		    galp::codec::device::CROSSRLEColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && expander == galp::format::Expander::StatefulAdvance) {
		return galp::kernels::host::decompress_column<
		    uint64_t,
		    1,
		    1,
		    galp::codec::device::CROSSRLEDecompressor<uint64_t,
		                                         1,
		                                         galp::codec::device::StatefulAdvanceCROSSRLEExpander<uint64_t, 1, 1>,
		                                         galp::codec::device::CROSSRLEColumn<uint64_t>>,
		    galp::codec::device::CROSSRLEColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && expander == galp::format::Expander::Dummy) {
		return galp::kernels::host::decompress_column<
		    uint64_t,
		    4,
		    1,
		    galp::codec::device::CROSSRLEDecompressor<uint64_t,
		                                         4,
		                                         galp::codec::device::DummyCROSSRLEExpander<uint64_t, 4, 1>,
		                                         galp::codec::device::CROSSRLEColumn<uint64_t>>,
		    galp::codec::device::CROSSRLEColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && expander == galp::format::Expander::Stateful) {
		return galp::kernels::host::decompress_column<
		    uint64_t,
		    4,
		    1,
		    galp::codec::device::CROSSRLEDecompressor<uint64_t,
		                                         4,
		                                         galp::codec::device::StatefulCROSSRLEExpander<uint64_t, 4, 1>,
		                                         galp::codec::device::CROSSRLEColumn<uint64_t>>,
		    galp::codec::device::CROSSRLEColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && expander == galp::format::Expander::StatefulCache) {
		return galp::kernels::host::decompress_column<
		    uint64_t,
		    4,
		    1,
		    galp::codec::device::CROSSRLEDecompressor<uint64_t,
		                                         4,
		                                         galp::codec::device::StatefulCacheCROSSRLEExpander<uint64_t, 4, 1>,
		                                         galp::codec::device::CROSSRLEColumn<uint64_t>>,
		    galp::codec::device::CROSSRLEColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && expander == galp::format::Expander::PrefetchStateful) {
		return galp::kernels::host::decompress_column<
		    uint64_t,
		    4,
		    1,
		    galp::codec::device::CROSSRLEDecompressor<uint64_t,
		                                         4,
		                                         galp::codec::device::PrefetchStatefulCROSSRLEExpander<uint64_t, 4, 1>,
		                                         galp::codec::device::CROSSRLEColumn<uint64_t>>,
		    galp::codec::device::CROSSRLEColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && expander == galp::format::Expander::StatefulShuffle) {
		return galp::kernels::host::decompress_column<
		    uint64_t,
		    4,
		    1,
		    galp::codec::device::CROSSRLEDecompressor<uint64_t,
		                                         4,
		                                         galp::codec::device::StatefulShuffleCROSSRLEExpander<uint64_t, 4, 1>,
		                                         galp::codec::device::CROSSRLEColumn<uint64_t>>,
		    galp::codec::device::CROSSRLEColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && expander == galp::format::Expander::StatefulAdvance) {
		return galp::kernels::host::decompress_column<
		    uint64_t,
		    4,
		    1,
		    galp::codec::device::CROSSRLEDecompressor<uint64_t,
		                                         4,
		                                         galp::codec::device::StatefulAdvanceCROSSRLEExpander<uint64_t, 4, 1>,
		                                         galp::codec::device::CROSSRLEColumn<uint64_t>>,
		    galp::codec::device::CROSSRLEColumn<uint64_t>>(column, n_samples);
	}
	throw std::invalid_argument("Could not find correct binding in decompress_column CROSSRLE<uint64_t>");
}

} // namespace galp::bench::bindings
