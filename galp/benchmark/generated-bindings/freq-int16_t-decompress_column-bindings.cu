// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/benchmark/generated-bindings/freq-int16_t-decompress_column-bindings.cu
// ────────────────────────────────────────────────────────
#include "engine/kernels/dispatch.cuh"
#include "galp_bench/generated/multi_column_host_kernels.cuh"
#include "generated-bindings/kernel-bindings.cuh"
#include <stdexcept>

namespace galp::bench::bindings {

template <>
int16_t*
decompress_column<int16_t, galp::codec::device::FREQColumn<int16_t>>(const galp::codec::device::FREQColumn<int16_t> column,
                                                                const unsigned        unpack_n_vectors,
                                                                const unsigned        unpack_n_values,
                                                                const galp::format::Unpacker unpacker,
                                                                const galp::format::Patcher  patcher,
                                                                const galp::format::Expander expander,
                                                                const uint32_t        n_samples) {
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && patcher == galp::format::Patcher::Dummy) {
		return galp::kernels::host::decompress_column<
		    int16_t,
		    1,
		    1,
		    galp::codec::device::FREQDecompressor<int16_t,
		                                     1,
		                                     galp::codec::device::DummyFREQExceptionPatcher<int16_t, 1, 1>,
		                                     galp::codec::device::FREQColumn<int16_t>>,
		    galp::codec::device::FREQColumn<int16_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && patcher == galp::format::Patcher::Stateless) {
		return galp::kernels::host::decompress_column<
		    int16_t,
		    1,
		    1,
		    galp::codec::device::FREQDecompressor<int16_t,
		                                     1,
		                                     galp::codec::device::StatelessFREQExceptionPatcher<int16_t, 1, 1>,
		                                     galp::codec::device::FREQColumn<int16_t>>,
		    galp::codec::device::FREQColumn<int16_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && patcher == galp::format::Patcher::Stateful) {
		return galp::kernels::host::decompress_column<
		    int16_t,
		    1,
		    1,
		    galp::codec::device::FREQDecompressor<int16_t,
		                                     1,
		                                     galp::codec::device::StatefulFREQExceptionPatcher<int16_t, 1, 1>,
		                                     galp::codec::device::FREQColumn<int16_t>>,
		    galp::codec::device::FREQColumn<int16_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && patcher == galp::format::Patcher::Dummy) {
		return galp::kernels::host::decompress_column<
		    int16_t,
		    4,
		    1,
		    galp::codec::device::FREQDecompressor<int16_t,
		                                     4,
		                                     galp::codec::device::DummyFREQExceptionPatcher<int16_t, 4, 1>,
		                                     galp::codec::device::FREQColumn<int16_t>>,
		    galp::codec::device::FREQColumn<int16_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && patcher == galp::format::Patcher::Stateless) {
		return galp::kernels::host::decompress_column<
		    int16_t,
		    4,
		    1,
		    galp::codec::device::FREQDecompressor<int16_t,
		                                     4,
		                                     galp::codec::device::StatelessFREQExceptionPatcher<int16_t, 4, 1>,
		                                     galp::codec::device::FREQColumn<int16_t>>,
		    galp::codec::device::FREQColumn<int16_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && patcher == galp::format::Patcher::Stateful) {
		return galp::kernels::host::decompress_column<
		    int16_t,
		    4,
		    1,
		    galp::codec::device::FREQDecompressor<int16_t,
		                                     4,
		                                     galp::codec::device::StatefulFREQExceptionPatcher<int16_t, 4, 1>,
		                                     galp::codec::device::FREQColumn<int16_t>>,
		    galp::codec::device::FREQColumn<int16_t>>(column, n_samples);
	}
	throw std::invalid_argument("Could not find correct binding in decompress_column FREQ<int16_t>");
}

} // namespace galp::bench::bindings
