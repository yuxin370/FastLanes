// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/benchmark/generated-bindings/freq-int16_t-decompress_column-bindings.cu
// ────────────────────────────────────────────────────────
#include "engine/kernels.cuh"
#include "engine/multi-column-host-kernels.cuh"
#include "generated-bindings/kernel-bindings.cuh"
#include <stdexcept>

namespace bindings {

template <>
int16_t*
decompress_column<int16_t, flsgpu::device::FREQColumn<int16_t>>(const flsgpu::device::FREQColumn<int16_t> column,
                                                                  const unsigned        unpack_n_vectors,
                                                                  const unsigned        unpack_n_values,
                                                                  const enums::Unpacker unpacker,
                                                                  const enums::Patcher  patcher,
                                                                  const enums::Expander expander,
                                                                  const uint32_t        n_samples) {
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && patcher == enums::Patcher::Dummy) {
		return kernels::host::decompress_column<
		    int16_t,
		    1,
		    1,
		    flsgpu::device::FREQDecompressor<int16_t,
		                                     1,
		                                     flsgpu::device::DummyFREQExceptionPatcher<int16_t, 1, 1>,
		                                     flsgpu::device::FREQColumn<int16_t>>,
		    flsgpu::device::FREQColumn<int16_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && patcher == enums::Patcher::Stateless) {
		return kernels::host::decompress_column<
		    int16_t,
		    1,
		    1,
		    flsgpu::device::FREQDecompressor<int16_t,
		                                     1,
		                                     flsgpu::device::StatelessFREQExceptionPatcher<int16_t, 1, 1>,
		                                     flsgpu::device::FREQColumn<int16_t>>,
		    flsgpu::device::FREQColumn<int16_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && patcher == enums::Patcher::Stateful) {
		return kernels::host::decompress_column<
		    int16_t,
		    1,
		    1,
		    flsgpu::device::FREQDecompressor<int16_t,
		                                     1,
		                                     flsgpu::device::StatefulFREQExceptionPatcher<int16_t, 1, 1>,
		                                     flsgpu::device::FREQColumn<int16_t>>,
		    flsgpu::device::FREQColumn<int16_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && patcher == enums::Patcher::Dummy) {
		return kernels::host::decompress_column<
		    int16_t,
		    4,
		    1,
		    flsgpu::device::FREQDecompressor<int16_t,
		                                     4,
		                                     flsgpu::device::DummyFREQExceptionPatcher<int16_t, 4, 1>,
		                                     flsgpu::device::FREQColumn<int16_t>>,
		    flsgpu::device::FREQColumn<int16_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && patcher == enums::Patcher::Stateless) {
		return kernels::host::decompress_column<
		    int16_t,
		    4,
		    1,
		    flsgpu::device::FREQDecompressor<int16_t,
		                                     4,
		                                     flsgpu::device::StatelessFREQExceptionPatcher<int16_t, 4, 1>,
		                                     flsgpu::device::FREQColumn<int16_t>>,
		    flsgpu::device::FREQColumn<int16_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && patcher == enums::Patcher::Stateful) {
		return kernels::host::decompress_column<
		    int16_t,
		    4,
		    1,
		    flsgpu::device::FREQDecompressor<int16_t,
		                                     4,
		                                     flsgpu::device::StatefulFREQExceptionPatcher<int16_t, 4, 1>,
		                                     flsgpu::device::FREQColumn<int16_t>>,
		    flsgpu::device::FREQColumn<int16_t>>(column, n_samples);
	}
	throw std::invalid_argument("Could not find correct binding in decompress_column FREQ<int16_t>");
}

} // namespace bindings
