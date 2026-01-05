
#include "engine/kernels.cuh"
#include "engine/multi-column-host-kernels.cuh"
#include "generated-bindings/kernel-bindings.cuh"
#include <stdexcept>

namespace bindings {

template <>
uint64_t*
decompress_column<uint64_t, flsgpu::device::FREQColumn<uint64_t>>(const flsgpu::device::FREQColumn<uint64_t> column,
                                                                  const unsigned        unpack_n_vectors,
                                                                  const unsigned        unpack_n_values,
                                                                  const enums::Unpacker unpacker,
                                                                  const enums::Patcher  patcher,
                                                                  const uint32_t        n_samples) {

	if (unpack_n_vectors == 1 && unpack_n_values == 1 && patcher == enums::Patcher::Dummy) {
		return kernels::host::decompress_column<
		    uint64_t,
		    1,
		    1,
		    flsgpu::device::FREQDecompressor<uint64_t,
		                                     1,
		                                     flsgpu::device::DummyFREQExceptionPatcher<uint64_t, 1, 1>,
		                                     flsgpu::device::FREQColumn<uint64_t>>,
		    flsgpu::device::FREQColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && patcher == enums::Patcher::Stateless) {
		return kernels::host::decompress_column<
		    uint64_t,
		    1,
		    1,
		    flsgpu::device::FREQDecompressor<uint64_t,
		                                     1,
		                                     flsgpu::device::StatelessFREQExceptionPatcher<uint64_t, 1, 1>,
		                                     flsgpu::device::FREQColumn<uint64_t>>,
		    flsgpu::device::FREQColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && patcher == enums::Patcher::Stateful) {
		return kernels::host::decompress_column<
		    uint64_t,
		    1,
		    1,
		    flsgpu::device::FREQDecompressor<uint64_t,
		                                     1,
		                                     flsgpu::device::StatefulFREQExceptionPatcher<uint64_t, 1, 1>,
		                                     flsgpu::device::FREQColumn<uint64_t>>,
		    flsgpu::device::FREQColumn<uint64_t>>(column, n_samples);
	}

	if (unpack_n_vectors == 4 && unpack_n_values == 1 && patcher == enums::Patcher::Dummy) {
		return kernels::host::decompress_column<
		    uint64_t,
		    4,
		    1,
		    flsgpu::device::FREQDecompressor<uint64_t,
		                                     4,
		                                     flsgpu::device::DummyFREQExceptionPatcher<uint64_t, 4, 1>,
		                                     flsgpu::device::FREQColumn<uint64_t>>,
		    flsgpu::device::FREQColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && patcher == enums::Patcher::Stateless) {
		return kernels::host::decompress_column<
		    uint64_t,
		    4,
		    1,
		    flsgpu::device::FREQDecompressor<uint64_t,
		                                     4,
		                                     flsgpu::device::StatelessFREQExceptionPatcher<uint64_t, 4, 1>,
		                                     flsgpu::device::FREQColumn<uint64_t>>,
		    flsgpu::device::FREQColumn<uint64_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && patcher == enums::Patcher::Stateful) {
		return kernels::host::decompress_column<
		    uint64_t,
		    4,
		    1,
		    flsgpu::device::FREQDecompressor<uint64_t,
		                                     4,
		                                     flsgpu::device::StatefulFREQExceptionPatcher<uint64_t, 4, 1>,
		                                     flsgpu::device::FREQColumn<uint64_t>>,
		    flsgpu::device::FREQColumn<uint64_t>>(column, n_samples);
	}
	throw std::invalid_argument("Could not find correct binding in decompress_column FREQ<uint64_t>");
}

} // namespace bindings
