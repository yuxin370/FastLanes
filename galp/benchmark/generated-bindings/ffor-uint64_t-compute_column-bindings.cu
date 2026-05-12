// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/benchmark/generated-bindings/ffor-uint64_t-compute_column-bindings.cu
// ────────────────────────────────────────────────────────
#include "engine/kernels.cuh"
#include "engine/multi-column-host-kernels.cuh"
#include "generated-bindings/kernel-bindings.cuh"
#include <stdexcept>

namespace galp::bench::bindings {

template <>
bool compute_column<uint64_t, galp::codec::device::FFORColumn<uint64_t>>(const galp::codec::device::FFORColumn<uint64_t> column,
                                                                    const unsigned        unpack_n_vectors,
                                                                    const unsigned        unpack_n_values,
                                                                    const galp::format::Unpacker unpacker,
                                                                    const galp::format::Patcher  patcher,
                                                                    const unsigned        n_repetitions,
                                                                    const uint32_t        n_samples) {
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::Dummy &&
	    patcher == galp::format::Patcher::None) {
		return galp::kernels::host::compute_column<
		    uint64_t,
		    1,
		    1,
		    galp::codec::device::FFORDecompressor<
		        uint64_t,
		        1,
		        galp::codec::device::BitUnpackerDummy<uint64_t, 1, 1, galp::codec::device::FFORFunctor<uint64_t, 1>>,
		        galp::codec::device::FFORColumn<uint64_t>>,
		    galp::codec::device::FFORColumn<uint64_t>,
		    10>(column, n_samples);
	}

	if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::None) {
		return galp::kernels::host::compute_column<
		    uint64_t,
		    1,
		    1,
		    galp::codec::device::FFORDecompressor<
		        uint64_t,
		        1,
		        galp::codec::device::BitUnpackerStatefulBranchless<uint64_t, 1, 1, galp::codec::device::FFORFunctor<uint64_t, 1>>,
		        galp::codec::device::FFORColumn<uint64_t>>,
		    galp::codec::device::FFORColumn<uint64_t>,
		    10>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::Dummy &&
	    patcher == galp::format::Patcher::None) {
		return galp::kernels::host::compute_column<
		    uint64_t,
		    4,
		    1,
		    galp::codec::device::FFORDecompressor<
		        uint64_t,
		        4,
		        galp::codec::device::BitUnpackerDummy<uint64_t, 4, 1, galp::codec::device::FFORFunctor<uint64_t, 4>>,
		        galp::codec::device::FFORColumn<uint64_t>>,
		    galp::codec::device::FFORColumn<uint64_t>,
		    10>(column, n_samples);
	}

	if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::None) {
		return galp::kernels::host::compute_column<
		    uint64_t,
		    4,
		    1,
		    galp::codec::device::FFORDecompressor<
		        uint64_t,
		        4,
		        galp::codec::device::BitUnpackerStatefulBranchless<uint64_t, 4, 1, galp::codec::device::FFORFunctor<uint64_t, 4>>,
		        galp::codec::device::FFORColumn<uint64_t>>,
		    galp::codec::device::FFORColumn<uint64_t>,
		    10>(column, n_samples);
	}
	throw std::invalid_argument("Could not find correct binding in compute_column FFOR<uint64_t>");
}

} // namespace galp::bench::bindings
