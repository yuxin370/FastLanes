// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/benchmark/generated-bindings/ffor-uint32_t-decompress_column-bindings.cu
// ────────────────────────────────────────────────────────
#include "engine/kernels.cuh"
#include "engine/multi-column-host-kernels.cuh"
#include "generated-bindings/kernel-bindings.cuh"
#include <stdexcept>

namespace galp::bench::bindings {

template <>
uint32_t*
decompress_column<uint32_t, galp::codec::device::FFORColumn<uint32_t>>(const galp::codec::device::FFORColumn<uint32_t> column,
                                                                  const unsigned        unpack_n_vectors,
                                                                  const unsigned        unpack_n_values,
                                                                  const galp::format::Unpacker unpacker,
                                                                  const galp::format::Patcher  patcher,
                                                                  const galp::format::Expander expander,
                                                                  const uint32_t        n_samples) {
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::Dummy &&
	    patcher == galp::format::Patcher::None) {
		return galp::kernels::host::decompress_column<
		    uint32_t,
		    1,
		    1,
		    galp::codec::device::FFORDecompressor<
		        uint32_t,
		        1,
		        galp::codec::device::BitUnpackerDummy<uint32_t, 1, 1, galp::codec::device::FFORFunctor<uint32_t, 1>>,
		        galp::codec::device::FFORColumn<uint32_t>>,
		    galp::codec::device::FFORColumn<uint32_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 32 && unpacker == galp::format::Unpacker::OldFls &&
	    patcher == galp::format::Patcher::None) {
		return galp::kernels::host::decompress_column<
		    uint32_t,
		    1,
		    32,
		    galp::codec::device::FFORDecompressor<
		        uint32_t,
		        1,
		        galp::codec::device::BitUnpackerOldFls<uint32_t, 1, 32, galp::codec::device::FFORFunctor<uint32_t, 1>>,
		        galp::codec::device::FFORColumn<uint32_t>>,
		    galp::codec::device::FFORColumn<uint32_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::None) {
		return galp::kernels::host::decompress_column<
		    uint32_t,
		    1,
		    1,
		    galp::codec::device::FFORDecompressor<
		        uint32_t,
		        1,
		        galp::codec::device::BitUnpackerStatefulBranchless<uint32_t, 1, 1, galp::codec::device::FFORFunctor<uint32_t, 1>>,
		        galp::codec::device::FFORColumn<uint32_t>>,
		    galp::codec::device::FFORColumn<uint32_t>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::Dummy &&
	    patcher == galp::format::Patcher::None) {
		return galp::kernels::host::decompress_column<
		    uint32_t,
		    4,
		    1,
		    galp::codec::device::FFORDecompressor<
		        uint32_t,
		        4,
		        galp::codec::device::BitUnpackerDummy<uint32_t, 4, 1, galp::codec::device::FFORFunctor<uint32_t, 4>>,
		        galp::codec::device::FFORColumn<uint32_t>>,
		    galp::codec::device::FFORColumn<uint32_t>>(column, n_samples);
	}

	if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::None) {
		return galp::kernels::host::decompress_column<
		    uint32_t,
		    4,
		    1,
		    galp::codec::device::FFORDecompressor<
		        uint32_t,
		        4,
		        galp::codec::device::BitUnpackerStatefulBranchless<uint32_t, 4, 1, galp::codec::device::FFORFunctor<uint32_t, 4>>,
		        galp::codec::device::FFORColumn<uint32_t>>,
		    galp::codec::device::FFORColumn<uint32_t>>(column, n_samples);
	}
	throw std::invalid_argument("Could not find correct binding in decompress_column FFOR<uint32_t>");
}

} // namespace galp::bench::bindings
