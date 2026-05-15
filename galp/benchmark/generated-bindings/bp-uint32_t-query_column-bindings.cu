// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/benchmark/generated-bindings/bp-uint32_t-query_column-bindings.cu
// ────────────────────────────────────────────────────────
#include "engine/kernels/dispatch.cuh"
#include "galp_bench/generated/multi_column_host_kernels.cuh"
#include "generated-bindings/kernel-bindings.cuh"
#include <stdexcept>

namespace galp::bench::bindings {

template <>
bool query_column<uint32_t, galp::codec::device::BPColumn<uint32_t>>(const galp::codec::device::BPColumn<uint32_t> column,
                                                                const unsigned        unpack_n_vectors,
                                                                const unsigned        unpack_n_values,
                                                                const galp::format::Unpacker unpacker,
                                                                const galp::format::Patcher  patcher,
                                                                const uint32_t        magic_value,
                                                                const uint32_t        n_samples) {
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::Dummy &&
	    patcher == galp::format::Patcher::None) {
		return galp::kernels::host::query_column<
		    uint32_t,
		    1,
		    1,
		    galp::codec::device::BPDecompressor<
		        uint32_t,
		        1,
		        galp::codec::device::BitUnpackerDummy<uint32_t, 1, 1, galp::codec::device::BPFunctor<uint32_t>>,
		        galp::codec::device::BPColumn<uint32_t>>,
		    galp::codec::device::BPColumn<uint32_t>>(column, magic_value, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 32 && unpacker == galp::format::Unpacker::OldFls &&
	    patcher == galp::format::Patcher::None) {
		return galp::kernels::host::query_column<
		    uint32_t,
		    1,
		    32,
		    galp::codec::device::BPDecompressor<
		        uint32_t,
		        1,
		        galp::codec::device::BitUnpackerOldFls<uint32_t, 1, 32, galp::codec::device::BPFunctor<uint32_t>>,
		        galp::codec::device::BPColumn<uint32_t>>,
		    galp::codec::device::BPColumn<uint32_t>>(column, magic_value, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::None) {
		return galp::kernels::host::query_column<
		    uint32_t,
		    1,
		    1,
		    galp::codec::device::BPDecompressor<
		        uint32_t,
		        1,
		        galp::codec::device::BitUnpackerStatefulBranchless<uint32_t, 1, 1, galp::codec::device::BPFunctor<uint32_t>>,
		        galp::codec::device::BPColumn<uint32_t>>,
		    galp::codec::device::BPColumn<uint32_t>>(column, magic_value, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::Dummy &&
	    patcher == galp::format::Patcher::None) {
		return galp::kernels::host::query_column<
		    uint32_t,
		    4,
		    1,
		    galp::codec::device::BPDecompressor<
		        uint32_t,
		        4,
		        galp::codec::device::BitUnpackerDummy<uint32_t, 4, 1, galp::codec::device::BPFunctor<uint32_t>>,
		        galp::codec::device::BPColumn<uint32_t>>,
		    galp::codec::device::BPColumn<uint32_t>>(column, magic_value, n_samples);
	}

	if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::None) {
		return galp::kernels::host::query_column<
		    uint32_t,
		    4,
		    1,
		    galp::codec::device::BPDecompressor<
		        uint32_t,
		        4,
		        galp::codec::device::BitUnpackerStatefulBranchless<uint32_t, 4, 1, galp::codec::device::BPFunctor<uint32_t>>,
		        galp::codec::device::BPColumn<uint32_t>>,
		    galp::codec::device::BPColumn<uint32_t>>(column, magic_value, n_samples);
	}
	throw std::invalid_argument("Could not find correct binding in query_column BP<uint32_t>");
}

} // namespace galp::bench::bindings
