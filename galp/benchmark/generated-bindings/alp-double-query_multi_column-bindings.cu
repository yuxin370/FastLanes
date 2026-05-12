// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/benchmark/generated-bindings/alp-double-query_multi_column-bindings.cu
// ────────────────────────────────────────────────────────
#include "engine/kernels.cuh"
#include "engine/multi-column-host-kernels.cuh"
#include "generated-bindings/kernel-bindings.cuh"
#include <stdexcept>

namespace galp::bench::bindings {

template <>
bool query_multi_column<double, galp::codec::host::ALPColumn<double>>(const galp::codec::host::ALPColumn<double>& column,
                                                                 const unsigned                        unpack_n_vectors,
                                                                 const unsigned                        unpack_n_values,
                                                                 const galp::format::Unpacker                 unpacker,
                                                                 const galp::format::Patcher                  patcher,
                                                                 const double                          magic_value,
                                                                 const uint32_t                        n_samples) {

	if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::Stateful) {
		return galp::kernels::host::query_multi_column<
		    double,
		    1,
		    1,
		    galp::codec::device::ALPDecompressor<
		        double,
		        1,
		        galp::codec::device::BitUnpackerStatefulBranchless<double, 1, 1, galp::codec::device::ALPFunctor<double, 1>>,
		        galp::codec::device::StatefulALPExceptionPatcher<double, 1, 1>,
		        galp::codec::device::ALPColumn<double>>,
		    galp::codec::host::ALPColumn<double>>(column, magic_value, n_samples);
	}

	if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::Stateful) {
		return galp::kernels::host::query_multi_column<
		    double,
		    4,
		    1,
		    galp::codec::device::ALPDecompressor<
		        double,
		        4,
		        galp::codec::device::BitUnpackerStatefulBranchless<double, 4, 1, galp::codec::device::ALPFunctor<double, 4>>,
		        galp::codec::device::StatefulALPExceptionPatcher<double, 4, 1>,
		        galp::codec::device::ALPColumn<double>>,
		    galp::codec::host::ALPColumn<double>>(column, magic_value, n_samples);
	}
	throw std::invalid_argument("Could not find correct binding in query_multi_column ALP<double>");
}

} // namespace galp::bench::bindings
