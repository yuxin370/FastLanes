// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/benchmark/generated-bindings/alp-double-decompress_column-bindings.cu
// ────────────────────────────────────────────────────────
#include "engine/kernels/dispatch.cuh"
#include "galp_bench/generated/multi_column_host_kernels.cuh"
#include "generated-bindings/kernel-bindings.cuh"
#include <stdexcept>

namespace galp::bench::bindings {

template <>
double* decompress_column<double, galp::codec::device::ALPColumn<double>>(const galp::codec::device::ALPColumn<double> column,
                                                                     const unsigned        unpack_n_vectors,
                                                                     const unsigned        unpack_n_values,
                                                                     const galp::format::Unpacker unpacker,
                                                                     const galp::format::Patcher  patcher,
                                                                     const galp::format::Expander expander,
                                                                     const uint32_t        n_samples) {

	if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::Dummy) {
		return galp::kernels::host::decompress_column<
		    double,
		    1,
		    1,
		    galp::codec::device::ALPDecompressor<
		        double,
		        1,
		        galp::codec::device::BitUnpackerStatefulBranchless<double, 1, 1, galp::codec::device::ALPFunctor<double, 1>>,
		        galp::codec::device::DummyALPExceptionPatcher<double, 1, 1>,
		        galp::codec::device::ALPColumn<double>>,
		    galp::codec::device::ALPColumn<double>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::Stateless) {
		return galp::kernels::host::decompress_column<
		    double,
		    1,
		    1,
		    galp::codec::device::ALPDecompressor<
		        double,
		        1,
		        galp::codec::device::BitUnpackerStatefulBranchless<double, 1, 1, galp::codec::device::ALPFunctor<double, 1>>,
		        galp::codec::device::StatelessALPExceptionPatcher<double, 1, 1>,
		        galp::codec::device::ALPColumn<double>>,
		    galp::codec::device::ALPColumn<double>>(column, n_samples);
	}
	if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::Stateful) {
		return galp::kernels::host::decompress_column<
		    double,
		    1,
		    1,
		    galp::codec::device::ALPDecompressor<
		        double,
		        1,
		        galp::codec::device::BitUnpackerStatefulBranchless<double, 1, 1, galp::codec::device::ALPFunctor<double, 1>>,
		        galp::codec::device::StatefulALPExceptionPatcher<double, 1, 1>,
		        galp::codec::device::ALPColumn<double>>,
		    galp::codec::device::ALPColumn<double>>(column, n_samples);
	}

	if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::Dummy) {
		return galp::kernels::host::decompress_column<
		    double,
		    4,
		    1,
		    galp::codec::device::ALPDecompressor<
		        double,
		        4,
		        galp::codec::device::BitUnpackerStatefulBranchless<double, 4, 1, galp::codec::device::ALPFunctor<double, 4>>,
		        galp::codec::device::DummyALPExceptionPatcher<double, 4, 1>,
		        galp::codec::device::ALPColumn<double>>,
		    galp::codec::device::ALPColumn<double>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::Stateless) {
		return galp::kernels::host::decompress_column<
		    double,
		    4,
		    1,
		    galp::codec::device::ALPDecompressor<
		        double,
		        4,
		        galp::codec::device::BitUnpackerStatefulBranchless<double, 4, 1, galp::codec::device::ALPFunctor<double, 4>>,
		        galp::codec::device::StatelessALPExceptionPatcher<double, 4, 1>,
		        galp::codec::device::ALPColumn<double>>,
		    galp::codec::device::ALPColumn<double>>(column, n_samples);
	}
	if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == galp::format::Unpacker::StatefulBranchless &&
	    patcher == galp::format::Patcher::Stateful) {
		return galp::kernels::host::decompress_column<
		    double,
		    4,
		    1,
		    galp::codec::device::ALPDecompressor<
		        double,
		        4,
		        galp::codec::device::BitUnpackerStatefulBranchless<double, 4, 1, galp::codec::device::ALPFunctor<double, 4>>,
		        galp::codec::device::StatefulALPExceptionPatcher<double, 4, 1>,
		        galp::codec::device::ALPColumn<double>>,
		    galp::codec::device::ALPColumn<double>>(column, n_samples);
	}
	throw std::invalid_argument("Could not find correct binding in decompress_column ALP<double>");
}

} // namespace galp::bench::bindings
