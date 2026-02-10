// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/dispatch/column.cu
// ────────────────────────────────────────────────────────
#include "engine/dispatch/column.cuh"
#include "engine/reader.cuh"
#include "flsgpu/fls.cuh"
#include <cstring>
#include <memory>

namespace dispatch {
namespace detail {

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
using BPUnpacker =
    flsgpu::device::BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, flsgpu::device::BPFunctor<T>>;

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
using FFORUnpacker = flsgpu::device::BitUnpackerStatefulBranchless<T,
                                                                   UNPACK_N_VECTORS,
                                                                   UNPACK_N_VALUES,
                                                                   flsgpu::device::FFORFunctor<T, UNPACK_N_VECTORS>>;

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
using DICTUnpacker = flsgpu::device::BitUnpackerStatefulBranchless<T,
                                                                   UNPACK_N_VECTORS,
                                                                   UNPACK_N_VALUES,
                                                                   flsgpu::device::DICTFunctor<T, UNPACK_N_VECTORS>>;

template <typename T, typename IndexT, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
using DICTUnpackerIdx =
    flsgpu::device::BitUnpackerStatefulBranchlessIdx<T,
                                                     IndexT,
                                                     UNPACK_N_VECTORS,
                                                     UNPACK_N_VALUES,
                                                     flsgpu::device::DICTFunctorIdx<T, IndexT, UNPACK_N_VECTORS>>;

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
using DefaultFREQPatcher = flsgpu::device::StatefulFREQExceptionPatcher<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>;

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
using DefaultCROSSRLEExpander = flsgpu::device::StatefulCROSSRLEExpander<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>;

template <typename ColumnT>
auto decompress_device(const ColumnT& column, const Config& cfg) -> typename column_value_type<ColumnT>::type* {
	using T = typename column_value_type<ColumnT>::type;

	// For now we only expose a single default configuration.
	constexpr unsigned UNPACK_N_VECTORS = 1;
	constexpr unsigned UNPACK_N_VALUES  = 1;
	(void)cfg;

	if constexpr (std::is_same_v<ColumnT, flsgpu::device::BPColumn<T>>) {
		using DecompressorT = flsgpu::device::BPDecompressor<T,
		                                                     UNPACK_N_VECTORS,
		                                                     BPUnpacker<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		                                                     flsgpu::device::BPColumn<T>>;
		return kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, cfg.n_samples);
	} else if constexpr (std::is_same_v<ColumnT, flsgpu::device::CONSTANTColumn<T>>) {
		using DecompressorT = flsgpu::device::
		    CONSTANTDecompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, flsgpu::device::CONSTANTColumn<T>>;
		return kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, cfg.n_samples);
	} else if constexpr (std::is_same_v<ColumnT, flsgpu::device::FFORColumn<T>>) {
		using DecompressorT = flsgpu::device::FFORDecompressor<T,
		                                                       UNPACK_N_VECTORS,
		                                                       FFORUnpacker<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		                                                       flsgpu::device::FFORColumn<T>>;
		return kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, cfg.n_samples);
	} else if constexpr (std::is_same_v<ColumnT, flsgpu::device::DICTFFORColumn<T, uint8_t>>) {
		using IndexT     = uint8_t;
		using ProcessorT = flsgpu::device::DICTFunctorIdx<T, IndexT, UNPACK_N_VECTORS>;
		using DecompressorT =
		    flsgpu::device::DICTDecompressor<T,
		                                     UNPACK_N_VECTORS,
		                                     DICTUnpackerIdx<T, IndexT, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		                                     flsgpu::device::DICTFFORColumn<T, IndexT>,
		                                     ProcessorT>;
		return kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, cfg.n_samples);
	} else if constexpr (std::is_same_v<ColumnT, flsgpu::device::DICTFFORColumn<T, uint16_t>>) {
		using ProcessorT    = flsgpu::device::DICTFunctor<T, UNPACK_N_VECTORS>;
		using DecompressorT = flsgpu::device::DICTDecompressor<T,
		                                                       UNPACK_N_VECTORS,
		                                                       DICTUnpacker<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		                                                       flsgpu::device::DICTFFORColumn<T, uint16_t>,
		                                                       ProcessorT>;
		return kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, cfg.n_samples);
	} else if constexpr (std::is_same_v<ColumnT, flsgpu::device::DICTSLPATCHColumn<T, uint8_t>>) {
		using IndexT     = uint8_t;
		using ProcessorT = flsgpu::device::DICTFunctorIdx<T, IndexT, UNPACK_N_VECTORS>;
		using UnpackerT  = DICTUnpackerIdx<T, IndexT, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
		using PatcherT =
		    flsgpu::device::StatefulSLPATCHDictExceptionPatcher<T, IndexT, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
		using DecompressorT = flsgpu::device::DICTSLPATCHDecompressor<T,
		                                                              UNPACK_N_VECTORS,
		                                                              UNPACK_N_VALUES,
		                                                              UnpackerT,
		                                                              PatcherT,
		                                                              flsgpu::device::DICTSLPATCHColumn<T, IndexT>,
		                                                              ProcessorT>;
		return kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, cfg.n_samples);
	} else if constexpr (std::is_same_v<ColumnT, flsgpu::device::DICTSLPATCHColumn<T, uint16_t>>) {
		using IndexT     = uint16_t;
		using ProcessorT = flsgpu::device::DICTFunctor<T, UNPACK_N_VECTORS>;
		using UnpackerT =
		    flsgpu::device::BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, ProcessorT>;
		using PatcherT =
		    flsgpu::device::StatefulSLPATCHDictExceptionPatcher<T, IndexT, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
		using DecompressorT = flsgpu::device::DICTSLPATCHDecompressor<T,
		                                                              UNPACK_N_VECTORS,
		                                                              UNPACK_N_VALUES,
		                                                              UnpackerT,
		                                                              PatcherT,
		                                                              flsgpu::device::DICTSLPATCHColumn<T, uint16_t>,
		                                                              ProcessorT>;
		return kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, cfg.n_samples);
	} else if constexpr (std::is_same_v<ColumnT, flsgpu::device::FREQColumn<T>>) {
		using DecompressorT = flsgpu::device::FREQDecompressor<T,
		                                                       UNPACK_N_VECTORS,
		                                                       DefaultFREQPatcher<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		                                                       flsgpu::device::FREQColumn<T>>;
		return kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, cfg.n_samples);
	} else if constexpr (std::is_same_v<ColumnT, flsgpu::device::CROSSRLEColumn<T>>) {
		using DecompressorT =
		    flsgpu::device::CROSSRLEDecompressor<T,
		                                         UNPACK_N_VECTORS,
		                                         DefaultCROSSRLEExpander<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		                                         flsgpu::device::CROSSRLEColumn<T>>;
		return kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, cfg.n_samples);
	} else if constexpr (std::is_same_v<ColumnT, flsgpu::device::SLPATCHColumn<T>>) {
		using DecompressorT = flsgpu::device::SLPATCHDecompressor<
		    T,
		    UNPACK_N_VECTORS,
		    flsgpu::device::BitUnpackerStatefulBranchless<T,
		                                                  UNPACK_N_VECTORS,
		                                                  UNPACK_N_VALUES,
		                                                  flsgpu::device::FFORFunctor<T, UNPACK_N_VECTORS>>,
		    flsgpu::device::StatefulSLPATCHExceptionPatcher<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		    flsgpu::device::SLPATCHColumn<T>>;
		return kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, cfg.n_samples);
	} else if constexpr (std::is_same_v<ColumnT, flsgpu::device::RLEColumn<T, uint16_t>>) {
		using UnpackerT =
		    flsgpu::device::BitUnpackerStatefulBranchless<uint16_t,
		                                                  UNPACK_N_VECTORS,
		                                                  UNPACK_N_VALUES,
		                                                  flsgpu::device::FFORFunctor<uint16_t, UNPACK_N_VECTORS>>;
		using DecompressorT = flsgpu::device::RLEDecompressor<T,
		                                                      uint16_t,
		                                                      UNPACK_N_VECTORS,
		                                                      UNPACK_N_VALUES,
		                                                      UnpackerT,
		                                                      flsgpu::device::RLEColumn<T, uint16_t>>;
		return kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, cfg.n_samples);
	} else {
		static_assert(always_false_v<ColumnT>, "Unsupported column type for dispatch");
		return nullptr;
	}
}

template <typename HostColT>
DecompressResult decompress_common(const HostColT& host_col, const Config& cfg) {
	using T = typename host_value_type<HostColT>::type;

	auto device_col = host_col.copy_to_device();
	flsgpu::memory::sync_h2d();
	auto* out = detail::decompress_device(device_col, cfg);
	flsgpu::host::free_column(device_col);

	return make_result<T>(out);
}

template <typename HostColT>
DecompressResult decompress_host(const HostColT& host_col, const PlanKind plan, const Config& cfg) {
	using T = typename host_value_type<HostColT>::type;
	static_assert(is_supported_type_v<T>, "dispatch::decompress only supports int8_t and int16_t columns");

	auto fail = []() -> DecompressResult {
		throw std::runtime_error("dispatch plan/column mismatch");
	};

	switch (plan) {
	case PlanKind::UNCOMPRESSED: {
		if constexpr (std::is_same_v<HostColT, flsgpu::host::BPColumn<T>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::CONSTANT: {
		if constexpr (std::is_same_v<HostColT, flsgpu::host::CONSTANTColumn<T>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::FREQUENCY: {
		if constexpr (std::is_same_v<HostColT, flsgpu::host::FREQColumn<T>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::UNFFOR: {
		if constexpr (std::is_same_v<HostColT, flsgpu::host::FFORColumn<T>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::UNFFOR_SLPATCH: {
		if constexpr (std::is_same_v<HostColT, flsgpu::host::SLPATCHColumn<T>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::DICT_FFOR: {
		if constexpr (std::is_same_v<HostColT, flsgpu::host::DICTFFORColumn<T, uint8_t>> ||
		              std::is_same_v<HostColT, flsgpu::host::DICTFFORColumn<T, uint16_t>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::DICT_FFOR_SLPATCH: {
		if constexpr (std::is_same_v<HostColT, flsgpu::host::DICTSLPATCHColumn<T, uint8_t>> ||
		              std::is_same_v<HostColT, flsgpu::host::DICTSLPATCHColumn<T, uint16_t>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::CROSS_RLE: {
		if constexpr (std::is_same_v<HostColT, flsgpu::host::CROSSRLEColumn<T>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::RLE: {
		if constexpr (std::is_same_v<HostColT, flsgpu::host::RLEColumn<T, uint16_t>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	default:
		return fail();
	}
}

} // namespace detail

DecompressResult decompress(const expr::Expression& expression, const Config& cfg) {
	auto& col = *expression.column;
	return std::visit(
	    [&](auto&& host_col) -> DecompressResult {
		    using HostColT = std::decay_t<decltype(host_col)>;
		    const auto plan =
		        col.skip_decompress ? detail::plan_for_host_col<HostColT>() : plan_for_ops(expression.ops);
		    return detail::decompress_host(host_col, plan, cfg);
	    },
	    col.host);
}

template DecompressResult detail::decompress_host(const flsgpu::host::BPColumn<int8_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::FFORColumn<int8_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::DICTFFORColumn<int8_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::DICTSLPATCHColumn<int8_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::CONSTANTColumn<int8_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::FREQColumn<int8_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::SLPATCHColumn<int8_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::CROSSRLEColumn<int8_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::RLEColumn<int8_t, uint16_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::BPColumn<int16_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::FFORColumn<int16_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::DICTFFORColumn<int16_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::DICTFFORColumn<int16_t, uint8_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::DICTSLPATCHColumn<int16_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::DICTSLPATCHColumn<int16_t, uint8_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::SLPATCHColumn<int16_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::FREQColumn<int16_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::RLEColumn<int16_t, uint16_t>&, const PlanKind, const Config&);

} // namespace dispatch
