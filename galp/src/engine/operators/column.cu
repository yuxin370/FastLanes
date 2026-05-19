// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/operators/column.cu
// ────────────────────────────────────────────────────────
#include "core/data/value_store.cuh"
#include "engine/operators/column.cuh"
#include "engine/operators/column_traits.cuh"
#include "engine/operators/expr_ops.cuh"
#include "engine/unpack_dispatch.cuh"
#include "cuda/launch/dispatch.cuh"
#include "format/reader.cuh"
#include "codecs/device_ops.cuh"
#include <cstring>
#include <memory>
#include <type_traits>

namespace galp::execution {
namespace detail {

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
using BPUnpacker =
    galp::codec::device::BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, galp::codec::device::BPFunctor<T>>;

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
using FFORUnpacker = galp::codec::device::BitUnpackerStatefulBranchless<T,
                                                                   UNPACK_N_VECTORS,
                                                                   UNPACK_N_VALUES,
                                                                   galp::codec::device::FFORFunctor<T, UNPACK_N_VECTORS>>;

template <typename T,
          unsigned UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES,
          typename IndexT = typename galp::codec::utils::same_width_uint<T>::type>
using DICTUnpacker =
    galp::codec::device::BitUnpackerStatefulBranchless<T,
                                                  UNPACK_N_VECTORS,
                                                  UNPACK_N_VALUES,
                                                  galp::codec::device::DICTFunctor<T, UNPACK_N_VECTORS, IndexT>,
                                                  IndexT>;

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
using DefaultFREQPatcher = galp::codec::device::StatefulFREQExceptionPatcher<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>;

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
using DefaultCROSSRLEExpander = galp::codec::device::StatefulCROSSRLEExpander<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>;

template <typename ColumnT, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
auto decompress_device_tiled(const ColumnT& column) ->
    typename column_value_type<ColumnT>::type* {
	using T = typename column_value_type<ColumnT>::type;

	if constexpr (std::is_same_v<ColumnT, galp::codec::device::BPColumn<T>>) {
		using DecompressorT = galp::codec::device::BPDecompressor<T,
		                                                     UNPACK_N_VECTORS,
		                                                     BPUnpacker<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		                                                     galp::codec::device::BPColumn<T>>;
		return galp::kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(column,
		                                                                                                      1);
	} else if constexpr (std::is_same_v<ColumnT, galp::codec::device::CONSTANTColumn<T>>) {
		using DecompressorT = galp::codec::device::
		    CONSTANTDecompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, galp::codec::device::CONSTANTColumn<T>>;
		return galp::kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(column,
		                                                                                                      1);
	} else if constexpr (std::is_same_v<ColumnT, galp::codec::device::FFORColumn<T>>) {
		using DecompressorT = galp::codec::device::FFORDecompressor<T,
		                                                       UNPACK_N_VECTORS,
		                                                       FFORUnpacker<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		                                                       galp::codec::device::FFORColumn<T>>;
		return galp::kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(column,
		                                                                                                      1);
	} else if constexpr (std::is_same_v<ColumnT, galp::codec::device::DICTFFORColumn<T, uint8_t>>) {
		using IndexT     = uint8_t;
		using ProcessorT = galp::codec::device::DICTFunctor<T, UNPACK_N_VECTORS, IndexT>;
		using DecompressorT =
		    galp::codec::device::DICTDecompressor<T,
		                                     UNPACK_N_VECTORS,
		                                     DICTUnpacker<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, IndexT>,
		                                     galp::codec::device::DICTFFORColumn<T, IndexT>,
		                                     ProcessorT>;
		return galp::kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(column,
		                                                                                                      1);
	} else if constexpr (std::is_same_v<ColumnT, galp::codec::device::DICTFFORColumn<T, uint16_t>>) {
		using ProcessorT    = galp::codec::device::DICTFunctor<T, UNPACK_N_VECTORS>;
		using DecompressorT = galp::codec::device::DICTDecompressor<T,
		                                                       UNPACK_N_VECTORS,
		                                                       DICTUnpacker<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		                                                       galp::codec::device::DICTFFORColumn<T, uint16_t>,
		                                                       ProcessorT>;
		return galp::kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(column,
		                                                                                                      1);
	} else if constexpr (std::is_same_v<ColumnT, galp::codec::device::DICTSLPATCHColumn<T, uint8_t>>) {
		using IndexT     = uint8_t;
		using ProcessorT = galp::codec::device::DICTFunctor<T, UNPACK_N_VECTORS, IndexT>;
		using UnpackerT  = DICTUnpacker<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, IndexT>;
		using PatcherT =
		    galp::codec::device::StatefulSLPATCHDictExceptionPatcher<T, IndexT, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
		using DecompressorT = galp::codec::device::DICTSLPATCHDecompressor<T,
		                                                              UNPACK_N_VECTORS,
		                                                              UNPACK_N_VALUES,
		                                                              UnpackerT,
		                                                              PatcherT,
		                                                              galp::codec::device::DICTSLPATCHColumn<T, IndexT>,
		                                                              ProcessorT>;
		return galp::kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(column,
		                                                                                                      1);
	} else if constexpr (std::is_same_v<ColumnT, galp::codec::device::DICTSLPATCHColumn<T, uint16_t>>) {
		using IndexT     = uint16_t;
		using ProcessorT = galp::codec::device::DICTFunctor<T, UNPACK_N_VECTORS>;
		using UnpackerT =
		    galp::codec::device::BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, ProcessorT>;
		using PatcherT =
		    galp::codec::device::StatefulSLPATCHDictExceptionPatcher<T, IndexT, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
		using DecompressorT = galp::codec::device::DICTSLPATCHDecompressor<T,
		                                                              UNPACK_N_VECTORS,
		                                                              UNPACK_N_VALUES,
		                                                              UnpackerT,
		                                                              PatcherT,
		                                                              galp::codec::device::DICTSLPATCHColumn<T, uint16_t>,
		                                                              ProcessorT>;
		return galp::kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(column,
		                                                                                                      1);
	} else if constexpr (std::is_same_v<ColumnT, galp::codec::device::FREQColumn<T>>) {
		using DecompressorT = galp::codec::device::FREQDecompressor<T,
		                                                       UNPACK_N_VECTORS,
		                                                       DefaultFREQPatcher<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		                                                       galp::codec::device::FREQColumn<T>>;
		return galp::kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(column,
		                                                                                                      1);
	} else if constexpr (std::is_same_v<ColumnT, galp::codec::device::CROSSRLEColumn<T>>) {
		using DecompressorT =
		    galp::codec::device::CROSSRLEDecompressor<T,
		                                         UNPACK_N_VECTORS,
		                                         DefaultCROSSRLEExpander<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		                                         galp::codec::device::CROSSRLEColumn<T>>;
		return galp::kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(column,
		                                                                                                      1);
	} else if constexpr (std::is_same_v<ColumnT, galp::codec::device::SLPATCHColumn<T>>) {
		using DecompressorT = galp::codec::device::SLPATCHDecompressor<
		    T,
		    UNPACK_N_VECTORS,
		    galp::codec::device::BitUnpackerStatefulBranchless<T,
		                                                  UNPACK_N_VECTORS,
		                                                  UNPACK_N_VALUES,
		                                                  galp::codec::device::FFORFunctor<T, UNPACK_N_VECTORS>>,
		    galp::codec::device::StatefulSLPATCHExceptionPatcher<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		    galp::codec::device::SLPATCHColumn<T>>;
		return galp::kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(column,
		                                                                                                      1);
	} else if constexpr (std::is_same_v<ColumnT, galp::codec::device::RLEColumn<T, uint8_t>> ||
	                     std::is_same_v<ColumnT, galp::codec::device::RLEColumn<T, uint16_t>>) {
		using IndexT =
		    std::conditional_t<std::is_same_v<ColumnT, galp::codec::device::RLEColumn<T, uint8_t>>, uint8_t, uint16_t>;
		constexpr unsigned RLE_UNPACK_N_VALUES = galp::codec::utils::get_values_per_lane<IndexT>();
		using UnpackerT =
		    galp::codec::device::BitUnpackerStatefulBranchless<IndexT,
		                                                  UNPACK_N_VECTORS,
		                                                  RLE_UNPACK_N_VALUES,
		                                                  galp::codec::device::FFORFunctor<IndexT, UNPACK_N_VECTORS>>;
		using ExpanderT     = galp::codec::device::DummyRLEExpander<T, IndexT, UNPACK_N_VECTORS, RLE_UNPACK_N_VALUES>;
		using DecompressorT = galp::codec::device::RLEDecompressor<T,
		                                                      IndexT,
		                                                      UNPACK_N_VECTORS,
		                                                      RLE_UNPACK_N_VALUES,
		                                                      UnpackerT,
		                                                      ExpanderT,
		                                                      galp::codec::device::RLEColumn<T, IndexT>>;
		return galp::kernels::host::decompress_column<T, UNPACK_N_VECTORS, RLE_UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, 1);
	} else {
		static_assert(always_false_v<ColumnT>, "Unsupported column type for dispatch");
		return nullptr;
	}
}

template <typename ColumnT>
auto decompress_device(const ColumnT& column, const ExecutionConfig& cfg) ->
    typename column_value_type<ColumnT>::type* {
	return runtime::with_unpack_config(cfg, [&](auto unpack_n_vectors, auto unpack_n_values) {
		return decompress_device_tiled<ColumnT,
		                               decltype(unpack_n_vectors)::value,
		                               decltype(unpack_n_values)::value>(column);
	});
}

template <typename HostColT>
ValueStore decompress_common(const HostColT& host_col, const ExecutionConfig& cfg) {
	using T = typename ColumnKindTraits<HostColT>::value_type;

	auto device_col = host_col.copy_to_device();
	galp::memory::sync_h2d();
	auto* out = detail::decompress_device(device_col, cfg);
	galp::codec::host::free_column(device_col);

	return make_value_store<T>(out);
}

template <typename HostColT>
ValueStore decompress_host(const HostColT& host_col, const PlanKind plan, const ExecutionConfig& cfg) {
	using T = typename ColumnKindTraits<HostColT>::value_type;
	static_assert(is_supported_type_v<T>, "galp::execution::decompress only supports int8_t and int16_t columns");

	auto fail = []() -> ValueStore {
		throw std::runtime_error("dispatch plan/column mismatch");
	};

	switch (plan) {
	case PlanKind::UNCOMPRESSED: {
		if constexpr (std::is_same_v<HostColT, galp::codec::host::BPColumn<T>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::CONSTANT: {
		if constexpr (std::is_same_v<HostColT, galp::codec::host::CONSTANTColumn<T>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::FREQUENCY: {
		if constexpr (std::is_same_v<HostColT, galp::codec::host::FREQColumn<T>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::UNFFOR: {
		if constexpr (std::is_same_v<HostColT, galp::codec::host::FFORColumn<T>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::UNFFOR_SLPATCH: {
		if constexpr (std::is_same_v<HostColT, galp::codec::host::SLPATCHColumn<T>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::DICT_FFOR_U8:
	case PlanKind::DICT_FFOR_U16: {
		if constexpr (std::is_same_v<HostColT, galp::codec::host::DICTFFORColumn<T, uint8_t>> ||
		              std::is_same_v<HostColT, galp::codec::host::DICTFFORColumn<T, uint16_t>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::DICT_FFOR_SLPATCH_U8:
	case PlanKind::DICT_FFOR_SLPATCH_U16: {
		if constexpr (std::is_same_v<HostColT, galp::codec::host::DICTSLPATCHColumn<T, uint8_t>> ||
		              std::is_same_v<HostColT, galp::codec::host::DICTSLPATCHColumn<T, uint16_t>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::CROSS_RLE: {
		if constexpr (std::is_same_v<HostColT, galp::codec::host::CROSSRLEColumn<T>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::RLE_U8:
	case PlanKind::RLE_U16: {
		if constexpr (std::is_same_v<HostColT, galp::codec::host::RLEColumn<T, uint8_t>> ||
		              std::is_same_v<HostColT, galp::codec::host::RLEColumn<T, uint16_t>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	default:
		return fail();
	}
}

} // namespace detail

ValueStore decompress(const galp::expression::Expression& expression, const ExecutionConfig& cfg) {
	auto& col = *expression.column;
	return std::visit(
	    [&](auto&& host_col) -> ValueStore {
		    using HostColT  = std::decay_t<decltype(host_col)>;
		    const auto plan = detail::plan_for_host_col<HostColT>();
		    return detail::decompress_host(host_col, plan, cfg);
	    },
	    col.host);
}

template ValueStore
detail::decompress_host(const galp::codec::host::BPColumn<int8_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const galp::codec::host::FFORColumn<int8_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const galp::codec::host::DICTFFORColumn<int8_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const galp::codec::host::DICTSLPATCHColumn<int8_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const galp::codec::host::CONSTANTColumn<int8_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const galp::codec::host::FREQColumn<int8_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const galp::codec::host::SLPATCHColumn<int8_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const galp::codec::host::CROSSRLEColumn<int8_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const galp::codec::host::RLEColumn<int8_t, uint16_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const galp::codec::host::RLEColumn<int8_t, uint8_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const galp::codec::host::BPColumn<int16_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const galp::codec::host::FFORColumn<int16_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const galp::codec::host::DICTFFORColumn<int16_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const galp::codec::host::DICTFFORColumn<int16_t, uint8_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const galp::codec::host::DICTSLPATCHColumn<int16_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore detail::decompress_host(const galp::codec::host::DICTSLPATCHColumn<int16_t, uint8_t>&,
                                            const PlanKind,
                                            const ExecutionConfig&);
template ValueStore
detail::decompress_host(const galp::codec::host::SLPATCHColumn<int16_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const galp::codec::host::FREQColumn<int16_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const galp::codec::host::CROSSRLEColumn<int16_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const galp::codec::host::RLEColumn<int16_t, uint16_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const galp::codec::host::RLEColumn<int16_t, uint8_t>&, const PlanKind, const ExecutionConfig&);

} // namespace galp::execution
