// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// fastlens/src/engine/operators/column.cu
// ────────────────────────────────────────────────────────
#include "codecs/device_ops.cuh"
#include "core/data/value_store.cuh"
#include "cuda/launch/dispatch.cuh"
#include "engine/operators/column.cuh"
#include "engine/operators/column_traits.cuh"
#include "engine/operators/expr_ops.cuh"
#include "engine/unpack_dispatch.cuh"
#include "format/reader.cuh"
#include <cstring>
#include <memory>
#include <type_traits>

namespace fastlens::execution {
namespace detail {

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
using BPUnpacker = fastlens::codec::device::
    BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, fastlens::codec::device::BPFunctor<T>>;

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
using FFORUnpacker =
    fastlens::codec::device::BitUnpackerStatefulBranchless<T,
                                                       UNPACK_N_VECTORS,
                                                       UNPACK_N_VALUES,
                                                       fastlens::codec::device::FFORFunctor<T, UNPACK_N_VECTORS>>;

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
using DELTAUnpacker = fastlens::codec::device::BitUnpackerStatefulBranchless<
    typename fastlens::codec::utils::same_width_uint<T>::type,
    UNPACK_N_VECTORS,
    UNPACK_N_VALUES,
    fastlens::codec::device::FFORFunctor<typename fastlens::codec::utils::same_width_uint<T>::type, UNPACK_N_VECTORS>>;

template <typename T,
          unsigned UNPACK_N_VECTORS,
          unsigned UNPACK_N_VALUES,
          typename IndexT = typename fastlens::codec::utils::same_width_uint<T>::type>
using DICTUnpacker =
    fastlens::codec::device::BitUnpackerStatefulBranchless<T,
                                                       UNPACK_N_VECTORS,
                                                       UNPACK_N_VALUES,
                                                       fastlens::codec::device::DICTFunctor<T, UNPACK_N_VECTORS, IndexT>,
                                                       IndexT>;

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
using DefaultFREQPatcher = fastlens::codec::device::StatefulFREQExceptionPatcher<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>;

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
using DefaultCROSSRLEExpander = fastlens::codec::device::StatefulCROSSRLEExpander<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>;

template <typename ColumnT, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
auto decompress_device_tiled(const ColumnT& column) -> typename column_value_type<ColumnT>::type* {
	using T = typename column_value_type<ColumnT>::type;

	if constexpr (std::is_same_v<ColumnT, fastlens::codec::device::BPColumn<T>>) {
		using DecompressorT = fastlens::codec::device::BPDecompressor<T,
		                                                          UNPACK_N_VECTORS,
		                                                          BPUnpacker<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		                                                          fastlens::codec::device::BPColumn<T>>;
		return fastlens::kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, 1);
	} else if constexpr (std::is_same_v<ColumnT, fastlens::codec::device::CONSTANTColumn<T>>) {
		using DecompressorT = fastlens::codec::device::
		    CONSTANTDecompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, fastlens::codec::device::CONSTANTColumn<T>>;
		return fastlens::kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, 1);
	} else if constexpr (std::is_same_v<ColumnT, fastlens::codec::device::DELTAColumn<T>>) {
		using DecompressorT =
		    fastlens::codec::device::DELTADecompressor<T,
		                                           UNPACK_N_VECTORS,
		                                           DELTAUnpacker<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		                                           fastlens::codec::device::DELTAColumn<T>>;
		return fastlens::kernels::host::decompress_column<T,
		                                              UNPACK_N_VECTORS,
		                                              UNPACK_N_VALUES,
		                                              DecompressorT,
		                                              ColumnT,
		                                              fastlens::codec::device::FastLanes1024InputUntransposer>(column, 1);
	} else if constexpr (std::is_same_v<ColumnT, fastlens::codec::device::FFORColumn<T>>) {
		using DecompressorT = fastlens::codec::device::FFORDecompressor<T,
		                                                            UNPACK_N_VECTORS,
		                                                            FFORUnpacker<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		                                                            fastlens::codec::device::FFORColumn<T>>;
		return fastlens::kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, 1);
	} else if constexpr (std::is_same_v<ColumnT, fastlens::codec::device::DICTFFORColumn<T, uint8_t>>) {
		using IndexT     = uint8_t;
		using ProcessorT = fastlens::codec::device::DICTFunctor<T, UNPACK_N_VECTORS, IndexT>;
		using DecompressorT =
		    fastlens::codec::device::DICTDecompressor<T,
		                                          UNPACK_N_VECTORS,
		                                          DICTUnpacker<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, IndexT>,
		                                          fastlens::codec::device::DICTFFORColumn<T, IndexT>,
		                                          ProcessorT>;
		return fastlens::kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, 1);
	} else if constexpr (std::is_same_v<ColumnT, fastlens::codec::device::DICTFFORColumn<T, uint16_t>>) {
		using ProcessorT    = fastlens::codec::device::DICTFunctor<T, UNPACK_N_VECTORS>;
		using DecompressorT = fastlens::codec::device::DICTDecompressor<T,
		                                                            UNPACK_N_VECTORS,
		                                                            DICTUnpacker<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		                                                            fastlens::codec::device::DICTFFORColumn<T, uint16_t>,
		                                                            ProcessorT>;
		return fastlens::kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, 1);
	} else if constexpr (std::is_same_v<ColumnT, fastlens::codec::device::DICTSLPATCHColumn<T, uint8_t>>) {
		using IndexT     = uint8_t;
		using ProcessorT = fastlens::codec::device::DICTFunctor<T, UNPACK_N_VECTORS, IndexT>;
		using UnpackerT  = DICTUnpacker<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, IndexT>;
		using PatcherT =
		    fastlens::codec::device::StatefulSLPATCHDictExceptionPatcher<T, IndexT, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
		using DecompressorT =
		    fastlens::codec::device::DICTSLPATCHDecompressor<T,
		                                                 UNPACK_N_VECTORS,
		                                                 UNPACK_N_VALUES,
		                                                 UnpackerT,
		                                                 PatcherT,
		                                                 fastlens::codec::device::DICTSLPATCHColumn<T, IndexT>,
		                                                 ProcessorT>;
		return fastlens::kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, 1);
	} else if constexpr (std::is_same_v<ColumnT, fastlens::codec::device::DICTSLPATCHColumn<T, uint16_t>>) {
		using IndexT     = uint16_t;
		using ProcessorT = fastlens::codec::device::DICTFunctor<T, UNPACK_N_VECTORS>;
		using UnpackerT =
		    fastlens::codec::device::BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, ProcessorT>;
		using PatcherT =
		    fastlens::codec::device::StatefulSLPATCHDictExceptionPatcher<T, IndexT, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
		using DecompressorT =
		    fastlens::codec::device::DICTSLPATCHDecompressor<T,
		                                                 UNPACK_N_VECTORS,
		                                                 UNPACK_N_VALUES,
		                                                 UnpackerT,
		                                                 PatcherT,
		                                                 fastlens::codec::device::DICTSLPATCHColumn<T, uint16_t>,
		                                                 ProcessorT>;
		return fastlens::kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, 1);
	} else if constexpr (std::is_same_v<ColumnT, fastlens::codec::device::FREQColumn<T>>) {
		using DecompressorT =
		    fastlens::codec::device::FREQDecompressor<T,
		                                          UNPACK_N_VECTORS,
		                                          DefaultFREQPatcher<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		                                          fastlens::codec::device::FREQColumn<T>>;
		return fastlens::kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, 1);
	} else if constexpr (std::is_same_v<ColumnT, fastlens::codec::device::CROSSRLEColumn<T>>) {
		using DecompressorT =
		    fastlens::codec::device::CROSSRLEDecompressor<T,
		                                              UNPACK_N_VECTORS,
		                                              DefaultCROSSRLEExpander<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		                                              fastlens::codec::device::CROSSRLEColumn<T>>;
		return fastlens::kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, 1);
	} else if constexpr (std::is_same_v<ColumnT, fastlens::codec::device::SLPATCHColumn<T>>) {
		using DecompressorT = fastlens::codec::device::SLPATCHDecompressor<
		    T,
		    UNPACK_N_VECTORS,
		    fastlens::codec::device::BitUnpackerStatefulBranchless<T,
		                                                       UNPACK_N_VECTORS,
		                                                       UNPACK_N_VALUES,
		                                                       fastlens::codec::device::FFORFunctor<T, UNPACK_N_VECTORS>>,
		    fastlens::codec::device::StatefulSLPATCHExceptionPatcher<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		    fastlens::codec::device::SLPATCHColumn<T>>;
		return fastlens::kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, 1);
	} else if constexpr (std::is_same_v<ColumnT, fastlens::codec::device::RLEColumn<T, uint8_t>> ||
	                     std::is_same_v<ColumnT, fastlens::codec::device::RLEColumn<T, uint16_t>>) {
		using IndexT =
		    std::conditional_t<std::is_same_v<ColumnT, fastlens::codec::device::RLEColumn<T, uint8_t>>, uint8_t, uint16_t>;
		constexpr unsigned RLE_UNPACK_N_VALUES = fastlens::codec::utils::get_values_per_lane<IndexT>();
		using UnpackerT                        = fastlens::codec::device::BitUnpackerStatefulBranchless<
		                           IndexT,
		                           UNPACK_N_VECTORS,
		                           RLE_UNPACK_N_VALUES,
		                           fastlens::codec::device::FFORFunctor<IndexT, UNPACK_N_VECTORS>>;
		using ExpanderT     = fastlens::codec::device::DummyRLEExpander<T, IndexT, UNPACK_N_VECTORS, RLE_UNPACK_N_VALUES>;
		using DecompressorT = fastlens::codec::device::RLEDecompressor<T,
		                                                           IndexT,
		                                                           UNPACK_N_VECTORS,
		                                                           RLE_UNPACK_N_VALUES,
		                                                           UnpackerT,
		                                                           ExpanderT,
		                                                           fastlens::codec::device::RLEColumn<T, IndexT>>;
		return fastlens::kernels::host::decompress_column<T, UNPACK_N_VECTORS, RLE_UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, 1);
	} else if constexpr (std::is_same_v<ColumnT, fastlens::codec::device::RLESLPATCHColumn<T, uint16_t>>) {
		using IndexT                           = uint16_t;
		constexpr unsigned RLE_UNPACK_N_VALUES = fastlens::codec::utils::get_values_per_lane<IndexT>();
		using UnpackerT                        = fastlens::codec::device::BitUnpackerStatefulBranchless<
		                           IndexT,
		                           UNPACK_N_VECTORS,
		                           RLE_UNPACK_N_VALUES,
		                           fastlens::codec::device::FFORFunctor<IndexT, UNPACK_N_VECTORS>>;
		using PatcherT =
		    fastlens::codec::device::StatefulSLPATCHExceptionPatcher<IndexT, UNPACK_N_VECTORS, RLE_UNPACK_N_VALUES>;
		using ExpanderT     = fastlens::codec::device::DummyRLEExpander<T, IndexT, UNPACK_N_VECTORS, RLE_UNPACK_N_VALUES>;
		using DecompressorT = fastlens::codec::device::RLESLPATCHDecompressor<T,
		                                                                  IndexT,
		                                                                  UNPACK_N_VECTORS,
		                                                                  RLE_UNPACK_N_VALUES,
		                                                                  UnpackerT,
		                                                                  PatcherT,
		                                                                  ExpanderT,
		                                                                  ColumnT>;
		return fastlens::kernels::host::decompress_column<T, UNPACK_N_VECTORS, RLE_UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, 1);
	} else {
		static_assert(always_false_v<ColumnT>, "Unsupported column type for dispatch");
		return nullptr;
	}
}

template <typename ColumnT>
auto decompress_device(const ColumnT& column, const ExecutionConfig& cfg) ->
    typename column_value_type<ColumnT>::type* {
	using T = typename column_value_type<ColumnT>::type;
	if constexpr (std::is_same_v<ColumnT, fastlens::codec::device::DELTAColumn<T>>) {
		if (delta_uses_register(cfg.delta_decoder)) {
			return runtime::with_unpack_config(cfg, [&](auto unpack_n_vectors, [[maybe_unused]] auto unpack_n_values) {
				constexpr unsigned UNPACK_N_VECTORS  = decltype(unpack_n_vectors)::value;
				constexpr unsigned REGISTER_N_VALUES = fastlens::codec::utils::get_values_per_lane<T>();
				using UIntT                          = typename fastlens::codec::utils::same_width_uint<T>::type;
				using UnpackerT                      = fastlens::codec::device::BitUnpackerLaneTile<
				                         UIntT,
				                         UNPACK_N_VECTORS,
				                         REGISTER_N_VALUES,
				                         fastlens::codec::device::FFORFunctor<UIntT, UNPACK_N_VECTORS>>;
				using DecompressorT = fastlens::codec::device::
				    DELTARegisterDecompressor<T, UNPACK_N_VECTORS, UnpackerT, fastlens::codec::device::DELTAColumn<T>>;
				return fastlens::kernels::host::decompress_delta_register_column<
				    T,
				    UNPACK_N_VECTORS,
				    DecompressorT,
				    ColumnT,
				    fastlens::codec::device::FastLanes1024InputUntransposer>(
				    column, 1);
			});
		}
	}
	return runtime::with_unpack_config(cfg, [&](auto unpack_n_vectors, auto unpack_n_values) {
		return decompress_device_tiled<ColumnT, decltype(unpack_n_vectors)::value, decltype(unpack_n_values)::value>(
		    column);
	});
}

template <typename HostColT>
ValueStore decompress_common(const HostColT& host_col, const ExecutionConfig& cfg) {
	using T = typename ColumnKindTraits<HostColT>::value_type;

	auto device_col = host_col.copy_to_device();
	fastlens::memory::sync_h2d();
	auto* out = detail::decompress_device(device_col, cfg);
	fastlens::codec::host::free_column(device_col);

	return make_value_store<T>(out);
}

template <typename HostColT>
ValueStore decompress_host(const HostColT& host_col, const PlanKind plan, const ExecutionConfig& cfg) {
	using T = typename ColumnKindTraits<HostColT>::value_type;
	static_assert(is_supported_type_v<T>, "fastlens::execution::decompress only supports int8_t and int16_t columns");

	auto fail = []() -> ValueStore {
		throw std::runtime_error("dispatch plan/column mismatch");
	};

	switch (plan) {
	case PlanKind::UNCOMPRESSED: {
		if constexpr (std::is_same_v<HostColT, fastlens::codec::host::BPColumn<T>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::CONSTANT: {
		if constexpr (std::is_same_v<HostColT, fastlens::codec::host::CONSTANTColumn<T>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::DELTA: {
		if constexpr (std::is_same_v<HostColT, fastlens::codec::host::DELTAColumn<T>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::FREQUENCY: {
		if constexpr (std::is_same_v<HostColT, fastlens::codec::host::FREQColumn<T>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::UNFFOR: {
		if constexpr (std::is_same_v<HostColT, fastlens::codec::host::FFORColumn<T>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::UNFFOR_SLPATCH: {
		if constexpr (std::is_same_v<HostColT, fastlens::codec::host::SLPATCHColumn<T>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::DICT_FFOR_U8:
	case PlanKind::DICT_FFOR_U16: {
		if constexpr (std::is_same_v<HostColT, fastlens::codec::host::DICTFFORColumn<T, uint8_t>> ||
		              std::is_same_v<HostColT, fastlens::codec::host::DICTFFORColumn<T, uint16_t>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::DICT_FFOR_SLPATCH_U8:
	case PlanKind::DICT_FFOR_SLPATCH_U16: {
		if constexpr (std::is_same_v<HostColT, fastlens::codec::host::DICTSLPATCHColumn<T, uint8_t>> ||
		              std::is_same_v<HostColT, fastlens::codec::host::DICTSLPATCHColumn<T, uint16_t>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::CROSS_RLE: {
		if constexpr (std::is_same_v<HostColT, fastlens::codec::host::CROSSRLEColumn<T>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::RLE_U8:
	case PlanKind::RLE_U16: {
		if constexpr (std::is_same_v<HostColT, fastlens::codec::host::RLEColumn<T, uint8_t>> ||
		              std::is_same_v<HostColT, fastlens::codec::host::RLEColumn<T, uint16_t>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::RLE_SLPATCH_U16: {
		if constexpr (std::is_same_v<HostColT, fastlens::codec::host::RLESLPATCHColumn<T, uint16_t>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	default:
		return fail();
	}
}

} // namespace detail

ValueStore decompress(const fastlens::expression::Expression& expression, const ExecutionConfig& cfg) {
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
detail::decompress_host(const fastlens::codec::host::BPColumn<int8_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const fastlens::codec::host::FFORColumn<int8_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const fastlens::codec::host::DICTFFORColumn<int8_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const fastlens::codec::host::DICTSLPATCHColumn<int8_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const fastlens::codec::host::CONSTANTColumn<int8_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const fastlens::codec::host::DELTAColumn<int8_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const fastlens::codec::host::FREQColumn<int8_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const fastlens::codec::host::SLPATCHColumn<int8_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const fastlens::codec::host::CROSSRLEColumn<int8_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const fastlens::codec::host::RLEColumn<int8_t, uint16_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const fastlens::codec::host::RLEColumn<int8_t, uint8_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore detail::decompress_host(const fastlens::codec::host::RLESLPATCHColumn<int8_t, uint16_t>&,
                                            const PlanKind,
                                            const ExecutionConfig&);
template ValueStore
detail::decompress_host(const fastlens::codec::host::BPColumn<int16_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const fastlens::codec::host::FFORColumn<int16_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const fastlens::codec::host::CONSTANTColumn<int16_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const fastlens::codec::host::DELTAColumn<int16_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const fastlens::codec::host::DICTFFORColumn<int16_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore detail::decompress_host(const fastlens::codec::host::DICTFFORColumn<int16_t, uint8_t>&,
                                            const PlanKind,
                                            const ExecutionConfig&);
template ValueStore
detail::decompress_host(const fastlens::codec::host::DICTSLPATCHColumn<int16_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore detail::decompress_host(const fastlens::codec::host::DICTSLPATCHColumn<int16_t, uint8_t>&,
                                            const PlanKind,
                                            const ExecutionConfig&);
template ValueStore
detail::decompress_host(const fastlens::codec::host::SLPATCHColumn<int16_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const fastlens::codec::host::FREQColumn<int16_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const fastlens::codec::host::CROSSRLEColumn<int16_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const fastlens::codec::host::RLEColumn<int16_t, uint16_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore
detail::decompress_host(const fastlens::codec::host::RLEColumn<int16_t, uint8_t>&, const PlanKind, const ExecutionConfig&);
template ValueStore detail::decompress_host(const fastlens::codec::host::RLESLPATCHColumn<int16_t, uint16_t>&,
                                            const PlanKind,
                                            const ExecutionConfig&);

} // namespace fastlens::execution
