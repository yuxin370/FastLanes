// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/operators/column_traits.cuh
// ────────────────────────────────────────────────────────
// Single source of truth for per-family host-column metadata. Each host column
// type (BPColumn, FFORColumn, …) maps to its value type, PlanKind tag, and the
// routine that packs its payload into a DeviceExpression's union slot.
//
// Before this trait, `value_type`, `plan_kind`, and the fill switch lived in
// three parallel tables that had to be updated in lockstep whenever a new
// column family was added.
#ifndef ENGINE_EXECUTION_COLUMN_TRAITS_CUH
#define ENGINE_EXECUTION_COLUMN_TRAITS_CUH

#include "core/expression.cuh"
#include "cuda/memory/device_arena.cuh"
#include "codecs/encodings/all.cuh"
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace galp::execution {

template <typename HostColT>
struct ColumnKindTraits;

template <typename T>
struct ColumnKindTraits<galp::codec::host::BPColumn<T>> {
	using value_type                    = T;
	static constexpr PlanKind plan_kind = PlanKind::UNCOMPRESSED;
	static void fill(DeviceExpression<T>&                expr,
	                 const galp::codec::host::BPColumn<T>&    host_col,
	                 galp::memory::DeviceArena&        arena,
	                 [[maybe_unused]] bool               freq_use_extended) {
		host_col.copy_to_device(arena, expr.col.bp);
	}
};

template <typename T>
struct ColumnKindTraits<galp::codec::host::CONSTANTColumn<T>> {
	using value_type                    = T;
	static constexpr PlanKind plan_kind = PlanKind::CONSTANT;
	static void fill(DeviceExpression<T>&                   expr,
	                 const galp::codec::host::CONSTANTColumn<T>& host_col,
	                 galp::memory::DeviceArena&           arena,
	                 [[maybe_unused]] bool                  freq_use_extended) {
		host_col.copy_to_device(arena, expr.col.constant);
	}
};

template <typename T>
struct ColumnKindTraits<galp::codec::host::FFORColumn<T>> {
	using value_type                    = T;
	static constexpr PlanKind plan_kind = PlanKind::UNFFOR;
	static void fill(DeviceExpression<T>&                expr,
	                 const galp::codec::host::FFORColumn<T>&  host_col,
	                 galp::memory::DeviceArena&        arena,
	                 [[maybe_unused]] bool               freq_use_extended) {
		host_col.copy_to_device(arena, expr.col.ffor);
	}
};

template <typename T>
struct ColumnKindTraits<galp::codec::host::SLPATCHColumn<T>> {
	using value_type                    = T;
	static constexpr PlanKind plan_kind = PlanKind::UNFFOR_SLPATCH;
	static void fill(DeviceExpression<T>&                  expr,
	                 const galp::codec::host::SLPATCHColumn<T>& host_col,
	                 galp::memory::DeviceArena&          arena,
	                 [[maybe_unused]] bool                 freq_use_extended) {
		host_col.copy_to_device(arena, expr.col.slpatch);
	}
};

template <typename T>
struct ColumnKindTraits<galp::codec::host::FREQColumn<T>> {
	using value_type                    = T;
	static constexpr PlanKind plan_kind = PlanKind::FREQUENCY;
	static void fill(DeviceExpression<T>&                expr,
	                 const galp::codec::host::FREQColumn<T>&  host_col,
	                 galp::memory::DeviceArena&        arena,
	                 bool                                freq_use_extended) {
		if (freq_use_extended) {
			auto extended = host_col.create_extended_column();
			auto owned_extended = std::make_shared<decltype(extended)>(std::move(extended));
			owned_extended->copy_to_device(arena, expr.col.freq_extended);
			expr.freq_use_extended = true;
			arena.defer_free([owned_extended]() {});
			return;
		}
		host_col.copy_to_device(arena, expr.col.freq);
		expr.freq_use_extended = false;
	}
};

template <typename T>
struct ColumnKindTraits<galp::codec::host::CROSSRLEColumn<T>> {
	using value_type                    = T;
	static constexpr PlanKind plan_kind = PlanKind::CROSS_RLE;
	static void fill(DeviceExpression<T>&                    expr,
	                 const galp::codec::host::CROSSRLEColumn<T>&  host_col,
	                 galp::memory::DeviceArena&            arena,
	                 [[maybe_unused]] bool                   freq_use_extended) {
		host_col.copy_to_device(arena, expr.col.crossrle);
	}
};

template <typename T, typename IndexT>
struct ColumnKindTraits<galp::codec::host::DICTFFORColumn<T, IndexT>> {
	using value_type                    = T;
	static constexpr PlanKind plan_kind =
	    std::is_same_v<IndexT, uint8_t> ? PlanKind::DICT_FFOR_U8 : PlanKind::DICT_FFOR_U16;
	static void fill(DeviceExpression<T>&                               expr,
	                 const galp::codec::host::DICTFFORColumn<T, IndexT>&     host_col,
	                 galp::memory::DeviceArena&                       arena,
	                 [[maybe_unused]] bool                              freq_use_extended) {
		if constexpr (std::is_same_v<IndexT, uint8_t>) {
			host_col.copy_to_device(arena, expr.col.dictffor_u8);
		} else {
			host_col.copy_to_device(arena, expr.col.dictffor_u16);
		}
	}
};

template <typename T, typename IndexT>
struct ColumnKindTraits<galp::codec::host::DICTSLPATCHColumn<T, IndexT>> {
	using value_type                    = T;
	static constexpr PlanKind plan_kind =
	    std::is_same_v<IndexT, uint8_t> ? PlanKind::DICT_FFOR_SLPATCH_U8 : PlanKind::DICT_FFOR_SLPATCH_U16;
	static void fill(DeviceExpression<T>&                                  expr,
	                 const galp::codec::host::DICTSLPATCHColumn<T, IndexT>&     host_col,
	                 galp::memory::DeviceArena&                          arena,
	                 [[maybe_unused]] bool                                 freq_use_extended) {
		if constexpr (std::is_same_v<IndexT, uint8_t>) {
			host_col.copy_to_device(arena, expr.col.dictslpatch_u8);
		} else {
			host_col.copy_to_device(arena, expr.col.dictslpatch_u16);
		}
	}
};

template <typename T, typename IndexT>
struct ColumnKindTraits<galp::codec::host::RLEColumn<T, IndexT>> {
	using value_type                    = T;
	static constexpr PlanKind plan_kind =
	    std::is_same_v<IndexT, uint8_t> ? PlanKind::RLE_U8 : PlanKind::RLE_U16;
	static void fill(DeviceExpression<T>&                          expr,
	                 const galp::codec::host::RLEColumn<T, IndexT>&     host_col,
	                 galp::memory::DeviceArena&                  arena,
	                 [[maybe_unused]] bool                         freq_use_extended) {
		if constexpr (std::is_same_v<IndexT, uint8_t>) {
			host_col.copy_to_device(arena, expr.col.rle_u8);
		} else {
			host_col.copy_to_device(arena, expr.col.rle_u16);
		}
	}
};

template <typename T, typename IndexT>
struct ColumnKindTraits<galp::codec::host::RLESLPATCHColumn<T, IndexT>> {
	using value_type                    = T;
	static constexpr PlanKind plan_kind = PlanKind::RLE_SLPATCH_U16;
	static void fill(DeviceExpression<T>&                                 expr,
	                 const galp::codec::host::RLESLPATCHColumn<T, IndexT>&     host_col,
	                 galp::memory::DeviceArena&                         arena,
	                 [[maybe_unused]] bool                                freq_use_extended) {
		static_assert(std::is_same_v<IndexT, uint16_t>, "RLESLPATCHColumn currently supports uint16_t indexes");
		host_col.copy_to_device(arena, expr.col.rle_slpatch_u16);
	}
};

// DICTREFColumn is resolved to DICTFFORColumn by resolve_dict_refs() before
// reaching fill_device_expr. Trait exists so the visitor template instantiates
// cleanly; fill() throws to match the prior runtime "plan/column mismatch".
template <typename T, typename IndexT>
struct ColumnKindTraits<galp::codec::host::DICTREFColumn<T, IndexT>> {
	using value_type                    = T;
	static constexpr PlanKind plan_kind =
	    std::is_same_v<IndexT, uint8_t> ? PlanKind::DICT_FFOR_U8 : PlanKind::DICT_FFOR_U16;
	static void fill(DeviceExpression<T>&,
	                 const galp::codec::host::DICTREFColumn<T, IndexT>&,
	                 galp::memory::DeviceArena&,
	                 bool) {
		throw std::runtime_error("ColumnKindTraits<DICTREFColumn>::fill: must resolve_dict_refs before dispatch");
	}
};

// Device-side companion: DeviceColT → value type. Only consumed by
// detail::decompress_device in column.cu, but kept alongside the host-side
// traits so all column-type reflection lives in one place.
template <typename ColumnT>
struct column_value_type;
template <typename T>
struct column_value_type<galp::codec::device::BPColumn<T>> {
	using type = T;
};
template <typename T>
struct column_value_type<galp::codec::device::CONSTANTColumn<T>> {
	using type = T;
};
template <typename T>
struct column_value_type<galp::codec::device::FFORColumn<T>> {
	using type = T;
};
template <typename T, typename IndexT>
struct column_value_type<galp::codec::device::DICTFFORColumn<T, IndexT>> {
	using type = T;
};
template <typename T, typename IndexT>
struct column_value_type<galp::codec::device::DICTSLPATCHColumn<T, IndexT>> {
	using type = T;
};
template <typename T>
struct column_value_type<galp::codec::device::FREQColumn<T>> {
	using type = T;
};
template <typename T>
struct column_value_type<galp::codec::device::CROSSRLEColumn<T>> {
	using type = T;
};
template <typename T>
struct column_value_type<galp::codec::device::SLPATCHColumn<T>> {
	using type = T;
};
template <typename T, typename IndexT>
struct column_value_type<galp::codec::device::RLEColumn<T, IndexT>> {
	using type = T;
};
template <typename T, typename IndexT>
struct column_value_type<galp::codec::device::RLESLPATCHColumn<T, IndexT>> {
	using type = T;
};

} // namespace galp::execution

#endif // ENGINE_EXECUTION_COLUMN_TRAITS_CUH
