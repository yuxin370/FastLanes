// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/execution/column_traits.cuh
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

#include "engine/expression.cuh"
#include "flsgpu/memory/device_arena.cuh"
#include "flsgpu/structs.cuh"
#include <stdexcept>
#include <type_traits>

namespace dispatch {

template <typename HostColT>
struct ColumnKindTraits;

template <typename T>
struct ColumnKindTraits<flsgpu::host::BPColumn<T>> {
	using value_type                    = T;
	static constexpr PlanKind plan_kind = PlanKind::UNCOMPRESSED;
	static void fill(DeviceExpression<T>&                expr,
	                 const flsgpu::host::BPColumn<T>&    host_col,
	                 flsgpu::memory::DeviceArena&        arena,
	                 [[maybe_unused]] bool               freq_use_extended) {
		host_col.copy_to_device(arena, expr.col.bp);
	}
};

template <typename T>
struct ColumnKindTraits<flsgpu::host::CONSTANTColumn<T>> {
	using value_type                    = T;
	static constexpr PlanKind plan_kind = PlanKind::CONSTANT;
	static void fill(DeviceExpression<T>&                   expr,
	                 const flsgpu::host::CONSTANTColumn<T>& host_col,
	                 flsgpu::memory::DeviceArena&           arena,
	                 [[maybe_unused]] bool                  freq_use_extended) {
		host_col.copy_to_device(arena, expr.col.constant);
	}
};

template <typename T>
struct ColumnKindTraits<flsgpu::host::FFORColumn<T>> {
	using value_type                    = T;
	static constexpr PlanKind plan_kind = PlanKind::UNFFOR;
	static void fill(DeviceExpression<T>&                expr,
	                 const flsgpu::host::FFORColumn<T>&  host_col,
	                 flsgpu::memory::DeviceArena&        arena,
	                 [[maybe_unused]] bool               freq_use_extended) {
		host_col.copy_to_device(arena, expr.col.ffor);
	}
};

template <typename T>
struct ColumnKindTraits<flsgpu::host::SLPATCHColumn<T>> {
	using value_type                    = T;
	static constexpr PlanKind plan_kind = PlanKind::UNFFOR_SLPATCH;
	static void fill(DeviceExpression<T>&                  expr,
	                 const flsgpu::host::SLPATCHColumn<T>& host_col,
	                 flsgpu::memory::DeviceArena&          arena,
	                 [[maybe_unused]] bool                 freq_use_extended) {
		host_col.copy_to_device(arena, expr.col.slpatch);
	}
};

template <typename T>
struct ColumnKindTraits<flsgpu::host::FREQColumn<T>> {
	using value_type                    = T;
	static constexpr PlanKind plan_kind = PlanKind::FREQUENCY;
	static void fill(DeviceExpression<T>&                expr,
	                 const flsgpu::host::FREQColumn<T>&  host_col,
	                 flsgpu::memory::DeviceArena&        arena,
	                 bool                                freq_use_extended) {
		if (freq_use_extended) {
			// Extended column is a temporary — capture by value in defer_free so its
			// arrays stay alive until arena.upload() finishes the DMA.
			auto extended = host_col.create_extended_column();
			extended.copy_to_device(arena, expr.col.freq_extended);
			expr.freq_use_extended = true;
			arena.defer_free([ext = extended]() {
				flsgpu::host::free_column(ext);
			});
			return;
		}
		host_col.copy_to_device(arena, expr.col.freq);
		expr.freq_use_extended = false;
	}
};

template <typename T>
struct ColumnKindTraits<flsgpu::host::CROSSRLEColumn<T>> {
	using value_type                    = T;
	static constexpr PlanKind plan_kind = PlanKind::CROSS_RLE;
	static void fill(DeviceExpression<T>&                    expr,
	                 const flsgpu::host::CROSSRLEColumn<T>&  host_col,
	                 flsgpu::memory::DeviceArena&            arena,
	                 [[maybe_unused]] bool                   freq_use_extended) {
		host_col.copy_to_device(arena, expr.col.crossrle);
	}
};

template <typename T, typename IndexT>
struct ColumnKindTraits<flsgpu::host::DICTFFORColumn<T, IndexT>> {
	using value_type                    = T;
	static constexpr PlanKind plan_kind =
	    std::is_same_v<IndexT, uint8_t> ? PlanKind::DICT_FFOR_U8 : PlanKind::DICT_FFOR_U16;
	static void fill(DeviceExpression<T>&                               expr,
	                 const flsgpu::host::DICTFFORColumn<T, IndexT>&     host_col,
	                 flsgpu::memory::DeviceArena&                       arena,
	                 [[maybe_unused]] bool                              freq_use_extended) {
		if constexpr (std::is_same_v<IndexT, uint8_t>) {
			host_col.copy_to_device(arena, expr.col.dictffor_u8);
		} else {
			host_col.copy_to_device(arena, expr.col.dictffor_u16);
		}
	}
};

template <typename T, typename IndexT>
struct ColumnKindTraits<flsgpu::host::DICTSLPATCHColumn<T, IndexT>> {
	using value_type                    = T;
	static constexpr PlanKind plan_kind =
	    std::is_same_v<IndexT, uint8_t> ? PlanKind::DICT_FFOR_SLPATCH_U8 : PlanKind::DICT_FFOR_SLPATCH_U16;
	static void fill(DeviceExpression<T>&                                  expr,
	                 const flsgpu::host::DICTSLPATCHColumn<T, IndexT>&     host_col,
	                 flsgpu::memory::DeviceArena&                          arena,
	                 [[maybe_unused]] bool                                 freq_use_extended) {
		if constexpr (std::is_same_v<IndexT, uint8_t>) {
			host_col.copy_to_device(arena, expr.col.dictslpatch_u8);
		} else {
			host_col.copy_to_device(arena, expr.col.dictslpatch_u16);
		}
	}
};

template <typename T, typename IndexT>
struct ColumnKindTraits<flsgpu::host::RLEColumn<T, IndexT>> {
	using value_type                    = T;
	static constexpr PlanKind plan_kind =
	    std::is_same_v<IndexT, uint8_t> ? PlanKind::RLE_U8 : PlanKind::RLE_U16;
	static void fill(DeviceExpression<T>&                          expr,
	                 const flsgpu::host::RLEColumn<T, IndexT>&     host_col,
	                 flsgpu::memory::DeviceArena&                  arena,
	                 [[maybe_unused]] bool                         freq_use_extended) {
		if constexpr (std::is_same_v<IndexT, uint8_t>) {
			host_col.copy_to_device(arena, expr.col.rle_u8);
		} else {
			host_col.copy_to_device(arena, expr.col.rle_u16);
		}
	}
};

// DICTREFColumn is resolved to DICTFFORColumn by resolve_dict_refs() before
// reaching fill_device_expr. Trait exists so the visitor template instantiates
// cleanly; fill() throws to match the prior runtime "plan/column mismatch".
template <typename T, typename IndexT>
struct ColumnKindTraits<flsgpu::host::DICTREFColumn<T, IndexT>> {
	using value_type                    = T;
	static constexpr PlanKind plan_kind =
	    std::is_same_v<IndexT, uint8_t> ? PlanKind::DICT_FFOR_U8 : PlanKind::DICT_FFOR_U16;
	static void fill(DeviceExpression<T>&,
	                 const flsgpu::host::DICTREFColumn<T, IndexT>&,
	                 flsgpu::memory::DeviceArena&,
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
struct column_value_type<flsgpu::device::BPColumn<T>> {
	using type = T;
};
template <typename T>
struct column_value_type<flsgpu::device::CONSTANTColumn<T>> {
	using type = T;
};
template <typename T>
struct column_value_type<flsgpu::device::FFORColumn<T>> {
	using type = T;
};
template <typename T, typename IndexT>
struct column_value_type<flsgpu::device::DICTFFORColumn<T, IndexT>> {
	using type = T;
};
template <typename T, typename IndexT>
struct column_value_type<flsgpu::device::DICTSLPATCHColumn<T, IndexT>> {
	using type = T;
};
template <typename T>
struct column_value_type<flsgpu::device::FREQColumn<T>> {
	using type = T;
};
template <typename T>
struct column_value_type<flsgpu::device::CROSSRLEColumn<T>> {
	using type = T;
};
template <typename T>
struct column_value_type<flsgpu::device::SLPATCHColumn<T>> {
	using type = T;
};
template <typename T, typename IndexT>
struct column_value_type<flsgpu::device::RLEColumn<T, IndexT>> {
	using type = T;
};

} // namespace dispatch

#endif // ENGINE_EXECUTION_COLUMN_TRAITS_CUH
