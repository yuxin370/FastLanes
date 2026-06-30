// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/operators/expr_ops.cuh
// ────────────────────────────────────────────────────────
// Per-DeviceExpression helpers: pack host-column payload into the expression's
// union slot (fill), tear it down (free), decide FREQ patcher mode, and push
// an expression into a host-side Batch. The arena-resolver callback path lives
// in DeviceArena; this header is the thin bridge between host columns and
// that arena.
#ifndef ENGINE_EXECUTION_INTERNAL_EXPR_OPS_CUH
#define ENGINE_EXECUTION_INTERNAL_EXPR_OPS_CUH

#include "codecs/consts.cuh"
#include "codecs/encodings/all.cuh"
#include "core/data/value_store.cuh"
#include "core/expression.cuh"
#include "core/lane_policy.cuh"
#include "cuda/memory/device_arena.cuh"
#include "engine/config.cuh"
#include "engine/operators/batch.cuh"
#include "engine/operators/column_traits.cuh"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace galp::execution {

namespace detail {

template <typename HostColT>
ValueStore decompress_host(const HostColT& host_col, const PlanKind plan, const ExecutionConfig& cfg);

template <typename HostColT>
constexpr PlanKind plan_for_host_col() {
	return ColumnKindTraits<HostColT>::plan_kind;
}

template <typename T>
bool should_use_freq_extended(const galp::codec::host::FREQColumn<T>& host_col,
                              const FreqPatcher                       mode,
                              const float                             branchless_threshold) {
	switch (mode) {
	case FreqPatcher::Stateful:
		return false;
	case FreqPatcher::Branchless:
		return true;
	case FreqPatcher::Hybrid:
		if (host_col.get_n_vecs() == 0) {
			return false;
		}
		const double exc_per_vec =
		    static_cast<double>(host_col.n_exceptions) / static_cast<double>(host_col.get_n_vecs());
		return exc_per_vec >= static_cast<double>(branchless_threshold);
	}
	return false;
}

// Arena-based overload: packs column data into shared arena; pointers resolved on arena.upload().
// The runtime `plan` argument is redundant with the trait's compile-time `plan_kind`,
// but is still checked to catch mismatches arising from type-erased call paths.
template <typename T, typename HostColT>
void fill_device_expr(DeviceExpression<T>&       expr,
                      const HostColT&            host_col,
                      const PlanKind             plan,
                      const bool                 freq_use_extended,
                      galp::memory::DeviceArena& arena) {
	if (plan != ColumnKindTraits<HostColT>::plan_kind) {
		throw std::runtime_error("fill_device_expr(arena): plan/column type mismatch, plan=" +
		                         std::to_string(static_cast<int>(plan)));
	}
	ColumnKindTraits<HostColT>::fill(expr, host_col, arena, freq_use_extended);
}

template <typename T>
void free_device_expr(const DeviceExpression<T>& expr) {
	switch (expr.plan) {
	case PlanKind::UNCOMPRESSED:
		galp::codec::host::free_column(expr.col.bp);
		break;
	case PlanKind::CONSTANT:
		galp::codec::host::free_column(expr.col.constant);
		break;
	case PlanKind::FREQUENCY:
		if (expr.freq_use_extended) {
			galp::codec::host::free_column(expr.col.freq_extended);
		} else {
			galp::codec::host::free_column(expr.col.freq);
		}
		break;
	case PlanKind::UNFFOR:
		galp::codec::host::free_column(expr.col.ffor);
		break;
	case PlanKind::UNFFOR_SLPATCH:
		galp::codec::host::free_column(expr.col.slpatch);
		break;
	case PlanKind::DICT_FFOR_U8:
		galp::codec::host::free_column(expr.col.dictffor_u8);
		break;
	case PlanKind::DICT_FFOR_U16:
		galp::codec::host::free_column(expr.col.dictffor_u16);
		break;
	case PlanKind::DICT_FFOR_SLPATCH_U8:
		galp::codec::host::free_column(expr.col.dictslpatch_u8);
		break;
	case PlanKind::DICT_FFOR_SLPATCH_U16:
		galp::codec::host::free_column(expr.col.dictslpatch_u16);
		break;
	case PlanKind::CROSS_RLE:
		galp::codec::host::free_column(expr.col.crossrle);
		break;
	case PlanKind::RLE_U8:
		galp::codec::host::free_column(expr.col.rle_u8);
		break;
	case PlanKind::RLE_U16:
		galp::codec::host::free_column(expr.col.rle_u16);
		break;
	case PlanKind::RLE_SLPATCH_U16:
		galp::codec::host::free_column(expr.col.rle_slpatch_u16);
		break;
	default:
		break;
	}
}

} // namespace detail

// Arena-based overload: emplaces expr into pre-reserved batch, packs column data into shared arena.
// batch.device_exprs MUST be pre-reserved to avoid reallocation (resolver callbacks capture &out).
template <typename T, typename HostColT>
void add_expression_to_batch(const size_t                 expr_index,
                             const HostColT&              host_col,
                             const PlanKind               plan,
                             Batch<T>&                    batch,
                             const size_t                 output_offset,
                             const FreqPatcher            freq_patcher,
                             const float                  freq_branchless_threshold,
                             const bool                   emit_typed_work_items,
                             galp::memory::DeviceArena&   arena,
                             const std::vector<uint32_t>* selected_vectors      = nullptr,
                             const uint32_t               selected_vector_width = 1) {
	// Emplace into pre-reserved vector — address is stable.
	batch.device_exprs.emplace_back();
	const size_t output_vector_width = selected_vector_width == 0 ? 1U : static_cast<size_t>(selected_vector_width);
	auto&        expr                = batch.device_exprs.back();
	expr.plan                        = plan;
	expr.n_values                    = host_col.get_n_values();
	expr.output_n_values = selected_vectors != nullptr ? selected_vectors->size() * output_vector_width *
	                                                         galp::codec::consts::VALUES_PER_VECTOR
	                                                   : expr.n_values;
	expr.out             = nullptr;
	batch.output_offsets.push_back(output_offset);

	bool use_freq_extended = false;
	if constexpr (std::is_same_v<HostColT, galp::codec::host::FREQColumn<T>>) {
		use_freq_extended = detail::should_use_freq_extended(host_col, freq_patcher, freq_branchless_threshold);
	}
	detail::fill_device_expr(expr, host_col, plan, use_freq_extended, arena);

	const auto device_idx = static_cast<uint32_t>(batch.device_exprs.size() - 1);
	batch.expr_indices.push_back(expr_index);

	if (!emit_typed_work_items && selected_vectors == nullptr) {
		return;
	}
	const size_t n_vecs = galp::codec::utils::get_n_vecs_from_size(expr.n_values);
	if (selected_vectors != nullptr) {
		batch.work_items_explicit = true;
		batch.work_items.reserve(batch.work_items.size() + selected_vectors->size());
		for (size_t idx = 0; idx < selected_vectors->size(); ++idx) {
			const uint32_t vec = (*selected_vectors)[idx];
			if (vec >= n_vecs) {
				throw std::out_of_range("selected FastLanes vector index out of range");
			}
			if (static_cast<size_t>(vec) + output_vector_width > n_vecs) {
				throw std::out_of_range("selected FastLanes vector chunk extends past column vectors");
			}
			batch.work_items.push_back(
			    WorkItemAny {device_idx, vec, type_tag_for<T>(), static_cast<uint32_t>(idx * output_vector_width)});
		}
		return;
	}
	if (emit_typed_work_items) {
		const size_t decode_vector_width = output_vector_width;
		batch.work_items.reserve(batch.work_items.size() + (n_vecs + decode_vector_width - 1U) / decode_vector_width);
		for (size_t vec = 0; vec < n_vecs; vec += decode_vector_width) {
			if (vec + decode_vector_width > n_vecs) {
				throw std::out_of_range("FastLanes decode chunk extends past column vectors");
			}
			batch.work_items.push_back(
			    WorkItemAny {device_idx, static_cast<uint32_t>(vec), type_tag_for<T>(), static_cast<uint32_t>(vec)});
		}
	}
}

} // namespace galp::execution

#endif // ENGINE_EXECUTION_INTERNAL_EXPR_OPS_CUH
