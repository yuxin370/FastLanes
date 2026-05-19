// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/workset/append.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_RUNTIME_WORKSET_APPEND_CUH
#define ENGINE_RUNTIME_WORKSET_APPEND_CUH

#include "engine/operators/dict_ref_resolver.cuh"
#include "engine/workset/streams.cuh"
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <variant>

namespace galp::runtime {

inline void reserve_batch_expr_storage(ExecutionWorkset& workset, const size_t additional_exprs) {
	if (additional_exprs == 0) {
		return;
	}
	galp::execution::for_each_type(galp::execution::SupportedTypes {}, [&](auto tag) {
		using T     = typename decltype(tag)::type;
		auto& batch = workset.buffers.host_batches.template get<T>();
		batch.device_exprs.reserve(batch.device_exprs.size() + additional_exprs);
		batch.output_offsets.reserve(batch.output_offsets.size() + additional_exprs);
		batch.expr_indices.reserve(batch.expr_indices.size() + additional_exprs);
	});
}

inline bool any_batch_has_device_exprs(const ExecutionWorkset& workset) {
	bool has = false;
	galp::execution::for_each_type(galp::execution::SupportedTypes {}, [&](auto tag) {
		using T = typename decltype(tag)::type;
		if (!workset.buffers.host_batches.template get<T>().device_exprs.empty()) {
			has = true;
		}
	});
	return has;
}

inline bool begin_workset_chunk_arena(ExecutionWorkset& workset, const size_t additional_exprs) {
	if (workset.buffers.chunk_arena) {
		if (!any_batch_has_device_exprs(workset)) {
			reserve_batch_expr_storage(workset, additional_exprs);
		}
		return true;
	}
	reserve_batch_expr_storage(workset, additional_exprs);
	workset.buffers.chunk_arena = std::make_unique<galp::memory::DeviceArena>(ensure_workset_h2d_stream(workset));
	return true;
}

template <typename T>
inline size_t reserve_workset_output_bytes(ExecutionWorkset& workset, const size_t n_values) {
	if (n_values == 0) {
		return workset.outputs.used_bytes;
	}
	constexpr size_t kAlign    = alignof(T) > 16 ? alignof(T) : 16U;
	workset.outputs.used_bytes = (workset.outputs.used_bytes + (kAlign - 1U)) & ~(kAlign - 1U);
	const size_t offset        = workset.outputs.used_bytes;
	workset.outputs.used_bytes += n_values * sizeof(T);
	return offset;
}

inline size_t count_work_items(const ExecutionWorkset& workset) {
	size_t total = 0;
	galp::execution::for_each_type(galp::execution::SupportedTypes {}, [&](auto tag) {
		using T = typename decltype(tag)::type;
		total += workset.buffers.host_batches.template get<T>().work_items.size();
	});
	return total;
}

inline size_t count_expr_work_items(const ExecutionWorkset& workset) {
	size_t total = 0;
	galp::execution::for_each_type(galp::execution::SupportedTypes {}, [&](auto tag) {
		using T = typename decltype(tag)::type;
		for (const auto& expr : workset.buffers.host_batches.template get<T>().device_exprs) {
			total += galp::codec::utils::get_n_vecs_from_size(expr.n_values);
		}
	});
	return total;
}

inline size_t count_active_columns(const std::vector<galp::expression::Expression>& expressions) {
	size_t total = 0;
	for (const auto& expr : expressions) {
		if (expr.column && !expr.column->skip_decompress) {
			++total;
		}
	}
	return total;
}

inline bool can_direct_append_column(const galp::execution::Column& column) {
	if (column.skip_decompress) {
		return true;
	}
	return !std::holds_alternative<galp::codec::host::DICTREFColumn<int8_t, uint8_t>>(column.host);
}

inline bool can_direct_append_rowgroup(const galp::execution::Rowgroup& rowgroup) {
	for (const auto& column : rowgroup.columns) {
		if (!can_direct_append_column(column)) {
			return false;
		}
	}
	return true;
}

inline size_t count_active_columns(const galp::execution::Rowgroup& rowgroup) {
	size_t total = 0;
	for (const auto& column : rowgroup.columns) {
		if (!column.skip_decompress) {
			++total;
		}
	}
	return total;
}

inline bool has_pinned_backing(const galp::execution::Column& column) {
	return column.host_owned_by_backing && column.backing_is_pinned && column.backing_base != nullptr &&
	       column.backing_bytes > 0;
}

#ifndef NDEBUG
inline void validate_unique_work_item(std::unordered_set<uint64_t>& seen, const galp::execution::WorkItemAny& work) {
	const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(work.type)) << 56) |
	                     (static_cast<uint64_t>(work.expr_index) << 28) | static_cast<uint64_t>(work.vector_index);
	if (!seen.insert(key).second) {
		throw std::runtime_error("duplicate work item detected (type,expr_index,vector_index)");
	}
}
#endif

inline void append_expressions(ExecutionWorkset&                          workset,
                               std::vector<galp::expression::Expression>& expressions,
                               const ExecutionConfig&                     cfg,
                               size_t*                                    out_total_bytes       = nullptr,
                               size_t*                                    out_n_exprs           = nullptr,
                               const size_t                               expr_index_base       = 0,
                               const bool                                 use_global_expr_index = false) {
	galp::execution::resolve_dict_refs(expressions);
	workset.outputs.required = workset.outputs.required || cfg.write_out;
	begin_workset_chunk_arena(workset, expressions.size());
	galp::memory::DeviceArena* active_chunk_arena = workset.buffers.chunk_arena.get();

	size_t      active_expr_count  = 0;
	const void* last_backing_base  = nullptr;
	size_t      last_backing_bytes = 0;

	for (size_t i = 0; i < expressions.size(); ++i) {
		const auto& expr = expressions[i];
		if (!expr.column || expr.column->skip_decompress) {
			continue;
		}
		if (has_pinned_backing(*expr.column) &&
		    (expr.column->backing_base != last_backing_base || expr.column->backing_bytes != last_backing_bytes)) {
			active_chunk_arena->register_backing(expr.column->backing_base, expr.column->backing_bytes);
			last_backing_base  = expr.column->backing_base;
			last_backing_bytes = expr.column->backing_bytes;
		}
		++active_expr_count;
		if (out_n_exprs) {
			++(*out_n_exprs);
		}

		std::visit(
		    [&](auto&& host_col) {
			    using HostColT      = std::decay_t<decltype(host_col)>;
			    using T             = typename galp::execution::ColumnKindTraits<HostColT>::value_type;
			    constexpr auto plan = galp::execution::detail::plan_for_host_col<HostColT>();
			    if constexpr (galp::execution::is_supported_type_v<T>) {
				    if (out_total_bytes) {
					    *out_total_bytes += host_col.get_n_values() * sizeof(T);
				    }
				    const size_t materialize_expr_index =
				        use_global_expr_index ? (expr_index_base + active_expr_count - 1U) : i;
				    const size_t output_offset =
				        cfg.write_out ? reserve_workset_output_bytes<T>(workset, host_col.get_n_values()) : 0U;
				    add_expression_to_batch<T>(materialize_expr_index,
				                               host_col,
				                               plan,
				                               workset.buffers.host_batches.template get<T>(),
				                               output_offset,
				                               cfg.freq_patcher,
				                               cfg.freq_branchless_threshold,
				                               cfg.launch_strategy != galp::execution::LaunchStrategy::MixedDispatch,
				                               *active_chunk_arena);
			    }
		    },
		    expr.column->host);
	}
}

inline void append_column_to_workset(ExecutionWorkset&              workset,
                                     const galp::execution::Column& column,
                                     const ExecutionConfig&         cfg,
                                     const size_t                   materialize_expr_index,
                                     galp::memory::DeviceArena&     active_chunk_arena,
                                     size_t*                        out_total_bytes       = nullptr,
                                     const bool                     emit_typed_work_items = true,
                                     const bool                     register_backing      = true) {
	if (column.skip_decompress) {
		return;
	}
	if (register_backing && has_pinned_backing(column)) {
		active_chunk_arena.register_backing(column.backing_base, column.backing_bytes);
	}

	std::visit(
	    [&](auto&& host_col) {
		    using HostColT      = std::decay_t<decltype(host_col)>;
		    using T             = typename galp::execution::ColumnKindTraits<HostColT>::value_type;
		    constexpr auto plan = galp::execution::detail::plan_for_host_col<HostColT>();
		    if constexpr (galp::execution::is_supported_type_v<T>) {
			    if (out_total_bytes) {
				    *out_total_bytes += host_col.get_n_values() * sizeof(T);
			    }
			    const size_t output_offset =
			        cfg.write_out ? reserve_workset_output_bytes<T>(workset, host_col.get_n_values()) : 0U;
			    add_expression_to_batch<T>(materialize_expr_index,
			                               host_col,
			                               plan,
			                               workset.buffers.host_batches.template get<T>(),
			                               output_offset,
			                               cfg.freq_patcher,
			                               cfg.freq_branchless_threshold,
			                               emit_typed_work_items,
			                               active_chunk_arena);
		    }
	    },
	    column.host);
}

inline void append_rowgroup_columns(ExecutionWorkset&                workset,
                                    const galp::execution::Rowgroup& rowgroup,
                                    const ExecutionConfig&           cfg,
                                    const size_t                     expr_index_base       = 0,
                                    const bool                       use_global_expr_index = false) {
	workset.outputs.required = workset.outputs.required || cfg.write_out;
	begin_workset_chunk_arena(workset, rowgroup.columns.size());
	galp::memory::DeviceArena* active_chunk_arena = workset.buffers.chunk_arena.get();

	size_t      active_expr_count  = 0;
	const void* last_backing_base  = nullptr;
	size_t      last_backing_bytes = 0;
	for (size_t i = 0; i < rowgroup.columns.size(); ++i) {
		const auto& column = rowgroup.columns[i];
		if (column.skip_decompress) {
			continue;
		}
		if (has_pinned_backing(column) &&
		    (column.backing_base != last_backing_base || column.backing_bytes != last_backing_bytes)) {
			active_chunk_arena->register_backing(column.backing_base, column.backing_bytes);
			last_backing_base  = column.backing_base;
			last_backing_bytes = column.backing_bytes;
		}
		++active_expr_count;
		const size_t materialize_expr_index = use_global_expr_index ? (expr_index_base + active_expr_count - 1U) : i;
		append_column_to_workset(workset,
		                         column,
		                         cfg,
		                         materialize_expr_index,
		                         *active_chunk_arena,
		                         nullptr,
		                         cfg.launch_strategy != galp::execution::LaunchStrategy::MixedDispatch,
		                         false);
	}
}

} // namespace galp::runtime

#endif // ENGINE_RUNTIME_WORKSET_APPEND_CUH
