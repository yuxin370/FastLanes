// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/workset/append.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_RUNTIME_WORKSET_APPEND_CUH
#define ENGINE_RUNTIME_WORKSET_APPEND_CUH

#include "codecs/consts.cuh"
#include "engine/operators/dict_ref_resolver.cuh"
#include "engine/workset/streams.cuh"
#include <algorithm>
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
		// Selected-vector callers emit at least one explicit work item per
		// expression. Reserve the batch floor up front so add_expression_to_batch
		// does not grow and copy the vector once per column.
		batch.work_items.reserve(batch.work_items.size() + additional_exprs);
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

inline void append_packed_rowgroup_device_scatter(ExecutionWorkset&                workset,
                                                  const galp::execution::Rowgroup& rowgroup,
                                                  galp::memory::DeviceArena&       arena) {
	const auto& payload = rowgroup.packed_device_payload;
	if (!payload) {
		return;
	}
	if (payload->packed_data == nullptr || payload->packed_bytes == 0U || payload->logical_data == nullptr ||
	    payload->logical_bytes == 0U || payload->ranges.empty()) {
		throw std::runtime_error("packed rowgroup device payload is incomplete");
	}

	arena.register_backing(payload->logical_data, payload->logical_bytes, /*upload=*/false);
	const auto logical_entry = arena.add<std::byte>(payload->logical_bytes, payload->logical_data);
	const auto packed_entry  = arena.add<std::byte>(payload->packed_bytes, payload->packed_data);

	const size_t plan_index = workset.buffers.pending_device_scatters.size();
	const size_t copy_begin = workset.buffers.device_scatter_copies.size();
	workset.buffers.pending_device_scatters.push_back(
	    PendingDeviceScatter {payload, nullptr, nullptr, copy_begin});
	workset.buffers.device_scatter_copies.resize(copy_begin + payload->ranges.size());
	auto& plan = workset.buffers.pending_device_scatters.back();
	arena.resolve_to(reinterpret_cast<void**>(&plan.device_logical), logical_entry);
	arena.resolve_to(reinterpret_cast<void**>(&plan.device_packed), packed_entry);
	arena.add_resolver([&workset, plan_index]() {
		auto& resolved = workset.buffers.pending_device_scatters.at(plan_index);
		const auto& ranges = resolved.payload->ranges;
		for (size_t index = 0; index < ranges.size(); ++index) {
			const auto& range = ranges[index];
			if (range.packed_offset > resolved.payload->packed_bytes ||
			    range.size > resolved.payload->packed_bytes - range.packed_offset ||
			    range.logical_offset > resolved.payload->logical_bytes ||
			    range.size > resolved.payload->logical_bytes - range.logical_offset) {
				throw std::runtime_error("packed rowgroup device scatter range exceeds its backing");
			}
			workset.buffers.device_scatter_copies[resolved.copy_begin + index] = DeviceScatterCopy {
			    resolved.device_packed + range.packed_offset,
			    resolved.device_logical + range.logical_offset,
			    range.size};
		}
	});
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
		const auto& batch = workset.buffers.host_batches.template get<T>();
		total += batch.work_items.size() + batch.scalar_tail_work_items.size();
	});
	return total;
}

inline size_t count_expr_work_items(const ExecutionWorkset& workset) {
	size_t total = 0;
	galp::execution::for_each_type(galp::execution::SupportedTypes {}, [&](auto tag) {
		using T           = typename decltype(tag)::type;
		const auto& batch = workset.buffers.host_batches.template get<T>();
		if (batch.work_items_explicit) {
			total += batch.work_items.size() + batch.scalar_tail_work_items.size();
			return;
		}
		for (const auto& expr : batch.device_exprs) {
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
	return !galp::execution::has_unresolved_dict_ref(column.host);
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
				                               *active_chunk_arena,
				                               /*selected_vectors=*/nullptr,
				                               /*selected_vector_width=*/std::max(1U, cfg.unpack_n_vectors));
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
                                     const bool                     register_backing      = true,
                                     const std::vector<uint32_t>*   selected_vectors      = nullptr,
                                     const uint32_t                 selected_vector_width = 1) {
	if (column.skip_decompress) {
		return;
	}
	if (galp::execution::has_unresolved_dict_ref(column.host)) {
		throw std::runtime_error("unresolved DICTREF reached GPU workset construction");
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
			    const uint32_t effective_vector_width =
			        selected_vectors != nullptr ? selected_vector_width : std::max(1U, cfg.unpack_n_vectors);
			    const size_t output_vector_width =
			        effective_vector_width == 0 ? 1U : static_cast<size_t>(effective_vector_width);
			    const size_t output_n_values = selected_vectors != nullptr
			                                       ? selected_vectors->size() *
			                                             output_vector_width * galp::codec::consts::VALUES_PER_VECTOR
			                                       : host_col.get_n_values();
			    if (out_total_bytes) {
				    *out_total_bytes += output_n_values * sizeof(T);
			    }
			    const size_t output_offset =
			        cfg.write_out ? reserve_workset_output_bytes<T>(workset, output_n_values) : 0U;
			    add_expression_to_batch<T>(materialize_expr_index,
			                               host_col,
			                               plan,
			                               workset.buffers.host_batches.template get<T>(),
			                               output_offset,
			                               cfg.freq_patcher,
			                               cfg.freq_branchless_threshold,
			                               emit_typed_work_items,
			                               active_chunk_arena,
			                               selected_vectors,
			                               effective_vector_width);
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

inline void append_rowgroup_columns_selected_vectors(ExecutionWorkset&                workset,
	                                                     galp::execution::Rowgroup&       rowgroup,
                                                     const ExecutionConfig&           cfg,
                                                     const std::vector<uint32_t>&     selected_vectors,
                                                     const size_t                     expr_index_base       = 0,
                                                     const bool                       use_global_expr_index = false) {
	auto expressions = galp::expression::assemble(rowgroup);
	galp::execution::resolve_dict_refs(expressions);
	workset.outputs.required = workset.outputs.required || cfg.write_out;
	begin_workset_chunk_arena(workset, rowgroup.columns.size());
	galp::memory::DeviceArena* active_chunk_arena = workset.buffers.chunk_arena.get();
	workset.buffers.pending_device_scatters.reserve(workset.buffers.pending_device_scatters.size() + 1U);
	if (rowgroup.packed_device_payload) {
		workset.buffers.device_scatter_copies.reserve(
		    workset.buffers.device_scatter_copies.size() + rowgroup.packed_device_payload->ranges.size());
	}
	append_packed_rowgroup_device_scatter(workset, rowgroup, *active_chunk_arena);

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
		const size_t   materialize_expr_index = use_global_expr_index ? (expr_index_base + active_expr_count - 1U) : i;
		const uint32_t selected_vector_width  = std::max(1U, cfg.unpack_n_vectors);
		append_column_to_workset(workset,
		                         column,
		                         cfg,
		                         materialize_expr_index,
		                         *active_chunk_arena,
		                         nullptr,
		                         /*emit_typed_work_items=*/true,
		                         /*register_backing=*/false,
		                         &selected_vectors,
		                         selected_vector_width);
	}
}

} // namespace galp::runtime

#endif // ENGINE_RUNTIME_WORKSET_APPEND_CUH
