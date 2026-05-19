// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/table/table_options.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_TABLE_OPTIONS_CUH
#define ENGINE_EXECUTION_TABLE_OPTIONS_CUH

#include "engine/table/table.cuh"
#include "galp/options.hpp"
#include <cstddef>

namespace galp::execution {

inline TableDecompressionScope to_table_scope(const galp::TableDecompressionScope scope) {
	return scope == galp::TableDecompressionScope::PerRowgroup ? TableDecompressionScope::PerRowgroup
	                                                           : TableDecompressionScope::WholeTable;
}

inline void apply_table_streaming_options(TableDecompressionConfig& cfg,
                                          const bool                enable_rowgroup_prefetch,
                                          const size_t              prefetch_depth,
                                          const size_t              prefetch_workers,
                                          const size_t              max_prefetch_storage_bytes,
                                          const size_t              streaming_target_work_items,
                                          const size_t              streaming_target_rowgroups,
                                          const size_t              compute_inflight_chunks = 0) {
	cfg.enable_rowgroup_prefetch    = enable_rowgroup_prefetch;
	cfg.prefetch_depth              = prefetch_depth;
	cfg.prefetch_workers            = prefetch_workers;
	cfg.max_prefetch_storage_bytes  = max_prefetch_storage_bytes;
	cfg.streaming_target_work_items = streaming_target_work_items;
	cfg.streaming_target_rowgroups  = streaming_target_rowgroups;
	cfg.compute_inflight_chunks     = compute_inflight_chunks;
}

inline TableDecompressionConfig make_table_decompression_config(const galp::DecompressOptions& options) {
	TableDecompressionConfig cfg {};
	cfg.execution.write_out = options.write_output;
	cfg.scope               = to_table_scope(options.scope);
	const galp::AdvancedOptions defaults {};
	const auto choose_bool = [](const bool top_level_value, const bool advanced_value, const bool default_value) {
		return top_level_value != default_value ? top_level_value : advanced_value;
	};
	const auto choose_size = [](const size_t top_level_value, const size_t advanced_value, const size_t default_value) {
		return top_level_value != default_value ? top_level_value : advanced_value;
	};
	apply_table_streaming_options(
	    cfg,
	    choose_bool(options.enable_rowgroup_prefetch,
	                options.advanced.enable_rowgroup_prefetch,
	                defaults.enable_rowgroup_prefetch),
	    choose_size(options.prefetch_depth, options.advanced.prefetch_depth, defaults.prefetch_depth),
	    choose_size(options.prefetch_workers, options.advanced.prefetch_workers, defaults.prefetch_workers),
	    choose_size(options.max_prefetch_storage_bytes,
	                options.advanced.max_prefetch_storage_bytes,
	                defaults.max_prefetch_storage_bytes),
	    choose_size(options.streaming_target_work_items,
	                options.advanced.streaming_target_work_items,
	                defaults.streaming_target_work_items),
	    choose_size(options.streaming_target_rowgroups,
	                options.advanced.streaming_target_rowgroups,
	                defaults.streaming_target_rowgroups));
	return cfg;
}

} // namespace galp::execution

#endif // ENGINE_EXECUTION_TABLE_OPTIONS_CUH
