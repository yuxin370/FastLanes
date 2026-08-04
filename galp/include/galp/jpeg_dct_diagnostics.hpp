#ifndef GALP_JPEG_DCT_DIAGNOSTICS_HPP
#define GALP_JPEG_DCT_DIAGNOSTICS_HPP

#include "galp/config.hpp"

#if GALP_WITH_JPEG_DCT

#include <cstddef>
#include <string>

namespace galp::jpeg {

struct JpegDctDeviceCacheStats {
	size_t capacity_bytes     = 0;
	size_t resident_bytes     = 0;
	size_t resident_rowgroups = 0;
	size_t peak_resident_bytes = 0;
	size_t peak_resident_rowgroups = 0;
	size_t hits               = 0;
	size_t misses             = 0;
	size_t inserts            = 0;
	size_t evictions          = 0;
};

struct JpegDctDeviceExecutionStats {
	size_t      planned_selected_vector_count                 = 0;
	size_t      selected_vector_count                         = 0;
	size_t      full_vector_count                             = 0;
	size_t      planned_saved_vector_count                    = 0;
	size_t      actual_saved_vector_count                     = 0;
	size_t      decoded_coefficient_bytes                     = 0;
	size_t      rowgroup_count                                = 0;
	size_t      workset_count                                 = 0;
	size_t      decode_kernel_launch_count                    = 0;
	size_t      gather_kernel_launch_count                    = 0;
	size_t      prefix_gather_kernel_launch_count             = 0;
	size_t      cached_gather_kernel_launch_count             = 0;
	size_t      materialize_kernel_launch_count               = 0;
	size_t      gather_item_count                             = 0;
	size_t      decoded_gather_item_count                     = 0;
	size_t      cached_gather_item_count                      = 0;
	size_t      workset_upload_count                          = 0;
	size_t      scratch_upload_count                          = 0;
	size_t      scratch_allocation_count                      = 0;
	size_t      internal_sync_count                           = 0;
	size_t      cached_gather_sync_count                      = 0;
	size_t      decoded_batch_sync_count                      = 0;
	size_t      cached_gather_event_handoff_count             = 0;
	size_t      sparse_vector_cache_hits                      = 0;
	size_t      sparse_vector_cache_misses                    = 0;
	size_t      plan_cache_hits                               = 0;
	size_t      plan_cache_misses                             = 0;
	size_t      plan_cache_evictions                          = 0;
	size_t      runtime_policy_selected_rowgroups             = 0;
	size_t      runtime_policy_full_rowgroups                 = 0;
	size_t      runtime_policy_tail_full_rowgroups            = 0;
	size_t      runtime_policy_ratio_full_rowgroups           = 0;
	size_t      runtime_policy_low_saving_full_rowgroups      = 0;
	size_t      runtime_policy_forced_full_rowgroups          = 0;
	size_t      runtime_policy_forced_selected_rowgroups      = 0;
	size_t      prefetch_initial_cache_hit_rowgroup_count     = 0;
	size_t      prefetch_candidate_rowgroup_count             = 0;
	size_t      prefetch_active_shard_count                   = 0;
	size_t      prefetch_config_disabled_shard_count          = 0;
	size_t      prefetch_all_hit_shard_count                  = 0;
	size_t      prefetch_small_batch_disabled_shard_count     = 0;
	size_t      prefetch_selected_vector_disabled_shard_count = 0;
	size_t      prefetch_selected_vector_miss_rowgroup_count  = 0;
	size_t      prefetch_initial_hit_runtime_miss_count       = 0;
	size_t      prefetch_skipped_repeated_runtime_miss_count  = 0;
	size_t      prefetched_rowgroup_count                     = 0;
	size_t      prefetch_consumed_as_hit_count                = 0;
	size_t      prefetch_skipped_repeated_rowgroup_count      = 0;
	double      prefetch_consumed_as_hit_read_ms              = 0.0;
	double      prefetch_consumed_as_hit_wait_ms              = 0.0;
	double      planning_ms                                   = 0.0;
	double      host_io_staging_ms                            = 0.0;
	size_t      host_io_staged_rowgroups                      = 0;
	double      workset_build_ms                              = 0.0;
	double      workset_upload_ms                             = 0.0;
	double      workset_upload_prep_ms                        = 0.0;
	double      workset_upload_arena_ms                       = 0.0;
	double      workset_upload_arena_pack_ms                  = 0.0;
	double      workset_upload_arena_layout_ms                = 0.0;
	double      workset_upload_arena_alloc_ms                 = 0.0;
	double      workset_upload_arena_resolve_ms               = 0.0;
	double      workset_upload_dma_issue_ms                   = 0.0;
	double      workset_upload_event_record_ms                = 0.0;
	size_t      workset_upload_dma_bytes                      = 0;
	size_t      workset_upload_dma_count                      = 0;
	double      decode_ms                                     = 0.0;
	double      gather_ms                                     = 0.0;
	double      decoded_gather_ms                             = 0.0;
	double      cached_gather_ms                              = 0.0;
	double      projection_ms                                 = 0.0;
	double      decoded_projection_ms                         = 0.0;
	double      projection_item_build_ms                      = 0.0;
	double      fixed_transform_ms                            = 0.0;
	double      fixed_grid_round_ms                           = 0.0;
	double      resize_weight_build_ms                        = 0.0;
	double      prefetch_wait_ms                              = 0.0;
	double      prefetch_depth_block_ms                       = 0.0;
	double      prefetch_queue_start_ms                       = 0.0;
	double      prefetch_rowgroup_read_ms                     = 0.0;
	double      prefetch_ready_ahead_ms                       = 0.0;
	double      sync_rowgroup_read_ms                         = 0.0;
	std::string runtime_policy_decision;
	std::string runtime_policy_reason;
	size_t      projection_item_count                   = 0;
	size_t      decoded_projection_item_count           = 0;
	size_t      fixed_transform_item_count              = 0;
	size_t      fixed_transform_image_count             = 0;
	size_t      fixed_transform_component_count         = 0;
	size_t      fixed_transform_source_block_count      = 0;
	size_t      fixed_transform_output_block_count      = 0;
	size_t      dct_resize_weight_cache_hits            = 0;
	size_t      dct_resize_weight_cache_misses          = 0;
	size_t      dct_conversion_matrix_cache_hits        = 0;
	size_t      dct_conversion_matrix_cache_misses      = 0;
	size_t      project_decoded_ycbcr_grid_launch_count = 0;
	size_t      jpeg_dct_projection_items_materialized  = 0;
	size_t      fixed_grid_round_event_handoff_count    = 0;
	bool        cache_enabled                           = false;
	// Phase-2 architecture counters are appended to preserve the positional
	// initialization order of the legacy public aggregate.
	bool   exact_batch_plan_cache_enabled           = false;
	bool   uses_planless_fixed_transform            = false;
	size_t host_expanded_transform_items_created    = 0;
	size_t host_output_block_source_lists_created   = 0;
	size_t host_global_transform_sort_items         = 0;
	size_t planless_image_descriptor_count          = 0;
	size_t planless_transform_output_block_count    = 0;
	size_t planless_transform_full_scan_output_block_count = 0;
	size_t planless_transform_skipped_output_block_count   = 0;
	size_t planless_transform_active_output_index_bytes    = 0;
	size_t planless_transform_active_output_offset_bytes   = 0;
	size_t planless_transform_active_output_schedule_peak_bytes = 0;
	size_t planless_transform_source_contribution_count    = 0;
	size_t planless_transform_active_output_workset_count  = 0;
	size_t planless_transform_active_output_schedule_build_count = 0;
	bool   planless_transform_active_output_offsets_valid  = false;
	double planless_transform_active_output_planning_ms    = 0.0;
	double planless_transform_gpu_kernel_ms                = 0.0;
	size_t planless_axis_program_count              = 0;
	size_t planless_axis_phase_matrix_count         = 0;
	size_t planless_axis_program_bytes              = 0;
	// Runtime-owned compact plan storage, excluding persistent reader
	// dictionaries and decode worksets. Keep this separate from allocator RSS.
	size_t compact_plan_bytes                       = 0;
	size_t compact_plan_peak_bytes                  = 0;
	size_t rowgroup_storage_bytes_read              = 0;
	size_t galp_native_device_in_use_bytes          = 0;
	size_t galp_native_device_peak_in_use_bytes     = 0;
	size_t galp_native_device_cached_bytes          = 0;
	size_t galp_native_device_allocation_requests   = 0;
	size_t galp_native_device_cuda_allocation_count = 0;
	size_t galp_native_device_cuda_allocation_bytes = 0;
	double device_mapping_ms                        = 0.0;
	bool   device_mapping_fused                     = false;
	// Scheduling/stream diagnostics (appended for aggregate compatibility).
	size_t      planless_transform_kernel_launch_count          = 0;
	size_t      planless_transform_max_blocks_per_launch        = 0;
	size_t      planless_transform_max_output_blocks_per_launch = 0;
	size_t      planless_transform_registers_per_thread         = 0;
	size_t      planless_transform_static_shared_bytes_per_cta  = 0;
	size_t      planless_transform_local_bytes_per_thread       = 0;
	size_t      planless_transform_threads_per_cta              = 0;
	size_t      planless_transform_max_active_ctas_per_sm       = 0;
	size_t      cuda_max_threads_per_sm                         = 0;
	size_t      cuda_warp_size                                  = 0;
	size_t      decode_to_transform_event_handoff_count         = 0;
	size_t      copy_to_decode_event_handoff_count              = 0;
	int         direct_dct_stream_priority                      = 0;
	int         direct_dct_h2d_stream_priority                  = 0;
	int         direct_dct_decode_stream_priority               = 0;
	int         direct_dct_transform_stream_priority            = 0;
	int         direct_dct_round_stream_priority                = 0;
	int         cuda_least_stream_priority                      = 0;
	int         cuda_greatest_stream_priority                   = 0;
	bool        direct_dct_low_priority_streams                 = false;
	std::string scheduling_policy;
	// Generic transformed-grid output diagnostics. Appended to keep older
	// aggregate field positions stable.
	size_t fixed_grid_finalize_kernel_launch_count = 0;
	bool   fixed_grid_output_float32               = false;
	bool   fixed_grid_output_affine_applied        = false;
	float  fixed_grid_output_add                   = 0.0F;
	float  fixed_grid_output_scale                 = 1.0F;
	// Crop-pushdown accounting is deliberately explicit about each stage.
	// `planned_vector_count` is the crop-selected count, while
	// `actual_vector_count` is the count actually submitted to decode after
	// runtime fallback.  Physical I/O is reported independently from decode
	// and transform work so a transform reduction cannot be presented as a
	// storage-read reduction.
	std::string storage_read_granularity              = "rowgroup";
	std::string decode_granularity                    = "rowgroup";
	size_t      requested_source_block_count          = 0;
	size_t      planned_vector_count                  = 0;
	size_t      actual_vector_count                   = 0;
	size_t      compressed_payload_bytes_read         = 0;
	size_t      full_compressed_payload_bytes         = 0;
	size_t      pread_count                           = 0;
	size_t      preadv_count                          = 0;
	size_t      vector_bundle_rowgroup_count          = 0;
	size_t      vector_bundle_envelope_rowgroup_count = 0;
	size_t      vector_bundle_pread_count             = 0;
	double      read_amplification                    = 0.0;
	size_t      source_blocks_transformed             = 0;
	bool        sparse_read_supported                 = false;
	size_t      sparse_read_fallback_rowgroup_count   = 0;
	std::string sparse_read_fallback_reason;
	// Cost-model decisions for automatic physical sparse reads. Decode
	// selection is accounted separately above.
	size_t automatic_sparse_storage_candidate_rowgroup_count = 0;
	size_t automatic_sparse_storage_selected_rowgroup_count  = 0;
	size_t automatic_sparse_storage_rejected_rowgroup_count  = 0;
	size_t automatic_sparse_storage_early_rejected_rowgroup_count = 0;
	size_t automatic_sparse_storage_full_bytes               = 0;
	size_t automatic_sparse_storage_candidate_bytes          = 0;
	size_t automatic_sparse_storage_candidate_pread_count    = 0;
	size_t automatic_sparse_storage_optimistic_bytes         = 0;
	size_t automatic_sparse_storage_optimistic_pread_count   = 0;
	double automatic_sparse_storage_full_estimated_ns        = 0.0;
	double automatic_sparse_storage_candidate_estimated_ns   = 0.0;
	double adaptive_run_interval_estimated_ns                 = 0.0;
	double adaptive_bitmap_estimated_ns                       = 0.0;
	double adaptive_full_rowgroup_estimated_ns                = 0.0;
	size_t adaptive_selected_memory_fit_rowgroup_count        = 0U;
	size_t adaptive_full_memory_fit_rowgroup_count            = 0U;
	// Final adaptive strategy after comparing exact sparse intervals, exact
	// selected-vector decode over a full physical rowgroup, and full decode.
	size_t run_interval_exact_rowgroup_count = 0U;
	size_t bitmap_exact_rowgroup_count       = 0U;
	size_t full_rowgroup_strategy_count      = 0U;
	// Rowgroups read directly into CUDA-pinned host backing. These bytes can
	// be DMA-uploaded without repacking through DeviceArena's staging buffer.
	size_t pinned_rowgroup_read_count = 0;
	size_t pinned_rowgroup_read_bytes = 0;
	// Compact-v3 batches lease one shared pinned arena per physical shard.
	// Growth is a high-water event; later batches should reuse an existing
	// slot when its capacity already covers the requested shard payload.
	size_t compact_batch_buffer_acquire_count    = 0;
	size_t compact_batch_buffer_growth_count     = 0;
	size_t compact_batch_buffer_reuse_count      = 0;
	size_t compact_batch_buffer_requested_bytes  = 0;
	size_t compact_batch_buffer_capacity_bytes   = 0;
	size_t compact_batch_buffer_high_water_bytes = 0;
	size_t compact_batch_buffer_pageable_fallback_count = 0;
	size_t compact_batch_read_group_count        = 0;
	size_t compact_batch_read_worker_count       = 0;
	// Process-global pinned host pool gauges/counters sampled after execution.
	size_t galp_native_pinned_in_use_bytes          = 0;
	size_t galp_native_pinned_peak_in_use_bytes     = 0;
	size_t galp_native_pinned_cached_bytes          = 0;
	size_t galp_native_pinned_allocation_requests   = 0;
	size_t galp_native_pinned_cuda_allocation_count = 0;
	size_t galp_native_pinned_cuda_allocation_bytes = 0;
	// Compact-v3 coefficient pushdown accounting. Logical bytes exclude
	// same-page gaps; range bytes are the actual bytes requested with pread.
	size_t coefficient_range_rowgroup_count       = 0;
	size_t coefficient_logical_bytes_requested    = 0;
	size_t coefficient_range_bytes_read           = 0;
	size_t physical_page_bytes_covered            = 0;
	size_t full_physical_page_bytes                = 0;
	size_t coalesced_read_run_count                = 0;
	size_t selected_coefficient_count              = 0;
	size_t full_coefficient_count                  = 0;
	double selected_coefficient_ratio              = 0.0;
	double physical_page_coverage_ratio            = 0.0;
	// Decode worksets are bounded independently of the decoded-rowgroup cache.
	// A single rowgroup may exceed the configured budget and is then executed
	// alone; that exceptional case is counted explicitly.
	size_t decode_workset_capacity_bytes          = 0U;
	size_t max_estimated_decode_workset_bytes      = 0U;
	size_t oversized_decode_rowgroup_count         = 0U;
	// The persistent output and chunk arenas have independent capacity plans
	// and growth counters. This separates high-water expansion from unrelated
	// process-global allocator activity.
	size_t decode_workset_capacity_plan_image_count          = 0U;
	size_t decode_workset_output_arena_capacity_plan_bytes   = 0U;
	size_t decode_workset_output_arena_requested_bytes       = 0U;
	size_t decode_workset_output_arena_capacity_bytes        = 0U;
	size_t decode_workset_output_arena_growth_count          = 0U;
	size_t decode_workset_output_arena_growth_bytes          = 0U;
	size_t decode_workset_chunk_arena_capacity_plan_bytes    = 0U;
	size_t decode_workset_chunk_arena_requested_bytes        = 0U;
	size_t decode_workset_chunk_arena_capacity_bytes         = 0U;
	size_t decode_workset_chunk_arena_growth_count           = 0U;
	size_t decode_workset_chunk_arena_growth_bytes           = 0U;
	bool   bounded_double_buffer_enabled            = false;
	std::string bounded_double_buffer_policy         = "automatic";
	bool   bounded_double_buffer_candidate           = false;
	size_t bounded_double_buffer_workset_count      = 0U;
	size_t bounded_double_buffer_peak_estimated_bytes = 0U;
	// Column binding is constructed once per decode batch. The expression
	// scan count makes the intended O(expressions + rowgroups) path auditable.
	double column_binding_ms                    = 0.0;
	size_t column_binding_expression_scan_count = 0U;
	size_t column_binding_rowgroup_count         = 0U;
	// Block-major plan-lifetime coordinate index and deterministic two-pass
	// active-output scheduler diagnostics. Appended for aggregate compatibility.
	size_t coordinate_group_lookup_count          = 0U;
	size_t coordinate_group_index_entries         = 0U;
	size_t coordinate_group_index_populated       = 0U;
	size_t coordinate_group_index_holes           = 0U;
	size_t coordinate_group_index_bytes           = 0U;
	double coordinate_group_index_density         = 0.0;
	size_t planless_transform_source_contribution_visit_count = 0U;
	size_t planless_transform_output_workset_ownership_count   = 0U;
	double planless_transform_group_workset_build_ms           = 0.0;
	double planless_transform_active_output_count_ms           = 0.0;
	double planless_transform_active_output_prefix_ms          = 0.0;
	double planless_transform_active_output_fill_ms            = 0.0;
};

} // namespace galp::jpeg

#endif // GALP_WITH_JPEG_DCT

#endif // GALP_JPEG_DCT_DIAGNOSTICS_HPP
