if (NOT DEFINED PYTHON_EXECUTABLE)
        message(FATAL_ERROR "PYTHON_EXECUTABLE is required")
endif ()
if (NOT DEFINED REPO_ROOT)
        message(FATAL_ERROR "REPO_ROOT is required")
endif ()
if (NOT DEFINED TEST_TMP)
        message(FATAL_ERROR "TEST_TMP is required")
endif ()

set(script "${REPO_ROOT}/scripts/my_tool/bench_pipeline_summary.py")
if (NOT EXISTS "${script}")
        message(FATAL_ERROR "pipeline benchmark summary script does not exist: ${script}")
endif ()

file(MAKE_DIRECTORY "${TEST_TMP}")
set(sample_log "${TEST_TMP}/pipeline_benchmark_sample.log")
file(WRITE "${sample_log}" [=[
Pipeline benchmark results:
  dataset_images: 10
  requested_images: 10
  mode: compare
  crop: 0,0,16,16
  outputs_match: 1
  dct_coeffs: all
  pushdown_selected_coefficient_count: 8
  pushdown_full_coefficient_count: 64
  pushdown_selected_coefficient_ratio: 0.125
  pushdown_output_coefficients: 80
  pushdown_output_bytes: 160
  pushdown_selected_vector_ratio: 0.25
  full_then_crop_selected_coefficient_count: 8
  full_then_crop_full_coefficient_count: 64
  full_then_crop_selected_coefficient_ratio: 0.125
  full_then_crop_output_coefficients: 80
  full_then_crop_output_bytes: 160
  full_then_crop_selected_vector_ratio: 1
  pushdown_total_ms: 10
  full_then_crop_total_ms: 25
  pushdown_plan_ms: 1
  pushdown_read_decode_ms: 9
  pushdown_decode_ms: 7
  pushdown_gather_ms: 2
  pushdown_decoded_gather_ms: 1.5
  pushdown_cached_gather_ms: 0.5
  pushdown_sync_rowgroup_read_ms: 5
  pushdown_prefetch_queue_start_ms: 0.1
  pushdown_prefetch_wait_ms: 0.2
  pushdown_prefetch_depth_block_ms: 0.3
  pushdown_prefetch_rowgroup_read_ms: 4
  pushdown_prefetch_ready_ahead_ms: 3
  pushdown_prefetch_initial_cache_hit_rowgroup_count: 2
  pushdown_prefetch_candidate_rowgroup_count: 8
  pushdown_prefetch_active_shard_count: 1
  pushdown_prefetch_config_disabled_shard_count: 0
  pushdown_prefetch_all_hit_shard_count: 0
  pushdown_prefetch_small_batch_disabled_shard_count: 0
  pushdown_prefetch_selected_vector_disabled_shard_count: 0
  pushdown_prefetch_selected_vector_miss_rowgroup_count: 0
  pushdown_prefetch_initial_hit_runtime_miss_count: 0
  pushdown_prefetch_skipped_repeated_runtime_miss_count: 0
  pushdown_prefetched_rowgroup_count: 8
  pushdown_prefetch_consumed_as_hit_count: 1
  pushdown_prefetch_skipped_repeated_rowgroup_count: 2
  pushdown_prefetch_consumed_as_hit_read_ms: 0.7
  pushdown_prefetch_consumed_as_hit_wait_ms: 0.05
  pushdown_workset_count: 3
  pushdown_decode_kernel_launch_count: 3
  pushdown_gather_kernel_launch_count: 3
  pushdown_scratch_allocation_count: 1
  pushdown_internal_sync_count: 3
  pushdown_runtime_policy_decision: selected-vector
Pipeline benchmark results:
  dataset_images: 20
  requested_images: 20
  mode: compare
  crop: 30,40,224,224
  outputs_match: 1
  dct_coeffs: all
  pushdown_selected_coefficient_count: 8
  pushdown_full_coefficient_count: 64
  pushdown_selected_coefficient_ratio: 0.125
  pushdown_output_coefficients: 160
  pushdown_output_bytes: 320
  pushdown_selected_vector_ratio: 0.5
  full_then_crop_selected_coefficient_count: 8
  full_then_crop_full_coefficient_count: 64
  full_then_crop_selected_coefficient_ratio: 0.125
  full_then_crop_output_coefficients: 160
  full_then_crop_output_bytes: 320
  full_then_crop_selected_vector_ratio: 1
  pushdown_total_ms: 20
  full_then_crop_total_ms: 30
  pushdown_plan_ms: 2
  pushdown_read_decode_ms: 18
  pushdown_decode_ms: 15
  pushdown_gather_ms: 3
  pushdown_decoded_gather_ms: 2.25
  pushdown_cached_gather_ms: 0.75
  pushdown_sync_rowgroup_read_ms: 8
  pushdown_prefetch_queue_start_ms: 0.2
  pushdown_prefetch_wait_ms: 0.4
  pushdown_prefetch_depth_block_ms: 0.6
  pushdown_prefetch_rowgroup_read_ms: 7
  pushdown_prefetch_ready_ahead_ms: 6
  pushdown_prefetch_initial_cache_hit_rowgroup_count: 4
  pushdown_prefetch_candidate_rowgroup_count: 16
  pushdown_prefetch_active_shard_count: 1
  pushdown_prefetch_config_disabled_shard_count: 0
  pushdown_prefetch_all_hit_shard_count: 0
  pushdown_prefetch_small_batch_disabled_shard_count: 0
  pushdown_prefetch_selected_vector_disabled_shard_count: 0
  pushdown_prefetch_selected_vector_miss_rowgroup_count: 0
  pushdown_prefetch_initial_hit_runtime_miss_count: 0
  pushdown_prefetch_skipped_repeated_runtime_miss_count: 0
  pushdown_prefetched_rowgroup_count: 16
  pushdown_prefetch_consumed_as_hit_count: 2
  pushdown_prefetch_skipped_repeated_rowgroup_count: 4
  pushdown_prefetch_consumed_as_hit_read_ms: 1.4
  pushdown_prefetch_consumed_as_hit_wait_ms: 0.1
  pushdown_workset_count: 4
  pushdown_decode_kernel_launch_count: 4
  pushdown_gather_kernel_launch_count: 4
  pushdown_scratch_allocation_count: 2
  pushdown_internal_sync_count: 4
  pushdown_runtime_policy_decision: mixed
]=])

file(APPEND "${sample_log}" [=[
Pipeline benchmark results:
  dataset_images: 10
  requested_images: 10
  mode: auto
  crop: 0,0,16,16
  outputs_match: 1
  dct_coeffs: list:0,2,5
  auto_pushdown_windows: 1
  auto_full_then_crop_windows: 1
  auto_policy_ms: 0.5
  auto_total_ms: 35
  auto_policy_reason: coefficient_selection_pushdown(selected_coefficients=3)
  auto_policy_fast_gate_windows: 0
  auto_policy_estimate_windows: 2
  auto_policy_selected_blocks: 10
  auto_policy_full_blocks: 20
  auto_policy_selected_block_ratio: 0.5
  auto_policy_selected_vectors: 10
  auto_policy_full_vectors: 20
  auto_policy_selected_vector_ratio: 0.5
  auto_policy_touched_rowgroups: 2
  auto_policy_full_rowgroups: 4
  auto_policy_estimated_pushdown_worksets: 2
  auto_policy_estimated_full_worksets: 4
  auto_policy_estimated_pushdown_gather_items: 10
  auto_policy_estimated_full_gather_items: 20
  auto_policy_pushdown_reuse_candidate_rowgroups: 1
  auto_policy_full_reuse_candidate_rowgroups: 1
  auto_policy_touched_rowgroup_ratio: 0.5
  auto_policy_pushdown_reuse_candidate_ratio: 0.5
  auto_policy_full_reuse_candidate_ratio: 0.25
  auto_policy_avg_full_blocks_per_rowgroup: 5
  auto_policy_empty_windows: 0
  auto_policy_crop_covers_full_windows: 0
  auto_policy_coefficient_pushdown_windows: 1
  auto_policy_very_small_crop_windows: 0
  auto_policy_small_window_full_windows: 0
  auto_policy_large_window_pushdown_windows: 0
  auto_policy_workset_overhead_full_windows: 0
  auto_policy_tiny_rowgroups_full_windows: 0
  auto_policy_touches_most_rowgroups_windows: 0
  auto_policy_gather_output_full_windows: 0
  auto_policy_saves_enough_blocks_windows: 0
  auto_policy_savings_too_small_windows: 1
  pushdown_selected_coefficient_count: 8
  pushdown_full_coefficient_count: 64
  pushdown_selected_coefficient_ratio: 0.125
  pushdown_output_coefficients: 80
  pushdown_output_bytes: 160
  pushdown_selected_vector_ratio: 0.25
  full_then_crop_selected_coefficient_count: 8
  full_then_crop_full_coefficient_count: 64
  full_then_crop_selected_coefficient_ratio: 0.125
  full_then_crop_output_coefficients: 80
  full_then_crop_output_bytes: 160
  full_then_crop_selected_vector_ratio: 1
  pushdown_total_ms: 10
  full_then_crop_total_ms: 25
  pushdown_plan_ms: 1
  pushdown_read_decode_ms: 9
  pushdown_decode_ms: 7
  pushdown_gather_ms: 2
  pushdown_decoded_gather_ms: 1.5
  pushdown_cached_gather_ms: 0.5
  pushdown_sync_rowgroup_read_ms: 5
  pushdown_prefetch_queue_start_ms: 0.1
  pushdown_prefetch_wait_ms: 0.2
  pushdown_prefetch_depth_block_ms: 0.3
  pushdown_prefetch_rowgroup_read_ms: 4
  pushdown_prefetch_ready_ahead_ms: 3
  pushdown_prefetch_initial_cache_hit_rowgroup_count: 2
  pushdown_prefetch_candidate_rowgroup_count: 8
  pushdown_prefetch_active_shard_count: 1
  pushdown_prefetch_config_disabled_shard_count: 0
  pushdown_prefetch_all_hit_shard_count: 0
  pushdown_prefetch_small_batch_disabled_shard_count: 0
  pushdown_prefetch_selected_vector_disabled_shard_count: 0
  pushdown_prefetch_selected_vector_miss_rowgroup_count: 0
  pushdown_prefetch_initial_hit_runtime_miss_count: 0
  pushdown_prefetch_skipped_repeated_runtime_miss_count: 0
  pushdown_prefetched_rowgroup_count: 8
  pushdown_prefetch_consumed_as_hit_count: 1
  pushdown_prefetch_skipped_repeated_rowgroup_count: 2
  pushdown_prefetch_consumed_as_hit_read_ms: 0.7
  pushdown_prefetch_consumed_as_hit_wait_ms: 0.05
  pushdown_workset_count: 3
  pushdown_decode_kernel_launch_count: 3
  pushdown_gather_kernel_launch_count: 3
  pushdown_scratch_allocation_count: 1
  pushdown_internal_sync_count: 3
  pushdown_runtime_policy_decision: selected-vector
]=])

execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" "${script}"
                --datasets small,large,auto
                --image-sizes 32x32,varies,16x16
                --require-match
                --require-default-fields
                "${sample_log}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
)
if (NOT result EQUAL 0)
        message(FATAL_ERROR "pipeline benchmark summary script failed: ${error}")
endif ()

foreach (expected
        "| dataset | image_size | crop_size | mode | outputs_match |"
        "| small | 32x32 | 16x16 | compare | 1 |"
        "| large | varies | 224x224 | compare | 1 |"
        "| auto | 16x16 | 16x16 | auto | 1 |"
        "| 10 | 25 | 2.5 | 15 |"
        "| 2 | 1.5 | 0.5 | 5 | 0.1 | 0.2 | 0.3 | 4 | 3 | 2 | 8 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 1 | 2 | 0.7 | 0.05 |"
        "selected-vector |"
        "coefficient_selection_pushdown(selected_coefficients=3)")
        string(FIND "${output}" "${expected}" found)
        if (found EQUAL -1)
                message(FATAL_ERROR
                        "pipeline benchmark summary output did not contain '${expected}'. Output:\n${output}")
        endif ()
endforeach ()

execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" "${script}"
                --format csv
                --datasets small,large,auto
                --image-sizes 32x32,varies,16x16
                "${sample_log}"
        RESULT_VARIABLE csv_result
        OUTPUT_VARIABLE csv_output
        ERROR_VARIABLE csv_error
)
if (NOT csv_result EQUAL 0)
        message(FATAL_ERROR "pipeline benchmark summary CSV output failed: ${csv_error}")
endif ()

foreach (expected
        "dataset,image_size,crop_size,mode,outputs_match"
        "small,32x32,16x16,compare,1"
        "large,varies,224x224,compare,1"
        "auto,16x16,16x16,auto,1"
        "20,30,1.5,10")
        string(FIND "${csv_output}" "${expected}" found)
        if (found EQUAL -1)
                message(FATAL_ERROR
                        "pipeline benchmark summary CSV output did not contain '${expected}'. Output:\n${csv_output}")
        endif ()
endforeach ()

execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" "${script}" --datasets only-one "${sample_log}"
        RESULT_VARIABLE mismatch_result
        OUTPUT_VARIABLE mismatch_output
        ERROR_VARIABLE mismatch_error
)
if (mismatch_result EQUAL 0)
        message(FATAL_ERROR
                "pipeline benchmark summary script should reject mismatched --datasets count. Output:\n${mismatch_output}")
endif ()
string(FIND "${mismatch_error}" "--datasets has 1 labels but found 3 result blocks" mismatch_found)
if (mismatch_found EQUAL -1)
        message(FATAL_ERROR
                "pipeline benchmark summary mismatch error was not explanatory. Error:\n${mismatch_error}")
endif ()

set(mismatch_log "${TEST_TMP}/pipeline_benchmark_mismatch.log")
file(WRITE "${mismatch_log}" [=[
Pipeline benchmark results:
  mode: compare
  crop: 0,0,16,16
  outputs_match: 0
  pushdown_total_ms: 10
  full_then_crop_total_ms: 20
]=])

execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" "${script}" --dataset mismatch --require-match "${mismatch_log}"
        RESULT_VARIABLE require_match_result
        OUTPUT_VARIABLE require_match_output
        ERROR_VARIABLE require_match_error
)
if (require_match_result EQUAL 0)
        message(FATAL_ERROR
                "pipeline benchmark summary script should reject outputs_match=0. Output:\n${require_match_output}")
endif ()
string(FIND "${require_match_error}" "mismatch: outputs_match=0" require_match_found)
if (require_match_found EQUAL -1)
        message(FATAL_ERROR
                "pipeline benchmark summary require-match error was not explanatory. Error:\n${require_match_error}")
endif ()

set(missing_field_log "${TEST_TMP}/pipeline_benchmark_missing_field.log")
file(WRITE "${missing_field_log}" [=[
Pipeline benchmark results:
  mode: compare
  crop: 0,0,16,16
  outputs_match: 1
  pushdown_total_ms: 10
  full_then_crop_total_ms: 20
]=])

execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" "${script}"
                --dataset missing
                --require-fields pushdown_workset_count
                "${missing_field_log}"
        RESULT_VARIABLE missing_field_result
        OUTPUT_VARIABLE missing_field_output
        ERROR_VARIABLE missing_field_error
)
if (missing_field_result EQUAL 0)
        message(FATAL_ERROR
                "pipeline benchmark summary script should reject missing required fields. Output:\n${missing_field_output}")
endif ()
string(FIND "${missing_field_error}" "missing: missing pushdown_workset_count" missing_field_found)
if (missing_field_found EQUAL -1)
        message(FATAL_ERROR
                "pipeline benchmark summary missing-field error was not explanatory. Error:\n${missing_field_error}")
endif ()
