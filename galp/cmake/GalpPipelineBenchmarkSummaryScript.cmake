if (NOT DEFINED PYTHON_EXECUTABLE)
        message(FATAL_ERROR "PYTHON_EXECUTABLE is required")
endif ()
if (NOT DEFINED REPO_ROOT)
        message(FATAL_ERROR "REPO_ROOT is required")
endif ()
if (NOT DEFINED TEST_TMP)
        message(FATAL_ERROR "TEST_TMP is required")
endif ()

set(script "${REPO_ROOT}/scripts/my_tool/summarize_pipeline_benchmark.py")
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
  pushdown_selected_vector_ratio: 0.25
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
  pushdown_selected_vector_ratio: 0.5
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

execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" "${script}"
                --datasets small,large
                --image-sizes 32x32,varies
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
        "| 10 | 25 | 2.5 | 15 |"
        "| 2 | 1.5 | 0.5 | 5 | 0.1 | 0.2 | 0.3 | 4 | 3 | 2 | 8 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 1 | 2 | 0.7 | 0.05 |"
        "selected-vector |")
        string(FIND "${output}" "${expected}" found)
        if (found EQUAL -1)
                message(FATAL_ERROR
                        "pipeline benchmark summary output did not contain '${expected}'. Output:\n${output}")
        endif ()
endforeach ()

execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" "${script}"
                --format csv
                --datasets small,large
                --image-sizes 32x32,varies
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
        "20,30,1.5,10")
        string(FIND "${csv_output}" "${expected}" found)
        if (found EQUAL -1)
                message(FATAL_ERROR
                        "pipeline benchmark summary CSV output did not contain '${expected}'. Output:\n${csv_output}")
        endif ()
endforeach ()

set(prefetch_runner "${REPO_ROOT}/scripts/my_tool/run_jpeg_device_prefetch_benchmark.py")
if (NOT EXISTS "${prefetch_runner}")
        message(FATAL_ERROR "JPEG device prefetch benchmark runner does not exist: ${prefetch_runner}")
endif ()

execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" "${prefetch_runner}"
                --dry-run
                --repeats 2
                --manifest "${TEST_TMP}/manifest.bin"
                --work-dir "${TEST_TMP}/runner"
                --summary-out "${TEST_TMP}/runner_summary.csv"
                --pairs-out "${TEST_TMP}/runner_pairs.csv"
        RESULT_VARIABLE dry_run_result
        OUTPUT_VARIABLE dry_run_output
        ERROR_VARIABLE dry_run_error
)
if (NOT dry_run_result EQUAL 0)
        message(FATAL_ERROR "JPEG device prefetch benchmark runner dry-run failed: ${dry_run_error}")
endif ()
string(REGEX MATCHALL "galp_cli pipeline_benchmark" dry_run_pipeline_matches "${dry_run_output}")
list(LENGTH dry_run_pipeline_matches dry_run_pipeline_count)
if (NOT dry_run_pipeline_count EQUAL 8)
        message(FATAL_ERROR
                "JPEG device prefetch benchmark runner dry-run should emit 8 pipeline_benchmark commands. Output:\n${dry_run_output}")
endif ()
foreach (expected
        "small_prefetch_on_sample1.log"
        "small_prefetch_off_sample1.log"
        "large_prefetch_on_sample2.log"
        "large_prefetch_off_sample2.log"
        "small_prefetch_on_sample1,small_prefetch_off_sample1"
        "--no-jpeg-device-rowgroup-prefetch"
        "--jpeg-device-prefetch-min-batches 2"
        "--require-match"
        "runner_summary.csv"
        "pair-prefetch-on-off"
        "runner_pairs.csv")
        string(FIND "${dry_run_output}" "${expected}" found)
        if (found EQUAL -1)
                message(FATAL_ERROR
                        "JPEG device prefetch benchmark runner dry-run did not contain '${expected}'. Output:\n${dry_run_output}")
        endif ()
endforeach ()

set(pair_summary_input "${TEST_TMP}/pair_summary_input.csv")
set(pair_summary_output "${TEST_TMP}/pair_summary_output.csv")
file(WRITE "${pair_summary_input}" [=[
dataset,pushdown_total_ms,pushdown_prefetch_wait_ms,pushdown_prefetch_queue_start_ms,pushdown_prefetch_depth_block_ms,pushdown_prefetch_ready_ahead_ms,pushdown_prefetch_consumed_as_hit_count,pushdown_prefetch_consumed_as_hit_read_ms,pushdown_prefetch_consumed_as_hit_wait_ms,pushdown_prefetch_selected_vector_disabled_shard_count,pushdown_prefetch_selected_vector_miss_rowgroup_count,pushdown_prefetch_initial_hit_runtime_miss_count,pushdown_prefetch_skipped_repeated_runtime_miss_count,pushdown_prefetched_rowgroup_count
small_prefetch_on_sample1,12,0.2,0.1,0.3,4,1,0.7,0.05,0,0,0,0,8
small_prefetch_off_sample1,10,,,,,,,,,,,,
small_prefetch_on_sample10,15,0.5,0.1,0.3,2,0,0,0,0,0,0,0,8
small_prefetch_off_sample10,12,,,,,,,,,,,,
small_prefetch_on_sample2,14,0.4,0.1,0.3,3,0,0,0,0,0,0,0,8
small_prefetch_off_sample2,11,,,,,,,,,,,,
large_prefetch_on_sample1,18,0.1,0.2,0.4,8,0,0,0,0,0,0,0,16
large_prefetch_off_sample1,24,,,,,,,,,,,,
]=])
execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" "${prefetch_runner}"
                --pair-summary-only "${pair_summary_input}"
                --pairs-out "${pair_summary_output}"
        RESULT_VARIABLE pair_result
        OUTPUT_VARIABLE pair_output
        ERROR_VARIABLE pair_error
)
if (NOT pair_result EQUAL 0)
        message(FATAL_ERROR "JPEG device prefetch pair summary failed: ${pair_error}")
endif ()
file(READ "${pair_summary_output}" pair_csv)
foreach (expected
        "window,sample,prefetch_on_total_ms,prefetch_off_total_ms,prefetch_delta_ms,prefetch_speedup_off_over_on"
	        "small,1,12,10,2,0.833333,0.2,0.1,0.3,4,1,0.7,0.05,0,0,0,0,8"
	        "small,2,14,11,3,0.785714,0.4,0.1,0.3,3,0,0,0,0,0,0,0,8"
	        "small,10,15,12,3,0.8,0.5,0.1,0.3,2,0,0,0,0,0,0,0,8"
	        "large,1,18,24,-6,1.33333,0.1,0.2,0.4,8,0,0,0,0,0,0,0,16")
        string(FIND "${pair_csv}" "${expected}" found)
        if (found EQUAL -1)
                message(FATAL_ERROR
                        "JPEG device prefetch pair summary output did not contain '${expected}'. Output:\n${pair_csv}")
	        endif ()
endforeach ()
string(FIND "${pair_csv}" "small,2," small_sample2_pos)
string(FIND "${pair_csv}" "small,10," small_sample10_pos)
if (small_sample2_pos EQUAL -1 OR small_sample10_pos EQUAL -1 OR small_sample2_pos GREATER small_sample10_pos)
        message(FATAL_ERROR
                "JPEG device prefetch pair summary should sort numeric sample labels before sample10. Output:\n${pair_csv}")
endif ()

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
string(FIND "${mismatch_error}" "--datasets has 1 labels but found 2 result blocks" mismatch_found)
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
