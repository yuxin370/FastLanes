// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/execution/common.cuh
// ────────────────────────────────────────────────────────
// Barrel header — re-exports the dispatch-layer execution pieces that used to
// live in this file directly. Prefer including the narrower headers when you
// know what you need:
//   - config.cuh            — ExecutionConfig, LaunchStrategy
//   - batch.cuh             — Batch, BatchSet, DeviceBatch, DeviceBatchSet
//   - column_traits.cuh     — ColumnKindTraits, column_value_type
//   - internal/expr_ops.cuh — fill/free_device_expr, add_expression_to_batch
//   - internal/batch_kernel.cuh — launch_batch, finalize_batch
#ifndef ENGINE_EXECUTION_COMMON_CUH
#define ENGINE_EXECUTION_COMMON_CUH

#include "engine/execution/batch.cuh"
#include "engine/execution/column_traits.cuh"
#include "engine/execution/config.cuh"
#include "engine/execution/internal/batch_kernel.cuh"
#include "engine/execution/internal/expr_ops.cuh"

#endif // ENGINE_EXECUTION_COMMON_CUH
