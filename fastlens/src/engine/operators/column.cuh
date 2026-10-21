// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// fastlens/src/engine/operators/column.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_COLUMN_CUH
#define ENGINE_EXECUTION_COLUMN_CUH

#include "core/data/value_store.cuh"
#include "engine/config.cuh"
#include "core/expression.cuh"

namespace fastlens::execution {

ValueStore decompress(const fastlens::expression::Expression& expression, const ExecutionConfig& cfg = {});

} // namespace fastlens::execution

#endif // ENGINE_EXECUTION_COLUMN_CUH
