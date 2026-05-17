// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/execution/column.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_COLUMN_CUH
#define ENGINE_EXECUTION_COLUMN_CUH

#include "core/data/value_store.cuh"
#include "execution/config.cuh"
#include "core/expression.cuh"

namespace galp::execution {

ValueStore decompress(const galp::expression::Expression& expression, const ExecutionConfig& cfg = {});

} // namespace galp::execution

#endif // ENGINE_EXECUTION_COLUMN_CUH
