// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/execution/column.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_COLUMN_CUH
#define ENGINE_EXECUTION_COLUMN_CUH

#include "engine/execution/common.cuh"

namespace galp::execution {

ValueStore decompress(const galp::expression::Expression& expression, const ExecutionConfig& cfg = {});

} // namespace galp::execution

#endif // ENGINE_EXECUTION_COLUMN_CUH
