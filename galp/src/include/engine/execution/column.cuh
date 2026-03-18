// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/execution/column.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_COLUMN_CUH
#define ENGINE_EXECUTION_COLUMN_CUH

#include "engine/execution/common.cuh"

namespace dispatch {

ValueStore decompress(const expr::Expression& expression, const ExecutionConfig& cfg = {});

} // namespace dispatch

#endif // ENGINE_EXECUTION_COLUMN_CUH
