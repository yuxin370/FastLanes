// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/dispatch/column.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_DISPATCH_COLUMN_CUH
#define ENGINE_DISPATCH_COLUMN_CUH

#include "engine/dispatch/common.cuh"

namespace dispatch {

DecompressResult decompress(const expr::Expression& expression, const Config& cfg = {});

} // namespace dispatch

#endif // ENGINE_DISPATCH_COLUMN_CUH
