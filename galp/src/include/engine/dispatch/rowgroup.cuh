// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/dispatch/rowgroup.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_DISPATCH_ROWGROUP_CUH
#define ENGINE_DISPATCH_ROWGROUP_CUH

#include "engine/dispatch/common.cuh"

namespace dispatch {

RowgroupDecompressResult decompress_rowgroup(const std::vector<expr::Expression>& expressions, const Config& cfg = {});
BenchmarkResult          benchmark_rowgroup(const std::vector<expr::Expression>& expressions, const Config& cfg = {});

} // namespace dispatch

#endif // ENGINE_DISPATCH_ROWGROUP_CUH
