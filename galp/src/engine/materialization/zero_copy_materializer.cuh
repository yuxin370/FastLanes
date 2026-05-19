// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/format/zero_copy_materializer.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_ENGINE_FORMAT_ZERO_COPY_MATERIALIZER_CUH
#define GALP_ENGINE_FORMAT_ZERO_COPY_MATERIALIZER_CUH

#include "format/rowgroup_io.cuh"

namespace galp::format::detail {

void     release_transient_materialized_rowgroup(Rowgroup& rowgroup);
Rowgroup make_owning_rowgroup(Rowgroup rowgroup);
Rowgroup materialize_zero_copy_rowgroup(ZeroCopyRowgroup zero_copy);

} // namespace galp::format::detail

#endif // GALP_ENGINE_FORMAT_ZERO_COPY_MATERIALIZER_CUH
