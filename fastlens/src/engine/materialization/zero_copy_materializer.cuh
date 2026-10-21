// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// fastlens/src/format/zero_copy_materializer.cuh
// ────────────────────────────────────────────────────────
#ifndef FASTLENS_ENGINE_FORMAT_ZERO_COPY_MATERIALIZER_CUH
#define FASTLENS_ENGINE_FORMAT_ZERO_COPY_MATERIALIZER_CUH

#include "format/rowgroup_io.cuh"

namespace fastlens::format::detail {

void     release_transient_materialized_rowgroup(Rowgroup& rowgroup);
Rowgroup make_owning_rowgroup(Rowgroup rowgroup);
Rowgroup materialize_zero_copy_rowgroup(ZeroCopyRowgroup zero_copy);

} // namespace fastlens::format::detail

#endif // FASTLENS_ENGINE_FORMAT_ZERO_COPY_MATERIALIZER_CUH
