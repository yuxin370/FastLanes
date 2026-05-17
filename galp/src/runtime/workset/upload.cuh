// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/runtime/workset/upload.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_RUNTIME_WORKSET_UPLOAD_CUH
#define ENGINE_RUNTIME_WORKSET_UPLOAD_CUH

#include "runtime/workset/model.cuh"

namespace galp::runtime {

void            clear_mixed_slots(WorksetSlots& slots);
UploadBreakdown upload_workset(ExecutionWorkset& workset, const ExecutionConfig& cfg);

} // namespace galp::runtime

#endif // ENGINE_RUNTIME_WORKSET_UPLOAD_CUH
