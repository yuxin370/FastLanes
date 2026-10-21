// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// fastlens/src/engine/workset/upload.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_RUNTIME_WORKSET_UPLOAD_CUH
#define ENGINE_RUNTIME_WORKSET_UPLOAD_CUH

#include "engine/workset/model.cuh"

namespace fastlens::runtime {

void            clear_mixed_slots(WorksetSlots& slots);
UploadBreakdown upload_workset(ExecutionWorkset& workset, const ExecutionConfig& cfg);

} // namespace fastlens::runtime

#endif // ENGINE_RUNTIME_WORKSET_UPLOAD_CUH
