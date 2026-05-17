// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/runtime/workset/streams.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_RUNTIME_WORKSET_STREAMS_CUH
#define ENGINE_RUNTIME_WORKSET_STREAMS_CUH

#include "runtime/workset/model.cuh"
#include <cuda_runtime.h>

namespace galp::runtime {

bool         use_async_h2d();
bool         force_h2d_stream_for_sync();
cudaStream_t ensure_workset_h2d_stream(ExecutionWorkset& workset);
cudaStream_t ensure_workset_compute_stream(ExecutionWorkset& workset);
cudaStream_t ensure_workset_d2h_stream(ExecutionWorkset& workset);
cudaEvent_t  ensure_workset_h2d_ready_event(ExecutionWorkset& workset);

} // namespace galp::runtime

#endif // ENGINE_RUNTIME_WORKSET_STREAMS_CUH
