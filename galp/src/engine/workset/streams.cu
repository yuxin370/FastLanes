// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/workset/streams.cu
// ────────────────────────────────────────────────────────
#include "engine/workset/streams.cuh"
#include <cstdlib>

namespace galp::runtime {

bool use_async_h2d() {
	static const bool enabled = (std::getenv("GALP_DISABLE_ASYNC_H2D") == nullptr);
	return enabled;
}

bool force_h2d_stream_for_sync() {
	static const bool enabled = (std::getenv("GALP_FORCE_H2D_STREAM") != nullptr);
	return enabled;
}

cudaStream_t ensure_workset_h2d_stream(ExecutionWorkset& workset) {
	if (!use_async_h2d() && !force_h2d_stream_for_sync()) {
		return nullptr;
	}
	if (!workset.transfer.h2d_stream) {
		workset.transfer.h2d_stream.create_with_priority(cudaStreamNonBlocking, workset.transfer.stream_priority);
	}
	return workset.transfer.h2d_stream.get();
}

cudaStream_t ensure_workset_compute_stream(ExecutionWorkset& workset) {
	if (!workset.transfer.compute_stream) {
		workset.transfer.compute_stream.create_with_priority(cudaStreamNonBlocking, workset.transfer.stream_priority);
	}
	return workset.transfer.compute_stream.get();
}

cudaStream_t ensure_workset_d2h_stream(ExecutionWorkset& workset) {
	if (!workset.transfer.d2h_stream) {
		workset.transfer.d2h_stream.create_with_priority(cudaStreamNonBlocking, workset.transfer.stream_priority);
	}
	return workset.transfer.d2h_stream.get();
}

cudaEvent_t ensure_workset_h2d_ready_event(ExecutionWorkset& workset) {
	if (!workset.transfer.h2d_ready_event) {
		workset.transfer.h2d_ready_event.create_with_flags(cudaEventDisableTiming);
	}
	return workset.transfer.h2d_ready_event.get();
}

} // namespace galp::runtime
