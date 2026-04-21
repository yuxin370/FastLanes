// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/memory/upload_metrics.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_MEMORY_UPLOAD_METRICS_CUH
#define FLSGPU_MEMORY_UPLOAD_METRICS_CUH

namespace flsgpu { namespace memory {

struct ArenaUploadMetrics {
	double layout_ms    = 0.0;
	double alloc_ms     = 0.0; // ensure_capacity + ensure_pinned_capacity
	double resolve_ms   = 0.0;
	double pack_ms      = 0.0; // std::memcpy into pinned
	double dma_issue_ms = 0.0; // cudaMemcpyAsync issue time
};

}} // namespace flsgpu::memory

#endif // FLSGPU_MEMORY_UPLOAD_METRICS_CUH
