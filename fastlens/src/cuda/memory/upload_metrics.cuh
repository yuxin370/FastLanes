// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// fastlens/src/cuda/memory/upload_metrics.cuh
// ────────────────────────────────────────────────────────
#ifndef FASTLENS_MEMORY_UPLOAD_METRICS_CUH
#define FASTLENS_MEMORY_UPLOAD_METRICS_CUH

#include <cstddef>

namespace fastlens::memory {

struct ArenaCapacityMetrics {
	size_t requested_bytes        = 0;
	size_t minimum_capacity_bytes = 0;
	size_t capacity_before_bytes  = 0;
	size_t capacity_bytes         = 0;
	size_t growth_count           = 0;
	size_t growth_bytes           = 0;
};

struct ArenaUploadMetrics {
	double layout_ms    = 0.0;
	double alloc_ms     = 0.0; // ensure_capacity + ensure_pinned_capacity
	double resolve_ms   = 0.0;
	double pack_ms      = 0.0; // std::memcpy into pinned
	double dma_issue_ms = 0.0; // cudaMemcpyAsync issue time
	double dma_gpu_ms   = 0.0; // optional GPU-event H2D duration, enabled by FASTLENS_MEASURE_H2D=1
	size_t dma_bytes    = 0;
	size_t dma_count    = 0;
	ArenaCapacityMetrics device_capacity {};
	ArenaCapacityMetrics pinned_capacity {};
};

} // namespace fastlens::memory

#endif // FASTLENS_MEMORY_UPLOAD_METRICS_CUH
