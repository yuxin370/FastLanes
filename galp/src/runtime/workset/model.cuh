// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/runtime/workset/model.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_RUNTIME_WORKSET_MODEL_CUH
#define ENGINE_RUNTIME_WORKSET_MODEL_CUH

#include "execution/batch.cuh"
#include "execution/config.cuh"
#include "execution/internal/expr_ops.cuh"
#include "cuda/memory/cuda_raii.cuh"
#include <memory>
#include <optional>
#include <vector>

namespace galp::runtime {

using galp::execution::add_expression_to_batch;
using galp::execution::ExecutionConfig;

struct UploadBreakdown {
	double                           prep_ms              = 0.0;
	double                           prep_reset_ms        = 0.0;
	double                           prep_output_arena_ms = 0.0;
	double                           prep_bind_ms         = 0.0;
	double                           prep_slots_ms        = 0.0;
	double                           arena_pack_ms        = 0.0;
	double                           arena_upload_ms      = 0.0;
	double                           event_record_ms      = 0.0;
	galp::memory::ArenaUploadMetrics arena {};
};

struct WorksetBuffers {
	using HostBatches   = typename galp::execution::BatchSetFromList<galp::execution::SupportedTypes>::type;
	using DeviceBatches = typename galp::execution::DeviceBatchSetFromList<galp::execution::SupportedTypes>::type;

	HostBatches                                host_batches;
	DeviceBatches                              device_batches;
	std::unique_ptr<galp::memory::DeviceArena> chunk_arena;
	size_t                                     payload_arena_bytes = 0;
};

struct WorksetOutputs {
	std::optional<GPUArray<uint8_t>> arena;
	size_t                           capacity_bytes = 0;
	size_t                           used_bytes     = 0;
	bool                             required       = false;
};

struct WorksetSlots {
	std::vector<galp::execution::MixedWorkSlot>             mixed;
	std::optional<GPUArray<galp::execution::MixedWorkSlot>> owned;
	galp::execution::MixedWorkSlot*                         d = nullptr;
};

struct WorksetTransfer {
	galp::memory::CudaStream h2d_stream;
	galp::memory::CudaStream compute_stream;
	galp::memory::CudaStream d2h_stream;
	galp::memory::CudaEvent  h2d_ready_event;
};

struct ExecutionWorkset {
	WorksetBuffers  buffers;
	WorksetOutputs  outputs;
	WorksetSlots    slots;
	WorksetTransfer transfer;
};

} // namespace galp::runtime

#endif // ENGINE_RUNTIME_WORKSET_MODEL_CUH
