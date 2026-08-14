// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/workset/model.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_RUNTIME_WORKSET_MODEL_CUH
#define ENGINE_RUNTIME_WORKSET_MODEL_CUH

#include "cuda/memory/cuda_raii.cuh"
#include "core/data/model.cuh"
#include "engine/config.cuh"
#include "engine/operators/batch.cuh"
#include "engine/operators/expr_ops.cuh"
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
	galp::memory::ArenaCapacityMetrics output_arena {};
	galp::memory::ArenaUploadMetrics arena {};
};

struct DeviceScatterCopy {
	const std::byte* source      = nullptr;
	std::byte*       destination = nullptr;
	size_t           size        = 0;
};

struct PendingDeviceScatter {
	std::shared_ptr<const galp::execution::PackedRowgroupDevicePayload> payload;
	std::byte*       device_packed  = nullptr;
	std::byte*       device_logical = nullptr;
	size_t           copy_begin     = 0;
};

struct WorksetBuffers {
	using HostBatches   = typename galp::execution::BatchSetFromList<galp::execution::SupportedTypes>::type;
	using DeviceBatches = typename galp::execution::DeviceBatchSetFromList<galp::execution::SupportedTypes>::type;

	HostBatches                                host_batches;
	DeviceBatches                              device_batches;
	std::unique_ptr<galp::memory::DeviceArena> chunk_arena;
	size_t                                     payload_arena_bytes = 0;
	std::vector<DeviceScatterCopy>             device_scatter_copies;
	std::vector<PendingDeviceScatter>          pending_device_scatters;
	DeviceScatterCopy*                         d_device_scatter_copies = nullptr;
};

struct WorksetOutputs {
	std::optional<GPUArray<uint8_t>> arena;
	size_t                           capacity_bytes = 0;
	size_t                           minimum_capacity_bytes = 0;
	size_t                           used_bytes     = 0;
	bool                             required       = false;
};

struct WorksetSlots {
	std::vector<galp::execution::MixedWorkSlot>             mixed;
	std::vector<galp::execution::MixedWorkSlot>             scalar_tail_mixed;
	std::optional<GPUArray<galp::execution::MixedWorkSlot>> owned;
	std::optional<GPUArray<galp::execution::MixedWorkSlot>> owned_scalar_tail;
	galp::execution::MixedWorkSlot*                         d = nullptr;
	galp::execution::MixedWorkSlot*                         d_scalar_tail = nullptr;

	void reserve_host_slots(const size_t mixed_required, const size_t scalar_tail_required) {
		if (mixed_required > mixed.capacity()) {
			mixed.reserve(mixed_required);
		}
		if (scalar_tail_required > scalar_tail_mixed.capacity()) {
			scalar_tail_mixed.reserve(scalar_tail_required);
		}
	}
};

struct WorksetTransfer {
	galp::memory::CudaStream h2d_stream;
	galp::memory::CudaStream compute_stream;
	galp::memory::CudaStream d2h_stream;
	galp::memory::CudaEvent  h2d_ready_event;
	galp::memory::CudaEvent  timing_queued_event;
	galp::memory::CudaEvent  timing_start_event;
	galp::memory::CudaEvent  timing_stop_event;
	// CUDA stream priorities are device-defined: numerically smaller values are
	// higher priority. JPEG Direct-DCT sets this to the device's least-priority
	// value before any persistent workset stream is created.
	int stream_priority = 0;
};

struct ExecutionWorkset {
	WorksetBuffers  buffers;
	WorksetOutputs  outputs;
	WorksetSlots    slots;
	WorksetTransfer transfer;
};

} // namespace galp::runtime

#endif // ENGINE_RUNTIME_WORKSET_MODEL_CUH
