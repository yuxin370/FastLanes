// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/execution/config.cuh
// ────────────────────────────────────────────────────────
// Dispatch-layer knobs: launch strategy, unpack tile sizes, FREQ-patcher
// thresholds. Kept in its own header so consumers that only need the config
// struct don't pull in batch/expression/CUDA types.
#ifndef ENGINE_EXECUTION_CONFIG_CUH
#define ENGINE_EXECUTION_CONFIG_CUH

#include "engine/types.cuh"

namespace dispatch {

template <typename>
inline constexpr bool always_false_v = false;

enum class LaunchStrategy {
	TypedBatches,
	MixedDispatch,
};

struct ExecutionConfig {
	unsigned       unpack_n_vectors             = 1;
	unsigned       unpack_n_values              = 1;
	LaunchStrategy launch_strategy              = LaunchStrategy::MixedDispatch;
	bool           write_out                    = true;
	bool           freq_prefetch_all_branchless = false;
	bool           freq_hybrid_patcher          = false;
	float          freq_branchless_threshold    = 6.0f;

	constexpr DecodeChunk chunk() const {
		return DecodeChunk {unpack_n_vectors, unpack_n_values};
	}
};

} // namespace dispatch

#endif // ENGINE_EXECUTION_CONFIG_CUH
