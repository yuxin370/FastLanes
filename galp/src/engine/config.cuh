// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/config.cuh
// ────────────────────────────────────────────────────────
// Dispatch-layer knobs: launch strategy, unpack tile sizes, FREQ-patcher
// mode. Kept in its own header so consumers that only need the config
// struct don't pull in batch/expression/CUDA types.
#ifndef ENGINE_EXECUTION_CONFIG_CUH
#define ENGINE_EXECUTION_CONFIG_CUH

#include "core/data/model.cuh"
#include "core/types.cuh"

namespace galp::execution {

template <typename>
inline constexpr bool always_false_v = false;

enum class LaunchStrategy {
	TypedBatches,
	MixedDispatch,
};

enum class FreqPatcher {
	Stateful,   // default: stateful patcher
	Branchless, // FREQ extended format + PrefetchAllBranchless
	Hybrid,     // per-column choice between Stateful/Branchless by exception density
};

enum class DeltaDecoder {
	Auto,     // production policy; currently resolves to Register for DELTA I8/I16
	Stateful, // compatibility/A-B: one-value-at-a-time implementation
	Register, // whole-lane reservoir + segmented-prefix implementation
};

constexpr DeltaDecoder resolve_delta_decoder(const DeltaDecoder decoder) {
	return decoder == DeltaDecoder::Auto ? DeltaDecoder::Register : decoder;
}

constexpr bool delta_uses_register(const DeltaDecoder decoder) {
	return resolve_delta_decoder(decoder) == DeltaDecoder::Register;
}

// Hybrid threshold: avg exceptions per vector at which we switch to branchless.
inline constexpr float kFreqHybridBranchlessThreshold = 6.0f;

struct ExecutionConfig {
	unsigned       unpack_n_vectors          = 1;
	unsigned       unpack_n_values           = 1;
	LaunchStrategy launch_strategy           = LaunchStrategy::MixedDispatch;
	bool           write_out                 = true;
	DeltaDecoder   delta_decoder             = DeltaDecoder::Auto;
	FreqPatcher    freq_patcher              = FreqPatcher::Stateful;
	float          freq_branchless_threshold = kFreqHybridBranchlessThreshold;

	constexpr DecodeChunk chunk() const {
		return DecodeChunk {unpack_n_vectors, unpack_n_values};
	}
};

} // namespace galp::execution

#endif // ENGINE_EXECUTION_CONFIG_CUH
