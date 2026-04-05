// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/execution/internal/streaming_pipeline.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_INTERNAL_STREAMING_PIPELINE_CUH
#define ENGINE_EXECUTION_INTERNAL_STREAMING_PIPELINE_CUH

#include <array>
#include <cstddef>
#include <optional>
#include <utility>

namespace dispatch::runtime {

inline constexpr size_t kStreamingTargetWorkItems = 1u << 18;

template <typename ChunkT>
class StreamingDoubleBuffer {
public:
	ChunkT& build_chunk() {
		return chunks_[build_idx_];
	}

	ChunkT& other_chunk() {
		return chunks_[1U - build_idx_];
	}

	template <typename ConsumeFn>
	void consume_inflight(ConsumeFn&& consume) {
		if (!inflight_idx_.has_value()) {
			return;
		}
		consume(chunks_[*inflight_idx_]);
		inflight_idx_.reset();
	}

	template <typename SubmitFn, typename ConsumeFn>
	void submit_build_and_rotate(SubmitFn&& submit, ConsumeFn&& consume) {
		submit(chunks_[build_idx_]);
		consume_inflight(std::forward<ConsumeFn>(consume));
		inflight_idx_ = build_idx_;
		build_idx_    = 1U - build_idx_;
	}

	template <typename SubmitFn, typename ConsumeFn>
	void flush(SubmitFn&& submit, ConsumeFn&& consume) {
		// Submit whatever is in the build slot, then consume inflight (if any),
		// then consume the just-submitted build slot.  Do NOT blindly consume
		// the other slot — consume_inflight already handled it.
		submit(chunks_[build_idx_]);
		consume_inflight(consume);
		consume(chunks_[build_idx_]);
	}

private:
	std::array<ChunkT, 2U> chunks_ {};
	size_t                 build_idx_ = 0;
	std::optional<size_t>  inflight_idx_;
};

} // namespace dispatch::runtime

#endif // ENGINE_EXECUTION_INTERNAL_STREAMING_PIPELINE_CUH
