// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/pipeline/streaming_pipeline.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_INTERNAL_STREAMING_PIPELINE_CUH
#define ENGINE_EXECUTION_INTERNAL_STREAMING_PIPELINE_CUH

#include <cstddef>
#include <deque>
#include <optional>
#include <vector>

namespace galp::runtime {

inline constexpr size_t kStreamingTargetWorkItems = 1u << 18;

template <typename ChunkT>
class StreamingChunkRing {
public:
	explicit StreamingChunkRing(const size_t max_inflight_chunks)
	    : chunks_((max_inflight_chunks == 0 ? 1U : max_inflight_chunks) + 1U)
	    , max_inflight_chunks_(max_inflight_chunks == 0 ? 1U : max_inflight_chunks) {
		build_idx_ = 0;
		for (size_t idx = 1; idx < chunks_.size(); ++idx) {
			idle_.push_back(idx);
		}
	}

	ChunkT& build_chunk() {
		return chunks_[*build_idx_];
	}

	template <typename ConsumeFn>
	bool drain_oldest(ConsumeFn&& consume) {
		if (inflight_.empty()) {
			return false;
		}
		const size_t idx = inflight_.front();
		inflight_.pop_front();
		consume(chunks_[idx]);
		idle_.push_back(idx);
		return true;
	}

	template <typename ReadyFn, typename ConsumeFn>
	size_t drain_ready(ReadyFn&& ready, ConsumeFn&& consume) {
		size_t drained = 0;
		while (!inflight_.empty()) {
			const size_t idx = inflight_.front();
			if (!ready(chunks_[idx])) {
				break;
			}
			drain_oldest(consume);
			++drained;
		}
		return drained;
	}

	template <typename SubmitFn, typename ConsumeFn>
	void submit_build_and_rotate(SubmitFn&& submit, ConsumeFn&& consume) {
		const size_t idx = *build_idx_;
		if (submit(chunks_[idx])) {
			inflight_.push_back(idx);
			build_idx_.reset();
		}
		ensure_build_slot(consume);
	}

	template <typename SubmitFn, typename ConsumeFn>
	void flush(SubmitFn&& submit, ConsumeFn&& consume) {
		if (build_idx_.has_value()) {
			const size_t idx = *build_idx_;
			if (submit(chunks_[idx])) {
				inflight_.push_back(idx);
				build_idx_.reset();
			}
		}

		while (drain_oldest(consume)) {}
	}

	template <typename Fn>
	void for_each_chunk(Fn&& fn) {
		for (auto& chunk : chunks_) {
			fn(chunk);
		}
	}

private:
	template <typename ConsumeFn>
	void ensure_build_slot(ConsumeFn&& consume) {
		if (build_idx_.has_value()) {
			return;
		}
		if (idle_.empty()) {
			drain_oldest(consume);
		}
		if (!idle_.empty()) {
			build_idx_ = idle_.front();
			idle_.pop_front();
		}
	}

	std::vector<ChunkT>   chunks_;
	size_t                max_inflight_chunks_ = 1;
	std::deque<size_t>    idle_;
	std::deque<size_t>    inflight_;
	std::optional<size_t> build_idx_;
};

} // namespace galp::runtime

#endif // ENGINE_EXECUTION_INTERNAL_STREAMING_PIPELINE_CUH
