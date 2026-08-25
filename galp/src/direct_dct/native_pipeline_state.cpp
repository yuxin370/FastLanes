#include "direct_dct/native_pipeline_state.hpp"
#include <algorithm>

namespace galp::direct_dct {

void NativePipelineTraceBuffer::reset(const size_t request_count) {
	std::lock_guard lock(mutex_);
	events_.clear();
	events_.reserve(std::max(events_.capacity(), request_count * 9U + 1U));
	next_prepare_ordinal_    = 1U;
	next_stage_ordinal_      = 1U;
	next_read_ordinal_       = 1U;
	next_submission_ordinal_ = 1U;
	next_completion_ordinal_ = 1U;
}

void NativePipelineTraceBuffer::emit(PipelineTraceEvent event) {
	std::lock_guard lock(mutex_);
	switch (event.stage) {
	case PipelineTraceEvent::Stage::kPreparing:
		event.prepare_ordinal = next_prepare_ordinal_++;
		break;
	case PipelineTraceEvent::Stage::kStaged:
		event.stage_ordinal = next_stage_ordinal_++;
		break;
	case PipelineTraceEvent::Stage::kReadStarted:
		event.read_ordinal = next_read_ordinal_++;
		break;
	case PipelineTraceEvent::Stage::kSubmitted:
		event.submission_ordinal = next_submission_ordinal_++;
		break;
	case PipelineTraceEvent::Stage::kCompleted:
		event.completion_ordinal = next_completion_ordinal_++;
		break;
	default:
		break;
	}
	events_.push_back(event);
}

std::vector<PipelineTraceEvent> NativePipelineTraceBuffer::snapshot() const {
	std::lock_guard lock(mutex_);
	return events_;
}

size_t NativePipelineTraceBuffer::size() const {
	std::lock_guard lock(mutex_);
	return events_.size();
}

} // namespace galp::direct_dct
