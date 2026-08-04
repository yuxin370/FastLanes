// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/include/fls/encoder/encoding_options.hpp
// ────────────────────────────────────────────────────────
#ifndef FLS_ENCODER_ENCODING_OPTIONS_HPP
#define FLS_ENCODER_ENCODING_OPTIONS_HPP

#include "fls/api/api.hpp"
#include "fls/common/alias.hpp"

namespace fastlanes {

struct FLS_API EncodingOptions {
	// A value of one preserves the historical single-threaded encoder path.
	n_t worker_count = 1;

	// Zero selects an adaptive bounded window based on worker_count.
	n_t max_inflight_rowgroups = 0;

	// Zero disables the completed-payload byte limit. A single rowgroup larger
	// than this value is admitted when necessary to guarantee forward progress.
	n_t max_inflight_bytes = 0;

	// Zero selects an adaptive contiguous chunk size.
	n_t rowgroups_per_task = 0;

	// Ordered commit is a format-preserving invariant. False is rejected.
	bool deterministic_ordered_commit = true;
};

struct FLS_API EncodingStats {
	n_t    requested_worker_count      = 1;
	n_t    effective_worker_count      = 0;
	n_t    encoded_rowgroups           = 0;
	n_t    peak_inflight_rowgroups     = 0;
	n_t    peak_inflight_bytes         = 0;
	n_t    resolved_inflight_rowgroups = 0;
	n_t    resolved_rowgroups_per_task = 0;
	double preparation_wall_seconds    = 0.0;
	double encoding_wall_seconds       = 0.0;
	double finalization_wall_seconds   = 0.0;
	double total_wall_seconds          = 0.0;
};

} // namespace fastlanes

#endif // FLS_ENCODER_ENCODING_OPTIONS_HPP
