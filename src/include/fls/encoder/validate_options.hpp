#ifndef FLS_ENCODER_VALIDATE_OPTIONS_HPP
#define FLS_ENCODER_VALIDATE_OPTIONS_HPP

#include "fls/encoder/encoding_options.hpp"
#include <stdexcept>

namespace fastlanes::detail {
inline void validate_encoding_options(const EncodingOptions& options) {
	if (options.worker_count == 0) {
		throw std::invalid_argument("EncodingOptions::worker_count must be greater than zero");
	}
	if (!options.deterministic_ordered_commit) {
		throw std::invalid_argument("FastLanes encoding requires deterministic ordered commit");
	}
}
} // namespace fastlanes::detail

#endif
