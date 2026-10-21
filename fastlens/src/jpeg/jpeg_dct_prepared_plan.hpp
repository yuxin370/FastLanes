#ifndef FASTLENS_JPEG_DCT_PREPARED_PLAN_HPP
#define FASTLENS_JPEG_DCT_PREPARED_PLAN_HPP

#include "jpeg/jpeg_dct_plan_types.hpp"
#include <cstdint>
#include <memory>
#include <utility>

namespace fastlens::jpeg {

struct JpegDctDeviceBatchPreparedPlan::Impl {
	Impl(detail::JpegDctDeviceBatchPlan plan_in,
	     JpegDctDeviceBatchOptions      options_in,
	     std::shared_ptr<const uint8_t> owner_token_in)
	    : plan(std::move(plan_in))
	    , options(std::move(options_in))
	    , owner_token(std::move(owner_token_in)) {
	}

	detail::JpegDctDeviceBatchPlan plan;
	JpegDctDeviceBatchOptions      options;
	std::shared_ptr<const uint8_t> owner_token;
};

} // namespace fastlens::jpeg

#endif // FASTLENS_JPEG_DCT_PREPARED_PLAN_HPP
