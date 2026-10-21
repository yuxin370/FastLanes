// Experimental interposition: change spatial execution only, using one native binary.
#include "fastlens/jpeg_dct_storage.hpp"
#include <atomic>
#include <cstdlib>
#include <dlfcn.h>
#include <stdexcept>

namespace {
std::atomic<uint64_t> calls {0}, selected {0}, full {0};
bool                  full_source = false;
} // namespace

extern "C" void crop_control_begin(int disabled) {
	full_source = disabled != 0;
	calls = selected = full = 0;
}

extern "C" void crop_control_stats(uint64_t* count, uint64_t* kept, uint64_t* all) {
	*count = calls;
	*kept  = selected;
	*all   = full;
}

namespace fastlens::jpeg {
JpegDctDeviceBatchPreparedPlan
JpegDctShardDatasetReader::PrepareDeviceDctBatch(const std::vector<JpegDctImageCropRequest>& requests,
                                                 const JpegDctDeviceBatchOptions&            options) {
	using Function = JpegDctDeviceBatchPreparedPlan (*)(
	    JpegDctShardDatasetReader*, const std::vector<JpegDctImageCropRequest>&, const JpegDctDeviceBatchOptions&);
	static auto original = [] {
		auto* handle = dlopen(std::getenv("FASTLENS_NATIVE_LIBRARY"), RTLD_NOW | RTLD_NOLOAD);
		if (!handle)
			throw std::runtime_error(dlerror());
		auto* symbol = dlsym(handle,
		                     "_ZN8fastlens4jpeg25JpegDctShardDatasetReader21PrepareDeviceDctBatchERKSt6vectorINS0_"
		                     "23JpegDctImageCropRequestESaIS3_EERKNS0_25JpegDctDeviceBatchOptionsE");
		if (!symbol)
			throw std::runtime_error(dlerror());
		return reinterpret_cast<Function>(symbol);
	}();
	auto control = options;
	if (full_source)
		control.crop_execution_mode = JpegDctCropExecutionMode::kFullSourceDecode;
	auto result = original(this, requests, control);
	++calls;
	selected += result.planned_selected_vector_count();
	full += result.full_vector_count();
	return result;
}
} // namespace fastlens::jpeg
