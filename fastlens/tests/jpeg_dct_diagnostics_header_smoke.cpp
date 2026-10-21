#include <fastlens/jpeg_dct_diagnostics.hpp>

#if FASTLENS_WITH_JPEG_DCT
static_assert(sizeof(fastlens::jpeg::JpegDctDeviceCacheStats) > 0);
static_assert(sizeof(fastlens::jpeg::JpegDctDeviceExecutionStats) > 0);
#endif
