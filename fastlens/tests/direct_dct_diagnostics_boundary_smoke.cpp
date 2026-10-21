#include <fastlens/diagnostics/direct_dct.hpp>

#if FASTLENS_WITH_JPEG_DCT
#ifndef FASTLENS_DIRECT_DCT_DIAGNOSTICS_API
#error "Direct-DCT diagnostics boundary marker is missing"
#endif

static_assert(sizeof(fastlens::jpeg::JpegDctDeviceCacheStats) > 0);
static_assert(sizeof(fastlens::jpeg::JpegDctDeviceExecutionStats) > 0);
#endif
