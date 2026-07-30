#include <galp/jpeg_dct_diagnostics.hpp>

#if GALP_WITH_JPEG_DCT
static_assert(sizeof(galp::jpeg::JpegDctDeviceCacheStats) > 0);
static_assert(sizeof(galp::jpeg::JpegDctDeviceExecutionStats) > 0);
#endif
