#include <galp/diagnostics/direct_dct.hpp>

#if GALP_WITH_JPEG_DCT
#ifndef GALP_DIRECT_DCT_DIAGNOSTICS_API
#error "Direct-DCT diagnostics boundary marker is missing"
#endif

static_assert(sizeof(galp::jpeg::JpegDctDeviceCacheStats) > 0);
static_assert(sizeof(galp::jpeg::JpegDctDeviceExecutionStats) > 0);
#endif
