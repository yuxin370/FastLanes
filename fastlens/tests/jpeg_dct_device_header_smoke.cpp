#include <fastlens/jpeg_dct_device.hpp>

#if FASTLENS_WITH_JPEG_DCT
static_assert(sizeof(fastlens::jpeg::JpegDctDeviceBatchOptions) > 0);
static_assert(sizeof(fastlens::jpeg::JpegDctDeviceBatch) > 0);
#endif
