#include <fastlens/jpeg_dct_format.hpp>

#if FASTLENS_WITH_JPEG_DCT
static_assert(sizeof(fastlens::jpeg::JpegDctSpatialOrder) > 0);
static_assert(sizeof(fastlens::jpeg::JpegDctDatasetMetadata) > 0);
#endif
