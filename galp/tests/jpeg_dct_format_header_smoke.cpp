#include <galp/jpeg_dct_format.hpp>

#if GALP_WITH_JPEG_DCT
static_assert(sizeof(galp::jpeg::JpegDctSpatialOrder) > 0);
static_assert(sizeof(galp::jpeg::JpegDctDatasetMetadata) > 0);
#endif
