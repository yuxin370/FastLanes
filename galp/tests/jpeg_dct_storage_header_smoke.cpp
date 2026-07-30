#include <galp/jpeg_dct_storage.hpp>

#if GALP_WITH_JPEG_DCT
static_assert(sizeof(galp::jpeg::JpegDctShardManifest) > 0);
static_assert(sizeof(galp::jpeg::JpegDctShardDatasetReader) > 0);
#endif
