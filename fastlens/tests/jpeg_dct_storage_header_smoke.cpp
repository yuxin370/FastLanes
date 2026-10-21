#include <fastlens/jpeg_dct_storage.hpp>

#if FASTLENS_WITH_JPEG_DCT
static_assert(sizeof(fastlens::jpeg::JpegDctShardManifest) > 0);
static_assert(sizeof(fastlens::jpeg::JpegDctShardDatasetReader) > 0);
#endif
