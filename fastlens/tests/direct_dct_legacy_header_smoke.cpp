#include <fastlens/direct_dct.hpp>

#if FASTLENS_WITH_JPEG_DCT
#ifndef FASTLENS_DIRECT_DCT_LEGACY_COMPATIBILITY_HEADER
#error "legacy Direct-DCT compatibility boundary marker is missing"
#endif

static_assert(sizeof(fastlens::jpeg::DirectDctBatch) > 0);
#endif
