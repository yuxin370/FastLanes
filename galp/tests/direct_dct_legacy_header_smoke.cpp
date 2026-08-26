#include <galp/direct_dct.hpp>

#if GALP_WITH_JPEG_DCT
#ifndef GALP_DIRECT_DCT_LEGACY_COMPATIBILITY_HEADER
#error "legacy Direct-DCT compatibility boundary marker is missing"
#endif

static_assert(sizeof(galp::jpeg::DirectDctBatch) > 0);
#endif
