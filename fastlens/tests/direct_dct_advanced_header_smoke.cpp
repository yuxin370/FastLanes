#include <fastlens/advanced/direct_dct.hpp>

#include <type_traits>

#if FASTLENS_WITH_JPEG_DCT
#ifndef FASTLENS_DIRECT_DCT_ADVANCED_API
#error "advanced Direct-DCT boundary marker is missing"
#endif

static_assert(!std::is_copy_constructible_v<fastlens::jpeg::DirectDctBatch>);
static_assert(!std::is_copy_assignable_v<fastlens::jpeg::DirectDctBatch>);
static_assert(std::is_move_constructible_v<fastlens::jpeg::DirectDctBatch>);
#endif
