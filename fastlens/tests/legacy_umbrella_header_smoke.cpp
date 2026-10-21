#include <fastlens/fastlens.hpp>

#include <type_traits>

// Existing consumers that obtained Direct-DCT through the historical umbrella
// remain source compatible during the deprecation window.
#if FASTLENS_WITH_JPEG_DCT
static_assert(std::is_move_constructible_v<fastlens::jpeg::DirectDctBatch>);
#endif
