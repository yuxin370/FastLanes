// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// data/include/data/DCTBlocks.hpp
// ────────────────────────────────────────────────────────
#ifndef DATA_DCTBLOCKS_HPP
#define DATA_DCTBLOCKS_HPP

#include <array>
#include <string_view>

namespace fastlanes {

using dct_blocks_dataset_t = std::array<std::pair<std::string_view, std::string_view>, 3>;

// clang-format off
class DCTBlocks {
public:   
    static constexpr std::string_view IMAGENET                             { FLS_CMAKE_SOURCE_DIR "/data/dct_blocks/imagenet" };                
    static constexpr std::string_view MNIST                             { FLS_CMAKE_SOURCE_DIR "/data/dct_blocks/mnist" };                
    static constexpr std::string_view SVHN                             { FLS_CMAKE_SOURCE_DIR "/data/dct_blocks/svhn" };                
    static constexpr dct_blocks_dataset_t dataset = {{
        { "imagenet",                        IMAGENET },
        { "mnist",                           MNIST },
        { "svhn",                            SVHN },
    }};
};

} // namespace fastlanes

#endif // DATA_DCTBLOCKS_HPP