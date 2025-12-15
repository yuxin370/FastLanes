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

using dct_blocks_dataset_t = std::array<std::pair<std::string_view, std::string_view>, 9>;

// clang-format off
class DCTBlocks {
public:   
    static constexpr std::string_view IMAGENET                             { FLS_CMAKE_SOURCE_DIR "/data/dct_blocks/imagenet" };                
    static constexpr std::string_view MNIST                             { FLS_CMAKE_SOURCE_DIR "/data/dct_blocks/mnist" };                
    static constexpr std::string_view SVHN                             { FLS_CMAKE_SOURCE_DIR "/data/dct_blocks/svhn" };                
    static constexpr std::string_view CELEBA                             { FLS_CMAKE_SOURCE_DIR "/data/dct_blocks/celebA" };                
    static constexpr std::string_view DIV2K                             { FLS_CMAKE_SOURCE_DIR "/data/dct_blocks/div2k" };                
    static constexpr std::string_view LFWA                             { FLS_CMAKE_SOURCE_DIR "/data/dct_blocks/lfwa" };                
    static constexpr std::string_view CIFAR10                             { FLS_CMAKE_SOURCE_DIR "/data/dct_blocks/cifar10" };                
    static constexpr std::string_view CXR8                             { FLS_CMAKE_SOURCE_DIR "/data/dct_blocks/cxr8" };                
    static constexpr std::string_view TID2013                             { FLS_CMAKE_SOURCE_DIR "/data/dct_blocks/tid2013" };                
    static constexpr dct_blocks_dataset_t dataset = {{
        { "imagenet",                        IMAGENET },
        { "mnist",                           MNIST },
        { "svhn",                            SVHN },
        { "celebA",                          CELEBA },
        { "div2k",                           DIV2K },
        { "lfwa",                            LFWA },
        { "cifar10",                         CIFAR10 },
        { "cxr8",                            CXR8 },
        { "tid2013",                         TID2013 }
    }};
};

} // namespace fastlanes

#endif // DATA_DCTBLOCKS_HPP