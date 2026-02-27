// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// data/include/data/dct_blocks.hpp
// ────────────────────────────────────────────────────────
#ifndef DATA_DCT_BLOCKS_HPP
#define DATA_DCT_BLOCKS_HPP

#include <array>
#include <string_view>

namespace fastlanes {

using dct_blocks_dataset_t = std::array<std::pair<std::string_view, std::string_view>, 4>;

// clang-format off
class dct_blocks {
public:
    static constexpr std::string_view MNIST                             { "/home/tangyuxin/cleanFastlanes/FastLanes/data/dct_blocks/mnist" };                      
    static constexpr std::string_view CIFAR10                             { "/home/tangyuxin/FastLanes/data/mnist/cifar10/csv/zigzag/all_split/original" };                      
    static constexpr std::string_view CELEBA                             { "/home/tangyuxin/FastLanes/data/mnist/celebA/csv/align-64/zigzag/all_split/original" };                      
    static constexpr std::string_view TINY_IMAGENET                             { "/home/tangyuxin/FastLanes/data/mnist/tiny-imagenet/csv/zigzag/all_split/original" };                      


    static constexpr dct_blocks_dataset_t dataset = {{
        { "mnist",                           MNIST },
        { "cifar10",                           CIFAR10 },
        { "celeba",                           CELEBA },
        { "tiny-imagenet",                           TINY_IMAGENET },
    }};
};
// clang-format on

} // namespace fastlanes

#endif // DATA_DCT_BLOCKS_HPP
