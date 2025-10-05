// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/include/jpeg_loader.hpp
// ────────────────────────────────────────────────────────
#ifndef JPEG_LOADER_HPP
#define JPEG_LOADER_HPP

#include <string>
#include <vector>
#include "fls/std/filesystem.hpp"

namespace fastlanes {

#define threshold_ratio 0.25

constexpr int zigzag_order_reverse[64] = {
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
};

constexpr int zigzag_order[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

// struct ZigzagBlock {
//     std::array<int, 64> values;  //  8x8 DCT block after zigzag
// };


struct ImageRGB {
    unsigned int width;
    unsigned int height;
    std::vector<unsigned char> data;  // RGBRGB...
};


struct QuantTable {
    uint8_t id;
    uint8_t precision;  // 0: 8-bit
    uint8_t data[64];   // Zigzag order
};

struct DCTBlockRow {
    int16_t data[64];  // One 8x8 block's DCT coefficients
};

struct ChannelDCT {
    uint8_t component_id;
    uint8_t qtable_id;
    uint8_t color_space_id;
    uint32_t width_in_blocks;
    uint32_t height_in_blocks;
    std::vector<DCTBlockRow> blocks;
};

struct ImageHeader {
    uint32_t width;
    uint32_t height;
    uint32_t quality;
    std::vector<std::string> color_spaces;
    // std::vector<uint8_t> rgb_data;  // Optional
    std::vector<QuantTable> quant_tables;
    std::vector<ChannelDCT> channel_dcts;
};

struct ZeroNonZeroPair {
    int zero_count;
    int nonzero_count;
};

struct ProcessedDCTChannel {
    std::vector<DCTBlockRow> raw_blocks;           
    std::vector<uint8_t> metadata;                   //（1=nonzero raws）
    size_t total_blocks = 0;

    std::vector<int16_t> DC_values;           // DC values
    std::vector<int16_t> AC_values;           // AC values
    std::vector<int16_t> mix_run_nonzero_values;           
    std::vector<ZeroNonZeroPair> mix_run_pattern;  
};

class JpegLoader {
public:
    static ImageHeader load_header(const std::string& path);
    static ImageRGB load_rgb(const std::string& path);
    static std::vector<std::vector<std::vector<std::vector<uint8_t>>>> to_rgb(const std::vector<std::vector<double>>& dct_blocks, const path& file_path);
    static ProcessedDCTChannel process_channel_plain(const ImageHeader& header); // without spliting
    static ProcessedDCTChannel process_channel(const ImageHeader& header);
    static ProcessedDCTChannel process_channel(const ImageHeader& header,
                                               size_t left, size_t mid, size_t right);
    static void print_image_header(const ImageHeader& header);
    static bool dump_ImageHeader(const ImageHeader& header, const char* filename);
    static bool load_ImageHeader(ImageHeader& header, const char* filename);
    // std::tuple<size_t, size_t, size_t> compute_adaptive_split(const ImageHeader& header,
    //                                                                     size_t total_blocks,
    //                                                                     size_t sample_count = 1000);
    // std::tuple<size_t,size_t,size_t> compute_split_over_dataset(const std::vector<std::string>& paths,
    //                                                                         size_t sample_budget = 10000);
private:
    ImageHeader image_header;
};
} // namespace fastlanes
#endif // JPEG_LOADER_H

