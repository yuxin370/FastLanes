// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/json/jpeg_loader.cpp
// ────────────────────────────────────────────────────────

#include "fls/jpeg/jpeg_loader.hpp"
#include <jpeglib.h>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <string.h>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <map>
#include <cassert>

namespace fastlanes {
 

// Helper: convert libjpeg color space to string
std::string jpeg_color_space_to_string(J_COLOR_SPACE cs) {
    switch (cs) {
        case JCS_GRAYSCALE: return "Grayscale";
        case JCS_RGB:       return "RGB";
        case JCS_YCbCr:     return "YCbCr";
        case JCS_CMYK:      return "CMYK";
        case JCS_YCCK:      return "YCCK";
        default:            return "Unknown";
    }
}

std::vector<ZeroNonZeroPair> count_zero_nonzero_pairs(const std::vector<int16_t>& sequence) {
    std::vector<ZeroNonZeroPair> result;
    size_t i = 0, n = sequence.size();
    while (i < n) {
        int zc = 0, nzc = 0;
        while (i < n && sequence[i] == 0) ++zc, ++i;
        while (i < n && sequence[i] != 0) ++nzc, ++i;
        if (zc > 0 || nzc > 0) {
            result.push_back({zc, nzc});
        }
    }
    return result;
}

std::tuple<size_t, size_t, size_t> compute_adaptive_split(const ImageHeader& header, size_t total_blocks, size_t sample_count = 1000) {
    sample_count = std::min(sample_count, total_blocks);

    std::vector<size_t> zero_counts(64, 0);
    size_t block_index = 0;
    size_t sampled = 0;

    size_t step = total_blocks / sample_count;
    if (step == 0) step = 1;

    for (const auto& channel : header.channel_dcts) {
        for (const auto& block : channel.blocks) {
            if (block_index % step != 0) {
                block_index++;
                continue;
            }

            for (size_t i = 0; i < 64; ++i) {
                if(block.data[i] == 0){
                // if(block.data[zigzag_order[i]] == 0){
                    zero_counts[i]++;
                }
            }
            sampled++;
            if (sampled >= sample_count) break;
        }
        if (sampled >= sample_count) break;
        block_index++;
    }

    std::vector<float> ratios(64);
    for (size_t i = 0; i < 64; ++i) {
        ratios[i] = static_cast<float>(zero_counts[i]) / static_cast<float>(sampled);
    }

    size_t l = 0, m = 0, r = 0;

    // left: <50%
    while (l < 64 && ratios[l] < 0.5f) l++;

    // right: >90% 
    while (r < 64 - l && ratios[63 - r] >= 0.9f) r++; 

    m = 64 - l - r;

    l = std::max(l, (size_t)1);
    m = std::max(m, (size_t)1);
    r = std::max(r, (size_t)1);
    if (l + m + r != 64) r = 64 - l - m;

    return {l,m,r};
}


ProcessedDCTChannel JpegLoader::process_channel(const ImageHeader& header) {
    // 先算总 block
    size_t total_blocks = 0;
    for (const auto& ch : header.channel_dcts) total_blocks += ch.blocks.size();

    auto [left, mid, right] = compute_adaptive_split(header, total_blocks, 1000);
    return JpegLoader::process_channel(header, left, mid, right);
}

ProcessedDCTChannel JpegLoader::process_channel(const ImageHeader& header,
                                                size_t left, size_t mid, size_t /*right*/) {
    ProcessedDCTChannel pro_dct_blocks;
    pro_dct_blocks.total_blocks = 0;
    std::vector<int16_t> low_value_ac;
    size_t block_idx = 0;


    // metadata（each block 1 bit → use uint8_t to compressed store）
    for (const auto& channel : header.channel_dcts) {
        pro_dct_blocks.total_blocks += channel.blocks.size();
    }
    pro_dct_blocks.metadata.resize((pro_dct_blocks.total_blocks + 7) / 8, 0);

    int left_c = static_cast<int>(left) - 1; // except for DC
    int mid_c  = static_cast<int>(mid);

    for (const auto& channel : header.channel_dcts) {
        for (const auto& block : channel.blocks) {
            int16_t dc = block.data[0];
            std::vector<int16_t> ac_coefs(block.data + 1, block.data + 63);

            pro_dct_blocks.DC_values.push_back(dc);

            auto nonzero_in_mid = std::count_if(ac_coefs.begin() + left_c,
                                                ac_coefs.begin() + left_c + mid_c,
                                                [](int16_t x) { return x != 0; });
            bool is_high_value_mid =
                (static_cast<float>(nonzero_in_mid) >= static_cast<float>(mid_c) * 0.75f);

            if (is_high_value_mid) {
                pro_dct_blocks.metadata[block_idx / 8] |= (1 << (block_idx % 8)); // bit=1
                // left + mid
                pro_dct_blocks.AC_values.insert(pro_dct_blocks.AC_values.end(),
                                        ac_coefs.begin(), ac_coefs.begin() + left_c + mid_c);
                // right 非零并入 mix_run_nonzero_values
                for (size_t i = static_cast<size_t>(left_c + mid_c); i < 63; ++i) {
                    if (ac_coefs[i] != 0) {
                        pro_dct_blocks.mix_run_nonzero_values.push_back(ac_coefs[i]);
                    }
                }
            } else {
                pro_dct_blocks.metadata[block_idx / 8] &= ~(1 << (block_idx % 8)); // bit=0
                // left
                pro_dct_blocks.AC_values.insert(pro_dct_blocks.AC_values.end(),
                                        ac_coefs.begin(), ac_coefs.begin() + left_c);
                // mid+right：非零值入 mix_run_nonzero_values；完整序列用于 run-length
                for (size_t i = static_cast<size_t>(left_c); i < 63; ++i) {
                    if (ac_coefs[i] != 0) {
                        pro_dct_blocks.mix_run_nonzero_values.push_back(ac_coefs[i]);
                    }
                }
                low_value_ac.insert(low_value_ac.end(),
                                    ac_coefs.begin() + left_c, ac_coefs.end());
            }

            block_idx++;
        }
    }
    auto pairs = count_zero_nonzero_pairs(low_value_ac);
    pro_dct_blocks.mix_run_pattern.insert(pro_dct_blocks.mix_run_pattern.end(), pairs.begin(), pairs.end());
    return pro_dct_blocks;
}


ProcessedDCTChannel JpegLoader::process_channel_plain(const ImageHeader& header) {
    ProcessedDCTChannel result;
    // std::vector<int> flattened;
    // int index = 0, last_index = 0;

    for (const auto& channel : header.channel_dcts) {
        for (const auto& block : channel.blocks) {
            result.raw_blocks.push_back(block); 
        }
    }
    
    // pro_dct_blocks.mixed_run_encoding_pattern = count_zero_nonzero_pairs(flattened);
    return result;
}


ImageRGB JpegLoader::load_rgb(const std::string& path) {
    FILE* infile = fopen(path.c_str(), "rb");
    if (!infile) throw std::runtime_error("Cannot open file: " + path);

    jpeg_decompress_struct cinfo;
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);

    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    unsigned int w = cinfo.output_width;
    unsigned int h = cinfo.output_height;
    int num_channels = cinfo.output_components;  // Usually 3 (RGB)

    std::vector<unsigned char> buffer(w * h * static_cast<unsigned int>(num_channels));
    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char* rowptr = &buffer[cinfo.output_scanline * w * static_cast<unsigned int>(num_channels)];
        jpeg_read_scanlines(&cinfo, &rowptr, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    return ImageRGB{w, h, std::move(buffer)};
}

ImageHeader JpegLoader::load_header(const std::string& path) {
    FILE* infile = fopen(path.c_str(), "rb");
    if (!infile) throw std::runtime_error("Cannot open file: " + path);

    jpeg_decompress_struct cinfo_dct;
    jpeg_error_mgr jerr_dct;
    cinfo_dct.err = jpeg_std_error(&jerr_dct);
    
    jpeg_create_decompress(&cinfo_dct);
    jpeg_stdio_src(&cinfo_dct, infile);
    jpeg_read_header(&cinfo_dct, TRUE);

    // read DCT coefficients
    jvirt_barray_ptr* coeff_arrays = jpeg_read_coefficients(&cinfo_dct);
    
    // get quantization tables
    std::vector<QuantTable> qtables;
    for (int i = 0; i < NUM_QUANT_TBLS; ++i) {
        if (cinfo_dct.quant_tbl_ptrs[i]) {
            QuantTable q;
            q.id = static_cast<uint8_t>(i);
            q.precision = 0;  
            for (int j = 0; j < 64; ++j) {
                q.data[j] = static_cast<uint8_t>(cinfo_dct.quant_tbl_ptrs[i]->quantval[j]);
            }
            qtables.push_back(q);
        }
    }

    std::vector<ChannelDCT> channels;
    for (int ci = 0; ci < cinfo_dct.num_components; ++ci) {
        jpeg_component_info* comp = &cinfo_dct.comp_info[ci];
        ChannelDCT cdct;
        cdct.component_id = static_cast<uint8_t>(comp->component_id);
        cdct.qtable_id = static_cast<uint8_t>(comp->quant_tbl_no); // <-- NEW
        cdct.width_in_blocks = comp->width_in_blocks;
        cdct.height_in_blocks = comp->height_in_blocks;

        for (JDIMENSION row = 0; row < comp->height_in_blocks; ++row) {
            JBLOCKARRAY buffer = (*cinfo_dct.mem->access_virt_barray)(
                (j_common_ptr)&cinfo_dct,
                coeff_arrays[ci],
                row, 1, FALSE
            );
            
            for (JDIMENSION col = 0; col < comp->width_in_blocks; ++col) {
                DCTBlockRow block;
                int16_t* src = buffer[0][col];  
                for (size_t i = 0; i < 64; ++i) {
                    block.data[i] = src[zigzag_order[i]];
                }
                cdct.blocks.push_back(block);
            }
        }
        channels.push_back(cdct);
    }
    
    unsigned int w = cinfo_dct.image_width;  
    unsigned int h = cinfo_dct.image_height; 

    // Estimate quality: optional. Here we set to 0 (unknown) since libjpeg doesn't store it.
    // You could implement a heuristic based on quant tables if needed.
    uint32_t quality = 0;

    std::string color_space = jpeg_color_space_to_string(cinfo_dct.jpeg_color_space);

    jpeg_destroy_decompress(&cinfo_dct);
    fclose(infile);

    return {
        .width = w,
        .height = h,
        .quality = quality,              // <-- NEW
        .color_space = color_space,      // <-- NEW
        .quant_tables = std::move(qtables),
        .channel_dcts = std::move(channels),
    };
}



void idct_8x8(const int* coeffs, uint8_t* output) {
    // coeffs: frequency domain, row-major (v*8 + u)
    // output: spatial 8x8, row-major (y*8 + x)
    const double pi = M_PI;
    double C[8];
    for (int i = 0; i < 8; ++i) {
        C[i] = (i == 0) ? (1.0 / std::sqrt(2.0)) : 1.0;
    }

    // Precompute cosines: cos_table[x][u] = cos((2x+1)*u*pi/16)
    double cos_table[8][8];
    for (int x = 0; x < 8; ++x) {
        for (int u = 0; u < 8; ++u) {
            cos_table[x][u] = std::cos((2.0 * x + 1.0) * u * pi / 16.0);
        }
    }

    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            double sum = 0.0;
            for (int v = 0; v < 8; ++v) {
                double cos_y_v = cos_table[y][v];
                double Cv = C[v];
                for (int u = 0; u < 8; ++u) {
                    double Cu = C[u];
                    double Fuv = static_cast<double>(coeffs[v * 8 + u]); // note indexing v*8 + u
                    sum += Cu * Cv * Fuv * cos_table[x][u] * cos_y_v;
                }
            }
            double val = 0.25 * sum + 128.0; // scale and shift back to [0,255]
            // clamp and round
            val = std::round(std::clamp(val, 0.0, 255.0));
            output[y * 8 + x] = static_cast<uint8_t>(val);
        }
    }
}


void upsample_chroma_bilinear(const std::vector<std::vector<uint8_t>>& src,
                              std::vector<std::vector<uint8_t>>& dst,
                              uint32_t dst_h, uint32_t dst_w) {
    if (src.empty() || src[0].empty()) {
        dst.clear();
        return;
    }

    const size_t src_h = src.size();
    const size_t src_w = src[0].size();

    dst.assign(dst_h, std::vector<uint8_t>(dst_w, 0));

    // Ratios: mapping from destination pixel centers to source coordinates
    // Use common formula: src_coord = (dst_coord + 0.5) * (src_size/dst_size) - 0.5
    const double y_ratio = static_cast<double>(src_h) / static_cast<double>(dst_h);
    const double x_ratio = static_cast<double>(src_w) / static_cast<double>(dst_w);

    for (uint32_t y = 0; y < dst_h; ++y) {
        // mapped source coordinate (floating)
        double src_y = (static_cast<double>(y) + 0.5) * y_ratio - 0.5;
        // clamp to valid range [0, src_h-1]
        if (src_y < 0.0) src_y = 0.0;
        if (src_y > static_cast<double>(src_h - 1)) src_y = static_cast<double>(src_h - 1);

        size_t y0 = static_cast<size_t>(std::floor(src_y));
        size_t y1 = (y0 + 1 < src_h) ? (y0 + 1) : y0;
        double dy = src_y - static_cast<double>(y0);

        for (uint32_t x = 0; x < dst_w; ++x) {
            double src_x = (static_cast<double>(x) + 0.5) * x_ratio - 0.5;
            if (src_x < 0.0) src_x = 0.0;
            if (src_x > static_cast<double>(src_w - 1)) src_x = static_cast<double>(src_w - 1);

            size_t x0 = static_cast<size_t>(std::floor(src_x));
            size_t x1 = (x0 + 1 < src_w) ? (x0 + 1) : x0;
            double dx = src_x - static_cast<double>(x0);

            // fetch four neighbors
            double v00 = static_cast<double>(src[y0][x0]);
            double v10 = static_cast<double>(src[y0][x1]);
            double v01 = static_cast<double>(src[y1][x0]);
            double v11 = static_cast<double>(src[y1][x1]);

            // bilinear interpolation
            double v0 = v00 + (v10 - v00) * dx; // interp along x at y0
            double v1 = v01 + (v11 - v01) * dx; // interp along x at y1
            double v  = v0 + (v1 - v0) * dy;    // interp along y

            // round and clamp
            int iv = static_cast<int>(std::round(v));
            iv = std::min(255, std::max(0, iv));
            dst[y][x] = static_cast<uint8_t>(iv);
        }
    }
}

void ycbcr_to_rgb(uint8_t y, uint8_t cb, uint8_t cr, uint8_t& r, uint8_t& g, uint8_t& b) {
    double Y = y;
    double Cb = cb - 128.0;
    double Cr = cr - 128.0;

    double R = Y + 1.402 * Cr;
    double G = Y - 0.344136286 * Cb - 0.714136286 * Cr;
    double B = Y + 1.772 * Cb;

    r = static_cast<uint8_t>(std::clamp(R, 0.0, 255.0));
    g = static_cast<uint8_t>(std::clamp(G, 0.0, 255.0));
    b = static_cast<uint8_t>(std::clamp(B, 0.0, 255.0));
}

// 把 src（src_h x src_w）裁剪或上采样到 dst_h x dst_w
// - 如果 src 大于目标：裁剪（左上对齐）
// - 如果 src 小于目标：使用双线性插值放大（调用 upsample_chroma_bilinear）
// - 如果大小相等：直接拷贝
static void resize_or_crop_plane(const std::vector<std::vector<uint8_t>>& src,
                                 std::vector<std::vector<uint8_t>>& dst,
                                 uint32_t dst_h, uint32_t dst_w) {
    if (src.empty() || src[0].empty()) {
        dst.clear();
        return;
    }
    const size_t src_h = src.size();
    const size_t src_w = src[0].size();

    if (static_cast<uint32_t>(src_h) == dst_h && static_cast<uint32_t>(src_w) == dst_w) {
        // 直接拷贝
        dst = src;
        return;
    }

    if (static_cast<uint32_t>(src_h) >= dst_h && static_cast<uint32_t>(src_w) >= dst_w) {
        // 都大于等于 → 裁剪到目标（左上角对齐）
        dst.assign(dst_h, std::vector<uint8_t>(dst_w));
        for (uint32_t y = 0; y < dst_h; ++y) {
            for (uint32_t x = 0; x < dst_w; ++x) {
                dst[y][x] = src[y][x];
            }
        }
        return;
    }

    // 其他情况（至少有一维小于目标） → 使用双线性上采样到目标
    // 注意：upsample_chroma_bilinear 支持任意 src/dst 大小
    upsample_chroma_bilinear(src, dst, dst_h, dst_w);
}

std::vector<std::vector<std::vector<std::vector<double>>>> JpegLoader::to_rgb(
    const std::vector<std::vector<double>>& dct_blocks,
    const path& file_path) {

    ImageHeader header;
    if (!load_ImageHeader(header, file_path.c_str())) {
        throw std::runtime_error("Failed to load ImageHeader from " + file_path.string());
    }

    JpegLoader::print_image_header(header);

    uint32_t img_width = header.width;
    uint32_t img_height = header.height;

    // 计算总块数并验证
    size_t total_blocks_expected = 0;
    for (const auto& ch : header.channel_dcts) {
        total_blocks_expected += static_cast<size_t>(ch.width_in_blocks) * ch.height_in_blocks;
    }
    if (dct_blocks.size() != total_blocks_expected) {
        throw std::runtime_error(
            "dct_blocks size (" + std::to_string(dct_blocks.size()) +
            ") != expected (" + std::to_string(total_blocks_expected) + ")");
    }

    // 构建量化表映射
    std::map<uint8_t, const QuantTable*> qt_map;
    for (const auto& qt : header.quant_tables) {
        qt_map[qt.id] = &qt;
    }

    // 逐通道重建
    std::vector<std::vector<std::vector<uint8_t>>> planes; // [channel][h][w]
    size_t block_idx = 0;

    for (const auto& ch : header.channel_dcts) {
        uint32_t w_blocks = ch.width_in_blocks;
        uint32_t h_blocks = ch.height_in_blocks;
        uint32_t plane_w = w_blocks * 8;
        uint32_t plane_h = h_blocks * 8;

        auto qt_it = qt_map.find(ch.qtable_id);
        if (qt_it == qt_map.end()) {
            throw std::runtime_error("Quant table not found for component_id=" + std::to_string(ch.qtable_id));
        }
        const uint8_t* qtable = qt_it->second->data;

        std::vector<std::vector<uint8_t>> plane(plane_h, std::vector<uint8_t>(plane_w));

        for (size_t by = 0; by < h_blocks; ++by) {
            for (size_t bx = 0; bx < w_blocks; ++bx) {
                const auto& zigzag_coeffs = dct_blocks[block_idx++];
                int16_t spatial_coeffs[64];
                int dequant_coeffs[64];

                // 1. Zigzag → spatial order 
                for (int i = 0; i < 64; ++i) {
                    spatial_coeffs[zigzag_order[i]] = static_cast<int16_t>(std::round(zigzag_coeffs.data()[i]));
                }

                // 2. 反量化
                for (int i = 0; i < 64; ++i) {
                    dequant_coeffs[i] = static_cast<int>(spatial_coeffs[i]) * static_cast<int>(qtable[i]);
                }

                // 3. IDCT → 8x8 像素块
                uint8_t pixels[64];
                idct_8x8(dequant_coeffs, pixels);

                // 4. 写入平面
                for (size_t y = 0; y < 8; ++y) {
                    for (size_t x = 0; x < 8; ++x) {
                        size_t py = by * 8 + y;
                        size_t px = bx * 8 + x;
                        if (py < plane_h && px < plane_w) {
                            plane[py][px] = pixels[y * 8 + x];
                        }
                    }
                }
            }
        }
        planes.push_back(std::move(plane));
    }

    if (planes.size() == 1) {
        // Grayscale → RGB (直接复制)
        planes.push_back(planes[0]);
        planes.push_back(planes[0]);
    } else if (planes.size() == 3) {
        // 假设 channel_dcts 中顺序为 Y, Cb, Cr（通常 libjpeg 保持此顺序）
        // 以 Y 平面为基准（plane 0）
        const auto& y_plane = planes[0];
        uint32_t y_plane_h = static_cast<uint32_t>(y_plane.size());
        uint32_t y_plane_w = static_cast<uint32_t>(y_plane.empty() ? 0 : y_plane[0].size());

        // 如果 Y 平面本身小于图像，先把 Y 放大到图像尺寸（极少见，通常 Y >= img）
        std::vector<std::vector<uint8_t>> y_fixed;
        resize_or_crop_plane(planes[0], y_fixed, img_height, img_width);
        planes[0] = std::move(y_fixed);

        // 对每个色度分量：先把 Cb/Cr 放缩到 Y 平面原始尺寸（y_plane_h x y_plane_w）
        // 再把结果裁剪或放缩到最终的 img_height x img_width（以处理 MCU padding 情况）
        for (size_t c = 1; c <= 2; ++c) {
            // 当前 chroma plane 的尺寸
            uint32_t c_h = static_cast<uint32_t>(planes[c].size());
            uint32_t c_w = static_cast<uint32_t>(planes[c].empty() ? 0 : planes[c][0].size());

            // 先把 chroma 放大到 Y plane 大小（如果 chroma 更小）
            std::vector<std::vector<uint8_t>> tmp;
            if (c_h != y_plane_h || c_w != y_plane_w) {
                // 如果 chroma 更大也没关系，resize_or_crop_plane 会裁剪
                resize_or_crop_plane(planes[c], tmp, y_plane_h, y_plane_w);
            } else {
                tmp = planes[c]; // 大小相等，直接使用
            }

            // 最后把该平面裁剪或上采样到最终图像像素大小（img_height x img_width）
            std::vector<std::vector<uint8_t>> final_plane;
            resize_or_crop_plane(tmp, final_plane, img_height, img_width);
            planes[c] = std::move(final_plane);
        }

    } else {
        throw std::runtime_error("Unsupported channel count: " + std::to_string(planes.size()));
    }

    // 转换为 [image][channel][height][width] 格式（double）
    std::vector<std::vector<std::vector<std::vector<double>>>> result;
    result.resize(1);                    // 1 image
    result[0].resize(3);                 // R, G, B
    for (size_t c = 0; c < 3; ++c) {
        result[0][c].resize(img_height, std::vector<double>(img_width));
    }

    for (uint32_t y = 0; y < img_height; ++y) {
        for (uint32_t x = 0; x < img_width; ++x) {
            uint8_t Y = planes[0][y][x];
            uint8_t Cb = planes[1][y][x];
            uint8_t Cr = planes[2][y][x];
            uint8_t r, g, b;
            ycbcr_to_rgb(Y, Cb, Cr, r, g, b);
            result[0][0][y][x] = static_cast<double>(r);
            result[0][1][y][x] = static_cast<double>(g);
            result[0][2][y][x] = static_cast<double>(b);
        }
    }

    return result; // shape: [1][3][H][W]
}



void JpegLoader::print_image_header(const ImageHeader& header) {
    printf("=== Image Header ===\n");
    printf("Width: %u pixels\n", header.width);
    printf("Height: %u pixels\n", header.height);
    printf("Quality: %u (0 = unknown)\n", header.quality);           // <-- NEW
    printf("Color Space: %s\n", header.color_space.c_str());         // <-- NEW
    printf("\n");
    
    printf("Quantization Tables (%zu tables):\n", header.quant_tables.size());
    for (size_t i = 0; i < header.quant_tables.size(); ++i) {
        const auto& qt = header.quant_tables[i];
        printf("  Table %zu (ID: %d, Precision: %d-bit):\n", 
               i, (int)qt.id, (qt.precision == 0) ? 8 : 16);
        for (int row = 0; row < 8; ++row) {
            printf("    ");
            for (int col = 0; col < 8; ++col) {
                printf("%4d ", (int)qt.data[row * 8 + col]);
            }
            printf("\n");
        }
        printf("\n");
    }
    
    printf("Channel DCT Data (%zu channels):\n", header.channel_dcts.size());
    for (size_t ch = 0; ch < header.channel_dcts.size(); ++ch) {
        const auto& channel = header.channel_dcts[ch];
        printf("  Channel %zu:\n", ch);
        printf("    Component ID: %d\n", (int)channel.component_id);
        printf("    Quant Table ID: %d\n", (int)channel.qtable_id);  // <-- NEW
        printf("    Width in blocks: %u\n", channel.width_in_blocks);
        printf("    Height in blocks: %u\n", channel.height_in_blocks);
        printf("    Total blocks: %zu\n", channel.blocks.size());
        
        const size_t max_blocks_to_show = 2;
        size_t blocks_to_show = std::min(max_blocks_to_show, channel.blocks.size());
        for (size_t block_idx = 0; block_idx < blocks_to_show; ++block_idx) {
            printf("    Block %zu:\n", block_idx);
            const auto& block = channel.blocks[block_idx];
            for (int row = 0; row < 8; ++row) {
                printf("      ");
                for (int col = 0; col < 8; ++col) {
                    printf("%6d ", (int)block.data[row * 8 + col]);
                }
                printf("\n");
            }
        }
        if (channel.blocks.size() > max_blocks_to_show) {
            printf("    ... (showing only first %zu blocks out of %zu)\n", 
                   max_blocks_to_show, channel.blocks.size());
        }
        printf("\n");
    }
}

bool JpegLoader::dump_ImageHeader(const ImageHeader& header, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        perror("fopen (write)");
        return false;
    }

    // Basic image info
    fwrite(&header.width, sizeof(uint32_t), 1, fp);
    fwrite(&header.height, sizeof(uint32_t), 1, fp);
    fwrite(&header.quality, sizeof(uint32_t), 1, fp); // <-- NEW

    // Color space: write length + string
    uint64_t color_space_len = header.color_space.size();
    fwrite(&color_space_len, sizeof(uint64_t), 1, fp);
    if (color_space_len > 0) {
        fwrite(header.color_space.data(), sizeof(char), color_space_len, fp);
    }

    // Quant tables
    uint64_t qt_count = header.quant_tables.size();
    fwrite(&qt_count, sizeof(uint64_t), 1, fp);
    for (const auto& qt : header.quant_tables) {
        fwrite(&qt.id, sizeof(uint8_t), 1, fp);
        fwrite(&qt.precision, sizeof(uint8_t), 1, fp);
        fwrite(qt.data, sizeof(uint8_t), 64, fp);
    }

    // Channels
    uint64_t ch_count = header.channel_dcts.size();
    fwrite(&ch_count, sizeof(uint64_t), 1, fp);
    for (const auto& ch : header.channel_dcts) {
        fwrite(&ch.component_id, sizeof(uint8_t), 1, fp);
        fwrite(&ch.qtable_id, sizeof(uint8_t), 1, fp);            // <-- NEW
        fwrite(&ch.width_in_blocks, sizeof(uint32_t), 1, fp);
        fwrite(&ch.height_in_blocks, sizeof(uint32_t), 1, fp);
        // Note: blocks are NOT saved in header dump (as before)
    }

    fclose(fp);
    printf("Successfully wrote header to %s\n", filename);
    return true;
}

bool JpegLoader::load_ImageHeader(ImageHeader& header, const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        perror("fopen (read)");
        return false;
    }

    fread(&header.width, sizeof(uint32_t), 1, fp);
    fread(&header.height, sizeof(uint32_t), 1, fp);
    fread(&header.quality, sizeof(uint32_t), 1, fp); // <-- NEW

    // Color space
    uint64_t color_space_len;
    fread(&color_space_len, sizeof(uint64_t), 1, fp);
    header.color_space.resize(color_space_len);
    if (color_space_len > 0) {
        fread(&header.color_space[0], sizeof(char), color_space_len, fp);
    }

    // Quant tables
    uint64_t qt_count;
    fread(&qt_count, sizeof(uint64_t), 1, fp);
    header.quant_tables.resize(qt_count);
    for (size_t i = 0; i < qt_count; ++i) {
        auto& qt = header.quant_tables[i];
        fread(&qt.id, sizeof(uint8_t), 1, fp);
        fread(&qt.precision, sizeof(uint8_t), 1, fp);
        fread(qt.data, sizeof(uint8_t), 64, fp);
    }

    // Channels
    uint64_t ch_count;
    fread(&ch_count, sizeof(uint64_t), 1, fp);
    header.channel_dcts.resize(ch_count);
    for (size_t ch_idx = 0; ch_idx < ch_count; ++ch_idx) {
        auto& ch = header.channel_dcts[ch_idx];
        fread(&ch.component_id, sizeof(uint8_t), 1, fp);
        fread(&ch.qtable_id, sizeof(uint8_t), 1, fp);             // <-- NEW
        fread(&ch.width_in_blocks, sizeof(uint32_t), 1, fp);
        fread(&ch.height_in_blocks, sizeof(uint32_t), 1, fp);
        ch.blocks.clear(); // blocks not stored in header file
    }

    fclose(fp);
    return true;
}


} // namespace fastlanes