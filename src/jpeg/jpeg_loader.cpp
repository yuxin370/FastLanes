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
#include <cstdint>
#include <cstdio>
#include <algorithm>

// ZigzagBlock zigzag_transform(const DCTBlockRow& block) {
//     ZigzagBlock zz;
//     for (int i = 0; i < 64; ++i) {
//         zz.values[static_cast<size_t>(i)] = block.data[zigzag_order[i]];
//     }
//     return zz;
// }

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
    ProcessedDCTChannel result;
    result.total_blocks = 0;
    std::vector<int16_t> low_value_ac;
    size_t block_idx = 0;


    // metadata（each block 1 bit → use uint8_t to compressed store）
    for (const auto& channel : header.channel_dcts) {
        result.total_blocks += channel.blocks.size();
    }
    result.metadata.resize((result.total_blocks + 7) / 8, 0); // each byte 8 bit

    auto [left, mid, right] = compute_adaptive_split(header, result.total_blocks, 1000);
    
    int left_c = static_cast<int>(left) - 1; // except for DC
    int mid_c  = static_cast<int>(mid);
    // int right_c = static_cast<int>(right);

    for (const auto& channel : header.channel_dcts) {
        for (const auto& block : channel.blocks) {

            // // --- 1. ZigZag transform ---
            // std::vector<int16_t> zigzagged(64);
            // for (size_t i = 0; i < 64; ++i) {
            //     zigzagged[i] = block.data[zigzag_order[i]];
            // }

            // --- 2. split DC & AC ---
            int16_t dc = block.data[0];
            std::vector<int16_t> ac_coefs(block.data + 1, block.data + 63); // 63 
            // std::vector<int16_t> ac_coefs(zigzagged.begin() + 1, zigzagged.end()); // 63 

            result.DC_values.push_back(dc);

            // --- 4. check if mid is nonzero row---
            auto nonzero_in_mid = std::count_if(ac_coefs.begin() + left_c, ac_coefs.begin() + left_c + mid_c, [](int16_t x) { return x != 0; });
            bool is_high_value_mid = (static_cast<float>(nonzero_in_mid) >= static_cast<float>(mid_c) * 0.75f);

            // --- 5. AC ---
            if (is_high_value_mid) {
                result.metadata[block_idx / 8] |= (1 << block_idx % 8);   // set the bit to 1

                // left + mid ac
                result.AC_values.insert(result.AC_values.end(), ac_coefs.begin(), ac_coefs.begin() + left_c + mid_c);
                
                for (size_t i = static_cast<size_t>(left_c + mid_c); i < 63; ++i) {
                    if (ac_coefs[i] != 0) {
                        result.mix_run_nonzero_values.push_back(ac_coefs[i]);
                    }
                }
            } else {
                result.metadata[block_idx / 8] &= ~(1 << block_idx % 8);  // set the bit to 0
                // left_ac only
                result.AC_values.insert(result.AC_values.end(), ac_coefs.begin(), ac_coefs.begin() + left_c);

                for (size_t i = static_cast<size_t>(left_c); i < 63; ++i) {  // mid + right 
                    if (ac_coefs[i] != 0) {
                        result.mix_run_nonzero_values.push_back(ac_coefs[i]);  
                    }
                }
                // mid + right 
                low_value_ac.insert(low_value_ac.end(), ac_coefs.begin() + left_c, ac_coefs.end());
            }
            
        block_idx ++;
        }
    }
    auto pairs = count_zero_nonzero_pairs(low_value_ac);
    result.mix_run_pattern.insert(result.mix_run_pattern.end(), pairs.begin(), pairs.end());
    return result;
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
    
    // result.mixed_run_encoding_pattern = count_zero_nonzero_pairs(flattened);
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
        cdct.component_id = static_cast<unsigned char>(comp->component_id);
        cdct.width_in_blocks = comp->width_in_blocks;
        cdct.height_in_blocks = comp->height_in_blocks;

        for (JDIMENSION row = 0; row < comp->height_in_blocks; ++row) {
            // one raw each request
            JBLOCKARRAY buffer = (*cinfo_dct.mem->access_virt_barray)(
                (j_common_ptr)&cinfo_dct,
                coeff_arrays[ci],
                row,  // start raw
                1,    // request 1 raw
                FALSE // read only
            );
            
            // process each block in the raw
            for (JDIMENSION col = 0; col < comp->width_in_blocks; ++col) {
                DCTBlockRow block;
                // memcpy(block.data, buffer[0][col], sizeof(block.data));

                int16_t* src = buffer[0][col];  

                // Zigzag transform
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
    
    jpeg_destroy_decompress(&cinfo_dct);
    fclose(infile);

    return {
        .width = w,
        .height = h,
        .quant_tables = std::move(qtables),
        .channel_dcts = std::move(channels),
    };
}