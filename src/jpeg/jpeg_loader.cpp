// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/json/jpeg_loader.cpp
// ────────────────────────────────────────────────────────

#include "fls/jpeg/jpeg_loader.hpp"
#include <jpeglib.h>
#include <stdexcept>
#include <cstdio>
#include <algorithm>

// ZigzagBlock zigzag_transform(const DCTBlockRow& block) {
//     ZigzagBlock zz;
//     for (int i = 0; i < 64; ++i) {
//         zz.values[static_cast<size_t>(i)] = block.data[zigzag_order[i]];
//     }
//     return zz;
// }

std::vector<ZeroNonZeroPair> count_zero_nonzero_pairs(const std::vector<int>& sequence) {
    std::vector<ZeroNonZeroPair> result;
    size_t i = 0, n = sequence.size();
    while (i < n) {
        int zc = 0, nzc = 0;
        while (i < n && sequence[i] == 0) ++zc, ++i;
        while (i < n && sequence[i] != 0) ++nzc, ++i;
        result.push_back({zc, nzc});
    }
    return result;
}

ProcessedDCTChannel JpegLoader::process_channel(const ImageHeader& header) {
    ProcessedDCTChannel result;
    // std::vector<int> flattened;
    // int index = 0, last_index = 0;

    for (const auto& channel : header.channel_dcts) {
        for (const auto& block : channel.blocks) {
        result.raw_blocks.push_back(block);
        //     // ZigzagBlock zz = zigzag_transform(block);
        //     // result.raw_blocks.push_back(zz);
            // index ++;
            // auto zero_count = std::count(block.data, block.data + 64, 0);
            // float zero_ratio = static_cast<float>(zero_count) / 64.0f;
            // if (zero_ratio < threshold_ratio) {
            //     result.raw_index.push_back(index-last_index);
            //     result.raw_blocks.push_back(block);
            //     last_index = index;
            // } else {
            //     for (int i = 0; i < 64; ++i) {
            //     // for (int val : zz.values) {
            //         int16_t val = block.data[zigzag_order[i]];
            //         flattened.push_back(val);
            //         if (val != 0) result.nonzero_values.push_back(val);
            //     }
            // }
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

    // 读取DCT系数（关键：必须使用jpeg_read_coefficients）
    jvirt_barray_ptr* coeff_arrays = jpeg_read_coefficients(&cinfo_dct);
    
    // 正确获取量化表
    std::vector<QuantTable> qtables;
    for (int i = 0; i < NUM_QUANT_TBLS; ++i) {
        if (cinfo_dct.quant_tbl_ptrs[i]) {
            QuantTable q;
            q.id = static_cast<uint8_t>(i);
            q.precision = 0;  // JPEG标准中量化表总是8位
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

        // 修复点：分段访问虚拟数组（每次1行）
        for (JDIMENSION row = 0; row < comp->height_in_blocks; ++row) {
            // 每次只请求1行
            JBLOCKARRAY buffer = (*cinfo_dct.mem->access_virt_barray)(
                (j_common_ptr)&cinfo_dct,
                coeff_arrays[ci],
                row,  // 起始行
                1,    // 请求行数
                FALSE // 不可写
            );
            
            // 处理当前行的所有block
            for (JDIMENSION col = 0; col < comp->width_in_blocks; ++col) {
                DCTBlockRow block;
                memcpy(block.data, buffer[0][col], sizeof(block.data));
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