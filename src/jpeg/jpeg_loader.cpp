// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/jpeg/jpeg_loader.cpp
// ────────────────────────────────────────────────────────

#include "fls/jpeg/jpeg_loader.hpp"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cctype>
#include <condition_variable>
#include <cuda_runtime.h>  
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <jpeglib.h>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>


#include "gpujpeg_decoder.h"
#include "gpujpeg_common_internal.h"   // 先把枚举/struct 都定义好
#include "gpujpeg_decoder_internal.h"
#include "gpujpeg_table.h"

namespace fastlanes {

// // Helper: convert libjpeg color space to string
// std::string jpeg_color_space_to_string(J_COLOR_SPACE cs) {
//     switch (cs) {
//         case JCS_GRAYSCALE: return "Grayscale";
//         case JCS_RGB:       return "RGB";
//         case JCS_YCbCr:     return "YCbCr";
//         case JCS_CMYK:      return "CMYK";
//         case JCS_YCCK:      return "YCCK";
//         default:            return "Unknown";
//     }
// }

// // Helper: get number of channels from color space name
// size_t get_channel_count_from_color_space(const std::string& cs) {
//     if (cs == "Grayscale" || cs == "GRAY" || cs == "L") {
//         return 1;
//     } else if (cs == "YCbCr" || cs == "YUV" || cs == "RGB") {
//         return 3;
//     } else {
//         // You can extend this
//         throw std::runtime_error("Unknown color space: " + cs);
//     }
// }

// helper: check if this is jpeg file
static bool is_jpeg_ext(const fs::path& p) {
    auto ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return (ext == ".jpg" || ext == ".jpeg" || ext == ".jfif");
}

// helper: read whole file to vector
static std::vector<uint8_t> read_file_to_vec(const fs::path& file) {
    std::ifstream ifs(file, std::ios::binary | std::ios::ate);
    if (!ifs) {
        throw std::runtime_error("Cannot open file: " + file.string());
    }
    std::streamsize size = ifs.tellg();
    if (size < 0) {
        throw std::runtime_error("Failed to stat file: " + file.string());
    }
    std::vector<uint8_t> buf(static_cast<size_t>(size));
    ifs.seekg(0, std::ios::beg);
    if (size > 0 && !ifs.read(reinterpret_cast<char*>(buf.data()), size)) {
        throw std::runtime_error("Failed to read file: " + file.string());
    }
    return buf;
}

static int get_comp_count(ColorSpace cs) {
    switch (cs) {
    case ColorSpace::Grayscale: return 1;
    case ColorSpace::RGB:
    case ColorSpace::YCbCr:     return 3;
    case ColorSpace::CMYK:
    case ColorSpace::YCCK:      return 4;
    default:
        throw std::runtime_error("Unknown ColorSpace");
    }
}

static gpujpeg_color_space to_gpujpeg_color_space(ColorSpace cs) {
    switch (cs) {
    case ColorSpace::Grayscale:
        // GPUJPEG 没有专门的灰度枚举，一般做法是：
        //  - color_space 用 YCbCr (BT.601 full range)
        //  - comp_count=1 + subsampling=4:0:0
        return GPUJPEG_YCBCR_BT601_256LVLS;

    case ColorSpace::YCbCr:
        // 原始 JPEG 一般就是 BT.601 full range（对应 JPEG 语义）
        // 你也可以返回 GPUJPEG_YCBCR（BT.709 限幅），取决于数据集约定
        return GPUJPEG_YCBCR_BT601_256LVLS; // == GPUJPEG_YCBCR_JPEG

    case ColorSpace::RGB:
        return GPUJPEG_RGB;

    case ColorSpace::CMYK:
    case ColorSpace::YCCK:
        // GPUJPEG 当前不支持 CMYK / YCCK
        // 这里直接报错比较安全，避免误解码
        throw std::runtime_error("GPUJPEG does not support CMYK/YCCK color spaces");

    default:
        throw std::runtime_error("Unknown ColorSpace for to_gpujpeg_color_space");
    }
}



static gpujpeg_sampling_factor_t to_gpujpeg_subsampling(SampleFactor sf) {
    switch (sf) {
    case SampleFactor::SF_444: return GPUJPEG_SUBSAMPLING_444;
    case SampleFactor::SF_422: return GPUJPEG_SUBSAMPLING_422;
    case SampleFactor::SF_420: return GPUJPEG_SUBSAMPLING_420;
    case SampleFactor::SF_400: return GPUJPEG_SUBSAMPLING_400;
    default:
        throw std::runtime_error("Unknown SampleFactor");
    }
}



static void
sync_quant_tables_from_header_to_decoder(const ImageHeader& header,
                                         const ImageInfo&   img,
                                         gpujpeg_decoder*   decoder)
{
    // 映射：全局 qtable_id (来自 unified_header/channel_dcts) -> GPUJPEG 本地槽位 [0..)
    std::unordered_map<uint8_t, uint8_t> qid_global_to_local;

    uint8_t next_local_qid = 0;

    // 对该图像的每个通道
    for (uint32_t c = 0; c < img.channel_count; ++c) {
        const ChannelDCT& ch = header.channel_dcts[img.first_channel_index + c];
        const uint8_t qid_global = ch.qtable_id;

        // 分配/获取本地槽位
        uint8_t qid_local;
        auto it = qid_global_to_local.find(qid_global);
        if (it == qid_global_to_local.end()) {
            // 第一次遇到这个全局 qid：分配一个新的本地槽位
            if (next_local_qid >= GPUJPEG_MAX_COMPONENT_COUNT) {
                throw std::runtime_error(
                    "Too many distinct quant tables used by one image: " +
                    std::to_string((int)next_local_qid));
            }
            qid_local = next_local_qid++;
            qid_global_to_local[qid_global] = qid_local;

            // 利用“id == index”的性质，直接按下标取 global table
            if (qid_global >= header.quant_tables.size()) {
                throw std::runtime_error(
                    "qtable_id out of range for this header: " +
                    std::to_string((int)qid_global));
            }
            const QuantTable& qt = header.quant_tables[qid_global];

            auto* tbl = &decoder->table_quantization[qid_local];

            // header.quant_tables[i].data 是 zigzag 顺序，
            // GPUJPEG 的 table_raw 也是 zigzag
            for (int i = 0; i < 64; ++i) {
                tbl->table_raw[i] = qt.data[i];
            }
            
            // 生成 natural 顺序 + 上传到 device
            if (gpujpeg_table_quantization_decoder_compute(tbl) != 0) {
                throw std::runtime_error(
                    "gpujpeg_table_quantization_decoder_compute failed for local qid="
                    + std::to_string((int)qid_local));
            }
        } else {
            qid_local = it->second;
        }

        // 告诉 decoder：第 c 个 component 使用哪个本地量化表槽位
        decoder->comp_table_quantization_map[c] = qid_local;
    }
}

// Helper: convert libjpeg color space to ColorSpace enum
ColorSpace jpeg_color_space_to_color_space(J_COLOR_SPACE cs) {
	switch (cs) {
	case JCS_GRAYSCALE:
		return ColorSpace::Grayscale;
	case JCS_RGB:
		return ColorSpace::RGB;
	case JCS_YCbCr:
		return ColorSpace::YCbCr;
	case JCS_CMYK:
		return ColorSpace::CMYK;
	case JCS_YCCK:
		return ColorSpace::YCCK;
	default:
		throw std::runtime_error("Unsupported JPEG color space");
	}
}

// Helper: get number of channels from ColorSpace enum
size_t get_channel_count_from_color_space(ColorSpace cs) {
	switch (cs) {
	case ColorSpace::Grayscale:
		return 1;
	case ColorSpace::RGB:
	case ColorSpace::YCbCr:
		return 3;
	case ColorSpace::CMYK:
	case ColorSpace::YCCK:
		return 4;
	default:
		throw std::runtime_error("Unknown color space");
	}
}

const char* color_space_to_cstring(ColorSpace cs) {
	switch (cs) {
	case ColorSpace::Grayscale:
		return "Grayscale";
	case ColorSpace::RGB:
		return "RGB";
	case ColorSpace::YCbCr:
		return "YCbCr";
	case ColorSpace::CMYK:
		return "CMYK";
	case ColorSpace::YCCK:
		return "YCCK";
	default:
		return "Unknown";
	}
}

// SampleFactor -> 字符串
const char* sample_factor_to_cstring(SampleFactor sf) {
    switch (sf) {
    case SampleFactor::SF_444: return "4:4:4";
    case SampleFactor::SF_422: return "4:2:2";
    case SampleFactor::SF_420: return "4:2:0";
    case SampleFactor::SF_400: return "4:0:0";
    default:                   return "Unknown";
    }
}

// 利用 ImageHeader + 某一张 ImageInfo 初始化一个 gpujpeg_decoder
static void init_decoder_for_image(const ImageHeader& header,
                                   const ImageInfo&   img,
                                   gpujpeg_decoder*   decoder)
{
    if (!decoder)
        throw std::runtime_error("init_decoder_for_image: decoder is null");

    if (img.color_space_id >= header.color_spaces.size())
        throw std::runtime_error("init_decoder_for_image: invalid color_space_id");

    if (img.sample_factor_id >= header.sample_factors.size())
        throw std::runtime_error("init_decoder_for_image: invalid sample_factor_id");

    ColorSpace   cs = header.color_spaces[img.color_space_id];
    SampleFactor sf = header.sample_factors[img.sample_factor_id];

    const int comp_count_from_cs = get_comp_count(cs);
    if (comp_count_from_cs != static_cast<int>(img.channel_count)) {
        throw std::runtime_error(
            "init_decoder_for_image: channel_count mismatch with color_space");
    }

    // 1) 默认参数
    gpujpeg_parameters       param;
    gpujpeg_image_parameters param_image;
    gpujpeg_set_default_parameters(&param);
    gpujpeg_image_set_default_parameters(&param_image);

    // 2) 设置 subsampling（决定 param.comp_count + sampling_factor）
    // gpujpeg_sampling_factor_t subsampling = to_gpujpeg_subsampling(sf);
    gpujpeg_sampling_factor_t subsampling =
        (cs == ColorSpace::Grayscale)
            ? GPUJPEG_SUBSAMPLING_400    // 灰度强制 4:0:0
            : to_gpujpeg_subsampling(sf); // 其他按 JPEG 真实 subsampling

    gpujpeg_parameters_chroma_subsampling(&param, subsampling);
    // param.comp_count 已在上面设置好了

    // 3) 图像尺寸
    param_image.width  = static_cast<int>(img.width);
    param_image.height = static_cast<int>(img.height);

    param_image.color_space = to_gpujpeg_color_space(cs);
    param.color_space_internal = to_gpujpeg_color_space(cs);
    // printf("init_decoder_for_image: image size: %ux%u\n",
    //        param_image.width, param_image.height);  

    // // 4) 输出格式：我们希望解码得到 RGBRGB... 的打包格式
    // printf("init_decoder_for_image: set color_space_internal and color_space=%s, sample_factor=%s, comp_count=%d\n",
    //        color_space_to_cstring(cs),
    //        sample_factor_to_cstring(sf),
    //        static_cast<int>(img.channel_count));
    if (comp_count_from_cs == 1) {
        // 灰度：单通道 U8
        // param_image.color_space  = GPUJPEG_YCBCR_BT601_256LVLS; // 实际上灰度也走这个分支
        param_image.pixel_format = GPUJPEG_U8;
    } else {
        // 多通道：让 GPUJPEG 做 YCbCr->RGB，输出 packed RGB
        // param_image.color_space  = GPUJPEG_RGB;
        param_image.pixel_format = GPUJPEG_444_U8_P012;
    }

    // 5) 典型 JPEG 是 interleaved
    param.interleaved = 1;

    // 6) 初始化 decoder（这一步会算好 coder->data_size、d_data_quantized 等）
    if (gpujpeg_decoder_init(decoder, &param, &param_image) != 0) {
        throw std::runtime_error("gpujpeg_decoder_init failed");
    }

    // 初始化好 decoder 之后，先把 header 里的量化表同步过去
    sync_quant_tables_from_header_to_decoder(header, img, decoder);

    // 7) 用 ImageHeader 里的 block 数校验一下 coefficients 个数是否一致
    size_t expected_blocks = 0;
    for (uint32_t c = 0; c < img.channel_count; ++c) {
        const auto& ch = header.channel_dcts[img.first_channel_index + c];
        expected_blocks += static_cast<size_t>(ch.width_in_blocks) *
                           static_cast<size_t>(ch.height_in_blocks);
    }
    const size_t expected_coeffs = expected_blocks * 64;

    size_t coder_coeffs = gpujpeg_decoder_get_coefficients_count(decoder);
    if (coder_coeffs == 0) {
        throw std::runtime_error(
            "init_decoder_for_image: decoder coefficients count is 0 (init failed?)");
    }
    if (coder_coeffs != expected_coeffs) {
        throw std::runtime_error(
            "init_decoder_for_image: coefficients count mismatch, header vs gpujpeg");
    }

    // 8) 设置**输出格式**（和真实 JPEG 色彩空间无关，是你想要的最终结果）
    // if (comp_count_from_cs == 1) {
    //     // 灰度：如果你想输出单通道，就这样：
    //     gpujpeg_decoder_set_output_format(
    //         decoder,
    //         GPUJPEG_YCBCR_BT601_256LVLS,   // 输出色彩空间（灰度/亮度）
    //         GPUJPEG_U8                      // 单通道 8bit
    //     );
    //     printf("Note: Decoding grayscale image as single-channel U8 output.\n");
    //     // 如果你想统一输出 RGB（灰度复制三通道），可以传 GPUJPEG_RGB + 444_U8_P012，
    //     // 但要确认 GPUJPEG 这条路径支持。否则就自己在 CPU 侧复制 Y→RGB。
    // } else {
    //     // 多通道图像：输出 RGB 打包
    //     gpujpeg_decoder_set_output_format(
    //         decoder,
    //         GPUJPEG_RGB,                    // 目标输出色彩空间
    //         GPUJPEG_444_U8_P012             // RGBRGB... packed
    //     );
    //     printf("Note: Decoding image as packed RGB output.\n");
    // }
}

// 从 libjpeg 的采样因子推断 SampleFactor
static SampleFactor detect_sample_factor(const jpeg_decompress_struct& cinfo)
{
    // 灰度：只有一个分量，直接认为 4:0:0
    if (cinfo.num_components == 1) {
        return SampleFactor::SF_400;
    }

    if (cinfo.num_components != 3) {
        throw std::runtime_error(
            "Unsupported component count for sampling factor detection: " +
            std::to_string(cinfo.num_components));
    }

    // 找 component_id == 1 作为 Y 分量；找不到就用第 0 个
    const jpeg_component_info* y_comp = nullptr;
    for (int ci = 0; ci < cinfo.num_components; ++ci) {
        if (cinfo.comp_info[ci].component_id == 1) {
            y_comp = &cinfo.comp_info[ci];
            break;
        }
    }
    if (!y_comp) {
        y_comp = &cinfo.comp_info[0];
    }

    int hy = y_comp->h_samp_factor;
    int vy = y_comp->v_samp_factor;

    if (hy == 1 && vy == 1) {
        return SampleFactor::SF_444;
    }
    if (hy == 2 && vy == 1) {
        return SampleFactor::SF_422;
    }
    if (hy == 2 && vy == 2) {
        return SampleFactor::SF_420;
    }

    throw std::runtime_error("Unsupported sampling factor combination (h=" +
                             std::to_string(hy) + ", v=" + std::to_string(vy) + ")");
}


std::vector<ZeroNonZeroPair> count_zero_nonzero_pairs(const std::vector<int16_t>& sequence) {
	std::vector<ZeroNonZeroPair> result;
	size_t                       i = 0, n = sequence.size();
	while (i < n) {
		int zc = 0, nzc = 0;
		while (i < n && sequence[i] == 0)
			++zc, ++i;
		while (i < n && sequence[i] != 0)
			++nzc, ++i;
		if (zc > 0 || nzc > 0) {
			result.push_back({zc, nzc});
		}
	}
	return result;
}

std::tuple<size_t, size_t, size_t>
compute_adaptive_split(const ImageHeader& header, size_t total_blocks, size_t sample_count = 1000) {
	sample_count = std::min(sample_count, total_blocks);

	std::vector<size_t> zero_counts(64, 0);
	size_t              block_index = 0;
	size_t              sampled     = 0;

	size_t step = total_blocks / sample_count;
	if (step == 0)
		step = 1;

	for (const auto& channel : header.channel_dcts) {
		for (const auto& block : channel.blocks) {
			if (block_index % step != 0) {
				block_index++;
				continue;
			}

			for (size_t i = 0; i < 64; ++i) {
				if (block.data[i] == 0) {
					// if(block.data[zigzag_order[i]] == 0){
					zero_counts[i]++;
				}
			}
			sampled++;
			if (sampled >= sample_count)
				break;
		}
		if (sampled >= sample_count)
			break;
		block_index++;
	}

	std::vector<float> ratios(64);
	for (size_t i = 0; i < 64; ++i) {
		ratios[i] = static_cast<float>(zero_counts[i]) / static_cast<float>(sampled);
	}

	size_t l = 0, m = 0, r = 0;

	// left: <50%
	while (l < 64 && ratios[l] < 0.5f)
		l++;

	// right: >90%
	while (r < 64 - l && ratios[63 - r] >= 0.9f)
		r++;

	m = 64 - l - r;

	l = std::max(l, (size_t)1);
	m = std::max(m, (size_t)1);
	r = std::max(r, (size_t)1);
	if (l + m + r != 64)
		r = 64 - l - m;

	return {l, m, r};
}

ProcessedDCTChannel JpegLoader::process_channel(const ImageHeader& header) {
	// 先算总 block
	size_t total_blocks = 0;
	for (const auto& ch : header.channel_dcts)
		total_blocks += ch.blocks.size();

	auto [left, mid, right] = compute_adaptive_split(header, total_blocks, 1000);
	return JpegLoader::process_channel(header, left, mid, right);
}

ProcessedDCTChannel JpegLoader::process_channel(const ImageHeader& header, size_t left, size_t mid, size_t /*right*/) {
	ProcessedDCTChannel pro_dct_blocks;
	pro_dct_blocks.total_blocks = 0;
	std::vector<int16_t> low_value_ac;
	size_t               block_idx = 0;

	// metadata（each block 1 bit → use uint8_t to compressed store）
	for (const auto& channel : header.channel_dcts) {
		pro_dct_blocks.total_blocks += channel.blocks.size();
	}
	pro_dct_blocks.metadata.resize((pro_dct_blocks.total_blocks + 7) / 8, 0);

	int left_c = static_cast<int>(left) - 1; // except for DC
	int mid_c  = static_cast<int>(mid);

	for (const auto& channel : header.channel_dcts) {
		for (const auto& block : channel.blocks) {
			int16_t              dc = block.data[0];
			std::vector<int16_t> ac_coefs(block.data + 1, block.data + 63);

			pro_dct_blocks.DC_values.push_back(dc);

			auto nonzero_in_mid = std::count_if(
			    ac_coefs.begin() + left_c, ac_coefs.begin() + left_c + mid_c, [](int16_t x) { return x != 0; });
			bool is_high_value_mid = (static_cast<float>(nonzero_in_mid) >= static_cast<float>(mid_c) * 0.75f);

			if (is_high_value_mid) {
				pro_dct_blocks.metadata[block_idx / 8] |= (1 << (block_idx % 8)); // bit=1
				// left + mid
				pro_dct_blocks.AC_values.insert(
				    pro_dct_blocks.AC_values.end(), ac_coefs.begin(), ac_coefs.begin() + left_c + mid_c);
				// right merge into mix_run_nonzero_values
				for (size_t i = static_cast<size_t>(left_c + mid_c); i < 63; ++i) {
					if (ac_coefs[i] != 0) {
						pro_dct_blocks.mix_run_nonzero_values.push_back(ac_coefs[i]);
					}
				}
			} else {
				pro_dct_blocks.metadata[block_idx / 8] &= ~(1 << (block_idx % 8)); // bit=0
				// left
				pro_dct_blocks.AC_values.insert(
				    pro_dct_blocks.AC_values.end(), ac_coefs.begin(), ac_coefs.begin() + left_c);
				// mid+right：nonzero value merge into mix_run_nonzero_values
				for (size_t i = static_cast<size_t>(left_c); i < 63; ++i) {
					if (ac_coefs[i] != 0) {
						pro_dct_blocks.mix_run_nonzero_values.push_back(ac_coefs[i]);
					}
				}
				low_value_ac.insert(low_value_ac.end(), ac_coefs.begin() + left_c, ac_coefs.end());
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

ImageRGB JpegLoader::load_rgb_gpu(const std::string& path) {
    // 1.  initialized libgpujpeg
    static bool initialized = false;
    if (!initialized) {
        if (gpujpeg_init_device(0, 0) != 0) {
            throw std::runtime_error("Failed to initialize GPUJPEG device");
        }
        initialized = true;
    }

    // 2. construct decoder
    struct gpujpeg_decoder* decoder = gpujpeg_decoder_create(nullptr);
    if (!decoder) {
        throw std::runtime_error("Failed to create GPUJPEG decoder");
    }

    // 3. set output format: RGB, interleaved (packed), 8-bit
    // GPUJPEG_RGB + GPUJPEG_444_U8_P012 = packed RGB (RGBRGB...)
    // note：P012N means "packed non-planar"
    // gpujpeg_decoder_set_output_format(decoder, GPUJPEG_RGB, GPUJPEG_444_U8_P012);

    // 4. read the whole jpeg file to memory
    FILE* infile = fopen(path.c_str(), "rb");
    if (!infile) {
        gpujpeg_decoder_destroy(decoder);
        throw std::runtime_error("Cannot open file: " + path);
    }

    fseek(infile, 0, SEEK_END);
    // long file_size = ftell(infile);
	size_t file_size = static_cast<size_t>(ftell(infile));
    fseek(infile, 0, SEEK_SET);

    std::vector<uint8_t> jpeg_data(file_size);
    size_t read_size = fread(jpeg_data.data(), 1, file_size, infile);
    fclose(infile);

    if (read_size != static_cast<size_t>(file_size)) {
        gpujpeg_decoder_destroy(decoder);
        throw std::runtime_error("Failed to read entire JPEG file");
    }

    // 5. set default output（libgpujpeg allocate host memory automatically）
    struct gpujpeg_decoder_output decoder_output;
    gpujpeg_decoder_output_set_default(&decoder_output);

    // 6. execute GPU decoding
    if (gpujpeg_decoder_decode(decoder, jpeg_data.data(), file_size, &decoder_output) != 0) {
        gpujpeg_decoder_destroy(decoder);
        throw std::runtime_error("GPUJPEG decoding failed");
    }

    // 7. get iamge parameters
    // unsigned int w = decoder_output.param_image.width;
    // unsigned int h = decoder_output.param_image.height;
	auto w = static_cast<unsigned int>(decoder_output.param_image.width);
	auto h = static_cast<unsigned int>(decoder_output.param_image.height);

    // 8. copy decoding results（libgpujpeg already in memory）
    std::vector<unsigned char> buffer(decoder_output.data, 
                                      decoder_output.data + decoder_output.data_size);

    // 9. clean
    gpujpeg_decoder_destroy(decoder);

    // 10. return the result
    return ImageRGB{w, h, std::move(buffer)};
}


std::vector<ImageRGB> JpegLoader::load_rgb_dir_gpu(const std::string& dir_path) {
    // 1) collect all JPEG files in the directory and sort it
    fs::path dir(dir_path);
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        throw std::runtime_error("Not a directory: " + dir.string());
    }

    std::vector<fs::path> files;
    files.reserve(1024);
    for (auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const auto& p = entry.path();
        if (is_jpeg_ext(p)) files.push_back(p);
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        throw std::runtime_error("No JPEG files found in: " + dir.string());
    }

    // 2) initialize GPUJPEG once
    static std::once_flag s_gpu_init_once;
    std::call_once(s_gpu_init_once, [] {
        if (gpujpeg_init_device(/*device_id*/0, /*flags*/0) != 0) {
            throw std::runtime_error("Failed to initialize GPUJPEG device");
        }
    });

    // 3) prefetch thread: read files into memory
    using WorkItem = std::pair<fs::path, std::vector<uint8_t>>;
    BoundedQueue<WorkItem> queue(/*capacity=*/8);
    std::atomic<bool> prefetch_ok{true};

    std::thread prefetcher([&] {
        try {
            for (const auto& p : files) {
                auto data = read_file_to_vec(p);
                queue.push(WorkItem{p, std::move(data)});
            }
        } catch (...) {
            prefetch_ok.store(false);
        }
        queue.close();
    });

    struct ThreadJoiner {
        std::thread& t;
        ~ThreadJoiner() { if (t.joinable()) t.join(); }
    } joiner{prefetcher};

    // 4) main thread, decode images one by one
    std::vector<ImageRGB> results;
    results.reserve(files.size());

    WorkItem item;
    while (queue.pop(item)) {
        const auto& path = item.first;
        auto& jpeg_data = item.second;

        struct gpujpeg_decoder* decoder = gpujpeg_decoder_create(nullptr);
        if (!decoder) {
            queue.close();
            throw std::runtime_error("Failed to create GPUJPEG decoder");
        }
        // gpujpeg_decoder_set_output_format(decoder, GPUJPEG_RGB, GPUJPEG_444_U8_P012);

        gpujpeg_decoder_output decoder_output;
        gpujpeg_decoder_output_set_default(&decoder_output);

        if (gpujpeg_decoder_decode(decoder,
                                   jpeg_data.data(),
                                   jpeg_data.size(),
                                   &decoder_output) != 0) {
            gpujpeg_decoder_destroy(decoder);
            queue.close();
            throw std::runtime_error("GPUJPEG decoding failed: " + path.string());
        }

        unsigned int w = static_cast<unsigned int>(decoder_output.param_image.width);
        unsigned int h = static_cast<unsigned int>(decoder_output.param_image.height);

        std::vector<unsigned char> buffer(
            decoder_output.data,
            decoder_output.data + decoder_output.data_size
        );

        gpujpeg_decoder_destroy(decoder);
        results.push_back(ImageRGB{w, h, std::move(buffer)});
    }

    if (!prefetch_ok.load()) {
        throw std::runtime_error("Prefetch thread failed while reading files.");
    }

    return results;
}


// std::vector<ImageRGB>
// JpegLoader::load_rgb_dir_gpu_mt_ms(const std::string& dir_path,
//                                 int num_workers,
//                                 size_t queue_capacity)
// {
// 	// printf("JpegLoader::load_rgb_dir_gpu_mt_ms(): dir_path=%s, num_workers=%d, queue_capacity=%zu\n",
// 	    //    dir_path.c_str(), num_workers, queue_capacity);
//     // 0) 合法化 worker 数
//     if (num_workers <= 0) {
//         num_workers = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
//     }

//     // 1) 收集并排序所有 JPEG 文件
//     fs::path dir(dir_path);
//     if (!fs::exists(dir) || !fs::is_directory(dir)) {
//         throw std::runtime_error("Not a directory: " + dir.string());
//     }

//     std::vector<fs::path> files;
//     files.reserve(1024);
//     for (auto& entry : fs::directory_iterator(dir)) {
//         if (!entry.is_regular_file()) continue;
//         if (is_jpeg_ext(entry.path())) {
//             files.push_back(entry.path());
//         }
//     }
//     std::sort(files.begin(), files.end());
//     if (files.empty()) {
//         throw std::runtime_error("No JPEG files found in: " + dir.string());
//     }

//     const size_t file_count = files.size();

//     // 2) 只初始化一次 GPU 设备
//     static std::once_flag s_gpu_init_once;
//     std::call_once(s_gpu_init_once, [] {
//         if (gpujpeg_init_device(/*device_id*/0, /*flags*/0) != 0) {
//             throw std::runtime_error("Failed to initialize GPUJPEG device");
//         }
//     });

//     // 3) 定义 WorkItem：带 index，方便保持输出顺序
//     struct WorkItem {
//         size_t index;
//         fs::path path;
//         std::vector<uint8_t> jpeg_data;
//     };

//     if (queue_capacity == 0) {
//         queue_capacity = 8; // 默认队列大小
//     }
//     BoundedQueue<WorkItem> queue(queue_capacity);

//     std::atomic<bool> prefetch_ok{true};
//     std::exception_ptr prefetch_ex;

//     // 4) 预取线程：读文件 -> 放入队列
//     std::thread prefetcher([&] {
//         try {
//             for (size_t i = 0; i < file_count; ++i) {
//                 const auto& p = files[i];
//                 auto data = read_file_to_vec(p);
//                 queue.push(WorkItem{i, p, std::move(data)});
//             }
//         } catch (...) {
//             prefetch_ok.store(false);
//             prefetch_ex = std::current_exception();
//         }
//         queue.close();
//     });

//     // RAII：保证异常或正常退出时 join 预取线程
//     struct ThreadJoiner {
//         std::thread& t;
//         ~ThreadJoiner() {
//             if (t.joinable()) t.join();
//         }
//     } prefetch_joiner{prefetcher};

//     // 5) 结果数组：预先 resize，多个线程按 index 写入，保持顺序
//     std::vector<ImageRGB> results(file_count);

//     // 用来传播 worker 线程里的异常
//     std::atomic<bool> worker_ok{true};
//     std::exception_ptr worker_ex;

//     // 6) worker 线程函数：每个线程一个 decoder（=> 每个线程一个 default stream）

// 	auto worker_func = [&](int worker_id) {
// 		try {
// 			// 1) 每个 worker 线程创建一个 CUDA stream
// 			cudaStream_t stream;
// 			cudaError_t cerr = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
// 			if (cerr != cudaSuccess) {
// 				throw std::runtime_error("Failed to create CUDA stream");
// 			}

// 			// 2) 用带参数的接口创建 decoder，指定刚刚这个 stream
// 			gpujpeg_decoder_init_parameters init_params = gpujpeg_decoder_default_init_parameters();
// 			init_params.stream      = stream;   // 关键！
// 			init_params.verbose     = 0;
// 			init_params.perf_stats  = 0;
// 			init_params.ff_cs_itu601_is_709 = 0;

// 			gpujpeg_decoder* decoder = gpujpeg_decoder_create_with_params(&init_params);
// 			if (!decoder) {
// 				cudaStreamDestroy(stream);
// 				throw std::runtime_error("Failed to create GPUJPEG decoder");
// 			}

// 			gpujpeg_decoder_output decoder_output;
// 			gpujpeg_decoder_output_set_default(&decoder_output);

// 			WorkItem item;
// 			while (queue.pop(item)) {
// 				int rc = gpujpeg_decoder_decode(decoder,
// 												item.jpeg_data.data(),
// 												item.jpeg_data.size(),
// 												&decoder_output);
// 				if (rc != 0) {
// 					throw std::runtime_error("GPUJPEG decoding failed: " + item.path.string());
// 				}

// 				ImageRGB img;
// 				img.width  = static_cast<unsigned int>(decoder_output.param_image.width);
// 				img.height = static_cast<unsigned int>(decoder_output.param_image.height);
// 				img.data.assign(decoder_output.data,
// 								decoder_output.data + decoder_output.data_size);

// 				results[item.index] = std::move(img);
// 			}

// 			gpujpeg_decoder_destroy(decoder);
// 			cudaStreamDestroy(stream);
// 		} catch (...) {
// 			worker_ok.store(false);
// 			worker_ex = std::current_exception();
// 			queue.close();
// 		}
// 	};


//     // 7) 启动多个 worker
//     std::vector<std::thread> workers;
//     workers.reserve(static_cast<size_t>(num_workers));
//     for (int i = 0; i < num_workers; ++i) {
//         workers.emplace_back(worker_func, i);
//     }

//     // RAII join 所有 worker
//     struct WorkerJoiner {
//         std::vector<std::thread>& w;
//         ~WorkerJoiner() {
//             for (auto& t : w) {
//                 if (t.joinable()) t.join();
//             }
//         }
//     } workers_joiner{workers};

//     // 8) 检查错误并返回
//     if (!prefetch_ok.load()) {
//         if (prefetch_ex) std::rethrow_exception(prefetch_ex);
//         throw std::runtime_error("Prefetch thread failed.");
//     }
//     if (!worker_ok.load()) {
//         if (worker_ex) std::rethrow_exception(worker_ex);
//         throw std::runtime_error("Worker thread failed.");
//     }

//     return results;
// }



std::vector<ImageRGB>
JpegLoader::load_rgb_dir_gpu_mt(const std::string& dir_path,
                                int num_workers,
                                size_t queue_capacity)
{
	// printf("JpegLoader::load_rgb_dir_gpu_mt(): dir_path=%s, num_workers=%d, queue_capacity=%zu\n",
	    //    dir_path.c_str(), num_workers, queue_capacity);
    // 0) 合法化 worker 数
    if (num_workers <= 0) {
        num_workers = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    }

    // 1) 收集并排序所有 JPEG 文件
    fs::path dir(dir_path);
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        throw std::runtime_error("Not a directory: " + dir.string());
    }

    std::vector<fs::path> files;
    files.reserve(1024);
    for (auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        if (is_jpeg_ext(entry.path())) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        throw std::runtime_error("No JPEG files found in: " + dir.string());
    }

    const size_t file_count = files.size();

    // 2) 只初始化一次 GPU 设备
    static std::once_flag s_gpu_init_once;
    std::call_once(s_gpu_init_once, [] {
        if (gpujpeg_init_device(/*device_id*/0, /*flags*/0) != 0) {
            throw std::runtime_error("Failed to initialize GPUJPEG device");
        }
    });

    // 3) 定义 WorkItem：带 index，方便保持输出顺序
    struct WorkItem {
        size_t index;
        fs::path path;
        std::vector<uint8_t> jpeg_data;
    };

    if (queue_capacity == 0) {
        queue_capacity = 8; // 默认队列大小
    }
    BoundedQueue<WorkItem> queue(queue_capacity);

    std::atomic<bool> prefetch_ok{true};
    std::exception_ptr prefetch_ex;

    // 4) 预取线程：读文件 -> 放入队列
    std::thread prefetcher([&] {
        try {
            for (size_t i = 0; i < file_count; ++i) {
                const auto& p = files[i];
                auto data = read_file_to_vec(p);
                queue.push(WorkItem{i, p, std::move(data)});
            }
        } catch (...) {
            prefetch_ok.store(false);
            prefetch_ex = std::current_exception();
        }
        queue.close();
    });

    // RAII：保证异常或正常退出时 join 预取线程
    struct ThreadJoiner {
        std::thread& t;
        ~ThreadJoiner() {
            if (t.joinable()) t.join();
        }
    } prefetch_joiner{prefetcher};

    // 5) 结果数组：预先 resize，多个线程按 index 写入，保持顺序
    std::vector<ImageRGB> results(file_count);

    // 用来传播 worker 线程里的异常
    std::atomic<bool> worker_ok{true};
    std::exception_ptr worker_ex;

    // 6) worker 线程函数：每个线程一个 decoder（=> 每个线程一个 default stream）
    auto worker_func = [&](int worker_id) {
        try {
            // 每个线程自建一个 decoder
            gpujpeg_decoder* decoder = gpujpeg_decoder_create(nullptr);
            if (!decoder) {
                throw std::runtime_error("Failed to create GPUJPEG decoder");
            }
            // gpujpeg_decoder_set_output_format(decoder, GPUJPEG_RGB, GPUJPEG_444_U8_P012);

            gpujpeg_decoder_output decoder_output;
            gpujpeg_decoder_output_set_default(&decoder_output);

            WorkItem item;
            while (queue.pop(item)) {
                // 同一个 decoder 在本线程内复用，多张图重复 decode
                int rc = gpujpeg_decoder_decode(decoder,
                                                item.jpeg_data.data(),
                                                item.jpeg_data.size(),
                                                &decoder_output);
                if (rc != 0) {
                    throw std::runtime_error("GPUJPEG decoding failed: " + item.path.string());
                }

                // 拷贝结果到自己的 ImageRGB，然后写入 results[index]
                ImageRGB img;
                img.width  = static_cast<unsigned int>(decoder_output.param_image.width);
                img.height = static_cast<unsigned int>(decoder_output.param_image.height);
                img.data.assign(decoder_output.data,
                                decoder_output.data + decoder_output.data_size);

                results[item.index] = std::move(img);
            }

            gpujpeg_decoder_destroy(decoder);
        } catch (...) {
            worker_ok.store(false);
            worker_ex = std::current_exception();
            // 让其他线程尽快停下来
            queue.close();
        }
    };

    // 7) 启动多个 worker
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(num_workers));
    for (int i = 0; i < num_workers; ++i) {
        workers.emplace_back(worker_func, i);
    }

    // RAII join 所有 worker
    struct WorkerJoiner {
        std::vector<std::thread>& w;
        ~WorkerJoiner() {
            for (auto& t : w) {
                if (t.joinable()) t.join();
            }
        }
    } workers_joiner{workers};

    // 8) 检查错误并返回
    if (!prefetch_ok.load()) {
        if (prefetch_ex) std::rethrow_exception(prefetch_ex);
        throw std::runtime_error("Prefetch thread failed.");
    }
    if (!worker_ok.load()) {
        if (worker_ex) std::rethrow_exception(worker_ex);
        throw std::runtime_error("Worker thread failed.");
    }

    return results;
}


ImageRGB JpegLoader::load_rgb(const std::string& path) {
	FILE* infile = fopen(path.c_str(), "rb");
	if (!infile)
		throw std::runtime_error("Cannot open file: " + path);

	jpeg_decompress_struct cinfo;
	jpeg_error_mgr         jerr;
	cinfo.err = jpeg_std_error(&jerr);

	jpeg_create_decompress(&cinfo);
	jpeg_stdio_src(&cinfo, infile);
	jpeg_read_header(&cinfo, TRUE);
	jpeg_start_decompress(&cinfo);

	unsigned int w            = cinfo.output_width;
	unsigned int h            = cinfo.output_height;
	int          num_channels = cinfo.output_components; // Usually 3 (RGB)

	std::vector<unsigned char> buffer(w * h * static_cast<unsigned int>(num_channels));
	while (cinfo.output_scanline < cinfo.output_height) {
		unsigned char* rowptr = &buffer[cinfo.output_scanline * w * static_cast<unsigned int>(num_channels)];
		jpeg_read_scanlines(&cinfo, &rowptr, 1);
	}

	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
	fclose(infile);

	return ImageRGB {w, h, std::move(buffer)};
}

// ImageHeader JpegLoader::load_header(const std::string& path) {
// 	FILE* infile = fopen(path.c_str(), "rb");
// 	if (!infile)
// 		throw std::runtime_error("Cannot open file: " + path);

// 	jpeg_decompress_struct cinfo_dct;
// 	jpeg_error_mgr         jerr_dct;
// 	cinfo_dct.err = jpeg_std_error(&jerr_dct);

// 	jpeg_create_decompress(&cinfo_dct);
// 	jpeg_stdio_src(&cinfo_dct, infile);
// 	jpeg_read_header(&cinfo_dct, TRUE);

// 	// read DCT coefficients
// 	jvirt_barray_ptr* coeff_arrays = jpeg_read_coefficients(&cinfo_dct);

// 	// get quantization tables
// 	std::vector<QuantTable> qtables;
// 	for (int i = 0; i < NUM_QUANT_TBLS; ++i) {
// 		if (cinfo_dct.quant_tbl_ptrs[i]) {
// 			QuantTable q;
// 			q.id        = static_cast<uint8_t>(i);
// 			q.precision = 0;
// 			for (int j = 0; j < 64; ++j) {
// 				q.data[j] = static_cast<uint8_t>(cinfo_dct.quant_tbl_ptrs[i]->quantval[j]);
// 			}
// 			qtables.push_back(q);
// 		}
// 	}

// 	std::vector<ChannelDCT> channels;
// 	for (int ci = 0; ci < cinfo_dct.num_components; ++ci) {
// 		jpeg_component_info* comp = &cinfo_dct.comp_info[ci];
// 		ChannelDCT           cdct;
// 		cdct.component_id     = static_cast<uint8_t>(comp->component_id);
// 		cdct.qtable_id        = static_cast<uint8_t>(comp->quant_tbl_no);
// 		cdct.color_space_id   = 0; // <-- one image only one color space configuration
// 		cdct.width_in_blocks  = comp->width_in_blocks;
// 		cdct.height_in_blocks = comp->height_in_blocks;

// 		for (JDIMENSION row = 0; row < comp->height_in_blocks; ++row) {
// 			JBLOCKARRAY buffer =
// 			    (*cinfo_dct.mem->access_virt_barray)((j_common_ptr)&cinfo_dct, coeff_arrays[ci], row, 1, FALSE);

// 			for (JDIMENSION col = 0; col < comp->width_in_blocks; ++col) {
// 				DCTBlockRow block;
// 				int16_t*    src = buffer[0][col];
// 				for (size_t i = 0; i < 64; ++i) {
// 					block.data[i] = src[zigzag_order[i]];
// 				}
// 				cdct.blocks.push_back(block);
// 			}
// 		}
// 		channels.push_back(cdct);
// 	}

// 	unsigned int w = cinfo_dct.image_width;
// 	unsigned int h = cinfo_dct.image_height;

// 	// Estimate quality: optional. Here we set to 0 (unknown) since libjpeg doesn't store it.
// 	// You could implement a heuristic based on quant tables if needed.
// 	uint8_t quality = 0;

// 	// std::string color_space = jpeg_color_space_to_string(cinfo_dct.jpeg_color_space);

// 	std::vector<ColorSpace> color_spaces;
// 	color_spaces.push_back(jpeg_color_space_to_color_space(cinfo_dct.jpeg_color_space));

// 	jpeg_destroy_decompress(&cinfo_dct);
// 	fclose(infile);

// 	return {
// 	    .width        = w,
// 	    .height       = h,
// 	    .quality      = quality,
// 	    .color_spaces = color_spaces,
// 	    .quant_tables = std::move(qtables),
// 	    .channel_dcts = std::move(channels),
// 	};
// }

ImageHeader JpegLoader::load_header(const std::string& path) {
    FILE* infile = fopen(path.c_str(), "rb");
    if (!infile)
        throw std::runtime_error("Cannot open file: " + path);

    jpeg_decompress_struct cinfo_dct;
    jpeg_error_mgr         jerr_dct;
    cinfo_dct.err = jpeg_std_error(&jerr_dct);

    jpeg_create_decompress(&cinfo_dct);
    jpeg_stdio_src(&cinfo_dct, infile);
    jpeg_read_header(&cinfo_dct, TRUE);

    // 读取 DCT 系数
    jvirt_barray_ptr* coeff_arrays = jpeg_read_coefficients(&cinfo_dct);

    // 推断采样因子 & 颜色空间
    SampleFactor sf = detect_sample_factor(cinfo_dct);
    ColorSpace   cs = jpeg_color_space_to_color_space(cinfo_dct.jpeg_color_space);

    std::vector<SampleFactor> sample_factors;
    sample_factors.push_back(sf);
    const uint8_t sample_factor_id = 0;

    std::vector<ColorSpace> color_spaces;
    color_spaces.push_back(cs);
    const uint8_t color_space_id = 0;

    // 量化表
    std::vector<QuantTable> qtables;
    for (int i = 0; i < NUM_QUANT_TBLS; ++i) {
        if (cinfo_dct.quant_tbl_ptrs[i]) {
            QuantTable q;
            q.id        = static_cast<uint8_t>(i);
            q.precision = 0;
            for (int j = 0; j < 64; ++j) {
                q.data[j] = static_cast<uint8_t>(cinfo_dct.quant_tbl_ptrs[i]->quantval[j]);
            }
            qtables.push_back(q);
        }
    }

    // 通道 DCT
    std::vector<ChannelDCT> channels;
    channels.reserve(static_cast<size_t>(cinfo_dct.num_components));

    for (int ci = 0; ci < cinfo_dct.num_components; ++ci) {
        jpeg_component_info* comp = &cinfo_dct.comp_info[ci];
        ChannelDCT           cdct;
        cdct.component_id     = static_cast<uint8_t>(comp->component_id);
        cdct.qtable_id        = static_cast<uint8_t>(comp->quant_tbl_no); // 视作索引到 quant_tables
        cdct.width_in_blocks  = comp->width_in_blocks;
        cdct.height_in_blocks = comp->height_in_blocks;

        for (JDIMENSION row = 0; row < comp->height_in_blocks; ++row) {
            JBLOCKARRAY buffer =
                (*cinfo_dct.mem->access_virt_barray)(
                    (j_common_ptr)&cinfo_dct, coeff_arrays[ci], row, 1, FALSE);

            for (JDIMENSION col = 0; col < comp->width_in_blocks; ++col) {
                DCTBlockRow block;
                int16_t*    src = buffer[0][col];
                for (size_t i = 0; i < 64; ++i) {
                    block.data[i] = src[zigzag_order[i]]; // 保存为 zigzag 顺序
                }
                cdct.blocks.push_back(block);
            }
        }
        channels.push_back(std::move(cdct));
    }

    // 图像尺寸 & 质量（质量先设 0）
    unsigned int w = cinfo_dct.image_width;
    unsigned int h = cinfo_dct.image_height;
    uint8_t      quality = 0;

    jpeg_destroy_decompress(&cinfo_dct);
    fclose(infile);

    // 构造单图像的 ImageInfo
    ImageInfo img;
    img.width              = w;
    img.height             = h;
    img.quality            = quality;
    img.color_space_id     = color_space_id;
    img.sample_factor_id   = sample_factor_id;
    img.first_channel_index = 0;
    img.channel_count       = static_cast<uint32_t>(channels.size());

    ImageHeader header;
    header.sample_factors = std::move(sample_factors);
    header.color_spaces   = std::move(color_spaces);
    header.quant_tables   = std::move(qtables);
    header.channel_dcts   = std::move(channels);
    header.images.push_back(img);

    return header;
}


void idct_8x8(const int* coeffs, uint8_t* output) {
	// coeffs: frequency domain, row-major (v*8 + u)
	// output: spatial 8x8, row-major (y*8 + x)
	const double pi = M_PI;
	double       C[8];
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
				double Cv      = C[v];
				for (int u = 0; u < 8; ++u) {
					double Cu  = C[u];
					double Fuv = static_cast<double>(coeffs[v * 8 + u]); // note indexing v*8 + u
					sum += Cu * Cv * Fuv * cos_table[x][u] * cos_y_v;
				}
			}
			double val = 0.25 * sum + 128.0; // scale and shift back to [0,255]
			// clamp and round
			val               = std::round(std::clamp(val, 0.0, 255.0));
			output[y * 8 + x] = static_cast<uint8_t>(val);
		}
	}
}

void upsample_chroma_bilinear(const std::vector<std::vector<uint8_t>>& src,
                              std::vector<std::vector<uint8_t>>&       dst,
                              uint32_t                                 dst_h,
                              uint32_t                                 dst_w) {
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
		if (src_y < 0.0)
			src_y = 0.0;
		if (src_y > static_cast<double>(src_h - 1))
			src_y = static_cast<double>(src_h - 1);

		size_t y0 = static_cast<size_t>(std::floor(src_y));
		size_t y1 = (y0 + 1 < src_h) ? (y0 + 1) : y0;
		double dy = src_y - static_cast<double>(y0);

		for (uint32_t x = 0; x < dst_w; ++x) {
			double src_x = (static_cast<double>(x) + 0.5) * x_ratio - 0.5;
			if (src_x < 0.0)
				src_x = 0.0;
			if (src_x > static_cast<double>(src_w - 1))
				src_x = static_cast<double>(src_w - 1);

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
			int iv    = static_cast<int>(std::round(v));
			iv        = std::min(255, std::max(0, iv));
			dst[y][x] = static_cast<uint8_t>(iv);
		}
	}
}

void ycbcr_to_rgb(uint8_t y, uint8_t cb, uint8_t cr, uint8_t& r, uint8_t& g, uint8_t& b) {
	double Y  = y;
	double Cb = cb - 128.0;
	double Cr = cr - 128.0;

	double R = Y + 1.402 * Cr;
	double G = Y - 0.344136286 * Cb - 0.714136286 * Cr;
	double B = Y + 1.772 * Cb;

	r = static_cast<uint8_t>(std::clamp(R, 0.0, 255.0));
	g = static_cast<uint8_t>(std::clamp(G, 0.0, 255.0));
	b = static_cast<uint8_t>(std::clamp(B, 0.0, 255.0));
}

static void resize_or_crop_plane(const std::vector<std::vector<uint8_t>>& src,
                                 std::vector<std::vector<uint8_t>>&       dst,
                                 uint32_t                                 dst_h,
                                 uint32_t                                 dst_w) {
	if (src.empty() || src[0].empty()) {
		dst.clear();
		return;
	}
	const size_t src_h = src.size();
	const size_t src_w = src[0].size();

	if (static_cast<uint32_t>(src_h) == dst_h && static_cast<uint32_t>(src_w) == dst_w) {
		dst = src;
		return;
	}

	if (static_cast<uint32_t>(src_h) >= dst_h && static_cast<uint32_t>(src_w) >= dst_w) {
		dst.assign(dst_h, std::vector<uint8_t>(dst_w));
		for (uint32_t y = 0; y < dst_h; ++y) {
			for (uint32_t x = 0; x < dst_w; ++x) {
				dst[y][x] = src[y][x];
			}
		}
		return;
	}

	upsample_chroma_bilinear(src, dst, dst_h, dst_w);
}


std::vector<std::vector<std::vector<std::vector<uint8_t>>>>
JpegLoader::to_rgb_gpu(const std::vector<std::vector<double>>& dct_blocks,
                       const path& file_path)
{
    using namespace std;
    ImageHeader header;
    if (!load_ImageHeader(header, file_path.string().c_str())) {
        throw runtime_error("Failed to load ImageHeader from " + file_path.string());
    }
    if (header.images.empty()) {
        throw runtime_error("ImageHeader contains no images.");
    }

    // 初始化 GPUJPEG
    static once_flag s_gpu_init;
    call_once(s_gpu_init, [] {
        if (gpujpeg_init_device(0, 0) != 0) {
            throw runtime_error("Failed to initialize GPUJPEG device");
        }
    });

    vector<vector<vector<vector<uint8_t>>>> results;
    results.resize(header.images.size());

    size_t global_block_idx = 0;


    // === 创建解码器 ===
    gpujpeg_decoder* decoder = gpujpeg_decoder_create(nullptr);
    if (!decoder) throw std::runtime_error("Failed to create GPUJPEG decoder");


    for (size_t img_idx = 0; img_idx < header.images.size(); ++img_idx) {
        const auto& img = header.images[img_idx];
        uint32_t img_w = img.width;
        uint32_t img_h = img.height;

        gpujpeg_decoder_output decoder_output;
        gpujpeg_decoder_output_set_default(&decoder_output);

        // ✅ 使用 ImageHeader + ImageInfo 初始化 gpujpeg_decoder
        init_decoder_for_image(header, img, decoder);

        // 如果想明确指定输出为 RGB packed，可以再调一次：
        // gpujpeg_decoder_set_output_format(decoder, GPUJPEG_RGB, GPUJPEG_444_U8_P012);

        // === 构建外部量化系数缓冲区 ===
        std::vector<int16_t> coeffs;
        coeffs.reserve(img.channel_count * 64);

        for (uint32_t c = 0; c < img.channel_count; ++c) {
            const auto& ch = header.channel_dcts[img.first_channel_index + c];

            // 查找量化表
            const QuantTable* qt = nullptr;
            for (const auto& q : header.quant_tables) {
                if (q.id == ch.qtable_id) { qt = &q; break; }
            }
            if (!qt) {
                gpujpeg_decoder_destroy(decoder);
                throw std::runtime_error("Quant table not found for channel " + std::to_string(c));
            }

            const uint32_t num_blocks =
                ch.width_in_blocks * ch.height_in_blocks;

            for (uint32_t b = 0; b < num_blocks; ++b) {
                if (global_block_idx >= dct_blocks.size()) {
                    gpujpeg_decoder_destroy(decoder);
                    throw std::runtime_error("Ran out of DCT blocks for GPU decode");
                }

                int16_t block_nat[64];
                const auto& src = dct_blocks[global_block_idx];

                for (size_t k = 0; k < 64; ++k) {
                    // dct_blocks[k] 是按 zigzag 存的，这里转成 natural 顺序
                    int zz = zigzag_order[k];
                    double val = (k < src.size()) ? src[k] : 0.0;
                    block_nat[zz] = static_cast<int16_t>(std::round(val));
                }

                // 这里只是把「量化后的系数」按 natural 顺序排好，真正的反量化（乘 qtable）
                // 会在 GPUJPEG 内部完成；如果你希望自己先乘一遍，再传给 GPU 也可以，
                // 那样就必须同步修改 GPU 端的 dequant 逻辑。

                coeffs.insert(coeffs.end(), block_nat, block_nat + 64);
                ++global_block_idx;
            }
        }

        // ✅ 校验一下和 gpujpeg 初始化时预期的系数个数是否一致
        size_t expect_coeffs = gpujpeg_decoder_get_coefficients_count(decoder);
        if (coeffs.size() != expect_coeffs) {
            gpujpeg_decoder_destroy(decoder);
            throw std::runtime_error(
                "coeffs.size() != gpujpeg_decoder_get_coefficients_count(decoder)");
        }

        // === 上传系数到 GPUJPEG ===
        // ⚠️ 注意：这里传的是「元素个数」，不要乘 sizeof(int16_t)
        if (gpujpeg_decoder_set_quantized_coefficients_host(
                decoder, coeffs.data(), coeffs.size()) != 0)
        {
            gpujpeg_decoder_destroy(decoder);
            throw std::runtime_error("Failed to set external coefficients to GPUJPEG");
        }

        // === 执行 GPU 解码（外部系数路径） ===
        if (gpujpeg_decoder_process_external_coefficients(decoder, &decoder_output) != 0) {
            gpujpeg_decoder_destroy(decoder);
            throw std::runtime_error("GPUJPEG external coefficients decoding failed");
        }

        // === 拷贝输出 ===
        unsigned char* out_data = decoder_output.data;
        // size_t out_size = decoder_output.data_size;

        // // 理论上这里 out_size 应该是 img_w * img_h * 3（RGB packed）
        // if (out_size != static_cast<size_t>(img_w) * img_h * 3) {
        //     fprintf(stderr,
        //             "Warning: decoded size mismatch (expected %u x %u x 3 = %zu, got %zu)\n",
        //             img_w, img_h,
        //             static_cast<size_t>(img_w) * img_h * 3,
        //             out_size);
        // }

        std::vector<std::vector<std::vector<uint8_t>>> rgb(
            3, std::vector<std::vector<uint8_t>>(img_h, std::vector<uint8_t>(img_w)));

        for (uint32_t y = 0; y < img_h; ++y) {
            for (uint32_t x = 0; x < img_w; ++x) {
                size_t idx = (static_cast<size_t>(y) * img_w + x) * 3;
                rgb[0][y][x] = out_data[idx + 0];
                rgb[1][y][x] = out_data[idx + 1];
                rgb[2][y][x] = out_data[idx + 2];
            }
        }

        results[img_idx] = std::move(rgb);
    }

    gpujpeg_decoder_destroy(decoder);

    return results;
}

// std::vector<std::vector<std::vector<std::vector<uint8_t>>>>
// JpegLoader::to_rgb(const std::vector<std::vector<double>>& dct_blocks, const std::filesystem::path& file_path) {

// 	ImageHeader header;
// 	if (!load_ImageHeader(header, file_path.c_str())) {
// 		throw std::runtime_error("Failed to load ImageHeader from " + file_path.string());
// 	}

// 	// printf("in to_rgb, we load image header:\n");
// 	// JpegLoader::print_image_header(header);

// 	if (header.channel_dcts.empty()) {
// 		throw std::runtime_error("No channels in header.");
// 	}

// 	// === Step 1: Group channels by image (contiguous same color_space_id) ===
// 	struct ImageRange {
// 		size_t  start_idx;     // start index in channel_dcts
// 		size_t  channel_count; // expected channels for this image
// 		uint8_t color_space_id;
// 	};

// 	std::vector<ImageRange> image_ranges;
// 	size_t                  i = 0;
// 	while (i < header.channel_dcts.size()) {
// 		uint8_t cs_id = header.channel_dcts[i].color_space_id;
// 		if (cs_id >= header.color_spaces.size()) {
// 			throw std::runtime_error("Invalid color_space_id: " + std::to_string(cs_id));
// 		}

// 		ColorSpace color_space_name  = header.color_spaces[cs_id];
// 		size_t     expected_channels = get_channel_count_from_color_space(color_space_name);

// 		// Check that next 'expected_channels' channels all have same cs_id
// 		if (i + expected_channels > header.channel_dcts.size()) {
// 			throw std::runtime_error("Incomplete channel group at end.");
// 		}

// 		for (size_t j = 0; j < expected_channels; ++j) {
// 			if (header.channel_dcts[i + j].color_space_id != cs_id) {
// 				throw std::runtime_error("Channel group has inconsistent color_space_id.");
// 			}
// 		}

// 		image_ranges.push_back({i, expected_channels, cs_id});
// 		i += expected_channels;
// 	}

// 	size_t num_images = image_ranges.size();

// 	// === Step 2: Validate total DCT block count ===
// 	size_t total_blocks_expected = 0;
// 	for (const auto& ch : header.channel_dcts) {
// 		total_blocks_expected += static_cast<size_t>(ch.width_in_blocks) * ch.height_in_blocks;
// 	}
// 	if (dct_blocks.size() != total_blocks_expected) {
// 		throw std::runtime_error("dct_blocks size (" + std::to_string(dct_blocks.size()) + ") != expected (" +
// 		                         std::to_string(total_blocks_expected) + ")");
// 	}

// 	// === Step 3: Build quant table map ===
// 	std::map<uint8_t, const QuantTable*> qt_map;
// 	for (const auto& qt : header.quant_tables) {
// 		qt_map[qt.id] = &qt;
// 	}

// 	// === Step 4: Decode each image ===
// 	std::vector<std::vector<std::vector<std::vector<uint8_t>>>> result;
// 	result.resize(num_images);

// 	size_t block_idx = 0;

// 	for (size_t img_idx = 0; img_idx < num_images; ++img_idx) {
// 		const auto& range      = image_ranges[img_idx];
// 		uint32_t    img_width  = header.width;
// 		uint32_t    img_height = header.height;

// 		std::vector<std::vector<std::vector<uint8_t>>> planes;
// 		planes.reserve(range.channel_count);

// 		// Decode each channel of this image
// 		for (size_t ch_offset = 0; ch_offset < range.channel_count; ++ch_offset) {
// 			const auto& ch = header.channel_dcts[range.start_idx + ch_offset];

// 			uint32_t w_blocks = ch.width_in_blocks;
// 			uint32_t h_blocks = ch.height_in_blocks;
// 			uint32_t plane_w  = w_blocks * 8;
// 			uint32_t plane_h  = h_blocks * 8;

// 			auto qt_it = qt_map.find(ch.qtable_id);
// 			if (qt_it == qt_map.end()) {
// 				throw std::runtime_error("Quant table not found for qtable_id=" + std::to_string(ch.qtable_id));
// 			}
// 			const uint8_t* qtable = qt_it->second->data;

// 			std::vector<std::vector<uint8_t>> plane(plane_h, std::vector<uint8_t>(plane_w));

// 			for (size_t by = 0; by < h_blocks; ++by) {
// 				for (size_t bx = 0; bx < w_blocks; ++bx) {
// 					if (block_idx >= dct_blocks.size()) {
// 						throw std::runtime_error("Ran out of DCT blocks while decoding.");
// 					}
// 					const auto& zigzag_coeffs = dct_blocks[block_idx++];
// 					int16_t     spatial_coeffs[64];
// 					int         dequant_coeffs[64];

// 					for (size_t k = 0; k < 64; ++k) {
// 						// Defensive: ensure zigzag_coeffs has at least 64 entries
// 						double coeff = (k < static_cast<int>(zigzag_coeffs.size())) ? zigzag_coeffs[k] : 0.0;
// 						spatial_coeffs[zigzag_order[k]] = static_cast<int16_t>(std::round(coeff));
// 					}

// 					for (size_t k = 0; k < 64; ++k) {
// 						dequant_coeffs[k] = static_cast<int>(spatial_coeffs[k]) * static_cast<int>(qtable[k]);
// 					}

// 					uint8_t pixels[64];
// 					idct_8x8(dequant_coeffs, pixels);

// 					for (size_t y = 0; y < 8; ++y) {
// 						for (size_t x = 0; x < 8; ++x) {
// 							size_t py = by * 8 + y;
// 							size_t px = bx * 8 + x;
// 							if (py < plane_h && px < plane_w) {
// 								plane[py][px] = pixels[y * 8 + x];
// 							}
// 						}
// 					}
// 				}
// 			}
// 			planes.push_back(std::move(plane));
// 		}

// 		// Handle grayscale → RGB
// 		if (planes.size() == 1) {
// 			planes.push_back(planes[0]);
// 			planes.push_back(planes[0]);
// 		} else if (planes.size() != 3) {
// 			throw std::runtime_error("Unsupported channel count: " + std::to_string(planes.size()));
// 		}

// 		// === NEW: ensure each plane is resized/cropped to img_height x img_width ===
// 		for (size_t c = 0; c < planes.size(); ++c) {
// 			std::vector<std::vector<uint8_t>> fixed;
// 			resize_or_crop_plane(planes[c], fixed, img_height, img_width);
// 			planes[c] = std::move(fixed);
// 		}

// 		uint32_t y_plane_h = static_cast<uint32_t>(planes[0].size());
// 		uint32_t y_plane_w = planes[0].empty() ? 0u : static_cast<uint32_t>(planes[0][0].size());

// 		for (size_t c = 1; c <= 2; ++c) {
// 			uint32_t c_h = static_cast<uint32_t>(planes[c].size());
// 			uint32_t c_w = planes[c].empty() ? 0u : static_cast<uint32_t>(planes[c][0].size());

// 			std::vector<std::vector<uint8_t>> tmp;
// 			if (c_h != y_plane_h || c_w != y_plane_w) {
// 				resize_or_crop_plane(planes[c], tmp, y_plane_h, y_plane_w);
// 			} else {
// 				tmp = planes[c];
// 			}

// 			std::vector<std::vector<uint8_t>> final_plane;
// 			resize_or_crop_plane(tmp, final_plane, img_height, img_width);
// 			planes[c] = std::move(final_plane);
// 		}

// 		// Prepare result storage for this image: [3][H][W] of uint8_t
// 		result[img_idx].resize(3);
// 		for (size_t c = 0; c < 3; ++c) {
// 			result[img_idx][c].assign(img_height, std::vector<uint8_t>(img_width));
// 		}

// 		// Convert to RGB uint8
// 		for (uint32_t y = 0; y < img_height; ++y) {
// 			for (uint32_t x = 0; x < img_width; ++x) {
// 				uint8_t Y  = planes[0][y][x];
// 				uint8_t Cb = planes[1][y][x];
// 				uint8_t Cr = planes[2][y][x];
// 				uint8_t r, g, b;
// 				ycbcr_to_rgb(Y, Cb, Cr, r, g, b);
// 				result[img_idx][0][y][x] = r;
// 				result[img_idx][1][y][x] = g;
// 				result[img_idx][2][y][x] = b;
// 			}
// 		}
// 	}

// 	if (block_idx != dct_blocks.size()) {
// 		throw std::runtime_error("Unused DCT blocks remain after decoding all images.");
// 	}

// 	return result; // shape: [N][3][H][W], values in uint8_t
// }

std::vector<std::vector<std::vector<std::vector<uint8_t>>>>
JpegLoader::to_rgb(const std::vector<std::vector<double>>& dct_blocks,
                   const std::filesystem::path& header_path)
{
    ImageHeader header;
    if (!load_ImageHeader(header, header_path.c_str())) {
        throw std::runtime_error("Failed to load ImageHeader from " + header_path.string());
    }

    if (header.channel_dcts.empty()) {
        throw std::runtime_error("No channels in header.");
    }
    if (header.images.empty()) {
        throw std::runtime_error("No images in header.");
    }

    // === Step 1: 校验 dct_blocks 数量 ===
    size_t total_blocks_expected = 0;
    for (const auto& ch : header.channel_dcts) {
        total_blocks_expected += static_cast<size_t>(ch.width_in_blocks) *
                                 static_cast<size_t>(ch.height_in_blocks);
    }
    if (dct_blocks.size() != total_blocks_expected) {
        throw std::runtime_error(
            "dct_blocks size (" + std::to_string(dct_blocks.size()) +
            ") != expected (" + std::to_string(total_blocks_expected) + ")");
    }

    // === Step 2: 量化表映射（qtable_id -> QuantTable*）===
    std::map<uint8_t, const QuantTable*> qt_map;
    for (const auto& qt : header.quant_tables) {
        qt_map[qt.id] = &qt;
    }

    // === Step 3: 对每一张图做反变换 ===
    const size_t num_images = header.images.size();
    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> result;
    result.resize(num_images);

    size_t block_idx = 0;  // 消费 dct_blocks 的游标（按 channel 顺序）

    for (size_t img_idx = 0; img_idx < num_images; ++img_idx) {
        const ImageInfo& img_info = header.images[img_idx];

        if (img_info.color_space_id >= header.color_spaces.size()) {
            throw std::runtime_error("Invalid color_space_id in ImageInfo: " +
                                     std::to_string(img_info.color_space_id));
        }
        ColorSpace color_space = header.color_spaces[img_info.color_space_id];

        uint32_t img_width  = img_info.width;
        uint32_t img_height = img_info.height;

        // 通道范围 [first_channel_index, first_channel_index + channel_count)
        uint32_t first_ch = img_info.first_channel_index;
        uint32_t ch_cnt   = img_info.channel_count;

        if (first_ch + ch_cnt > header.channel_dcts.size()) {
            throw std::runtime_error("Image channel range out of bounds.");
        }

        // 根据颜色空间检查通道数是否合理
        size_t expected_ch = get_channel_count_from_color_space(color_space);
        // 对于灰度，expected_ch=1；数据里如果就是 1，就 OK。
        if (ch_cnt != expected_ch) {
            // 这里可以按需更宽松一点；现在先严格检查。
            throw std::runtime_error(
                "Channel count mismatch for image " + std::to_string(img_idx) +
                ": got " + std::to_string(ch_cnt) +
                ", expected " + std::to_string(expected_ch));
        }

        // planes[c][h][w]，c in [0, ch_cnt)
        std::vector<std::vector<std::vector<uint8_t>>> planes;
        planes.reserve(ch_cnt);

        // === 3.1 解出每个通道的 spatial plane ===
        for (uint32_t ch_offset = 0; ch_offset < ch_cnt; ++ch_offset) {
            const ChannelDCT& ch = header.channel_dcts[first_ch + ch_offset];

            uint32_t w_blocks = ch.width_in_blocks;
            uint32_t h_blocks = ch.height_in_blocks;
            uint32_t plane_w  = w_blocks * 8;
            uint32_t plane_h  = h_blocks * 8;

            auto qt_it = qt_map.find(ch.qtable_id);
            if (qt_it == qt_map.end()) {
                throw std::runtime_error("Quant table not found for qtable_id=" +
                                         std::to_string(ch.qtable_id));
            }
            const uint8_t* qtable = qt_it->second->data;

            std::vector<std::vector<uint8_t>> plane(plane_h,
                                                    std::vector<uint8_t>(plane_w));

            for (uint32_t by = 0; by < h_blocks; ++by) {
                for (uint32_t bx = 0; bx < w_blocks; ++bx) {
                    if (block_idx >= dct_blocks.size()) {
                        throw std::runtime_error("Ran out of DCT blocks while decoding.");
                    }

                    const auto& zigzag_coeffs = dct_blocks[block_idx++];
                    int16_t     spatial_coeffs[64];
                    int         dequant_coeffs[64];

                    // 你的 dct_blocks 是 zigzag 顺序，这里还原到自然顺序
                    for (size_t k = 0; k < 64; ++k) {
                        double coeff = (k < zigzag_coeffs.size()) ? zigzag_coeffs[k] : 0.0;
                        spatial_coeffs[zigzag_order[k]] =
                            static_cast<int16_t>(std::round(coeff));
                    }

                    // 反量化
                    for (size_t k = 0; k < 64; ++k) {
                        dequant_coeffs[k] = static_cast<int>(spatial_coeffs[k]) *
                                            static_cast<int>(qtable[k]);
                    }

                    // IDCT -> 8x8 像素
                    uint8_t pixels[64];
                    idct_8x8(dequant_coeffs, pixels);

                    // 写回 plane
                    for (size_t y = 0; y < 8; ++y) {
                        for (size_t x = 0; x < 8; ++x) {
                            size_t py = static_cast<size_t>(by) * 8 + y;
                            size_t px = static_cast<size_t>(bx) * 8 + x;
                            if (py < plane_h && px < plane_w) {
                                plane[py][px] = pixels[y * 8 + x];
                            }
                        }
                    }
                }
            }

            // 把 plane resize/crop 到真实图像大小
            std::vector<std::vector<uint8_t>> fixed;
            resize_or_crop_plane(plane, fixed, img_height, img_width);
            planes.push_back(std::move(fixed));
        }

        // 这里 planes.size() == ch_cnt

        // === 3.2 根据颜色空间进行最终 RGB 合成 ===
        result[img_idx].resize(3); // [3][H][W]
        for (size_t c = 0; c < 3; ++c) {
            result[img_idx][c].assign(img_height, std::vector<uint8_t>(img_width));
        }

        if (color_space == ColorSpace::Grayscale) {
            // 单通道灰度，直接复制到 RGB 三个通道
            if (planes.size() != 1) {
                throw std::runtime_error("Grayscale image with channel_count != 1");
            }
            const auto& Y = planes[0];
            for (uint32_t y = 0; y < img_height; ++y) {
                for (uint32_t x = 0; x < img_width; ++x) {
                    uint8_t v = Y[y][x];
                    result[img_idx][0][y][x] = v;
                    result[img_idx][1][y][x] = v;
                    result[img_idx][2][y][x] = v;
                }
            }
        } else if (color_space == ColorSpace::YCbCr) {
            if (planes.size() != 3) {
                throw std::runtime_error("YCbCr image with channel_count != 3");
            }
            const auto& Y  = planes[0];
            const auto& Cb = planes[1];
            const auto& Cr = planes[2];
            for (uint32_t y = 0; y < img_height; ++y) {
                for (uint32_t x = 0; x < img_width; ++x) {
                    uint8_t r, g, b;
                    ycbcr_to_rgb(Y[y][x], Cb[y][x], Cr[y][x], r, g, b);
                    result[img_idx][0][y][x] = r;
                    result[img_idx][1][y][x] = g;
                    result[img_idx][2][y][x] = b;
                }
            }
        } else if (color_space == ColorSpace::RGB) {
            if (planes.size() != 3) {
                throw std::runtime_error("RGB image with channel_count != 3");
            }
            const auto& R = planes[0];
            const auto& G = planes[1];
            const auto& B = planes[2];
            for (uint32_t y = 0; y < img_height; ++y) {
                for (uint32_t x = 0; x < img_width; ++x) {
                    result[img_idx][0][y][x] = R[y][x];
                    result[img_idx][1][y][x] = G[y][x];
                    result[img_idx][2][y][x] = B[y][x];
                }
            }
        } else {
            // CMYK / YCCK 等暂时不支持
            throw std::runtime_error(
                "Unsupported color space in to_rgb (only Grayscale/YCbCr/RGB supported).");
        }
    }

    if (block_idx != dct_blocks.size()) {
        throw std::runtime_error("Unused DCT blocks remain after decoding all images.");
    }

    // shape: [N][3][H][W]
    return result;
}



// void JpegLoader::print_image_header(const ImageHeader& header) {
// 	printf("=== Image Header ===\n");
// 	printf("Width: %u pixels\n", header.width);
// 	printf("Height: %u pixels\n", header.height);
// 	printf("Quality: %u (0 = unknown)\n", header.quality);
// 	// printf("Color Space: %s\n", header.color_space.c_str());
// 	printf("\n");

// 	printf("Color Space (%zu config):\n", header.color_spaces.size());
// 	for (size_t i = 0; i < header.color_spaces.size(); ++i) {
// 		printf("Color Space %zu: %s\n", i, color_space_to_cstring(header.color_spaces[i]));
// 	}

// 	printf("Quantization Tables (%zu tables):\n", header.quant_tables.size());
// 	for (size_t i = 0; i < header.quant_tables.size(); ++i) {
// 		const auto& qt = header.quant_tables[i];
// 		printf("  Table %zu (ID: %d, Precision: %d-bit):\n", i, (int)qt.id, (qt.precision == 0) ? 8 : 16);
// 		for (int row = 0; row < 8; ++row) {
// 			printf("    ");
// 			for (int col = 0; col < 8; ++col) {
// 				printf("%4d ", (int)qt.data[row * 8 + col]);
// 			}
// 			printf("\n");
// 		}
// 		printf("\n");
// 	}

// 	printf("Channel DCT Data (%zu channels):\n", header.channel_dcts.size());
// 	for (size_t ch = 0; ch < header.channel_dcts.size(); ++ch) {
// 		const auto& channel = header.channel_dcts[ch];
// 		printf("  Channel %zu:\n", ch);
// 		printf("    Component ID: %d\n", (int)channel.component_id);
// 		printf("    Quant Table ID: %d\n", (int)channel.qtable_id);
// 		printf("    Color Space ID: %d\n", (int)channel.color_space_id);
// 		printf("    Width in blocks: %u\n", channel.width_in_blocks);
// 		printf("    Height in blocks: %u\n", channel.height_in_blocks);
// 		printf("    Total blocks: %zu\n", channel.blocks.size());

// 		const size_t max_blocks_to_show = 2;
// 		size_t       blocks_to_show     = std::min(max_blocks_to_show, channel.blocks.size());
// 		for (size_t block_idx = 0; block_idx < blocks_to_show; ++block_idx) {
// 			printf("    Block %zu:\n", block_idx);
// 			const auto& block = channel.blocks[block_idx];
// 			for (int row = 0; row < 8; ++row) {
// 				printf("      ");
// 				for (int col = 0; col < 8; ++col) {
// 					printf("%6d ", (int)block.data[row * 8 + col]);
// 				}
// 				printf("\n");
// 			}
// 		}
// 		if (channel.blocks.size() > max_blocks_to_show) {
// 			printf("    ... (showing only first %zu blocks out of %zu)\n", max_blocks_to_show, channel.blocks.size());
// 		}
// 		printf("\n");
// 	}
// }

void JpegLoader::print_image_header(const ImageHeader& header) {
    printf("=== Image Header ===\n");

    // 全局采样配置
    printf("Sample Factors (%zu configs):\n", header.sample_factors.size());
    for (size_t i = 0; i < header.sample_factors.size(); ++i) {
        printf("  [%zu] %s\n", i, sample_factor_to_cstring(header.sample_factors[i]));
    }
    printf("\n");

    // 全局颜色空间配置
    printf("Color Spaces (%zu configs):\n", header.color_spaces.size());
    for (size_t i = 0; i < header.color_spaces.size(); ++i) {
        printf("  [%zu] %s\n", i, color_space_to_cstring(header.color_spaces[i]));
    }
    printf("\n");

    // 量化表
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

    // 图像信息
    printf("Images (%zu):\n", header.images.size());
    for (size_t i = 0; i < header.images.size(); ++i) {
        const auto& img = header.images[i];
        printf("  Image %zu:\n", i);
        printf("    Size: %u x %u\n", img.width, img.height);
        printf("    Quality: %u\n", img.quality);
        printf("    Color Space ID: %u",
               (unsigned)img.color_space_id);
        if (img.color_space_id < header.color_spaces.size()) {
            printf(" (%s)", color_space_to_cstring(header.color_spaces[img.color_space_id]));
        }
        printf("\n");
        printf("    Sample Factor ID: %u",
               (unsigned)img.sample_factor_id);
        if (img.sample_factor_id < header.sample_factors.size()) {
            printf(" (%s)", sample_factor_to_cstring(header.sample_factors[img.sample_factor_id]));
        }
        printf("\n");
        printf("    First Channel Index: %u\n", img.first_channel_index);
        printf("    Channel Count: %u\n", img.channel_count);
    }
    printf("\n");

    // 通道 DCT
    printf("Channel DCT Data (%zu channels):\n", header.channel_dcts.size());
    for (size_t ch = 0; ch < header.channel_dcts.size(); ++ch) {
        const auto& channel = header.channel_dcts[ch];
        printf("  Channel %zu:\n", ch);
        printf("    Component ID: %d\n", (int)channel.component_id);
        printf("    Quant Table ID: %d\n", (int)channel.qtable_id);
        printf("    Width in blocks: %u\n", channel.width_in_blocks);
        printf("    Height in blocks: %u\n", channel.height_in_blocks);
        printf("    Total blocks: %zu\n", channel.blocks.size());

        const size_t max_blocks_to_show = 2;
        size_t       blocks_to_show     = std::min(max_blocks_to_show, channel.blocks.size());
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


// bool JpegLoader::dump_ImageHeader(const ImageHeader& header, const char* filename) {
// 	FILE* fp = fopen(filename, "wb");
// 	if (!fp) {
// 		perror("fopen (write)");
// 		return false;
// 	}

// 	// Basic image info
// 	fwrite(&header.width, sizeof(uint32_t), 1, fp);
// 	fwrite(&header.height, sizeof(uint32_t), 1, fp);
// 	fwrite(&header.quality, sizeof(uint8_t), 1, fp);

// 	// uint64_t color_space_len = header.color_space.size();
// 	// fwrite(&color_space_len, sizeof(uint64_t), 1, fp);
// 	// if (color_space_len > 0) {
// 	//     fwrite(header.color_space.data(), sizeof(char), color_space_len, fp);

// 	// Color space: write length + string
// 	uint64_t cs_count = header.color_spaces.size();
// 	fwrite(&cs_count, sizeof(uint64_t), 1, fp);
// 	for (const auto& cs : header.color_spaces) {
// 		uint8_t cs_val = static_cast<uint8_t>(cs);
// 		fwrite(&cs_val, sizeof(uint8_t), 1, fp);
// 	}

// 	// Quant tables
// 	uint64_t qt_count = header.quant_tables.size();
// 	fwrite(&qt_count, sizeof(uint64_t), 1, fp);
// 	for (const auto& qt : header.quant_tables) {
// 		fwrite(&qt.id, sizeof(uint8_t), 1, fp);
// 		fwrite(&qt.precision, sizeof(uint8_t), 1, fp);
// 		fwrite(qt.data, sizeof(uint8_t), 64, fp);
// 	}

// 	// Channels
// 	uint64_t ch_count = header.channel_dcts.size();
// 	fwrite(&ch_count, sizeof(uint64_t), 1, fp);
// 	for (const auto& ch : header.channel_dcts) {
// 		fwrite(&ch.component_id, sizeof(uint8_t), 1, fp);
// 		fwrite(&ch.qtable_id, sizeof(uint8_t), 1, fp);
// 		fwrite(&ch.color_space_id, sizeof(uint8_t), 1, fp);
// 		fwrite(&ch.width_in_blocks, sizeof(uint32_t), 1, fp);
// 		fwrite(&ch.height_in_blocks, sizeof(uint32_t), 1, fp);
// 		// Note: blocks are NOT saved in header dump (as before)
// 	}

// 	fclose(fp);
// 	printf("Successfully wrote header to %s\n", filename);
// 	return true;
// }

bool JpegLoader::dump_ImageHeader(const ImageHeader& header, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        perror("fopen (write)");
        return false;
    }

    // Sample factors
    uint64_t sf_count = header.sample_factors.size();
    fwrite(&sf_count, sizeof(uint64_t), 1, fp);
    for (const auto& sf : header.sample_factors) {
        uint8_t sf_val = static_cast<uint8_t>(sf);
        fwrite(&sf_val, sizeof(uint8_t), 1, fp);
    }

    // Color spaces
    uint64_t cs_count = header.color_spaces.size();
    fwrite(&cs_count, sizeof(uint64_t), 1, fp);
    for (const auto& cs : header.color_spaces) {
        uint8_t cs_val = static_cast<uint8_t>(cs);
        fwrite(&cs_val, sizeof(uint8_t), 1, fp);
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
        fwrite(&ch.qtable_id, sizeof(uint8_t), 1, fp);
        fwrite(&ch.width_in_blocks, sizeof(uint32_t), 1, fp);
        fwrite(&ch.height_in_blocks, sizeof(uint32_t), 1, fp);
        // blocks 不保存
    }

    // Images
    uint64_t img_count = header.images.size();
    fwrite(&img_count, sizeof(uint64_t), 1, fp);
    for (const auto& img : header.images) {
        fwrite(&img.width,  sizeof(uint32_t), 1, fp);
        fwrite(&img.height, sizeof(uint32_t), 1, fp);
        fwrite(&img.quality, sizeof(uint8_t), 1, fp);
        fwrite(&img.color_space_id, sizeof(uint8_t), 1, fp);
        fwrite(&img.sample_factor_id, sizeof(uint8_t), 1, fp);
        fwrite(&img.first_channel_index, sizeof(uint32_t), 1, fp);
        fwrite(&img.channel_count, sizeof(uint32_t), 1, fp);
    }

    fclose(fp);
    printf("Successfully wrote header to %s\n", filename);
    return true;
}


// bool JpegLoader::load_ImageHeader(ImageHeader& header, const char* filename) {
// 	FILE* fp = fopen(filename, "rb");
// 	if (!fp) {
// 		perror("fopen (read)");
// 		return false;
// 	}

// 	fread(&header.width, sizeof(uint32_t), 1, fp);
// 	fread(&header.height, sizeof(uint32_t), 1, fp);
// 	fread(&header.quality, sizeof(uint8_t), 1, fp);

// 	// uint64_t color_space_len;
// 	// fread(&color_space_len, sizeof(uint64_t), 1, fp);
// 	// header.color_space.resize(color_space_len);
// 	// if (color_space_len > 0) {
// 	//     fread(&header.color_space[0], sizeof(char), color_space_len, fp);

// 	// Color space
// 	uint64_t cs_count;
// 	fread(&cs_count, sizeof(uint64_t), 1, fp);
// 	header.color_spaces.resize(cs_count);
// 	for (size_t i = 0; i < cs_count; ++i) {
// 		uint8_t cs_val;
// 		fread(&cs_val, sizeof(uint8_t), 1, fp);
// 		if (cs_val > static_cast<uint8_t>(ColorSpace::YCCK)) {
// 			fclose(fp);
// 			throw std::runtime_error("Invalid ColorSpace value in header file: " + std::to_string(cs_val));
// 		}
// 		header.color_spaces[i] = static_cast<ColorSpace>(cs_val);
// 	}

// 	// Quant tables
// 	uint64_t qt_count;
// 	fread(&qt_count, sizeof(uint64_t), 1, fp);
// 	header.quant_tables.resize(qt_count);
// 	for (size_t i = 0; i < qt_count; ++i) {
// 		auto& qt = header.quant_tables[i];
// 		fread(&qt.id, sizeof(uint8_t), 1, fp);
// 		fread(&qt.precision, sizeof(uint8_t), 1, fp);
// 		fread(qt.data, sizeof(uint8_t), 64, fp);
// 	}

// 	// Channels
// 	uint64_t ch_count;
// 	fread(&ch_count, sizeof(uint64_t), 1, fp);
// 	header.channel_dcts.resize(ch_count);
// 	for (size_t ch_idx = 0; ch_idx < ch_count; ++ch_idx) {
// 		auto& ch = header.channel_dcts[ch_idx];
// 		fread(&ch.component_id, sizeof(uint8_t), 1, fp);
// 		fread(&ch.qtable_id, sizeof(uint8_t), 1, fp);
// 		fread(&ch.color_space_id, sizeof(uint8_t), 1, fp);
// 		fread(&ch.width_in_blocks, sizeof(uint32_t), 1, fp);
// 		fread(&ch.height_in_blocks, sizeof(uint32_t), 1, fp);
// 		ch.blocks.clear(); // blocks not stored in header file
// 	}

// 	fclose(fp);
// 	return true;
// }

bool JpegLoader::load_ImageHeader(ImageHeader& header, const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        perror("fopen (read)");
        return false;
    }

    // Sample factors
    uint64_t sf_count;
    fread(&sf_count, sizeof(uint64_t), 1, fp);
    header.sample_factors.resize(sf_count);
    for (size_t i = 0; i < sf_count; ++i) {
        uint8_t sf_val;
        fread(&sf_val, sizeof(uint8_t), 1, fp);
        if (sf_val > static_cast<uint8_t>(SampleFactor::SF_400)) {
            fclose(fp);
            throw std::runtime_error("Invalid SampleFactor value in header file: " +
                                     std::to_string(sf_val));
        }
        header.sample_factors[i] = static_cast<SampleFactor>(sf_val);
    }

    // Color spaces
    uint64_t cs_count;
    fread(&cs_count, sizeof(uint64_t), 1, fp);
    header.color_spaces.resize(cs_count);
    for (size_t i = 0; i < cs_count; ++i) {
        uint8_t cs_val;
        fread(&cs_val, sizeof(uint8_t), 1, fp);
        if (cs_val > static_cast<uint8_t>(ColorSpace::YCCK)) {
            fclose(fp);
            throw std::runtime_error("Invalid ColorSpace value in header file: " +
                                     std::to_string(cs_val));
        }
        header.color_spaces[i] = static_cast<ColorSpace>(cs_val);
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
        fread(&ch.qtable_id, sizeof(uint8_t), 1, fp);
        fread(&ch.width_in_blocks, sizeof(uint32_t), 1, fp);
        fread(&ch.height_in_blocks, sizeof(uint32_t), 1, fp);
        ch.blocks.clear(); // blocks 不存
    }

    // Images
    uint64_t img_count;
    fread(&img_count, sizeof(uint64_t), 1, fp);
    header.images.resize(img_count);
    for (size_t i = 0; i < img_count; ++i) {
        auto& img = header.images[i];
        fread(&img.width,  sizeof(uint32_t), 1, fp);
        fread(&img.height, sizeof(uint32_t), 1, fp);
        fread(&img.quality, sizeof(uint8_t), 1, fp);
        fread(&img.color_space_id, sizeof(uint8_t), 1, fp);
        fread(&img.sample_factor_id, sizeof(uint8_t), 1, fp);
        fread(&img.first_channel_index, sizeof(uint32_t), 1, fp);
        fread(&img.channel_count, sizeof(uint32_t), 1, fp);
    }

    fclose(fp);
    return true;
}


} // namespace fastlanes