// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// examples/jpeg_gpu_load_rgb_test.cpp
// ────────────────────────────────────────────────────────
#include "fastlanes.hpp"
#include "fls/connection.hpp"
#include "fls/jpeg/jpeg_loader.hpp"
#include "fls/printer/az_printer.hpp"
#include "fls/reader/table_reader.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <vector>
#include <cctype>
#include <cstdio>

extern "C" {
#include <jpeglib.h>   // 使用 libjpeg 读取 JPEG 头信息
}

using namespace fastlanes; // NOLINT
namespace fs = std::filesystem;
using Clock  = std::chrono::high_resolution_clock;

// Helper: compare two images that may be RGB(3ch) or GRAY(1ch)
// For GRAY we treat it as RGB with r=g=b=gray.
bool images_approx_equal(const ImageRGB& img1,
                         const ImageRGB& img2,
                         double tol = 5.0)
{
    // 1) 尺寸检查
    if (img1.width != img2.width || img1.height != img2.height) {
        std::cerr << "Size mismatch: (" << img1.width << "x" << img1.height
                  << ") vs (" << img2.width << "x" << img2.height << ")\n";
        return false;
    }

    const size_t W = img1.width;
    const size_t H = img1.height;
    const size_t pixels = W * H;

    const size_t size1 = img1.data.size();
    const size_t size2 = img2.data.size();

    // 2) 推断通道数
    auto deduce_channels = [&](size_t sz, const char* name) -> int {
        if (sz == pixels) {
            return 1; // 灰度
        }
        if (sz == pixels * 3) {
            return 3; // RGB
        }
        std::cerr << "[ERROR] " << name << " buffer size doesn't match 1ch or 3ch layout!\n"
                  << "        size       = " << sz << "\n"
                  << "        width*height = " << pixels << "\n"
                  << "        expected sz == pixels or sz == pixels*3\n";
        return 0; // 表示异常
    };

    const int ch1 = deduce_channels(size1, "img1");
    const int ch2 = deduce_channels(size2, "img2");
    if (ch1 == 0 || ch2 == 0) {
        return false; // 数据布局异常，避免越界
    }

    const unsigned char* data1 = img1.data.data();
    const unsigned char* data2 = img2.data.data();

    // 3) 一个小工具：给定像素索引，取“逻辑上的 RGB”
    auto get_rgb = [](const unsigned char* data, int ch, size_t pixel_idx,
                      int& r, int& g, int& b) {
        if (ch == 3) {
            size_t base = pixel_idx * 3;
            r = static_cast<int>(data[base + 0]);
            g = static_cast<int>(data[base + 1]);
            b = static_cast<int>(data[base + 2]);
        } else { // ch == 1，灰度
            unsigned char v = data[pixel_idx];
            r = g = b = static_cast<int>(v);
        }
    };

    bool   match          = true;
    size_t mismatch_count = 0;
    int    max_diff       = 0;

    // 最多打印多少个 mismatch；设成 0 表示不打印具体像素
    // constexpr size_t MAX_MISMATCH_PRINT = 20;

    for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
            const size_t p = h * W + w;

            int r1, g1, b1;
            int r2, g2, b2;
            get_rgb(data1, ch1, p, r1, g1, b1);
            get_rgb(data2, ch2, p, r2, g2, b2);

            const int dr = std::abs(r1 - r2);
            const int dg = std::abs(g1 - g2);
            const int db = std::abs(b1 - b2);
            const int diff = std::max({dr, dg, db});

            if (diff > max_diff) {
                max_diff = diff;
            }

            if (dr > tol || dg > tol || db > tol) {
                match = false;
                ++mismatch_count;

                // if (mismatch_count <= MAX_MISMATCH_PRINT) {
                //     std::cout << "Mismatch at (" << h << "," << w << "): "
                //               << "(" << r1 << "," << g1 << "," << b1 << ") vs "
                //               << "(" << r2 << "," << g2 << "," << b2 << ")\n";
                // }
            }
        }
    }

    if (!match) {
        std::cout << "Total mismatches: " << mismatch_count
                  << " (max channel diff = " << max_diff << ")\n";
    }

    return match;
}


double compute_mse_rgb(const ImageRGB& img1,
                         const ImageRGB& img2)
{
    // 1) 尺寸检查
    if (img1.width != img2.width || img1.height != img2.height) {
        std::cerr << "Size mismatch: (" << img1.width << "x" << img1.height
                  << ") vs (" << img2.width << "x" << img2.height << ")\n";
        return false;
    }

    const size_t W = img1.width;
    const size_t H = img1.height;
    const size_t pixels = W * H;

    const size_t size1 = img1.data.size();
    const size_t size2 = img2.data.size();

    // 2) 推断通道数
    auto deduce_channels = [&](size_t sz, const char* name) -> int {
        if (sz == pixels) {
            return 1; // 灰度
        }
        if (sz == pixels * 3) {
            return 3; // RGB
        }
        std::cerr << "[ERROR] " << name << " buffer size doesn't match 1ch or 3ch layout!\n"
                  << "        size       = " << sz << "\n"
                  << "        width*height = " << pixels << "\n"
                  << "        expected sz == pixels or sz == pixels*3\n";
        return 0; // 表示异常
    };

    const int ch1 = deduce_channels(size1, "img1");
    const int ch2 = deduce_channels(size2, "img2");
    if (ch1 == 0 || ch2 == 0) {
        return false; // 数据布局异常，避免越界
    }

    const unsigned char* data1 = img1.data.data();
    const unsigned char* data2 = img2.data.data();

    // 3) 一个小工具：给定像素索引，取“逻辑上的 RGB”
    auto get_rgb = [](const unsigned char* data, int ch, size_t pixel_idx,
                      int& r, int& g, int& b) {
        if (ch == 3) {
            size_t base = pixel_idx * 3;
            r = static_cast<int>(data[base + 0]);
            g = static_cast<int>(data[base + 1]);
            b = static_cast<int>(data[base + 2]);
        } else { // ch == 1，灰度
            unsigned char v = data[pixel_idx];
            r = g = b = static_cast<int>(v);
        }
    };

    double sum_sq = 0.0;
    for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
            const size_t p = h * W + w;
            int r1, g1, b1;
            int r2, g2, b2;
            get_rgb(data1, ch1, p, r1, g1, b1);
            get_rgb(data2, ch2, p, r2, g2, b2);
            const int dr = std::abs(r1 - r2);
            sum_sq += double(dr * dr);
            if(ch1 == 3){
                const int dg = std::abs(g1 - g2);
                sum_sq += double(dg * dg);
                const int db = std::abs(b1 - b2);
                sum_sq += double(db * db);
            }
        }
    }
    return sum_sq / double(pixels * 3);
}

double compute_psnr_rgb(const ImageRGB& a, const ImageRGB& b) {
    double mse = compute_mse_rgb(a, b);
    if (mse == 0.0) return std::numeric_limits<double>::infinity();
    const double maxI = 255.0;
    return 10.0 * std::log10((maxI * maxI) / mse);
}

// Helper: 是否是 JPEG 文件
bool is_jpeg_file(const fs::directory_entry& entry) {
    if (!entry.is_regular_file()) {
        return false;
    }
    auto ext = entry.path().extension().string();
    // 简单大小写处理
    for (auto& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return (ext == ".jpg" || ext == ".jpeg");
}

// Helper: 收集目录中所有 JPEG 文件路径（按文件名排序）
std::vector<fs::path> collect_jpeg_files(const std::string& dir_path) {
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (is_jpeg_file(entry)) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

const char* jpeg_color_space_name(J_COLOR_SPACE cs) {
    switch (cs) {
        case JCS_UNKNOWN:   return "JCS_UNKNOWN";
        case JCS_GRAYSCALE: return "JCS_GRAYSCALE";
        case JCS_RGB:       return "JCS_RGB";
        case JCS_YCbCr:     return "JCS_YCbCr";
        case JCS_CMYK:      return "JCS_CMYK";
        case JCS_YCCK:      return "JCS_YCCK";
        default:            return "JCS_???";
    }
}

// 使用 libjpeg 打印目录中每个 JPEG 的：宽、高、分量数、采样、restart interval、interleaved、internal color space
void print_jpeg_dir_parameters(const std::string& dir_path) {
    auto jpeg_files = collect_jpeg_files(dir_path);
    if (jpeg_files.empty()) {
        std::cout << "\n[Info] No JPEG files to inspect in directory: " << dir_path << "\n";
        return;
    }

    std::cout << "\n=== JPEG Header Info in Directory (libjpeg) ===\n";
    std::cout << "Directory: " << dir_path << "\n";

    for (const auto& p : jpeg_files) {
        std::cout << "--------------------------------------------------\n";
        std::cout << p.filename().string() << "\n";

        FILE* infile = std::fopen(p.string().c_str(), "rb");
        if (!infile) {
            std::cout << "  [Error] Failed to open file.\n";
            continue;
        }

        // 标准 libjpeg 解码结构体
        jpeg_decompress_struct cinfo;
        jpeg_error_mgr jerr;

        cinfo.err = jpeg_std_error(&jerr);
        jpeg_create_decompress(&cinfo);
        jpeg_stdio_src(&cinfo, infile);

        // 只读 header，不真正解压图像数据
        int rc = jpeg_read_header(&cinfo, TRUE);
        if (rc != JPEG_HEADER_OK) {
            std::cout << "  [Error] jpeg_read_header failed, rc = " << rc << "\n";
            jpeg_destroy_decompress(&cinfo);
            std::fclose(infile);
            continue;
        }

        unsigned int width  = cinfo.image_width;
        unsigned int height = cinfo.image_height;
        int comp_count      = cinfo.num_components;

        // 采样因子
        int max_h = 0, max_v = 0;
        std::vector<std::pair<int,int>> sampling;
        sampling.reserve(static_cast<size_t>(comp_count));
        for (int i = 0; i < comp_count; ++i) {
            int h = cinfo.comp_info[i].h_samp_factor;
            int v = cinfo.comp_info[i].v_samp_factor;
            sampling.emplace_back(h, v);
            if (h > max_h) max_h = h;
            if (v > max_v) max_v = v;
        }

        // restart interval：RST 标记之间的 MCU 数
        unsigned int restart_interval = cinfo.restart_interval;

        // “interleaved” 粗略判断：多分量就算 interleaved
        bool interleaved = (comp_count > 1);

        // internal color space 对应 libjpeg 的 jpeg_color_space
        J_COLOR_SPACE cs = cinfo.jpeg_color_space;

        std::cout << "  size (WxH)      : " << width << " x " << height << "\n";
        std::cout << "  components      : " << comp_count << "\n";

        std::cout << "  sampling        : ";
        for (size_t i = 0; i < comp_count; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << "C" << i << "="
                      << sampling[i].first << "x" << sampling[i].second;
        }
        std::cout << "\n";

        std::cout << "  restart interval: " << restart_interval << "\n";
        std::cout << "  interleaved     : " << (interleaved ? "yes" : "no") << "\n";
        std::cout << "  internal CS     : " << jpeg_color_space_name(cs)
                  << " (" << static_cast<int>(cs) << ")\n";

        jpeg_destroy_decompress(&cinfo);
        std::fclose(infile);
    }

    std::cout << "--------------------------------------------------\n\n";
}

// // 将 ImageRGB 保存为 PPM(P6) 文件，方便肉眼对比
// void save_imagergb_as_ppm(const ImageRGB& img, const std::string& filename)
// {
//     const size_t W = img.width;
//     const size_t H = img.height;
//     const size_t pixels = W * H;
//     const size_t sz = img.data.size();

//     if (W == 0 || H == 0 || sz == 0) {
//         std::cerr << "[save_imagergb_as_ppm] Empty image, skip: " << filename << "\n";
//         return;
//     }

//     // 推断通道数：1ch (灰度) or 3ch (RGB)
//     int channels = 0;
//     if (sz == pixels) {
//         channels = 1; // 灰度
//     } else if (sz == pixels * 3) {
//         channels = 3; // RGB
//     } else {
//         std::cerr << "[save_imagergb_as_ppm] Unexpected buffer size for "
//                   << filename << ": data.size=" << sz
//                   << ", width*height=" << pixels << "\n";
//         return;
//     }

//     std::ofstream ofs(filename, std::ios::binary);
//     if (!ofs) {
//         std::cerr << "[save_imagergb_as_ppm] Failed to open file for write: "
//                   << filename << "\n";
//         return;
//     }

//     // PPM P6 头
//     ofs << "P6\n" << W << " " << H << "\n255\n";

//     const unsigned char* data = img.data.data();

//     if (channels == 3) {
//         // 直接写 RGBRGB...
//         ofs.write(reinterpret_cast<const char*>(data),
//                   static_cast<std::streamsize>(sz));
//     } else {
//         // 灰度 → 扩展成 RGB (r=g=b)
//         std::vector<unsigned char> tmp(pixels * 3);
//         for (size_t i = 0; i < pixels; ++i) {
//             unsigned char v = data[i];
//             tmp[3 * i + 0] = v;
//             tmp[3 * i + 1] = v;
//             tmp[3 * i + 2] = v;
//         }
//         ofs.write(reinterpret_cast<const char*>(tmp.data()),
//                   static_cast<std::streamsize>(tmp.size()));
//     }

//     ofs.close();
//     std::cout << "[save_imagergb_as_ppm] Saved image to " << filename << "\n";
// }


int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: ./jpeg_gpu_load_rgb_test <image.jpg> <worker_num> <queue_num> [image_dir]" << std::endl;
        return 1;
    }

    std::string jpeg_path = argv[1];
    int         worker_num = std::atoi(argv[2]);
    size_t      queue_num  = static_cast<size_t>(std::stoul(argv[3]));
    
    printf("Worker num: %d, Queue num: %zu\n", worker_num, queue_num);
    // 若未提供 image_dir，则默认用 image.jpg 的父目录
    std::string dir_path;
    if (argc == 3) {
        dir_path = argv[2];
    } else {
        dir_path = fs::path(jpeg_path).parent_path().string();
        if (dir_path.empty()) {
            dir_path = "."; // 当前目录
        }
    }

    std::cout << "Single image path: " << jpeg_path << std::endl;
    std::cout << "Directory path   : " << dir_path << std::endl;

    // ─────────────── 先打印目录中 JPEG 的头部参数信息（libjpeg） ───────────────
    // print_jpeg_dir_parameters(dir_path);

    // ─────────────── Step 1: Load original RGB (CPU) ───────────────
    auto t1 = Clock::now();
    auto original_rgb = JpegLoader::load_rgb(jpeg_path);
    auto t2 = Clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    std::cout << "\n=== Single Image Test ===" << std::endl;
    std::cout << "  CPU decode time: " << cpu_ms << " ms" << std::endl;

    // ─────────────── Step 2: Load RGB using GPU version ───────────────
    auto t3 = Clock::now();
    auto gpujpeg_rgb = JpegLoader::load_rgb_gpu(jpeg_path);
    auto t4 = Clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();

    std::cout << " GPU decode time: " << gpu_ms << " ms" << std::endl;

    // ─────────────── Step 3: Validate correctness (single image) ───────────────

    // fs::path pd = dir_path;
    // std::string stem = pd.stem().string();   // 比如 9_59854
    // std::string cpu_out = stem + "_cpu.ppm";
    // std::string gpu_out = stem + "_gpu.ppm";

    // save_imagergb_as_ppm(original_rgb, cpu_out);
    // save_imagergb_as_ppm(gpujpeg_rgb, gpu_out);

    if (!images_approx_equal(original_rgb, gpujpeg_rgb)) {
        std::cout << "  Reconstruction differs from original." << std::endl;
    }

    std::cout << "\n Single Image Summary:" << std::endl;
    std::cout << "   CPU decode: " << cpu_ms << " ms" << std::endl;
    std::cout << "   GPU decode: " << gpu_ms << " ms" << std::endl;
    std::cout << "   Speedup:    " << (cpu_ms / gpu_ms) << "× faster (approx.)" << std::endl;

    // ─────────────── Step 4: Directory GPU batch test ───────────────
    std::cout << "\n=== Directory Batch Test ===" << std::endl;
    // 4-1. 收集目录中的 JPEG 文件
    auto jpeg_files = collect_jpeg_files(dir_path);
    if (jpeg_files.empty()) {
        std::cerr << "No JPEG files found in directory: " << dir_path << std::endl;
        return 0; 
    }

    std::cout << "Found " << jpeg_files.size() << " JPEG file(s) in directory." << std::endl;

    // 4-2. CPU：逐张载入
    auto t5 = Clock::now();
    std::vector<ImageRGB> cpu_images;
    cpu_images.reserve(jpeg_files.size());
    for (const auto& p : jpeg_files) {
        auto img = JpegLoader::load_rgb(p.string());
        cpu_images.push_back(std::move(img));
    }
    auto t6 = Clock::now();
    double cpu_dir_ms = std::chrono::duration<double, std::milli>(t6 - t5).count();

    // 4-3. GPU：批量载入
    auto t7 = Clock::now();
    std::vector<ImageRGB> gpu_images = JpegLoader::load_rgb_dir_gpu_mt(dir_path,worker_num,queue_num);
    auto t8 = Clock::now();
    double gpu_dir_ms = std::chrono::duration<double, std::milli>(t8 - t7).count();

    if (cpu_images.size() != gpu_images.size()) {
        std::cerr << " Image count mismatch between CPU("<<cpu_images.size()<<") and GPU("<<gpu_images.size()<<") batch load!" << std::endl;
    }

    // 4-4. 按顺序逐张比较（假设 load_rgb_dir_gpu 使用相同排序）
    size_t n = std::min(cpu_images.size(), gpu_images.size());
    bool all_match = true;

    for (size_t i = 0; i < n; ++i) {
        const auto& cpu_img = cpu_images[i];
        const auto& gpu_img = gpu_images[i];

        if (!images_approx_equal(cpu_img, gpu_img)) {
            std::cout << " Image " << i << "("<<jpeg_files[i]<<")" << " mismatch!" << std::endl;
            double mse  = compute_mse_rgb(cpu_img, gpu_img);
            double psnr = compute_psnr_rgb(cpu_img, gpu_img);
            std::cout << "MSE = " << mse << ", PSNR = " << psnr << " dB\n";

            // 例如：PSNR >= 40 dB 就认为 OK
            if (psnr < 40.0) {
                std::cout << " PSNR too low, potential issue.\n";
            }
            all_match = false;
        } 
    }

    std::cout << "\n Directory Batch Summary:" << std::endl;
    std::cout << "   CPU batch decode: " << cpu_dir_ms << " ms for " << cpu_images.size() << " image(s)" << std::endl;
    std::cout << "   GPU batch decode: " << gpu_dir_ms << " ms for " << gpu_images.size() << " image(s)" << std::endl;
    if (gpu_dir_ms > 0.0) {
        std::cout << "   Speedup:          " << (cpu_dir_ms / gpu_dir_ms) << "× faster (approx.)" << std::endl;
    }

    if (all_match && cpu_images.size() == gpu_images.size()) {
        std::cout << "\n load_rgb_dir_gpu() batch reconstruction matches CPU (within tolerance)!" << std::endl;
    } else {
        std::cout << "\n  load_rgb_dir_gpu_mt() batch reconstruction has differences." << std::endl;
    }

    return 0;
}
