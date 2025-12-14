// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// examples/jpeg_decoding_dir_test.cpp
// ────────────────────────────────────────────────────────
#include "fastlanes.hpp"
#include "fls/connection.hpp"
#include "fls/jpeg/jpeg_loader.hpp"
#include "fls/printer/az_printer.hpp"
#include "fls/reader/table_reader.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>  // 新增：std::transform
#include <chrono>     // 新增：计时 to_rgb

using namespace fastlanes; // NOLINT
namespace fs = std::filesystem;

// Helper: compare an ImageRGB (loaded from libjpeg) with reconstructed RGB plane [3][H][W] of uint8_t
bool images_approx_equal(const ImageRGB& img1, const std::vector<std::vector<std::vector<uint8_t>>>& img2_rgb) {
	if (img2_rgb.size() != 3) {
		std::cerr << "Reconstructed image does not have 3 channels" << std::endl;
		return false;
	}

	const size_t H = img1.height;
	const size_t W = img1.width;

	if (img2_rgb[0].size() != H || img2_rgb[0][0].size() != W) {
		std::cerr << "Size mismatch! original: " << W << "x" << H
		          << " reconstructed: " << (img2_rgb[0].empty() ? 0 : img2_rgb[0][0].size()) << "x"
		          << img2_rgb[0].size() << std::endl;
		return false;
	}

	const unsigned char* orig = img1.data.data();

	bool match = true;
	for (size_t h = 0; h < H; ++h) {
		for (size_t w = 0; w < W; ++w) {
			unsigned char r_orig = orig[(h * W + w) * 3 + 0];
			unsigned char g_orig = orig[(h * W + w) * 3 + 1];
			unsigned char b_orig = orig[(h * W + w) * 3 + 2];

			unsigned char r_rec = img2_rgb[0][h][w];
			unsigned char g_rec = img2_rgb[1][h][w];
			unsigned char b_rec = img2_rgb[2][h][w];

			// 允许少量误差（IDCT 四舍五入或量化带来的微小差异）
			const int tol = 5;
			if (std::abs((int)r_orig - (int)r_rec) > tol || std::abs((int)g_orig - (int)g_rec) > tol ||
			    std::abs((int)b_orig - (int)b_rec) > tol) {
				// std::cout << "Mismatch at (" << h << "," << w << "): "
				//           << "(" << (int)r_orig << "," << (int)g_orig << "," << (int)b_orig << ") vs "
				//           << "(" << (int)r_rec << "," << (int)g_rec << "," << (int)b_rec << ")" << std::endl;
				match = false;
				// 继续查找其它 mismatch（不要立刻返回），以便获取调试信息
			}
		}
	}
	return match;
}

// Helper: case-insensitive extension check
bool is_jpeg_file(const std::filesystem::path& p) {
	static const std::set<std::string> jpeg_exts = {".jpg", ".jpeg", ".JPG", ".JPEG"};
	auto                               ext       = p.extension().string();
	std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
	return jpeg_exts.count(ext) > 0;
}

// image_dir = "/home/tangyuxin/FastLanes/data/ILSVRC/jpeg/96x96"

int main(int argc, char** argv) {
	if (argc < 4) {
		std::cerr << "Usage: ./jpeg_loader_demo_dir <jpeg_directory> <fls_output_dir> <is_gpu>" << std::endl;
		return 1;
	}

	std::string jpeg_dir = argv[1];
	fs::path    fls_dir(argv[2]);
	bool is_gpu = (std::string(argv[3]) == "1");

	if (!fs::exists(jpeg_dir) || !fs::is_directory(jpeg_dir)) {
		std::cerr << "JPEG directory does not exist: " << jpeg_dir << std::endl;
		return 1;
	}

	// 收集目录中 JPEG 文件路径（与 Connection::read_jpeg_dir 内部迭代顺序一致）
	std::vector<std::string> jpeg_paths;
	for (const auto& entry : fs::directory_iterator(jpeg_dir)) {
		if (!entry.is_regular_file())
			continue;
		const auto& p = entry.path();
		// 使用你工程里的 is_jpeg_file() 判定函数
		if (is_jpeg_file(p)) {
			jpeg_paths.push_back(p.string());
		}
	}

	if (jpeg_paths.empty()) {
		std::cerr << "No JPEG files found in directory: " << jpeg_dir << std::endl;
		return 1;
	}

	std::sort(jpeg_paths.begin(), jpeg_paths.end(), [](const std::string& a, const std::string& b) {
		const auto fa = fs::path(a).filename().string();
		const auto fb = fs::path(b).filename().string();
		if (fa != fb)
			return fa < fb;
		// 同名时用完整路径打破并列（极少见，但更稳妥）
		return a < b;
	});

	// 预先载入所有原始 JPEG（用于后续验证）
	std::vector<ImageRGB> originals;
	originals.reserve(jpeg_paths.size());
	for (const auto& jp : jpeg_paths) {
		try {
			// printf("Loading original JPEG: %s\n", jp.c_str());
			auto rgb = JpegLoader::load_rgb(jp);
			originals.push_back(std::move(rgb));
			// std::cout << "Loaded original: " << jp << " (" << originals.back().width << "x" <<
			// originals.back().height << ")" << std::endl;
		} catch (const std::exception& ex) {
			az_printer::bold_red_cout << "-- Warning: failed to load original JPEG '" << jp << "': " << ex.what()
			                          << std::endl;
			// 推进一个空占位（以保持索引对齐），但记录尺寸为0
			originals.emplace_back();
		}
	}

	// 确保输出目录存在
	try {
		if (!fs::exists(fls_dir)) {
			fs::create_directories(fls_dir);
		}
	} catch (const std::exception& ex) {
		az_printer::bold_red_cout << "-- Error creating fls output dir: " << ex.what() << std::endl;
		return EXIT_FAILURE;
	}

	// ─────────────── Step 1: Use Connection::read_jpeg_dir to read directory and write header.meta / image.fls
	// ───────────────
	try {
		auto con = connect(); // 或者 Connection con; 根据你的 API
		fs::path header_meta = fls_dir / "header.meta";

		// ====== 计时：写 header.meta（read_jpeg_dir） ======
		auto t_header_begin = std::chrono::high_resolution_clock::now();

		// read_jpeg_dir 会遍历目录、读取 headers、deduplicate 并把 unified header 写入 header.meta
		con->read_jpeg_dir(jpeg_dir, header_meta.string());

		auto t_header_end = std::chrono::high_resolution_clock::now();
		auto header_ms =
			std::chrono::duration_cast<std::chrono::milliseconds>(t_header_end - t_header_begin).count();

		std::cout << "✅ header written : " << header_meta << std::endl;
		std::cout << "⏱ read_jpeg_dir (write header.meta) elapsed: "
				<< header_ms << " ms";
		if (!jpeg_paths.empty()) {
			double per_img = static_cast<double>(header_ms) / static_cast<double>(jpeg_paths.size());
			std::cout << " (" << per_img << " ms/image)";
		}
		std::cout << std::endl;

		// ====== 计时：写 image.fls（to_fls） ======
		auto t_fls_begin = std::chrono::high_resolution_clock::now();

		// 将内存中的 DCT table 写到 image.fls
		con->to_fls((fls_dir / "image.fls").string());

		auto t_fls_end = std::chrono::high_resolution_clock::now();
		auto fls_ms =
			std::chrono::duration_cast<std::chrono::milliseconds>(t_fls_end - t_fls_begin).count();

		std::cout << "✅ Wrote unified header and FLS to: " << fls_dir << std::endl;
		std::cout << "⏱ to_fls (write image.fls) elapsed: "
				<< fls_ms << " ms";
		if (!jpeg_paths.empty()) {
			double per_img = static_cast<double>(fls_ms) / static_cast<double>(jpeg_paths.size());
			std::cout << " (" << per_img << " ms/image)";
		}
		std::cout << std::endl;

	} catch (const std::exception& ex) {
		az_printer::bold_red_cout << "-- Error during FLS write: " << ex.what() << std::endl;
		return EXIT_FAILURE;
	}


	// ─────────────── Step 2: Read FLS and reconstruct RGBs ───────────────
	try {
		Connection con2;
		// reset().read_fls(...) 根据你的 API 返回 TableReader / reader 对象
		const auto fls_reader = con2.reset().read_fls((fls_dir / "image.fls").string());

		// to_rgb 现在返回 std::vector<std::vector<std::vector<std::vector<uint8_t>>>> => [N][3][H][W]
		std::cout << "🔄 Reconstructing RGB from DCT blocks in FLS... is_gpu = "<< is_gpu << std::endl;

		// ############### 这里开始计时 to_rgb ###############
		auto t_begin = std::chrono::high_resolution_clock::now();

		auto reconstructed_collection = fls_reader->to_rgb((fls_dir / "header.meta").c_str(), is_gpu);

		auto t_end     = std::chrono::high_resolution_clock::now();
		auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_begin).count();

		std::cout << "⏱ to_rgb elapsed time: " << elapsedMs << " ms" << std::endl;
		if (!reconstructed_collection.empty()) {
			double per_img = static_cast<double>(elapsedMs) / static_cast<double>(reconstructed_collection.size());
			std::cout << "⏱ to_rgb per image: " << per_img << " ms/image" << std::endl;
		}
		// ############### 计时结束 ###############

		// auto reconstructed_collection = fls_reader->to_rgb((fls_dir / "header.meta").c_str(),is_gpu);
		// if (reconstructed_collection.empty()) {
		// 	throw std::runtime_error("to_rgb returned empty collection");
		// }

		const size_t N_rec = reconstructed_collection.size();
		std::cout << "✅ Reconstructed " << N_rec << " images from FLS." << std::endl;

		// 遍历每张重建的图像，输出形状和首像素，并与原图（若已加载）比较
		for (size_t idx = 0; idx < N_rec; ++idx) {
			const auto& rec = reconstructed_collection[idx]; // [3][H][W]
			if (rec.size() != 3) {
				std::cerr << "Reconstructed image " << idx << " does not have 3 channels, skipping." << std::endl;
				continue;
			}
			// const size_t H = rec[0].size();
			// const size_t W = (H > 0) ? rec[0][0].size() : 0;

			// std::cout << "Image[" << idx << "] shape: [3][" << H << "][" << W << "]" << std::endl;
			// if (H > 0 && W > 0) {
			// 	std::cout << "   First pixel: (" << static_cast<int>(rec[0][0][0]) << ", "
			// 	          << static_cast<int>(rec[1][0][0]) << ", " << static_cast<int>(rec[2][0][0]) << ")"
			// 	          << std::endl;
			// }

			// 如果我们事先成功加载了原图并且索引存在，则进行近似比较
			if (idx < originals.size() && originals[idx].width > 0 && originals[idx].height > 0) {
				try {
					bool ok = images_approx_equal(originals[idx], rec);
					if (ok) {
						// std::cout << "   ✅ Reconstruction matches original (within tolerance)." << std::endl;
					} else {
						// std::cout << "   ⚠️  Reconstruction differs from original." << std::endl;
					}
				} catch (const std::exception& ex) {
					az_printer::bold_red_cout << "-- Error comparing images for index " << idx << ": " << ex.what()
					                          << std::endl;
				}
			} else {
				std::cout << "   (No original available to compare for this index)" << std::endl;
			}
		}

	} catch (const std::exception& ex) {
		az_printer::bold_red_cout << "-- Error during FLS read/reconstruction: " << ex.what() << std::endl;
		return EXIT_FAILURE;
	}

	return 0;
}
