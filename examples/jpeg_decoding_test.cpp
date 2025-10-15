// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// examples/jpeg_decoding_test.cpp
// ────────────────────────────────────────────────────────
#include "fastlanes.hpp"
#include "fls/connection.hpp"
#include "fls/jpeg/jpeg_loader.hpp"
#include "fls/printer/az_printer.hpp"
#include "fls/reader/table_reader.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>

using namespace fastlanes; // NOLINT
namespace fs = std::filesystem;

// Helper: compare two RGB images (allow small error due to IDCT rounding)
bool images_approx_equal(const ImageRGB& img1, const std::vector<std::vector<std::vector<uint8_t>>>& img2_rgb) {
	bool match = true;
	if (img1.width != img2_rgb[0][0].size() || img1.height != img2_rgb[0].size()) {
		std::cerr << "Size mismatch!" << std::endl;
		return false;
	}

	const size_t         H    = img1.height;
	const size_t         W    = img1.width;
	const unsigned char* orig = img1.data.data();

	for (size_t h = 0; h < H; ++h) {
		for (size_t w = 0; w < W; ++w) {
			// Original RGB
			unsigned char r_orig = orig[(h * W + w) * 3 + 0];
			unsigned char g_orig = orig[(h * W + w) * 3 + 1];
			unsigned char b_orig = orig[(h * W + w) * 3 + 2];

			// Reconstructed RGB (from to_rgb)
			double r_rec = img2_rgb[0][h][w]; // R
			double g_rec = img2_rgb[1][h][w]; // G
			double b_rec = img2_rgb[2][h][w]; // B

			// Allow small error (IDCT is lossy if quantized, but if quant=1, should be near lossless)
			if (std::abs(r_orig - r_rec) > 5.0 || std::abs(g_orig - g_rec) > 5.0 || std::abs(b_orig - b_rec) > 5.0) {
				std::cout << "Mismatch at (" << h << "," << w << "): "
				          << "(" << (int)r_orig << "," << (int)g_orig << "," << (int)b_orig << ") vs "
				          << "(" << r_rec << "," << g_rec << "," << b_rec << ")" << std::endl;
				match = false;
			}
		}
	}
	return match;
}

int main(int argc, char** argv) {
	if (argc < 3) {
		std::cerr << "Usage: ./jpeg_loader_demo <image.jpg> <fls_output_dir>" << std::endl;
		return 1;
	}

	std::string jpeg_path = argv[1];
	fs::path    fls_dir(argv[2]);

	// ─────────────── Step 1: Load original RGB ───────────────
	auto original_rgb = JpegLoader::load_rgb(jpeg_path);
	std::cout << "✅ Loaded original JPEG: " << original_rgb.width << "x" << original_rgb.height << std::endl;
	std::cout << "   First pixel RGB: " << (int)original_rgb.data[0] << ", " << (int)original_rgb.data[1] << ", "
	          << (int)original_rgb.data[2] << std::endl;

	// ─────────────── Step 2: Encode to FLS (DCT) ───────────────
	try {
		auto con = connect();
		if (!fs::exists(fls_dir)) {
			fs::create_directories(fls_dir);
		}

		con->read_jpeg(jpeg_path, fls_dir / "header.meta");
		con->to_fls(fls_dir / "image.fls");
		std::cout << "✅ Wrote FLS files to: " << fls_dir << std::endl;
	} catch (std::exception& ex) {
		az_printer::bold_red_cout << "-- Error during FLS write: " << ex.what() << std::endl;
		return EXIT_FAILURE;
	}

	// ─────────────── Step 3: Read FLS and reconstruct RGB ───────────────
	try {

		Connection con2;
		const auto fls_reader = con2.reset().read_fls(fls_dir / "image.fls");

		// This calls TableReader::to_rgb(), which uses header.meta to reconstruct
		auto reconstructed_rgb_collection = fls_reader->to_rgb((fls_dir / "header.meta").c_str()); // returns [3][H][W]
		auto reconstructed_rgb            = reconstructed_rgb_collection[0];

		std::cout << "✅ Reconstructed RGB from DCT blocks. Shape: [3][" << reconstructed_rgb[0].size() << "]["
		          << reconstructed_rgb[0][0].size() << "]" << std::endl;

		// Print first pixel of reconstructed image
		double r = reconstructed_rgb[0][0][0];
		double g = reconstructed_rgb[1][0][0];
		double b = reconstructed_rgb[2][0][0];
		std::cout << "   Reconstructed first pixel: (" << r << ", " << g << ", " << b << ")" << std::endl;

		// ─────────────── Step 4: Validate correctness ───────────────
		if (images_approx_equal(original_rgb, reconstructed_rgb)) {
			std::cout << "✅ Reconstruction matches original (within tolerance)!" << std::endl;
		} else {
			std::cout << "⚠️  Reconstruction differs from original." << std::endl;
		}

	} catch (std::exception& ex) {
		az_printer::bold_red_cout << "-- Error during FLS read/reconstruction: " << ex.what() << std::endl;
		return EXIT_FAILURE;
	}

	return 0;
}