// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// examples/jpeg_loader_example.cpp
// ────────────────────────────────────────────────────────
#include "fastlanes.hpp"
#include "fls/connection.hpp"
#include "fls/jpeg/jpeg_loader.hpp"
#include "fls/printer/az_printer.hpp"
#include <filesystem>
#include <iostream>

using namespace fastlanes; // NOLINT
namespace fs = std::filesystem;

int main(int argc, char** argv) {
	if (argc < 3) {
		std::cerr << "Usage: ./jpeg_loader_demo <image.jpg> <fls_path>" << std::endl;
		return 1;
	}

	// example 1: load jpeg and decompress it to RGB
	auto image_rgb = JpegLoader::load_rgb(argv[1]);

	std::cout << "Loaded JPEG image: " << image_rgb.width << "x" << image_rgb.height << std::endl;
	std::cout << "First pixel RGB: " << (int)image_rgb.data[0] << ", " << (int)image_rgb.data[1] << ", "
	          << (int)image_rgb.data[2] << std::endl;

	// example 2: load jpeg and write it into fls
	auto image_header = JpegLoader::load_header(argv[1]);
	std::cout << "\n[DCT] Image size: " << image_header.width << "x" << image_header.height << std::endl;

	auto ProcessedDCT = JpegLoader::process_channel(image_header);

	try {
		// auto       con1             = connect(); //DC
		// auto       con2             = connect(); //AC
		// auto       con3             = connect(); //mix_run_nonzero_values
		// auto       con4             = connect(); //mix_run_pattern
		// fs::path fls_file_base_path(argv[2]);

		// if(!fs::exists(fls_file_base_path)){
		//     fs::create_directories(fls_file_base_path);
		// }

		// // Step 1: Read the CSV file from the specified directory path
		// con1->set_n_vectors_per_rowgroup(64).read_dct(ProcessedDCT,1); // DC
		// con2->set_n_vectors_per_rowgroup(64).read_dct(ProcessedDCT,2); // AC
		// con3->set_n_vectors_per_rowgroup(64).read_dct(ProcessedDCT,3); // mix_run_nonzero_values
		// con4->set_n_vectors_per_rowgroup(64).read_dct(ProcessedDCT,4); // mix_run_pattern

		// // Step 2: Write the data to the FastLanes file format in the specified directory
		// con1->to_fls(fls_file_base_path / "DC.fls");
		// con2->to_fls(fls_file_base_path / "AC.fls");
		// con3->to_fls(fls_file_base_path / "mix_run_nonzero_values.fls");
		// con4->to_fls(fls_file_base_path / "mix_run_pattern.fls");

		auto     con1 = connect(); // DC
		fs::path fls_file_base_path(argv[2]);

		if (!fs::exists(fls_file_base_path)) {
			fs::create_directories(fls_file_base_path);
		}

		// Step 1: Read the CSV file from the specified directory path
		con1->set_n_vectors_per_rowgroup(64).read_dct(ProcessedDCT, 5); // DC

		// Step 2: Write the data to the FastLanes file format in the specified directory
		con1->to_fls(fls_file_base_path / "image.fls");

		exit(EXIT_SUCCESS);
	} catch (std::exception& ex) {
		az_printer::bold_red_cout << "-- Error: " << ex.what() << std::endl;
		return EXIT_FAILURE;
	}

	if (false) {
		// Print quantization tables
		for (const auto& qt : image_header.quant_tables) {
			std::cout << "[DCT] Quant Table ID: " << (int)qt.id
			          << " (precision: " << (qt.precision == 0 ? "8-bit" : "16-bit") << ")\n[DCT] Zigzag: ";
			for (int i = 0; i < 64; ++i) {
				std::cout << (int)qt.data[i] << " ";
			}
			std::cout << "\n";
		}

		// Print DCT info of each channel
		for (const auto& channel : image_header.channel_dcts) {
			std::cout << "\n[DCT] Component ID: " << (int)channel.component_id << " with " << channel.width_in_blocks
			          << "x" << channel.height_in_blocks << " blocks\n";

			// Print first 1 or 2 blocks
			size_t print_blocks = std::min<size_t>(2, channel.blocks.size());
			for (size_t i = 0; i < print_blocks; ++i) {
				std::cout << "[DCT] Block " << i << " coeffs: ";
				for (int j = 0; j < 64; ++j) {
					std::cout << channel.blocks[i].data[j] << " ";
				}
				std::cout << "\n";
			}
		}
	}
	return 0;
}
