// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// examples/jpeg_loader_example.cpp
// ────────────────────────────────────────────────────────
#include "fastlanes.hpp"
#include "fls/connection.hpp"
#include "fls/jpeg/jpeg_loader.hpp"
#include "fls/printer/az_printer.hpp"

#include <iostream>
#include <filesystem>

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
    std::cout << "First pixel RGB: "
              << (int)image_rgb.data[0] << ", "
              << (int)image_rgb.data[1] << ", "
              << (int)image_rgb.data[2] << std::endl;


   // example 2: load jpeg and write it into fls
   std::string jpeg_path = argv[1];

	try {
		auto       con1             = connect(); //DC
        fs::path fls_file_base_path(argv[2]);

        if(!fs::exists(fls_file_base_path)){
            fs::create_directories(fls_file_base_path);
        }

		// Step 1: Read the CSV file from the specified directory path
		con1->read_jpeg(jpeg_path); // DC

		// Step 2: Write the data to the FastLanes file format in the specified directory
		con1->to_fls(fls_file_base_path / "image.fls");

		exit(EXIT_SUCCESS);
	} catch (std::exception& ex) {
		az_printer::bold_red_cout << "-- Error: " << ex.what() << std::endl;
		return EXIT_FAILURE;
	}
    
    return 0;
}
