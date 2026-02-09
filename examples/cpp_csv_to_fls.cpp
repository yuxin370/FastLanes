// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// examples/cpp_csv_to_fls.cpp
// ────────────────────────────────────────────────────────
#include "fastlanes.hpp"
#include "fls/connection.hpp"
#include <filesystem>
#include <iostream>
#include <string_view>

using namespace fastlanes; // NOLINT

namespace {

void print_usage(const char* prog) {
	std::cerr << "Usage:\n"
	          << "  " << prog << " <input.csv> <output.fls> [--overwrite]\n"
	          << "\n"
	          << "Notes:\n"
	          << "  - The input CSV directory must contain schema.json\n";
}

} // namespace

int main(int argc, char** argv) {
	try {
		if (argc < 3) {
			print_usage(argv[0]);
			return EXIT_FAILURE;
		}
		printf(" -   ----------------- - \n");
		const std::filesystem::path input_path = argv[1];
		const std::filesystem::path output_fls = argv[2];
		const bool                  overwrite  = (argc >= 4 && std::string_view(argv[3]) == "--overwrite");

		if (!std::filesystem::exists(input_path)) {
			throw std::runtime_error("input path does not exist: " + input_path.string());
		}

		std::filesystem::path csv_dir = input_path;
		if (std::filesystem::is_regular_file(input_path)) {
			csv_dir = input_path.parent_path();
		}

		const auto schema_path = csv_dir / SCHEMA_FILE_NAME;
		if (!std::filesystem::exists(schema_path)) {
			throw std::runtime_error("schema.json not found in directory: " + csv_dir.string());
		}

		if (std::filesystem::exists(output_fls)) {
			if (!overwrite) {
				throw std::runtime_error("output fls already exists (use --overwrite): " + output_fls.string());
			}
			std::error_code ec;
			std::filesystem::remove(output_fls, ec);
			if (ec) {
				throw std::runtime_error("failed to remove existing fls: " + ec.message());
			}
		}

		if (!output_fls.parent_path().empty()) {
			std::error_code ec;
			std::filesystem::create_directories(output_fls.parent_path(), ec);
			if (ec) {
				throw std::runtime_error("failed to create output directory: " + ec.message());
			}
		}

		const auto con = connect();
		std::cout << "-- FastLanes Version: " << con->get_version() << "\n";
		std::cout << "-- Reading CSV from directory: " << csv_dir << "\n";
		con->read_csv(csv_dir);
		std::cout << "-- Writing FLS to: " << output_fls << "\n";
		con->to_fls(output_fls);
		std::cout << "-- Done.\n";
		return EXIT_SUCCESS;
	} catch (const std::exception& ex) {
		std::cerr << "-- Error: " << ex.what() << "\n";
		return EXIT_FAILURE;
	}
}
