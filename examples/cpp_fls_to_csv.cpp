// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// examples/cpp_fls_to_csv.cpp
// ────────────────────────────────────────────────────────
#include "fls/connection.hpp"
#include <filesystem>
#include <iostream>
#include <string_view>

namespace {

void print_usage(const char* prog) {
	std::cerr << "Usage:\n"
	          << "  " << prog << " <input.fls> <output.csv>\n";
}

} // namespace

int main(int argc, char** argv) {
	if (argc != 3) {
		print_usage(argv[0]);
		return 1;
	}

	const std::filesystem::path input(argv[1]);
	const std::filesystem::path output(argv[2]);
	if (!std::filesystem::exists(input)) {
		std::cerr << "Input file does not exist: " << input << "\n";
		return 2;
	}

	try {
		fastlanes::Connection conn;
		auto                  reader = conn.reset().read_fls(input);
		reader->to_csv(output);
		std::cout << "Wrote " << output << "\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "Error: " << ex.what() << "\n";
		return 3;
	}
}
