#include "engine/operators/rowgroup.cuh"
#include "format/reader.cuh"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>

int main(const int argc, char** argv) {
	try {
		if (argc < 2 || argc > 3) {
			std::cerr << "Usage: " << argv[0] << " input.fls [repeats]\n";
			return 1;
		}
		const std::filesystem::path path = argv[1];
		const size_t repeats = argc == 3 ? std::stoull(argv[2]) : 7U;
		if (repeats == 0U) {
			throw std::invalid_argument("repeats must be greater than zero");
		}
		galp::format::FlsReader reader(path, /*load_column_names=*/false);
		for (size_t repeat = 0; repeat < repeats; ++repeat) {
			galp::format::ZeroCopyReadTiming aggregate {};
			const auto start = std::chrono::steady_clock::now();
			for (size_t rowgroup = 0; rowgroup < reader.rowgroup_count(); ++rowgroup) {
				galp::format::ZeroCopyReadTiming timing {};
				auto zero_copy = reader.read_rowgroup_zero_copy(rowgroup, &timing);
				auto materialized = reader.materialize_zero_copy_rowgroup(std::move(zero_copy));
				galp::execution::free_rowgroup(materialized);
				aggregate.storage_bytes += timing.storage_bytes;
				aggregate.pread_count += timing.pread_count;
				aggregate.pread_ms += timing.pread_ms;
				aggregate.zero_copy_view_setup_ms += timing.zero_copy_view_setup_ms;
			}
			const auto end = std::chrono::steady_clock::now();
			std::cout << "repeat=" << repeat
			          << " wall_ms=" << std::chrono::duration<double, std::milli>(end - start).count()
			          << " pread_ms=" << aggregate.pread_ms
			          << " view_setup_ms=" << aggregate.zero_copy_view_setup_ms
			          << " bytes=" << aggregate.storage_bytes
			          << " preads=" << aggregate.pread_count << '\n';
		}
		return 0;
	} catch (const std::exception& error) {
		std::cerr << "galp_jpeg_dct_rowgroup_read_bench: " << error.what() << '\n';
		return 2;
	}
}
