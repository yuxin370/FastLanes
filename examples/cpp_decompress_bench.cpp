// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// examples/cpp_decompress_bench.cpp
// ────────────────────────────────────────────────────────
#include "fls/connection.hpp"
#include "fls/cor/lyt/buf.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/file/file_header.hpp"
#include "fls/footer/datatype_generated.h"
#include "fls/footer/table_descriptor.hpp"
#include "fls/io/file.hpp"
#include "fls/io/io.hpp"
#include "fls/reader/table_reader.hpp"
#include "fls/std/filesystem.hpp"
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

namespace {

struct Options {
	std::filesystem::path input;
	std::optional<size_t> rowgroup;
	uint32_t              samples = 1;
};

void print_usage(const char* prog) {
	std::cerr << "Usage:\n"
	          << "  " << prog << " benchmark <input.fls> [--rowgroup N] [--samples N]\n"
	          << "\n"
	          << "Options:\n"
	          << "  --rowgroup N   Only process the given rowgroup\n"
	          << "  --samples N    Number of benchmark repetitions (default: 1)\n";
}

bool parse_args(int argc, char** argv, Options& opt) {
	if (argc < 3) {
		return false;
	}

	std::string_view mode_arg = argv[1];
	if (mode_arg != "benchmark" && mode_arg != "bench") {
		return false;
	}

	for (int i = 2; i < argc; ++i) {
		std::string_view arg = argv[i];
		if (arg == "--help" || arg == "-h") {
			return false;
		}
		if (arg == "--rowgroup" && i + 1 < argc) {
			opt.rowgroup = static_cast<size_t>(std::stoul(argv[++i]));
			continue;
		}
		if (arg == "--samples" && i + 1 < argc) {
			opt.samples = static_cast<uint32_t>(std::stoul(argv[++i]));
			continue;
		}

		if (opt.input.empty()) {
			opt.input = std::filesystem::path(arg);
			continue;
		}

		return false;
	}

	return !opt.input.empty();
}

size_t data_type_size(const fastlanes::DataType dt) {
	using fastlanes::DataType;
	switch (dt) {
	case DataType::INT8:
	case DataType::UINT8:
	case DataType::BOOLEAN:
		return 1;
	case DataType::INT16:
	case DataType::UINT16:
		return 2;
	case DataType::INT32:
	case DataType::UINT32:
	case DataType::FLOAT:
	case DataType::DATE:
		return 4;
	case DataType::INT64:
	case DataType::UINT64:
	case DataType::DOUBLE:
	case DataType::TIMESTAMP:
		return 8;
	default:
		return 0;
	}
}

std::string format_bytes(double bytes) {
	static constexpr const char* kUnits[] = {"B", "KiB", "MiB", "GiB", "TiB"};
	size_t                       unit     = 0;
	while (bytes >= 1024.0 && unit < (sizeof(kUnits) / sizeof(kUnits[0]) - 1)) {
		bytes /= 1024.0;
		++unit;
	}
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(bytes < 10.0 ? 2 : (bytes < 100.0 ? 1 : 0)) << bytes << " " << kUnits[unit];
	return oss.str();
}

fastlanes::TableDescriptorHandle load_table_descriptor(const std::filesystem::path& file_path) {
	fastlanes::FileHeader header {};
	fastlanes::FileFooter footer {};

	fastlanes::FileHeader::Load(header, file_path);
	fastlanes::FileFooter::Load(footer, file_path);

	if (header.settings.inline_footer) {
		return fastlanes::TableDescriptorHandle::FromFileSlice(
		    file_path, footer.table_descriptor_offset, footer.table_descriptor_size, /*verify=*/true);
	}

	return fastlanes::TableDescriptorHandle::FromFile(file_path.parent_path() / "table_descriptor.fbb",
	                                                  /*verify=*/true);
}

double measure_rowgroup_io_ms(const std::filesystem::path& file_path, const fastlanes::RowgroupDescriptor* rg) {
	if (!rg) {
		return 0.0;
	}
	using Clock = std::chrono::steady_clock;
	fastlanes::Buf buf(rg->m_size());
	fastlanes::io  io    = fastlanes::make_unique<fastlanes::File>(file_path);
	const auto     start = Clock::now();
	fastlanes::IO::range_read(io, buf, rg->m_offset(), rg->m_size());
	const auto end = Clock::now();
	return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace

int main(int argc, char** argv) {
	Options opt;
	if (!parse_args(argc, argv, opt)) {
		print_usage(argv[0]);
		return 1;
	}

	try {
		fastlanes::Connection conn;
		auto                  table_reader = conn.reset().read_fls(opt.input);
		const auto            td_handle    = load_table_descriptor(opt.input);
		const auto*           td           = td_handle.Get();
		if (!td) {
			throw std::runtime_error("failed to load table descriptor");
		}

		const auto*  rowgroups   = td->m_rowgroup_descriptors();
		const size_t n_rowgroups = rowgroups ? rowgroups->size() : 0;
		if (n_rowgroups == 0) {
			throw std::runtime_error("no rowgroups found");
		}

		size_t start = 0;
		size_t end   = n_rowgroups;
		if (opt.rowgroup.has_value()) {
			if (*opt.rowgroup >= n_rowgroups) {
				throw std::out_of_range("rowgroup index out of range");
			}
			start = *opt.rowgroup;
			end   = start + 1;
		}

		double end_to_end_ms = 0.0;
		double kernel_ms     = 0.0;
		double setup_ms      = 0.0;
		double h2d_ms        = 0.0;
		double teardown_ms   = 0.0;
		double file_read_ms  = 0.0;
		size_t total_columns = 0;
		size_t total_items   = 0;
		size_t total_bytes   = 0;
		size_t total_rgs     = 0;

		for (size_t rg_idx = start; rg_idx < end; ++rg_idx) {
			const auto*  rg    = rowgroups->Get(static_cast<flatbuffers::uoffset_t>(rg_idx));
			const double io_ms = measure_rowgroup_io_ms(opt.input, rg);

			const auto*  cols       = rg->m_column_descriptors();
			const size_t rg_columns = cols ? cols->size() : 0;
			const size_t rg_vectors = rg_columns * static_cast<size_t>(rg->m_n_vec());

			size_t rg_bytes = 0;
			if (cols) {
				for (const auto* col_desc : *cols) {
					if (!col_desc) {
						continue;
					}
					const size_t element_size = data_type_size(col_desc->data_type());
					if (element_size == 0) {
						continue;
					}
					rg_bytes += static_cast<size_t>(rg->m_n_tuples()) * element_size;
				}
			}

			const auto ctor_start      = std::chrono::steady_clock::now();
			auto       rowgroup_reader = table_reader->get_rowgroup_reader(static_cast<fastlanes::n_t>(rg_idx));
			const auto ctor_end        = std::chrono::steady_clock::now();

			const double ctor_ms = std::chrono::duration<double, std::milli>(ctor_end - ctor_start).count();

			double build_ms = ctor_ms - io_ms;
			if (build_ms < 0.0) {
				build_ms = 0.0;
			}

			const auto           decode_start = std::chrono::steady_clock::now();
			const fastlanes::n_t n_vec        = static_cast<fastlanes::n_t>(rg->m_n_vec());
			for (uint32_t sample = 0; sample < opt.samples; ++sample) {
				for (fastlanes::n_t vec_idx {0}; vec_idx < n_vec; ++vec_idx) {
					rowgroup_reader->get_chunk(vec_idx);
				}
			}
			const auto decode_end = std::chrono::steady_clock::now();

			const double decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();

			end_to_end_ms += (build_ms + decode_ms);
			kernel_ms += decode_ms;
			setup_ms += build_ms;
			h2d_ms += 0.0;
			teardown_ms += 0.0;
			file_read_ms += io_ms;

			total_columns += rg_columns;
			total_items += rg_vectors;
			total_bytes += rg_bytes;
			++total_rgs;
		}

		const double total_samples =
		    (total_rgs > 0) ? (static_cast<double>(opt.samples) * static_cast<double>(total_rgs)) : 0.0;
		const double avg_us = (total_samples > 0.0) ? (kernel_ms * 1000.0 / total_samples) : 0.0;
		const double total_bytes_processed =
		    (total_rgs > 0) ? (static_cast<double>(total_bytes) * static_cast<double>(opt.samples)) : 0.0;
		const double kernel_seconds         = kernel_ms / 1000.0;
		const double kernel_throughput_bps  = (kernel_seconds > 0.0) ? (total_bytes_processed / kernel_seconds) : 0.0;
		const double kernel_throughput_gbps = kernel_throughput_bps / 1e9;
		const double kernel_throughput_gibps =
		    (kernel_seconds > 0.0) ? (total_bytes_processed / (1024.0 * 1024.0 * 1024.0 * kernel_seconds)) : 0.0;
		const double e2e_no_teardown_seconds = end_to_end_ms / 1000.0;
		const double e2e_no_teardown_bps =
		    (e2e_no_teardown_seconds > 0.0) ? (total_bytes_processed / e2e_no_teardown_seconds) : 0.0;
		const double e2e_no_teardown_gbps = e2e_no_teardown_bps / 1e9;
		const double e2e_no_teardown_gibps =
		    (e2e_no_teardown_seconds > 0.0)
		        ? (total_bytes_processed / (1024.0 * 1024.0 * 1024.0 * e2e_no_teardown_seconds))
		        : 0.0;

		const double end_to_end_with_teardown_ms = end_to_end_ms + teardown_ms;
		const double e2e_with_teardown_seconds   = end_to_end_with_teardown_ms / 1000.0;
		const double e2e_with_teardown_bps =
		    (e2e_with_teardown_seconds > 0.0) ? (total_bytes_processed / e2e_with_teardown_seconds) : 0.0;
		const double e2e_with_teardown_gbps = e2e_with_teardown_bps / 1e9;
		const double e2e_with_teardown_gibps =
		    (e2e_with_teardown_seconds > 0.0)
		        ? (total_bytes_processed / (1024.0 * 1024.0 * 1024.0 * e2e_with_teardown_seconds))
		        : 0.0;
		const double cols_per_rg =
		    (total_rgs > 0) ? (static_cast<double>(total_columns) / static_cast<double>(total_rgs)) : 0.0;
		const double vectors_per_rg =
		    (total_rgs > 0) ? (static_cast<double>(total_items) / static_cast<double>(total_rgs)) : 0.0;
		// const double vecs_per_col = (total_columns > 0) ? (static_cast<double>(total_items) /
		// static_cast<double>(total_columns)) : 0.0;

		std::cout << "Benchmark results:\n";
		std::cout << "  rowgroups: " << total_rgs << "\n";
		std::cout << "  columns:   " << total_columns << " (avg " << cols_per_rg << " per rowgroup)\n";
		std::cout << "  vectors: " << total_items << " (avg " << vectors_per_rg << " per rowgroup)\n";
		std::cout << "  samples:   " << opt.samples << "\n";
		std::cout << "  bytes:     " << total_bytes << " (" << format_bytes(static_cast<double>(total_bytes)) << ")\n";
		// benchmark_wall_ms mirrors galp_cli's semantics (file I/O + build + decode),
		// so the shared CSV column compares apples-to-apples across tools.
		std::cout << "  benchmark_wall_ms: " << (end_to_end_with_teardown_ms + file_read_ms) << "\n";
		std::cout << "  end_to_end_ms: " << end_to_end_with_teardown_ms << "\n";
		std::cout << "  end_to_end_ms (no teardown): " << end_to_end_ms << "\n";
		std::cout << "  kernel_ms:     " << kernel_ms << "\n";
		std::cout << "  setup_ms:      " << setup_ms << "\n";
		std::cout << "  h2d_ms:        " << h2d_ms << "\n";
		std::cout << "  teardown_ms:   " << teardown_ms << "\n";
		std::cout << "  file_read_ms:  " << file_read_ms << "\n";
		std::cout << "  avg_us:    " << avg_us << " (per rowgroup per sample)\n";
		std::cout << "  kernel_throughput: " << kernel_throughput_gbps << " (GB/s), " << kernel_throughput_gibps
		          << " (GiB/s)\n";
		std::cout << "  end_to_end_throughput: " << e2e_with_teardown_gbps << " (GB/s), " << e2e_with_teardown_gibps
		          << " (GiB/s)\n";
		std::cout << "  end_to_end_throughput (no teardown): " << e2e_no_teardown_gbps << " (GB/s), "
		          << e2e_no_teardown_gibps << " (GiB/s)\n";
		// std::cout << "  avg_vecs_per_col: " << vecs_per_col << "\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "Error: " << ex.what() << "\n";
		return 1;
	}
}
