#include "fls/connection.hpp"
#include "fls/table/memory_table.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace {

struct BenchmarkOptions {
	fastlanes::n_t              rowgroup_count         = 32;
	fastlanes::n_t              column_count           = 8;
	fastlanes::n_t              rows_per_rowgroup      = 64U * fastlanes::CFG::VEC_SZ;
	fastlanes::n_t              cardinality            = 0;
	std::string                 type                   = "i64";
	std::string                 schema                 = "auto";
	std::vector<fastlanes::n_t> workers                = {1, 2, 4, 8};
	fastlanes::n_t              repetitions            = 3;
	fastlanes::n_t              max_inflight_rowgroups = 0;
	fastlanes::n_t              max_inflight_bytes     = 0;
};

fastlanes::n_t parse_unsigned(const std::string& text, const std::string& option) {
	std::size_t parsed = 0;
	const auto  value  = std::stoull(text, &parsed);
	if (parsed != text.size()) {
		throw std::invalid_argument(option + " requires an unsigned integer");
	}
	return static_cast<fastlanes::n_t>(value);
}

std::vector<fastlanes::n_t> parse_workers(const std::string& value) {
	std::vector<fastlanes::n_t> result;
	std::size_t                 begin = 0;
	while (begin <= value.size()) {
		const auto end          = value.find(',', begin);
		const auto part         = value.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
		const auto worker_count = parse_unsigned(part, "--workers");
		if (worker_count == 0) {
			throw std::invalid_argument("--workers values must be greater than zero");
		}
		result.push_back(worker_count);
		if (end == std::string::npos) {
			break;
		}
		begin = end + 1;
	}
	if (result.empty()) {
		throw std::invalid_argument("--workers requires at least one value");
	}
	return result;
}

BenchmarkOptions parse_options(const int argc, char** argv) {
	BenchmarkOptions options;
	for (int arg_idx = 1; arg_idx < argc; ++arg_idx) {
		const std::string_view arg(argv[arg_idx]);
		if (arg == "--help") {
			std::cout << "Usage: bench_parallel_encoding [options]\n"
			             "  --rowgroups N\n"
			             "  --columns N\n"
			             "  --rows-per-rowgroup N\n"
			             "  --cardinality N (zero uses the type-specific default)\n"
			             "  --type i8|i16|i32|i64|float|double|string\n"
			             "  --schema auto|compressed|uncompressed\n"
			             "  --workers 1,2,4,8\n"
			             "  --repetitions N\n"
			             "  --max-inflight-rowgroups N\n"
			             "  --max-inflight-bytes N\n";
			std::exit(0);
		}
		if (arg_idx + 1 >= argc) {
			throw std::invalid_argument(std::string(arg) + " requires a value");
		}
		const std::string value(argv[++arg_idx]);
		if (arg == "--rowgroups") {
			options.rowgroup_count = parse_unsigned(value, std::string(arg));
		} else if (arg == "--columns") {
			options.column_count = parse_unsigned(value, std::string(arg));
		} else if (arg == "--rows-per-rowgroup") {
			options.rows_per_rowgroup = parse_unsigned(value, std::string(arg));
		} else if (arg == "--cardinality") {
			options.cardinality = parse_unsigned(value, std::string(arg));
		} else if (arg == "--type") {
			options.type = value;
		} else if (arg == "--schema") {
			options.schema = value;
		} else if (arg == "--workers") {
			options.workers = parse_workers(value);
		} else if (arg == "--repetitions") {
			options.repetitions = parse_unsigned(value, std::string(arg));
		} else if (arg == "--max-inflight-rowgroups") {
			options.max_inflight_rowgroups = parse_unsigned(value, std::string(arg));
		} else if (arg == "--max-inflight-bytes") {
			options.max_inflight_bytes = parse_unsigned(value, std::string(arg));
		} else {
			throw std::invalid_argument("unknown option: " + std::string(arg));
		}
	}
	if (options.rowgroup_count == 0 || options.column_count == 0 || options.rows_per_rowgroup == 0 ||
	    options.repetitions == 0) {
		throw std::invalid_argument("rowgroups, columns, rows-per-rowgroup, and repetitions must be nonzero");
	}
	if (options.schema != "auto" && options.schema != "compressed" && options.schema != "uncompressed") {
		throw std::invalid_argument("--schema must be auto, compressed, or uncompressed");
	}
	if (options.rows_per_rowgroup > std::numeric_limits<fastlanes::n_t>::max() / options.rowgroup_count) {
		throw std::invalid_argument("requested row count overflows");
	}
	return options;
}

class TemporaryDirectory {
public:
	TemporaryDirectory() {
		const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
		path =
		    std::filesystem::temp_directory_path() / ("fastlanes_parallel_encoding_benchmark_" + std::to_string(stamp));
		std::filesystem::create_directories(path);
	}

	~TemporaryDirectory() {
		std::error_code ec;
		std::filesystem::remove_all(path, ec);
	}

	std::filesystem::path path;
};

uint64_t process_peak_rss_bytes() {
#if defined(__unix__) || defined(__APPLE__)
	rusage usage {};
	if (getrusage(RUSAGE_SELF, &usage) != 0) {
		return 0;
	}
#if defined(__APPLE__)
	return static_cast<uint64_t>(usage.ru_maxrss);
#else
	return static_cast<uint64_t>(usage.ru_maxrss) * 1024U;
#endif
#else
	return 0;
#endif
}

template <typename T>
fastlanes::OperatorToken uncompressed_token() {
	if constexpr (std::is_same_v<T, int8_t>) {
		return fastlanes::OperatorToken::EXP_UNCOMPRESSED_I08;
	} else if constexpr (std::is_same_v<T, int16_t>) {
		return fastlanes::OperatorToken::EXP_UNCOMPRESSED_I16;
	} else if constexpr (std::is_same_v<T, int32_t>) {
		return fastlanes::OperatorToken::EXP_UNCOMPRESSED_I32;
	} else if constexpr (std::is_same_v<T, int64_t>) {
		return fastlanes::OperatorToken::EXP_UNCOMPRESSED_I64;
	} else if constexpr (std::is_same_v<T, float>) {
		return fastlanes::OperatorToken::EXP_UNCOMPRESSED_FLT;
	} else if constexpr (std::is_same_v<T, double>) {
		return fastlanes::OperatorToken::EXP_UNCOMPRESSED_DBL;
	} else {
		return fastlanes::OperatorToken::EXP_UNCOMPRESSED_STR;
	}
}

template <typename T>
fastlanes::OperatorToken compressed_token() {
	if constexpr (std::is_same_v<T, int8_t>) {
		return fastlanes::OperatorToken::EXP_FFOR_I08;
	} else if constexpr (std::is_same_v<T, int16_t>) {
		return fastlanes::OperatorToken::EXP_FFOR_I16;
	} else if constexpr (std::is_same_v<T, int32_t>) {
		return fastlanes::OperatorToken::EXP_FFOR_I32;
	} else if constexpr (std::is_same_v<T, int64_t>) {
		return fastlanes::OperatorToken::EXP_FFOR_I64;
	} else if constexpr (std::is_same_v<T, float>) {
		return fastlanes::OperatorToken::EXP_ALP_FLT;
	} else if constexpr (std::is_same_v<T, double>) {
		return fastlanes::OperatorToken::EXP_ALP_DBL;
	} else {
		return fastlanes::OperatorToken::EXP_FSST;
	}
}

template <typename T>
T make_value(const fastlanes::n_t row, const fastlanes::n_t column, const fastlanes::n_t requested_cardinality) {
	if constexpr (std::is_same_v<T, fastlanes::string>) {
		const auto cardinality = requested_cardinality == 0 ? 1009U : requested_cardinality;
		return "value_" + std::to_string((row * 17U + column * 31U) % cardinality);
	} else if constexpr (std::is_floating_point_v<T>) {
		const auto cardinality = requested_cardinality == 0 ? 8191U : requested_cardinality;
		const auto centered =
		    static_cast<int64_t>((row + column * 13U) % cardinality) - static_cast<int64_t>(cardinality / 2U);
		return static_cast<T>(centered) * static_cast<T>(0.125);
	} else {
		const auto cardinality = requested_cardinality == 0 ? 1000003U : requested_cardinality;
		return static_cast<T>((row * 7919U + column * 104729U) % cardinality);
	}
}

template <typename T>
class SyntheticTable {
public:
	explicit SyntheticTable(const BenchmarkOptions& benchmark_options) {
		const auto row_count = benchmark_options.rowgroup_count * benchmark_options.rows_per_rowgroup;
		storage.resize(static_cast<std::size_t>(benchmark_options.column_count));
		columns.reserve(static_cast<std::size_t>(benchmark_options.column_count));
		for (fastlanes::n_t column_idx = 0; column_idx < benchmark_options.column_count; ++column_idx) {
			auto& values = storage[static_cast<std::size_t>(column_idx)];
			values.resize(static_cast<std::size_t>(row_count));
			for (fastlanes::n_t row_idx = 0; row_idx < row_count; ++row_idx) {
				values[static_cast<std::size_t>(row_idx)] =
				    make_value<T>(row_idx, column_idx, benchmark_options.cardinality);
			}
			columns.push_back({"column_" + std::to_string(column_idx), std::span<const T>(values)});
		}

		rowgroups.assign(static_cast<std::size_t>(benchmark_options.rowgroup_count),
		                 benchmark_options.rows_per_rowgroup);
		options.n_vectors_per_rowgroup =
		    (benchmark_options.rows_per_rowgroup + fastlanes::CFG::VEC_SZ - 1U) / fastlanes::CFG::VEC_SZ;
		options.rowgroup_n_tuples = std::span<const fastlanes::n_t>(rowgroups);
		options.force_schema      = benchmark_options.schema != "auto";
		if (options.force_schema) {
			options.forced_schema.assign(static_cast<std::size_t>(benchmark_options.column_count),
			                             benchmark_options.schema == "compressed" ? compressed_token<T>()
			                                                                      : uncompressed_token<T>());
		}
	}

	fastlanes::MemoryTable table() const {
		return {std::span<const fastlanes::MemoryColumn>(columns)};
	}

	std::vector<std::vector<T>>          storage;
	std::vector<fastlanes::MemoryColumn> columns;
	std::vector<fastlanes::n_t>          rowgroups;
	fastlanes::MemoryTableOptions        options;
};

template <typename T>
void run_benchmark(const BenchmarkOptions& options) {
	SyntheticTable<T>  data(options);
	TemporaryDirectory temp;
	const auto         total_rows = options.rowgroup_count * options.rows_per_rowgroup;

	std::cout << "type,schema,worker_count,repetition,rowgroups,columns,rows,output_bytes,wall_seconds,"
	             "preparation_seconds,encoding_seconds,finalization_seconds,total_to_fls_seconds,rows_per_second,"
	             "encoding_rows_per_second,bytes_per_second,encoding_bytes_per_second,peak_inflight_rowgroups,"
	             "peak_inflight_bytes,resolved_inflight_rowgroups,rowgroups_per_task,process_peak_rss_bytes\n";
	for (const auto worker_count : options.workers) {
		for (fastlanes::n_t repetition = 0; repetition < options.repetitions; ++repetition) {
			fastlanes::Config config;
			config.inline_footer = fastlanes::FLS_TRUE;
			fastlanes::Connection connection(config);
			connection.read_memory(data.table(), data.options);

			fastlanes::EncodingOptions encoding_options;
			encoding_options.worker_count           = worker_count;
			encoding_options.max_inflight_rowgroups = options.max_inflight_rowgroups;
			encoding_options.max_inflight_bytes     = options.max_inflight_bytes;
			const auto output  = temp.path / ("workers_" + std::to_string(worker_count) + "_repeat_" +
                                             std::to_string(repetition) + ".fls");
			const auto started = std::chrono::steady_clock::now();
			connection.to_fls(output, encoding_options);
			const auto   finished     = std::chrono::steady_clock::now();
			const double seconds      = std::chrono::duration<double>(finished - started).count();
			const auto   output_bytes = std::filesystem::file_size(output);
			const auto&  stats        = connection.get_last_encoding_stats();

			std::cout << options.type << ',' << options.schema << ',' << worker_count << ',' << repetition << ','
			          << options.rowgroup_count << ',' << options.column_count << ',' << total_rows << ','
			          << output_bytes << ',' << seconds << ',' << stats.preparation_wall_seconds << ','
			          << stats.encoding_wall_seconds << ',' << stats.finalization_wall_seconds << ','
			          << stats.total_wall_seconds << ',' << static_cast<double>(total_rows) / seconds << ','
			          << static_cast<double>(total_rows) / stats.encoding_wall_seconds << ','
			          << static_cast<double>(output_bytes) / seconds << ','
			          << static_cast<double>(output_bytes) / stats.encoding_wall_seconds << ','
			          << stats.peak_inflight_rowgroups << ',' << stats.peak_inflight_bytes << ','
			          << stats.resolved_inflight_rowgroups << ',' << stats.resolved_rowgroups_per_task << ','
			          << process_peak_rss_bytes() << '\n';
		}
	}
}

} // namespace

int main(const int argc, char** argv) {
	try {
		const auto options = parse_options(argc, argv);
		if (options.type == "i8") {
			run_benchmark<int8_t>(options);
		} else if (options.type == "i16") {
			run_benchmark<int16_t>(options);
		} else if (options.type == "i32") {
			run_benchmark<int32_t>(options);
		} else if (options.type == "i64") {
			run_benchmark<int64_t>(options);
		} else if (options.type == "float") {
			run_benchmark<float>(options);
		} else if (options.type == "double") {
			run_benchmark<double>(options);
		} else if (options.type == "string") {
			run_benchmark<fastlanes::string>(options);
		} else {
			throw std::invalid_argument("unsupported --type: " + options.type);
		}
		return 0;
	} catch (const std::exception& exception) {
		std::cerr << "bench_parallel_encoding: " << exception.what() << '\n';
		return 1;
	}
}
