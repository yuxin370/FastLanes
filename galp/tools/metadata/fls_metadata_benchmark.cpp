#include "fls/common/status.hpp"
#include "fls/connection.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/file/file_header.hpp"
#include "fls/footer/table_descriptor.hpp"
#include "fls/info.hpp"
#include "fls/io/file.hpp"
#include "fls/json/nlohmann/json.hpp"
#include "fls/reader/rowgroup_reader.hpp"
#include "fls/reader/table_reader.hpp"
#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using json  = nlohmann::json;

struct Options {
	std::filesystem::path input;
	std::filesystem::path output;
	std::string           phase       = "metadata";
	std::string           cache_state = "uncontrolled";
	uint64_t              iterations  = 1000U;
	uint64_t              warmup      = 20U;
	uint64_t              seed        = 20260720U;
	uint32_t              threads     = 4U;
	bool                  pretty      = false;
};

[[noreturn]] void usage_error(const std::string& message) {
	throw std::invalid_argument(message + "\nRun galp_fls_metadata_benchmark --help for usage.");
}

void print_help() {
	std::cout << "Usage:\n"
	          << "  galp_fls_metadata_benchmark --input FILE --phase metadata|decode [options]\n\n"
	          << "Options:\n"
	          << "  --iterations N              measured operations per single-thread phase (default 1000)\n"
	          << "  --warmup N                  unmeasured operations (default 20)\n"
	          << "  --threads N                 multithread workers (default 4)\n"
	          << "  --seed N                    random seed (default 20260720)\n"
	          << "  --cache-state LABEL         cold|warm|uncontrolled provenance label\n"
	          << "  --output JSON --pretty      optional artifact\n\n"
	          << "metadata measures verified descriptor open and direct O(1) rowgroup/column lookup.\n"
	          << "decode measures current TableReader rowgroup range-read plus all-vector CPU decode.\n"
	          << "Each multithread decode worker owns a Connection/TableReader, matching the current API.\n";
}

uint64_t parse_u64(const char* value, const std::string_view name) {
	try {
		std::size_t parsed = 0;
		const auto  result = std::stoull(value, &parsed);
		if (parsed != std::string_view(value).size()) {
			usage_error(std::string(name) + " is not an unsigned integer");
		}
		return result;
	} catch (const std::exception&) { usage_error(std::string(name) + " is not an unsigned integer"); }
}

Options parse_options(const int argc, char** argv) {
	if (argc == 2 && std::string_view(argv[1]) == "--help") {
		print_help();
		std::exit(0);
	}
	Options options;
	for (int index = 1; index < argc; ++index) {
		const std::string_view argument(argv[index]);
		auto                   next = [&]() -> const char* {
            if (++index >= argc) {
                usage_error(std::string(argument) + " requires a value");
            }
            return argv[index];
		};
		if (argument == "--input") {
			options.input = next();
		} else if (argument == "--output") {
			options.output = next();
		} else if (argument == "--phase") {
			options.phase = next();
		} else if (argument == "--cache-state") {
			options.cache_state = next();
		} else if (argument == "--iterations") {
			options.iterations = parse_u64(next(), argument);
		} else if (argument == "--warmup") {
			options.warmup = parse_u64(next(), argument);
		} else if (argument == "--seed") {
			options.seed = parse_u64(next(), argument);
		} else if (argument == "--threads") {
			const auto value = parse_u64(next(), argument);
			if (value == 0U || value > std::numeric_limits<uint32_t>::max()) {
				usage_error("--threads is out of range");
			}
			options.threads = static_cast<uint32_t>(value);
		} else if (argument == "--pretty") {
			options.pretty = true;
		} else if (argument == "--help") {
			print_help();
			std::exit(0);
		} else {
			usage_error("unknown argument: " + std::string(argument));
		}
	}
	if (options.input.empty()) {
		usage_error("--input is required");
	}
	if (options.phase != "metadata" && options.phase != "decode") {
		usage_error("--phase must be metadata or decode");
	}
	if (options.iterations == 0U) {
		usage_error("--iterations must be positive");
	}
	if (options.cache_state != "cold" && options.cache_state != "warm" && options.cache_state != "uncontrolled") {
		usage_error("--cache-state must be cold, warm, or uncontrolled");
	}
	return options;
}

void require_status(const fastlanes::Status status, const std::string_view context) {
	if (!status.success) {
		throw std::runtime_error(std::string(context) + ": " +
		                         std::string(fastlanes::Status::message_for(status.code)));
	}
}

uint64_t current_rss_bytes() {
	std::ifstream status("/proc/self/statm");
	uint64_t      total_pages    = 0;
	uint64_t      resident_pages = 0;
	if (!(status >> total_pages >> resident_pages)) {
		return 0U;
	}
	static_cast<void>(total_pages);
	const auto page_size = ::sysconf(_SC_PAGESIZE);
	return page_size <= 0 ? 0U : resident_pages * static_cast<uint64_t>(page_size);
}

uint64_t peak_rss_bytes() {
	rusage usage {};
	if (::getrusage(RUSAGE_SELF, &usage) != 0) {
		return 0U;
	}
	return static_cast<uint64_t>(usage.ru_maxrss) * 1024U;
}

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
	return std::chrono::duration<double, std::milli>(end - start).count();
}

fastlanes::TableDescriptorHandle load_descriptor(const std::filesystem::path& input) {
	fastlanes::File       file(input);
	fastlanes::FileHeader header {};
	fastlanes::FileFooter footer {};
	require_status(fastlanes::FileHeader::Load(header, file), "read header");
	require_status(fastlanes::FileFooter::Load(footer, file), "read footer");
	if (header.magic_bytes != fastlanes::Info::get_magic_bytes() ||
	    footer.magic_bytes != fastlanes::Info::get_magic_bytes()) {
		throw std::runtime_error("FLS magic mismatch");
	}
	if (header.settings.inline_footer) {
		return fastlanes::TableDescriptorHandle::FromFileSlice(
		    file, footer.table_descriptor_offset, footer.table_descriptor_size, true);
	}
	return fastlanes::TableDescriptorHandle::FromFile(input.parent_path() / "table_descriptor.fbb", true);
}

json distribution(std::vector<double> values) {
	if (values.empty()) {
		return {{"count", 0U}};
	}
	std::sort(values.begin(), values.end());
	double total = 0.0;
	for (const auto value : values) {
		total += value;
	}
	auto percentile = [&](const double quantile) {
		const auto rank = static_cast<std::size_t>(std::ceil(quantile * static_cast<double>(values.size())) - 1.0);
		return values[std::min(rank, values.size() - 1U)];
	};
	return {{"count", values.size()},
	        {"mean_ms", total / static_cast<double>(values.size())},
	        {"p50_ms", percentile(0.50)},
	        {"p95_ms", percentile(0.95)},
	        {"p99_ms", percentile(0.99)},
	        {"min_ms", values.front()},
	        {"max_ms", values.back()}};
}

std::vector<uint64_t> random_indices(const uint64_t count, const uint64_t limit, const uint64_t seed) {
	if (limit == 0U) {
		throw std::runtime_error("table descriptor has no rowgroups");
	}
	std::mt19937_64                         generator(seed);
	std::uniform_int_distribution<uint64_t> distribution(0U, limit - 1U);
	std::vector<uint64_t>                   result;
	result.reserve(static_cast<std::size_t>(count));
	for (uint64_t index = 0; index < count; ++index) {
		result.push_back(distribution(generator));
	}
	return result;
}

uint64_t metadata_lookup(const fastlanes::TableDescriptor& descriptor, const uint64_t rowgroup_index) {
	const auto* rowgroup =
	    descriptor.m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rowgroup_index));
	const auto* columns  = rowgroup->m_column_descriptors();
	uint64_t    checksum = rowgroup->m_offset() ^ rowgroup->m_size() ^ rowgroup->m_n_tuples();
	if (columns != nullptr && !columns->empty()) {
		const auto  column_index = static_cast<flatbuffers::uoffset_t>(rowgroup_index % columns->size());
		const auto* column       = columns->Get(column_index);
		checksum ^= column->column_offset() ^ column->total_size() ^ column->idx();
	}
	return checksum;
}

json benchmark_metadata(const Options& options) {
	const auto  rss_before = current_rss_bytes();
	const auto  open_start = Clock::now();
	auto        descriptor = load_descriptor(options.input);
	const auto  open_end   = Clock::now();
	const auto  rss_open   = current_rss_bytes();
	const auto* rowgroups  = descriptor->m_rowgroup_descriptors();
	if (rowgroups == nullptr || rowgroups->empty()) {
		throw std::runtime_error("table descriptor has no rowgroups");
	}
	const auto rowgroup_count = static_cast<uint64_t>(rowgroups->size());
	const auto random         = random_indices(options.warmup + options.iterations, rowgroup_count, options.seed);
	uint64_t   checksum       = 0U;
	for (uint64_t index = 0; index < options.warmup; ++index) {
		checksum ^= metadata_lookup(*descriptor, random[static_cast<std::size_t>(index)]);
	}

	std::vector<double> random_latencies;
	random_latencies.reserve(static_cast<std::size_t>(options.iterations));
	const auto random_wall_start = Clock::now();
	for (uint64_t index = 0; index < options.iterations; ++index) {
		const auto start = Clock::now();
		checksum ^= metadata_lookup(*descriptor, random[static_cast<std::size_t>(options.warmup + index)]);
		const auto end = Clock::now();
		random_latencies.push_back(elapsed_ms(start, end));
	}
	const auto random_wall_end = Clock::now();

	std::vector<double> sequential_latencies;
	sequential_latencies.reserve(static_cast<std::size_t>(options.iterations));
	const auto sequential_wall_start = Clock::now();
	for (uint64_t index = 0; index < options.iterations; ++index) {
		const auto start = Clock::now();
		checksum ^= metadata_lookup(*descriptor, index % rowgroup_count);
		const auto end = Clock::now();
		sequential_latencies.push_back(elapsed_ms(start, end));
	}
	const auto sequential_wall_end = Clock::now();

	std::barrier                     ready_barrier(static_cast<std::ptrdiff_t>(options.threads + 1U));
	std::barrier                     start_barrier(static_cast<std::ptrdiff_t>(options.threads + 1U));
	std::vector<std::vector<double>> thread_latencies(options.threads);
	std::vector<uint64_t>            thread_checksums(options.threads, 0U);
	std::vector<std::thread>         workers;
	workers.reserve(options.threads);
	const uint64_t per_thread = (options.iterations + options.threads - 1U) / options.threads;
	for (uint32_t thread_index = 0; thread_index < options.threads; ++thread_index) {
		workers.emplace_back([&, thread_index]() {
			auto indices =
			    random_indices(per_thread, rowgroup_count, options.seed + 0x9e3779b97f4a7c15ULL * (thread_index + 1U));
			auto& latencies = thread_latencies[thread_index];
			latencies.reserve(static_cast<std::size_t>(per_thread));
			ready_barrier.arrive_and_wait();
			start_barrier.arrive_and_wait();
			for (const auto rowgroup_index : indices) {
				const auto start = Clock::now();
				thread_checksums[thread_index] ^= metadata_lookup(*descriptor, rowgroup_index);
				const auto end = Clock::now();
				latencies.push_back(elapsed_ms(start, end));
			}
		});
	}
	ready_barrier.arrive_and_wait();
	const auto multithread_wall_start = Clock::now();
	start_barrier.arrive_and_wait();
	for (auto& worker : workers) {
		worker.join();
	}
	const auto          multithread_wall_end = Clock::now();
	std::vector<double> multithread_latencies;
	for (uint32_t thread_index = 0; thread_index < options.threads; ++thread_index) {
		checksum ^= thread_checksums[thread_index];
		multithread_latencies.insert(
		    multithread_latencies.end(), thread_latencies[thread_index].begin(), thread_latencies[thread_index].end());
	}

	const auto random_wall_ms     = elapsed_ms(random_wall_start, random_wall_end);
	const auto sequential_wall_ms = elapsed_ms(sequential_wall_start, sequential_wall_end);
	const auto threaded_wall_ms   = elapsed_ms(multithread_wall_start, multithread_wall_end);
	return {
	    {"phase", "metadata"},
	    {"descriptor_open_ms", elapsed_ms(open_start, open_end)},
	    {"descriptor_loaded_bytes", descriptor.size()},
	    {"rowgroups", rowgroup_count},
	    {"lookup_granularity", "rowgroup directory plus one column descriptor"},
	    {"random",
	     {{"latency", distribution(std::move(random_latencies))},
	      {"wall_ms", random_wall_ms},
	      {"operations_per_second", static_cast<double>(options.iterations) * 1000.0 / random_wall_ms}}},
	    {"sequential",
	     {{"latency", distribution(std::move(sequential_latencies))},
	      {"wall_ms", sequential_wall_ms},
	      {"operations_per_second", static_cast<double>(options.iterations) * 1000.0 / sequential_wall_ms}}},
	    {"multithread_random",
	     {{"threads", options.threads},
	      {"operations", per_thread * options.threads},
	      {"latency", distribution(std::move(multithread_latencies))},
	      {"wall_ms", threaded_wall_ms},
	      {"operations_per_second", static_cast<double>(per_thread * options.threads) * 1000.0 / threaded_wall_ms}}},
	    {"rss_before_bytes", rss_before},
	    {"rss_after_open_bytes", rss_open},
	    {"peak_rss_bytes", peak_rss_bytes()},
	    {"checksum", checksum}};
}

struct RowgroupInfo {
	uint64_t stored_bytes = 0;
	uint64_t vectors      = 0;
};

std::vector<RowgroupInfo> load_rowgroup_info(const std::filesystem::path& input) {
	auto        descriptor = load_descriptor(input);
	const auto* rowgroups  = descriptor->m_rowgroup_descriptors();
	if (rowgroups == nullptr || rowgroups->empty()) {
		throw std::runtime_error("table descriptor has no rowgroups");
	}
	std::vector<RowgroupInfo> result;
	result.reserve(rowgroups->size());
	for (const auto* rowgroup : *rowgroups) {
		result.push_back({rowgroup->m_size(), rowgroup->m_n_vec()});
	}
	return result;
}

uint64_t decode_rowgroup(fastlanes::TableReader& reader, const uint64_t rowgroup_index, const uint64_t vectors) {
	auto     rowgroup = reader.get_rowgroup_reader(rowgroup_index);
	uint64_t checksum = rowgroup->get_descriptor().m_offset() ^ rowgroup->get_descriptor().m_size();
	for (uint64_t vector_index = 0; vector_index < vectors; ++vector_index) {
		checksum ^= rowgroup->get_chunk(vector_index).size();
	}
	return checksum;
}

struct DecodeMeasurement {
	std::vector<double> latencies;
	double              wall_ms      = 0.0;
	uint64_t            stored_bytes = 0;
	uint64_t            checksum     = 0;
};

DecodeMeasurement measure_decode(fastlanes::TableReader&          reader,
                                 const std::vector<RowgroupInfo>& info,
                                 const std::vector<uint64_t>&     indices) {
	DecodeMeasurement result;
	result.latencies.reserve(indices.size());
	const auto wall_start = Clock::now();
	for (const auto index : indices) {
		const auto start = Clock::now();
		result.checksum ^= decode_rowgroup(reader, index, info[static_cast<std::size_t>(index)].vectors);
		const auto end = Clock::now();
		result.latencies.push_back(elapsed_ms(start, end));
		result.stored_bytes += info[static_cast<std::size_t>(index)].stored_bytes;
	}
	result.wall_ms = elapsed_ms(wall_start, Clock::now());
	return result;
}

json decode_result(DecodeMeasurement measurement) {
	const auto wall_seconds = measurement.wall_ms / 1000.0;
	return {{"latency", distribution(std::move(measurement.latencies))},
	        {"wall_ms", measurement.wall_ms},
	        {"stored_bytes", measurement.stored_bytes},
	        {"rowgroups_per_second",
	         wall_seconds == 0.0 ? 0.0 : static_cast<double>(measurement.latencies.size()) / wall_seconds},
	        {"stored_gib_per_second",
	         wall_seconds == 0.0
	             ? 0.0
	             : static_cast<double>(measurement.stored_bytes) / (1024.0 * 1024.0 * 1024.0 * wall_seconds)},
	        {"checksum", measurement.checksum}};
}

json benchmark_decode(const Options& options) {
	const auto            info           = load_rowgroup_info(options.input);
	const auto            rowgroup_count = static_cast<uint64_t>(info.size());
	fastlanes::Connection connection;
	const auto            open_start = Clock::now();
	auto                  reader     = connection.reset().read_fls(options.input);
	const auto            open_end   = Clock::now();
	const auto            rss_open   = current_rss_bytes();

	const auto warmup_indices = random_indices(options.warmup, rowgroup_count, options.seed ^ 0x51f15e5dU);
	for (const auto index : warmup_indices) {
		static_cast<void>(decode_rowgroup(*reader, index, info[static_cast<std::size_t>(index)].vectors));
	}
	const auto            random = random_indices(options.iterations, rowgroup_count, options.seed);
	std::vector<uint64_t> sequential;
	sequential.reserve(static_cast<std::size_t>(options.iterations));
	for (uint64_t index = 0; index < options.iterations; ++index) {
		sequential.push_back(index % rowgroup_count);
	}
	auto random_result     = measure_decode(*reader, info, random);
	auto sequential_result = measure_decode(*reader, info, sequential);

	std::barrier                   ready_barrier(static_cast<std::ptrdiff_t>(options.threads + 1U));
	std::barrier                   start_barrier(static_cast<std::ptrdiff_t>(options.threads + 1U));
	std::vector<DecodeMeasurement> thread_results(options.threads);
	std::vector<double>            thread_open_ms(options.threads, 0.0);
	std::vector<std::thread>       workers;
	const uint64_t                 per_thread = (options.iterations + options.threads - 1U) / options.threads;
	workers.reserve(options.threads);
	for (uint32_t thread_index = 0; thread_index < options.threads; ++thread_index) {
		workers.emplace_back([&, thread_index]() {
			fastlanes::Connection local_connection;
			const auto            local_open_start = Clock::now();
			auto                  local_reader     = local_connection.reset().read_fls(options.input);
			thread_open_ms[thread_index]           = elapsed_ms(local_open_start, Clock::now());
			auto indices =
			    random_indices(per_thread, rowgroup_count, options.seed + 0x9e3779b97f4a7c15ULL * (thread_index + 1U));
			ready_barrier.arrive_and_wait();
			start_barrier.arrive_and_wait();
			thread_results[thread_index] = measure_decode(*local_reader, info, indices);
		});
	}
	ready_barrier.arrive_and_wait();
	const auto threaded_wall_start = Clock::now();
	start_barrier.arrive_and_wait();
	for (auto& worker : workers) {
		worker.join();
	}
	const auto        threaded_wall_ms = elapsed_ms(threaded_wall_start, Clock::now());
	DecodeMeasurement merged;
	for (auto& result : thread_results) {
		merged.stored_bytes += result.stored_bytes;
		merged.checksum ^= result.checksum;
		merged.latencies.insert(merged.latencies.end(), result.latencies.begin(), result.latencies.end());
	}
	merged.wall_ms = threaded_wall_ms;
	return {{"phase", "decode"},
	        {"table_reader_open_ms", elapsed_ms(open_start, open_end)},
	        {"rowgroups", rowgroup_count},
	        {"read_granularity", "one rowgroup range-read plus decode of every vector and column"},
	        {"random", decode_result(std::move(random_result))},
	        {"sequential_batch", decode_result(std::move(sequential_result))},
	        {"multithread_random",
	         {{"threads", options.threads},
	          {"reader_open_ms_per_thread", thread_open_ms},
	          {"measurement", decode_result(std::move(merged))}}},
	        {"rss_after_reader_open_bytes", rss_open},
	        {"peak_rss_bytes", peak_rss_bytes()}};
}

void emit(const json& result, const Options& options) {
	const auto output = result.dump(options.pretty ? 2 : -1) + '\n';
	if (options.output.empty()) {
		std::cout << output;
		return;
	}
	if (!options.output.parent_path().empty()) {
		std::filesystem::create_directories(options.output.parent_path());
	}
	std::ofstream stream(options.output);
	if (!stream) {
		throw std::runtime_error("cannot open output: " + options.output.string());
	}
	stream << output;
}

} // namespace

int main(const int argc, char** argv) {
	try {
		const auto options = parse_options(argc, argv);
		json       result;
		result["schema_version"] = "galp_fls_metadata_benchmark_v1";
		result["input"]          = std::filesystem::absolute(options.input).string();
		result["parameters"]     = {{"phase", options.phase},
		                            {"iterations", options.iterations},
		                            {"warmup", options.warmup},
		                            {"threads", options.threads},
		                            {"seed", options.seed},
		                            {"cache_state", options.cache_state},
		                            {"cache_state_is_operator_supplied", true}};
		result["result"] = options.phase == "metadata" ? benchmark_metadata(options) : benchmark_decode(options);
		emit(result, options);
		return 0;
	} catch (const std::exception& error) {
		std::cerr << "galp_fls_metadata_benchmark: " << error.what() << '\n';
		return 1;
	}
}
