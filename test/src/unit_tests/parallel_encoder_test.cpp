// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// test/src/unit_tests/parallel_encoder_test.cpp
// ────────────────────────────────────────────────────────
#include "alp/config.hpp"
#include "alp/encoder.hpp"
#include "alp/state.hpp"
#include "file/staged_output.hpp"
#include "fls/connection.hpp"
#include "fls/encoder/encoder.hpp"
#include "fls/encoder/parallel_rowgroups.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/file/file_header.hpp"
#include "fls/footer/table_descriptor.hpp"
#include "fls/table/memory_table.hpp"
#include "fls/table/rowgroup.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

class TemporaryDirectory {
public:
	explicit TemporaryDirectory(const std::string& label) {
		const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
		path = std::filesystem::temp_directory_path() / ("fastlanes_" + label + "_" + std::to_string(stamp));
		std::filesystem::create_directories(path);
	}

	~TemporaryDirectory() {
		std::error_code ec;
		std::filesystem::remove_all(path, ec);
	}

	TemporaryDirectory(const TemporaryDirectory&)            = delete;
	TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

	std::filesystem::path path;
};

std::vector<uint8_t> read_bytes(const std::filesystem::path& path) {
	std::ifstream input(path, std::ios::binary);
	if (!input) {
		throw std::runtime_error("failed to open test artifact: " + path.string());
	}
	return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

TEST(AlpEncoderScratch, ExceptionPlaceholdersComeFromTheCurrentVector) {
	using Encoder = alp::encoder<float>;
	using State   = alp::state<float>;
	using Encoded = Encoder::ST;

	std::array<float, alp::config::VECTOR_SIZE>    input {};
	std::array<float, alp::config::VECTOR_SIZE>    exceptions {};
	std::array<uint16_t, alp::config::VECTOR_SIZE> exception_positions {};
	std::array<Encoded, alp::config::VECTOR_SIZE>  encoded {};
	State                                          state;
	state.exp = 0;
	state.fac = 0;

	// Seed a preceding call with a longer exception prefix.  Scratch contents
	// from that call must not influence the next vector.
	input.fill(std::numeric_limits<float>::infinity());
	Encoder::encode_simdized(
	    input.data(), exceptions.data(), exception_positions.data(), encoded.data(), state, nullptr);
	ASSERT_EQ(state.n_exceptions, alp::config::VECTOR_SIZE);

	input.fill(17.0F);
	constexpr uint16_t exception_count = 31U;
	std::fill_n(input.begin(), exception_count, std::numeric_limits<float>::infinity());
	Encoder::encode_simdized(
	    input.data(), exceptions.data(), exception_positions.data(), encoded.data(), state, nullptr);

	ASSERT_EQ(state.n_exceptions, exception_count);
	ASSERT_EQ(encoded[exception_count], Encoded {17});
	for (uint16_t index = 0; index < exception_count; ++index) {
		EXPECT_EQ(exception_positions[index], index);
		EXPECT_EQ(encoded[index], encoded[exception_count]);
	}
}

struct RowgroupMetadata {
	uint64_t size;
	uint64_t offset;
	uint64_t tuple_count;
	uint64_t vector_count;

	bool operator==(const RowgroupMetadata&) const = default;
};

struct DescriptorSnapshot {
	uint64_t                      table_binary_size;
	uint64_t                      footer_offset;
	uint64_t                      footer_size;
	uint64_t                      footer_magic;
	std::vector<RowgroupMetadata> rowgroups;

	bool operator==(const DescriptorSnapshot&) const = default;
};

DescriptorSnapshot descriptor_snapshot(const std::filesystem::path& fls_path) {
	fastlanes::FileHeader header {};
	fastlanes::FileFooter footer {};
	if (!fastlanes::FileHeader::Load(header, fls_path).success ||
	    !fastlanes::FileFooter::Load(footer, fls_path).success) {
		throw std::runtime_error("failed to load encoded file header/footer");
	}

	fastlanes::up<fastlanes::TableDescriptorHandle> handle;
	if (static_cast<bool>(header.settings.inline_footer)) {
		handle =
		    fastlanes::make_table_descriptor(fls_path, footer.table_descriptor_offset, footer.table_descriptor_size);
	} else {
		handle = fastlanes::make_table_descriptor(fls_path.parent_path() / "table_descriptor.fbb");
	}
	const auto         native = handle->Unpack();
	DescriptorSnapshot result {
	    native->m_table_binary_size,
	    footer.table_descriptor_offset,
	    footer.table_descriptor_size,
	    footer.magic_bytes,
	    {},
	};
	result.rowgroups.reserve(native->m_rowgroup_descriptors.size());
	for (const auto& rowgroup : native->m_rowgroup_descriptors) {
		result.rowgroups.push_back({rowgroup->m_size, rowgroup->m_offset, rowgroup->m_n_tuples, rowgroup->m_n_vec});
	}
	return result;
}

struct MixedData {
	explicit MixedData(std::vector<fastlanes::n_t> rowgroup_sizes)
	    : rowgroups(std::move(rowgroup_sizes)) {
		const auto row_count =
		    static_cast<std::size_t>(std::accumulate(rowgroups.begin(), rowgroups.end(), fastlanes::n_t {0}));
		i8.resize(row_count);
		i16.resize(row_count);
		i32.resize(row_count);
		i64.resize(row_count);
		f32.resize(row_count);
		f64.resize(row_count);
		strings.resize(row_count);
		for (std::size_t row = 0; row < row_count; ++row) {
			i8[row]      = static_cast<int8_t>(static_cast<int>(row % 101U) - 50);
			i16[row]     = static_cast<int16_t>(static_cast<int>(row % 2003U) - 1001);
			i32[row]     = static_cast<int32_t>(row * 7919U) - 1000003;
			i64[row]     = static_cast<int64_t>(row) * 32452843 - 49979687;
			f32[row]     = static_cast<float>(static_cast<int>(row % 409U) - 204) * 0.25F;
			f64[row]     = static_cast<double>(static_cast<int>(row % 8191U) - 4095) * 0.125;
			strings[row] = "synthetic_" + std::to_string(row % 97U) + "_" + std::to_string(row % 11U);
		}

		columns = {
		    fastlanes::MemoryColumn {"i8", std::span<const int8_t>(i8)},
		    fastlanes::MemoryColumn {"i16", std::span<const int16_t>(i16)},
		    fastlanes::MemoryColumn {"i32", std::span<const int32_t>(i32)},
		    fastlanes::MemoryColumn {"i64", std::span<const int64_t>(i64)},
		    fastlanes::MemoryColumn {"f32", std::span<const float>(f32)},
		    fastlanes::MemoryColumn {"f64", std::span<const double>(f64)},
		    fastlanes::MemoryColumn {"string", std::span<const fastlanes::string>(strings)},
		};

		options.n_vectors_per_rowgroup = 2;
		options.rowgroup_n_tuples      = std::span<const fastlanes::n_t>(rowgroups);
		options.force_schema           = true;
		options.forced_schema          = {
		             fastlanes::OperatorToken::EXP_UNCOMPRESSED_I08,
		             fastlanes::OperatorToken::EXP_UNCOMPRESSED_I16,
		             fastlanes::OperatorToken::EXP_UNCOMPRESSED_I32,
		             fastlanes::OperatorToken::EXP_UNCOMPRESSED_I64,
		             fastlanes::OperatorToken::EXP_UNCOMPRESSED_FLT,
		             fastlanes::OperatorToken::EXP_UNCOMPRESSED_DBL,
		             fastlanes::OperatorToken::EXP_UNCOMPRESSED_STR,
        };
	}

	fastlanes::MemoryTable table() const {
		return {std::span<const fastlanes::MemoryColumn>(columns)};
	}

	std::vector<int8_t>                    i8;
	std::vector<int16_t>                   i16;
	std::vector<int32_t>                   i32;
	std::vector<int64_t>                   i64;
	std::vector<float>                     f32;
	std::vector<double>                    f64;
	std::vector<fastlanes::string>         strings;
	std::array<fastlanes::MemoryColumn, 7> columns;
	std::vector<fastlanes::n_t>            rowgroups;
	fastlanes::MemoryTableOptions          options;
};

fastlanes::EncodingStats write_and_verify(const std::filesystem::path&         path,
                                          const fastlanes::MemoryTable&        table,
                                          const fastlanes::MemoryTableOptions& table_options,
                                          const fastlanes::EncodingOptions&    encoding_options,
                                          const bool                           inline_footer) {
	std::filesystem::create_directories(path.parent_path());
	fastlanes::Config config;
	config.inline_footer = inline_footer ? fastlanes::FLS_TRUE : fastlanes::FLS_FALSE;
	fastlanes::Connection writer(config);
	writer.read_memory(table, table_options).to_fls(path, encoding_options);

	fastlanes::Connection reader_connection;
	auto                  reader     = reader_connection.read_fls(path);
	auto                  decoded    = reader->materialize();
	const auto            comparison = writer.get_table() == *decoded;
	EXPECT_TRUE(comparison.is_equal) << comparison.description;
	EXPECT_TRUE(writer.verify_fls(path).success);
	return writer.get_last_encoding_stats();
}

fastlanes::EncodingOptions
encoding_options(const fastlanes::n_t workers, const fastlanes::n_t window = 0, const fastlanes::n_t bytes = 0) {
	fastlanes::EncodingOptions options;
	options.worker_count             = workers;
	options.max_inflight_rowgroups   = window;
	options.max_pending_output_bytes = bytes;
	return options;
}

TEST(ParallelEncoder, MixedTypesAreByteIdenticalAcrossWorkerCountsAndRepeatedRuns) {
	TemporaryDirectory temp("parallel_encoder_mixed");
	MixedData          data({1024, 1537, 17, 900, 2048, 81, 777, 1301, 33});

	const auto serial_path = temp.path / "serial" / "data.fls";
	write_and_verify(serial_path, data.table(), data.options, encoding_options(1), true);
	const auto serial_bytes    = read_bytes(serial_path);
	const auto serial_snapshot = descriptor_snapshot(serial_path);

	for (const fastlanes::n_t workers : std::array<fastlanes::n_t, 4> {
	         fastlanes::n_t {1}, fastlanes::n_t {2}, fastlanes::n_t {4}, fastlanes::n_t {8}}) {
		const auto parallel_path = temp.path / ("workers_" + std::to_string(workers)) / "data.fls";
		const auto stats = write_and_verify(parallel_path, data.table(), data.options, encoding_options(workers), true);
		const auto parallel_bytes = read_bytes(parallel_path);
		EXPECT_EQ(parallel_bytes, serial_bytes);
		EXPECT_EQ(std::filesystem::file_size(parallel_path), std::filesystem::file_size(serial_path));
		EXPECT_EQ(descriptor_snapshot(parallel_path), serial_snapshot);
		EXPECT_EQ(stats.encoded_rowgroups, data.rowgroups.size());
		EXPECT_LE(stats.effective_worker_count, workers);
	}

	for (int repeat = 0; repeat < 5; ++repeat) {
		const auto repeat_path = temp.path / ("repeat_" + std::to_string(repeat)) / "data.fls";
		write_and_verify(repeat_path, data.table(), data.options, encoding_options(4, 5), true);
		const auto repeat_bytes = read_bytes(repeat_path);
		EXPECT_EQ(repeat_bytes, serial_bytes);
		EXPECT_EQ(descriptor_snapshot(repeat_path), serial_snapshot);
	}
}

TEST(ParallelEncoder, CompressedFloatingPointIsByteIdenticalAcrossWorkers) {
	TemporaryDirectory                temp("parallel_encoder_alp");
	const std::vector<fastlanes::n_t> rowgroups {2048, 1537, 4096, 901, 3073, 17, 1900, 1024};
	const auto                        row_count =
	    static_cast<std::size_t>(std::accumulate(rowgroups.begin(), rowgroups.end(), fastlanes::n_t {0}));
	std::vector<float>  float_values(row_count);
	std::vector<double> double_values(row_count);
	for (std::size_t row = 0; row < row_count; ++row) {
		float_values[row]  = static_cast<float>(static_cast<int64_t>(row % 8191U) - 4095) * 0.125F;
		double_values[row] = static_cast<double>(static_cast<int64_t>(row % 16381U) - 8190) * 0.0625;
	}
	const std::array<fastlanes::MemoryColumn, 2> columns {
	    fastlanes::MemoryColumn {"float", std::span<const float>(float_values)},
	    fastlanes::MemoryColumn {"double", std::span<const double>(double_values)},
	};
	fastlanes::MemoryTableOptions table_options;
	table_options.n_vectors_per_rowgroup = 4;
	table_options.rowgroup_n_tuples      = rowgroups;
	table_options.force_schema           = true;
	table_options.forced_schema = {fastlanes::OperatorToken::EXP_ALP_FLT, fastlanes::OperatorToken::EXP_ALP_DBL};
	const fastlanes::MemoryTable table {columns};

	const auto serial_path = temp.path / "serial" / "data.fls";
	write_and_verify(serial_path, table, table_options, encoding_options(1), true);
	const auto serial_bytes = read_bytes(serial_path);

	for (const fastlanes::n_t workers : std::array<fastlanes::n_t, 3> {2, 4, 8}) {
		for (int repeat = 0; repeat < 3; ++repeat) {
			const auto output =
			    temp.path / ("workers_" + std::to_string(workers) + "_repeat_" + std::to_string(repeat)) / "data.fls";
			write_and_verify(output, table, table_options, encoding_options(workers, 8), true);
			const auto bytes = read_bytes(output);
			EXPECT_EQ(bytes, serial_bytes);
		}
	}
}

TEST(ParallelEncoder, AutomaticWizardMatchesSerialAndOriginalInput) {
	TemporaryDirectory temp("parallel_wizard");
	MixedData          data({1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 137});
	data.options.force_schema = false;
	data.options.forced_schema.clear();
	std::fill(data.i8.begin(), data.i8.end(), 7); // constant
	for (std::size_t row = 0; row < data.i16.size(); ++row) {
		data.i16[row] = static_cast<int16_t>(row % 5); // low cardinality
	}
	std::vector<fastlanes::MemoryColumn> columns(data.columns.begin(), data.columns.end());
	columns.push_back({"equal_i32", std::span<const int32_t>(data.i32)});
	const fastlanes::MemoryTable input {columns};
	std::vector<uint8_t>         serial_bytes;
	std::vector<uint8_t>         serial_descriptor;
	for (const fastlanes::n_t workers : std::array<fastlanes::n_t, 4> {1, 2, 4, 8}) {
		for (int repeat = 0; repeat < 2; ++repeat) {
			SCOPED_TRACE("workers=" + std::to_string(workers) + " repeat=" + std::to_string(repeat));
			const auto directory = temp.path / (std::to_string(workers) + "_" + std::to_string(repeat));
			std::filesystem::create_directory(directory);
			const auto        output = directory / "data.fls";
			fastlanes::Config config;
			config.inline_footer = fastlanes::FLS_FALSE;
			fastlanes::Connection writer(config);
			writer.read_memory(input, data.options).to_fls(output, encoding_options(workers));
			fastlanes::Connection reader;
			auto                  decoded = reader.read_fls(output)->materialize();
			std::size_t           offset  = 0;
			for (const auto& rowgroup : decoded->m_rowgroups) {
				const auto& cols          = rowgroup->internal_rowgroup;
				std::size_t string_offset = 0;
				for (std::size_t row = 0; row < rowgroup->m_descriptor.m_n_tuples; ++row) {
					// Compare directly to source arrays, never to the prepared writer table.
					const auto check_number = [&](std::size_t col, auto expected) {
						std::visit(
						    [&](const auto& value) {
							    if constexpr (requires { value->data[row]; }) {
								    using Value = std::decay_t<decltype(value->data[row])>;
								    if constexpr (std::is_arithmetic_v<Value>) {
									    EXPECT_EQ(value->data[row], expected);
								    } else {
									    ADD_FAILURE() << "unexpected non-numeric column";
								    }
							    } else {
								    ADD_FAILURE() << "unexpected column type";
							    }
						    },
						    cols[col]);
					};
					check_number(0, data.i8[offset + row]);
					check_number(1, data.i16[offset + row]);
					check_number(2, data.i32[offset + row]);
					check_number(3, data.i64[offset + row]);
					check_number(4, data.f32[offset + row]);
					check_number(5, data.f64[offset + row]);
					check_number(7, data.i32[offset + row]);
					const auto& strings = *std::get<fastlanes::up<fastlanes::FLSStrColumn>>(cols[6]);
					EXPECT_EQ(std::string(reinterpret_cast<const char*>(strings.byte_arr.data() + string_offset),
					                      strings.length_arr[row]),
					          data.strings[offset + row]);
					string_offset += strings.length_arr[row];
				}
				offset += rowgroup->m_descriptor.m_n_tuples;
			}
			EXPECT_EQ(offset, data.i8.size());
			const auto bytes      = read_bytes(output);
			const auto descriptor = read_bytes(directory / "table_descriptor.fbb");
			if (serial_bytes.empty()) {
				serial_bytes      = bytes;
				serial_descriptor = descriptor;
			} else {
				EXPECT_EQ(bytes, serial_bytes);
				// Includes every selected operator/operand, type and offset.
				EXPECT_EQ(descriptor, serial_descriptor);
			}
		}
	}
}

TEST(ParallelEncoder, DefaultApiMatchesExplicitSerialOptions) {
	TemporaryDirectory   temp("parallel_encoder_default_api");
	std::vector<int32_t> values(4096U + 137U);
	for (std::size_t idx = 0; idx < values.size(); ++idx) {
		values[idx] = static_cast<int32_t>((idx * 7919U) % 65521U);
	}
	const std::array<fastlanes::MemoryColumn, 1> columns {
	    fastlanes::MemoryColumn {"value", std::span<const int32_t>(values)},
	};
	const std::array<fastlanes::n_t, 3> rowgroups {2048, 2048, 137};
	fastlanes::MemoryTableOptions       table_options;
	table_options.n_vectors_per_rowgroup = 2;
	table_options.rowgroup_n_tuples      = rowgroups;
	table_options.force_schema           = true;
	table_options.forced_schema          = {fastlanes::OperatorToken::EXP_FFOR_I32};
	const fastlanes::MemoryTable table {columns};

	const auto default_path  = temp.path / "default" / "data.fls";
	const auto explicit_path = temp.path / "explicit" / "data.fls";
	std::filesystem::create_directories(default_path.parent_path());
	std::filesystem::create_directories(explicit_path.parent_path());
	fastlanes::Config config;
	config.inline_footer = fastlanes::FLS_TRUE;
	fastlanes::Connection default_writer(config);
	default_writer.read_memory(table, table_options).to_fls(default_path);
	write_and_verify(explicit_path, table, table_options, encoding_options(1), true);

	EXPECT_EQ(read_bytes(default_path), read_bytes(explicit_path));
	EXPECT_EQ(descriptor_snapshot(default_path), descriptor_snapshot(explicit_path));
	EXPECT_EQ(default_writer.get_last_encoding_stats().effective_worker_count, 1U);
}

TEST(ParallelEncoder, ExternalFooterAndRowgroupOffsetsAreByteIdentical) {
	TemporaryDirectory temp("parallel_encoder_external_footer");
	MixedData          data({1024, 333, 2048, 7, 901});
	const auto         serial_path   = temp.path / "serial" / "data.fls";
	const auto         parallel_path = temp.path / "parallel" / "data.fls";

	write_and_verify(serial_path, data.table(), data.options, encoding_options(1), false);
	write_and_verify(parallel_path, data.table(), data.options, encoding_options(8, 3), false);

	EXPECT_EQ(read_bytes(parallel_path), read_bytes(serial_path));
	EXPECT_EQ(read_bytes(parallel_path.parent_path() / "table_descriptor.fbb"),
	          read_bytes(serial_path.parent_path() / "table_descriptor.fbb"));
	EXPECT_EQ(descriptor_snapshot(parallel_path), descriptor_snapshot(serial_path));
}

TEST(ReadSourceTest, MaterializesInlineAndExternalDescriptorsFromMemory) {
	TemporaryDirectory temp("memory_source");
	MixedData          data({1024, 17});
	for (bool inlined : {true, false}) {
		const auto dir = temp.path / (inlined ? "inline" : "external");
		std::filesystem::create_directory(dir);
		const auto        path = dir / "data.fls";
		fastlanes::Config config;
		config.inline_footer = inlined ? fastlanes::FLS_TRUE : fastlanes::FLS_FALSE;
		fastlanes::Connection writer(config);
		writer.read_memory(data.table(), data.options).to_fls(path);
		const auto                  bytes = read_bytes(path);
		const fastlanes::ReadSource source(std::as_bytes(std::span(bytes)));
		fastlanes::Connection       reader;
		ASSERT_TRUE(reader.verify_fls(source).success);
		fastlanes::up<fastlanes::TableReader> table;
		if (inlined) {
			table = reader.read_fls(source);
		} else {
			EXPECT_THROW(reader.read_fls(source), std::invalid_argument);
			const auto metadata = read_bytes(dir / "table_descriptor.fbb");
			table               = reader.read_fls(source, fastlanes::ReadSource(std::as_bytes(std::span(metadata))));
		}
		std::filesystem::remove_all(dir);
		EXPECT_TRUE((writer.get_table() == *table->materialize()).is_equal);
		auto rowgroup = table->get_rowgroup_reader(1);
		table.reset();
		EXPECT_TRUE((writer.get_table().GetRowgroup(1) == *rowgroup->materialize()).is_equal);
	}
}

TEST(StagedOutput, PreservesAndReportsBackupWhenRollbackIsBlocked) {
	TemporaryDirectory temp("staged_recovery");
	const auto         main = temp.path / "data.fls";
	const auto         json = temp.path / "data.fls.json";
	std::ofstream(main) << "existing main";
	std::ofstream(json) << "old sidecar";
	std::filesystem::path retained;
	{
		fastlanes::detail::StagedOutput staged(main);
		// A nonempty directory at the published sidecar path makes removal fail on
		// every platform, without depending on permissions, races or open handles.
		auto staged_json = staged.file_path();
		staged_json += ".json";
		ASSERT_TRUE(std::filesystem::create_directory(staged_json));
		std::ofstream(staged_json / "occupied") << "keep";
		try {
			staged.publish(false, true);
			FAIL() << "expected a rollback error";
		} catch (const std::filesystem::filesystem_error& err) {
			retained = err.path1();
			EXPECT_EQ(err.path2(), json);
			EXPECT_FALSE(retained.empty());
			EXPECT_NE(std::string(err.what()).find(retained.string()), std::string::npos);
		}
	}
	const auto backup = read_bytes(retained);
	EXPECT_EQ(std::string(backup.begin(), backup.end()), "old sidecar");
	const auto previous = read_bytes(main);
	EXPECT_EQ(std::string(previous.begin(), previous.end()), "existing main");
}

TEST(StagedOutput, RestoresSidecarAndRemovesStagingAfterOrdinaryPublishFailure) {
	TemporaryDirectory temp("staged_rollback");
	const auto         main = temp.path / "data.fls";
	const auto         json = temp.path / "data.fls.json";
	std::ofstream(json) << "previous";
	std::filesystem::path stage;
	{
		fastlanes::detail::StagedOutput output(main);
		stage            = output.file_path().parent_path().parent_path();
		auto staged_json = output.file_path();
		staged_json += ".json";
		std::ofstream(staged_json) << "replacement";
		// The missing main file fails after the sidecar has been published.
		EXPECT_THROW(output.publish(false, true), std::filesystem::filesystem_error);
	}
	const auto restored = read_bytes(json);
	EXPECT_EQ(std::string(restored.begin(), restored.end()), "previous");
	EXPECT_FALSE(std::filesystem::exists(stage));
	EXPECT_FALSE(std::filesystem::exists(main));
}

TEST(ParallelEncoder, SingleRowgroupAndWorkerOversubscriptionAreSafe) {
	TemporaryDirectory   temp("parallel_encoder_single");
	std::vector<int16_t> values(fastlanes::CFG::VEC_SZ + 19U);
	for (std::size_t idx = 0; idx < values.size(); ++idx) {
		values[idx] = static_cast<int16_t>(static_cast<int>(idx % 503U) - 251);
	}
	const std::array<fastlanes::MemoryColumn, 1> columns {
	    fastlanes::MemoryColumn {"value", std::span<const int16_t>(values)},
	};
	const std::array<fastlanes::n_t, 1> rowgroups {values.size()};
	fastlanes::MemoryTableOptions       table_options;
	table_options.n_vectors_per_rowgroup = 2;
	table_options.rowgroup_n_tuples      = rowgroups;
	table_options.force_schema           = true;
	table_options.forced_schema          = {fastlanes::OperatorToken::EXP_UNCOMPRESSED_I16};
	const fastlanes::MemoryTable table {columns};

	const auto serial_path   = temp.path / "serial" / "data.fls";
	const auto parallel_path = temp.path / "parallel" / "data.fls";
	write_and_verify(serial_path, table, table_options, encoding_options(1), true);
	const auto stats = write_and_verify(parallel_path, table, table_options, encoding_options(8, 8), true);

	EXPECT_EQ(read_bytes(parallel_path), read_bytes(serial_path));
	EXPECT_EQ(stats.effective_worker_count, 1U);
	EXPECT_EQ(stats.peak_inflight_rowgroups, 1U);
}

TEST(ParallelEncoder, InflightWindowAndCompletedPayloadBytesAreBounded) {
	TemporaryDirectory       temp("parallel_encoder_bounded");
	MixedData                data({1024, 511, 2048, 19, 777, 1301, 23, 900});
	constexpr fastlanes::n_t max_bytes = 4U * 1024U * 1024U;
	const auto               stats     = write_and_verify(
        temp.path / "bounded" / "data.fls", data.table(), data.options, encoding_options(8, 2, max_bytes), true);

	EXPECT_EQ(stats.resolved_inflight_rowgroups, 2U);
	EXPECT_LE(stats.peak_inflight_rowgroups, 2U);
	EXPECT_LE(stats.peak_pending_output_bytes, max_bytes);
}

TEST(ParallelEncoder, EmptyTableIsSafeWithMoreWorkersThanRowgroups) {
	TemporaryDirectory                           temp("parallel_encoder_empty");
	const std::vector<int32_t>                   values;
	const std::array<fastlanes::MemoryColumn, 1> columns {
	    fastlanes::MemoryColumn {"empty", std::span<const int32_t>(values)},
	};
	fastlanes::MemoryTableOptions table_options;
	table_options.force_schema  = true;
	table_options.forced_schema = {fastlanes::OperatorToken::EXP_UNCOMPRESSED_I32};
	const fastlanes::MemoryTable table {columns};

	const auto serial_path   = temp.path / "serial" / "data.fls";
	const auto parallel_path = temp.path / "parallel" / "data.fls";
	write_and_verify(serial_path, table, table_options, encoding_options(1), true);
	const auto stats = write_and_verify(parallel_path, table, table_options, encoding_options(8), true);

	EXPECT_EQ(read_bytes(parallel_path), read_bytes(serial_path));
	EXPECT_EQ(stats.encoded_rowgroups, 0U);
	EXPECT_EQ(stats.effective_worker_count, 0U);
}

TEST(ParallelEncoder, InvalidOptionsFailBeforePreparationOrOutput) {
	TemporaryDirectory                           temp("parallel_encoder_exception");
	std::vector<int32_t>                         values(fastlanes::CFG::VEC_SZ, 7);
	const std::array<fastlanes::MemoryColumn, 1> columns {
	    fastlanes::MemoryColumn {"value", std::span<const int32_t>(values)},
	};
	fastlanes::MemoryTableOptions table_options;
	table_options.force_schema  = true;
	table_options.forced_schema = {fastlanes::OperatorToken::EXP_UNCOMPRESSED_I32};

	fastlanes::Connection connection;
	connection.read_memory(fastlanes::MemoryTable {columns}, table_options);
	auto& column = *std::get<fastlanes::up<fastlanes::col_i32>>(
	    connection.get_table().m_rowgroups.front()->internal_rowgroup.front());
	const auto original_data = column.data;
	const auto output_path   = temp.path / "missing" / "data.fls";
	EXPECT_THROW(connection.to_fls(output_path, encoding_options(0)), std::invalid_argument);
	EXPECT_TRUE(std::filesystem::is_empty(temp.path));
	EXPECT_EQ(column.data, original_data);
}

TEST(ParallelPreparation, CancelsNewRowgroupsAndJoinsEveryWorker) {
	for (const char* stage : {"prepare_table", "Wizard::Spell"}) {
		std::barrier     entered(4);
		std::atomic<int> started {0};
		std::atomic<int> finished {0};
		try {
			fastlanes::detail::parallel_rowgroups(64, 4, stage, [&](fastlanes::n_t) {
				++started;
				entered.arrive_and_wait();
				++finished;
				throw std::runtime_error("injected rowgroup failure");
			});
			FAIL() << "expected failure";
		} catch (const std::runtime_error& error) {
			EXPECT_NE(std::string(error.what()).find(stage), std::string::npos);
			EXPECT_NE(std::string(error.what()).find("rowgroup"), std::string::npos);
			try {
				std::rethrow_if_nested(error);
				FAIL() << "missing original exception";
			} catch (const std::runtime_error& cause) { EXPECT_STREQ(cause.what(), "injected rowgroup failure"); }
		}
		EXPECT_EQ(started.load(), 4);
		EXPECT_EQ(finished.load(), 4);
	}
}

TEST(ParallelPreparation, FailedPreparationRequiresReload) {
	TemporaryDirectory    temp("prepare_failure");
	MixedData             data({1024, 1024, 17});
	fastlanes::Connection connection;
	auto                  automatic = data.options;
	automatic.force_schema          = false;
	connection.read_memory(data.table(), automatic);
	// Force preparation to attempt a numeric cast of a nonnumeric string.
	auto& columns = connection.get_table().GetRowgroup(0).internal_rowgroup;
	std::get<fastlanes::up<fastlanes::FLSStrColumn>>(columns.back())->m_stats.is_numeric = true;
	EXPECT_THROW(connection.to_fls(temp.path / "data.fls", encoding_options(2)), std::runtime_error);
	EXPECT_THROW(static_cast<void>(connection.get_table()), std::runtime_error);
	EXPECT_THROW(connection.to_fls(temp.path / "data.fls"), std::runtime_error);
	EXPECT_TRUE(std::filesystem::is_empty(temp.path));
	EXPECT_NO_THROW(connection.read_memory(data.table(), data.options).to_fls(temp.path / "data.fls"));
}

TEST(ParallelEncoder, OutputFailureJoinsWorkers) {
	TemporaryDirectory    temp("encoder_write_failure");
	MixedData             data({1024, 1024, 1024, 17});
	fastlanes::Connection connection;
	connection.read_memory(data.table(), data.options).to_fls(temp.path / "prepared.fls");
	// Isolate a writer failure after worker launch using already prepared input.
	EXPECT_THROW(static_cast<void>(fastlanes::Encoder::encode(connection, temp.path, encoding_options(4, 1))),
	             std::runtime_error);
}

} // namespace
