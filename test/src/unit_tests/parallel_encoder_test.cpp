#include "alp/config.hpp"
#include "alp/encoder.hpp"
#include "alp/state.hpp"
#include "fls/connection.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/file/file_header.hpp"
#include "fls/footer/table_descriptor.hpp"
#include "fls/table/memory_table.hpp"
#include <algorithm>
#include <array>
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

uint32_t rotate_right(const uint32_t value, const uint32_t bits) {
	return (value >> bits) | (value << (32U - bits));
}

std::array<uint8_t, 32> sha256(const std::vector<uint8_t>& input) {
	static constexpr std::array<uint32_t, 64> round_constants {
	    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
	    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
	    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
	    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
	    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
	    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
	    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
	    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
	};
	std::array<uint32_t, 8> state {
	    0x6a09e667U,
	    0xbb67ae85U,
	    0x3c6ef372U,
	    0xa54ff53aU,
	    0x510e527fU,
	    0x9b05688cU,
	    0x1f83d9abU,
	    0x5be0cd19U,
	};

	std::vector<uint8_t> message(input);
	const auto           bit_length = static_cast<uint64_t>(message.size()) * 8U;
	message.push_back(0x80U);
	while (message.size() % 64U != 56U) {
		message.push_back(0U);
	}
	for (int shift = 56; shift >= 0; shift -= 8) {
		message.push_back(static_cast<uint8_t>(bit_length >> static_cast<uint32_t>(shift)));
	}

	for (std::size_t offset = 0; offset < message.size(); offset += 64U) {
		std::array<uint32_t, 64> words {};
		for (std::size_t word_idx = 0; word_idx < 16U; ++word_idx) {
			const auto byte_idx = offset + word_idx * 4U;
			words[word_idx]     = (static_cast<uint32_t>(message[byte_idx]) << 24U) |
			                  (static_cast<uint32_t>(message[byte_idx + 1U]) << 16U) |
			                  (static_cast<uint32_t>(message[byte_idx + 2U]) << 8U) |
			                  static_cast<uint32_t>(message[byte_idx + 3U]);
		}
		for (std::size_t word_idx = 16; word_idx < words.size(); ++word_idx) {
			const auto x    = words[word_idx - 15U];
			const auto y    = words[word_idx - 2U];
			const auto s0   = rotate_right(x, 7U) ^ rotate_right(x, 18U) ^ (x >> 3U);
			const auto s1   = rotate_right(y, 17U) ^ rotate_right(y, 19U) ^ (y >> 10U);
			words[word_idx] = words[word_idx - 16U] + s0 + words[word_idx - 7U] + s1;
		}

		auto a = state[0];
		auto b = state[1];
		auto c = state[2];
		auto d = state[3];
		auto e = state[4];
		auto f = state[5];
		auto g = state[6];
		auto h = state[7];
		for (std::size_t round = 0; round < words.size(); ++round) {
			const auto sigma1   = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
			const auto choose   = (e & f) ^ ((~e) & g);
			const auto temp1    = h + sigma1 + choose + round_constants[round] + words[round];
			const auto sigma0   = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
			const auto majority = (a & b) ^ (a & c) ^ (b & c);
			const auto temp2    = sigma0 + majority;
			h                   = g;
			g                   = f;
			f                   = e;
			e                   = d + temp1;
			d                   = c;
			c                   = b;
			b                   = a;
			a                   = temp1 + temp2;
		}
		state[0] += a;
		state[1] += b;
		state[2] += c;
		state[3] += d;
		state[4] += e;
		state[5] += f;
		state[6] += g;
		state[7] += h;
	}

	std::array<uint8_t, 32> digest {};
	for (std::size_t word_idx = 0; word_idx < state.size(); ++word_idx) {
		for (std::size_t byte_idx = 0; byte_idx < 4U; ++byte_idx) {
			digest[word_idx * 4U + byte_idx] =
			    static_cast<uint8_t>(state[word_idx] >> static_cast<uint32_t>(24U - byte_idx * 8U));
		}
	}
	return digest;
}

TEST(ParallelEncoderHash, Sha256HelperMatchesTheStandardEmptyDigest) {
	const std::array<uint8_t, 32> expected {
	    0xe3U, 0xb0U, 0xc4U, 0x42U, 0x98U, 0xfcU, 0x1cU, 0x14U, 0x9aU, 0xfbU, 0xf4U, 0xc8U, 0x99U, 0x6fU, 0xb9U, 0x24U,
	    0x27U, 0xaeU, 0x41U, 0xe4U, 0x64U, 0x9bU, 0x93U, 0x4cU, 0xa4U, 0x95U, 0x99U, 0x1bU, 0x78U, 0x52U, 0xb8U, 0x55U,
	};
	EXPECT_EQ(sha256({}), expected);
}

TEST(AlpEncoderScratch, ExceptionPlaceholdersComeFromTheCurrentVector) {
	using Encoder = alp::encoder<float>;
	using State   = alp::state<float>;
	using Encoded = Encoder::ST;

	std::array<float, alp::config::VECTOR_SIZE>    input {};
	std::array<float, alp::config::VECTOR_SIZE>    exceptions {};
	std::array<uint16_t, alp::config::VECTOR_SIZE> exception_positions {};
	std::array<Encoded, alp::config::VECTOR_SIZE>  encoded {};
	State                                           state;
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
	options.worker_count           = workers;
	options.max_inflight_rowgroups = window;
	options.max_inflight_bytes     = bytes;
	return options;
}

TEST(ParallelEncoder, MixedTypesAreByteIdenticalAcrossWorkerCountsAndRepeatedRuns) {
	TemporaryDirectory temp("parallel_encoder_mixed");
	MixedData          data({1024, 1537, 17, 900, 2048, 81, 777, 1301, 33});

	const auto serial_path = temp.path / "serial" / "data.fls";
	write_and_verify(serial_path, data.table(), data.options, encoding_options(1), true);
	const auto serial_bytes    = read_bytes(serial_path);
	const auto serial_sha256   = sha256(serial_bytes);
	const auto serial_snapshot = descriptor_snapshot(serial_path);

	for (const fastlanes::n_t workers : std::array<fastlanes::n_t, 4> {
	         fastlanes::n_t {1}, fastlanes::n_t {2}, fastlanes::n_t {4}, fastlanes::n_t {8}}) {
		const auto parallel_path = temp.path / ("workers_" + std::to_string(workers)) / "data.fls";
		const auto stats = write_and_verify(parallel_path, data.table(), data.options, encoding_options(workers), true);
		const auto parallel_bytes = read_bytes(parallel_path);
		EXPECT_EQ(parallel_bytes, serial_bytes);
		EXPECT_EQ(sha256(parallel_bytes), serial_sha256);
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
		EXPECT_EQ(sha256(repeat_bytes), serial_sha256);
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
	const auto serial_bytes  = read_bytes(serial_path);
	const auto serial_digest = sha256(serial_bytes);

	for (const fastlanes::n_t workers : std::array<fastlanes::n_t, 3> {2, 4, 8}) {
		for (int repeat = 0; repeat < 3; ++repeat) {
			const auto output =
			    temp.path / ("workers_" + std::to_string(workers) + "_repeat_" + std::to_string(repeat)) / "data.fls";
			write_and_verify(output, table, table_options, encoding_options(workers, 8), true);
			const auto bytes = read_bytes(output);
			EXPECT_EQ(bytes, serial_bytes);
			EXPECT_EQ(sha256(bytes), serial_digest);
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
	EXPECT_LE(stats.peak_inflight_bytes, max_bytes);
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

TEST(ParallelEncoder, RejectsUnorderedCommitAndCleansStagedOutput) {
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
	fastlanes::EncodingOptions options;
	options.worker_count                 = 4;
	options.deterministic_ordered_commit = false;
	const auto output_path               = temp.path / "data.fls";
	EXPECT_THROW(connection.to_fls(output_path, options), std::invalid_argument);
	EXPECT_FALSE(std::filesystem::exists(output_path));

	for (const auto& entry : std::filesystem::directory_iterator(temp.path)) {
		EXPECT_FALSE(entry.path().filename().string().starts_with(".fastlanes-stage-"));
	}
}

} // namespace
