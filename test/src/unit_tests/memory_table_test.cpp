#include "fls/connection.hpp"
#include "fls/reader/rowgroup_reader.hpp"
#include "fls/reader/table_reader.hpp"
#include "fls/table/memory_table.hpp"
#include "fls/table/rowgroup.hpp"
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

TEST(MemoryTable, RejectsUnsafeUint64UnderDefaultCast) {
	const std::array<uint64_t, 2> values {
	    0,
	    static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1U,
	};
	const std::array<fastlanes::MemoryColumn, 1> columns {
	    fastlanes::MemoryColumn {
	        "u64",
	        std::span<const uint64_t> {values},
	    },
	};

	fastlanes::Connection        connection;
	const fastlanes::MemoryTable table {std::span<const fastlanes::MemoryColumn> {columns}};
	EXPECT_THROW(connection.read_memory(table), std::runtime_error);
}

TEST(MemoryTable, AcceptsUint16Columns) {
	const std::array<uint16_t, 3>                values {0, 1024, 65535};
	const std::array<fastlanes::MemoryColumn, 1> columns {
	    fastlanes::MemoryColumn {
	        "u16",
	        std::span<const uint16_t> {values},
	    },
	};

	fastlanes::Connection        connection;
	const fastlanes::MemoryTable table {std::span<const fastlanes::MemoryColumn> {columns}};
	EXPECT_NO_THROW(connection.read_memory(table));
}

TEST(MemoryTable, UsesExplicitRowgroupTupleCounts) {
	const std::array<int32_t, 5>                 values {1, 2, 3, 4, 5};
	const std::array<fastlanes::MemoryColumn, 1> columns {
	    fastlanes::MemoryColumn {
	        "i32",
	        std::span<const int32_t> {values},
	    },
	};
	const std::array<fastlanes::n_t, 3> rowgroups {2, 1, 2};
	fastlanes::MemoryTableOptions       options;
	options.rowgroup_n_tuples = std::span<const fastlanes::n_t> {rowgroups};

	fastlanes::Connection connection;
	connection.read_memory(fastlanes::MemoryTable {std::span<const fastlanes::MemoryColumn> {columns}}, options);

	const auto& table = connection.get_table();
	ASSERT_EQ(table.m_rowgroups.size(), rowgroups.size());
	for (size_t rowgroup_idx = 0; rowgroup_idx < rowgroups.size(); ++rowgroup_idx) {
		EXPECT_EQ(table.m_rowgroups[rowgroup_idx]->m_descriptor.m_n_tuples, rowgroups[rowgroup_idx]);
	}
}

TEST(MemoryTable, RejectsExplicitRowgroupTupleCountMismatch) {
	const std::array<int32_t, 3>                 values {1, 2, 3};
	const std::array<fastlanes::MemoryColumn, 1> columns {
	    fastlanes::MemoryColumn {
	        "i32",
	        std::span<const int32_t> {values},
	    },
	};
	const std::array<fastlanes::n_t, 2> rowgroups {1, 1};
	fastlanes::MemoryTableOptions       options;
	options.rowgroup_n_tuples = std::span<const fastlanes::n_t> {rowgroups};

	fastlanes::Connection connection;
	EXPECT_THROW(
	    connection.read_memory(fastlanes::MemoryTable {std::span<const fastlanes::MemoryColumn> {columns}}, options),
	    std::runtime_error);
}

TEST(MemoryTable, ClearsForcedSchemaBetweenLoads) {
	const std::array<int64_t, 2>                 first_values {1, 2};
	const std::array<fastlanes::MemoryColumn, 1> first_columns {
	    fastlanes::MemoryColumn {
	        "first",
	        std::span<const int64_t> {first_values},
	    },
	};
	fastlanes::MemoryTableOptions forced_options;
	forced_options.force_schema = true;
	forced_options.forced_schema.push_back(fastlanes::OperatorToken::EXP_UNCOMPRESSED_I64);

	fastlanes::Connection connection;
	connection.force_schema_pool({fastlanes::OperatorToken::EXP_UNCOMPRESSED_I64});
	ASSERT_TRUE(connection.is_forced_schema_pool());
	ASSERT_EQ(connection.get_forced_schema_pool().size(), 1);

	connection.read_memory(fastlanes::MemoryTable {std::span<const fastlanes::MemoryColumn> {first_columns}},
	                       forced_options);
	ASSERT_TRUE(connection.is_forced_schema());
	ASSERT_EQ(connection.get_forced_schema().size(), 1);
	ASSERT_FALSE(connection.is_forced_schema_pool());
	ASSERT_TRUE(connection.get_forced_schema_pool().empty());

	const std::array<int64_t, 2>                 second_values {3, 4};
	const std::array<int32_t, 2>                 third_values {5, 6};
	const std::array<fastlanes::MemoryColumn, 2> second_columns {
	    fastlanes::MemoryColumn {
	        "second",
	        std::span<const int64_t> {second_values},
	    },
	    fastlanes::MemoryColumn {
	        "third",
	        std::span<const int32_t> {third_values},
	    },
	};

	connection.read_memory(fastlanes::MemoryTable {std::span<const fastlanes::MemoryColumn> {second_columns}});

	EXPECT_FALSE(connection.is_forced_schema());
	EXPECT_TRUE(connection.get_forced_schema().empty());
	EXPECT_FALSE(connection.is_forced_schema_pool());
	EXPECT_TRUE(connection.get_forced_schema_pool().empty());
}

TEST(MemoryTable, MissingNullMapRemainsValidBeyondLegacy64VectorBoundary) {
	constexpr size_t row_count = 65U * fastlanes::CFG::VEC_SZ + 17U;
	std::vector<int16_t> values(row_count);
	for (size_t row = 0; row < values.size(); ++row) {
		values[row] = static_cast<int16_t>(static_cast<int>((row * 17U) % 257U) - 128);
	}
	const std::array<fastlanes::MemoryColumn, 1> columns {
	    fastlanes::MemoryColumn {
	        "i16",
	        std::span<const int16_t> {values},
	    },
	};
	const std::array<fastlanes::n_t, 1> rowgroups {row_count};
	fastlanes::MemoryTableOptions       options;
	options.n_vectors_per_rowgroup = 66;
	options.rowgroup_n_tuples      = std::span<const fastlanes::n_t> {rowgroups};
	options.force_schema           = true;
	options.forced_schema.push_back(fastlanes::OperatorToken::EXP_FFOR_I16);

	fastlanes::Connection writer;
	writer.read_memory(fastlanes::MemoryTable {std::span<const fastlanes::MemoryColumn> {columns}}, options);
	ASSERT_EQ(writer.get_table().m_rowgroups.size(), 1U);
	const auto& encoded_rowgroup = *writer.get_table().m_rowgroups.front();
	fastlanes::NullMapView null_map(encoded_rowgroup.internal_rowgroup.front());
	null_map.PointTo(0);
	const auto* first_zero_vector = null_map.NullMap();
	null_map.PointTo(64);
	const auto* boundary_zero_vector = null_map.NullMap();
	EXPECT_EQ(boundary_zero_vector, first_zero_vector);
	for (size_t row = 0; row < fastlanes::CFG::VEC_SZ; ++row) {
		ASSERT_EQ(boundary_zero_vector[row], 0U);
	}

	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto path =
	    std::filesystem::temp_directory_path() / ("fastlanes_memory_table_large_rowgroup_" + std::to_string(suffix) + ".fls");
	writer.to_fls(path);

	fastlanes::Connection reader_connection;
	auto                  table_reader    = reader_connection.read_fls(path);
	auto                  rowgroup_reader = table_reader->get_rowgroup_reader(0);
	auto                  decoded         = rowgroup_reader->materialize();
	const auto& decoded_column = std::get<fastlanes::up<fastlanes::col_i16>>(decoded->internal_rowgroup.front());
	ASSERT_NE(decoded_column, nullptr);
	ASSERT_GE(decoded_column->data.size(), values.size());
	EXPECT_TRUE(std::equal(values.begin(), values.end(), decoded_column->data.begin()));

	std::filesystem::remove(path);
}

} // namespace
