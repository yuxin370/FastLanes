#include "fls/connection.hpp"
#include "fls/table/memory_table.hpp"
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

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

} // namespace
