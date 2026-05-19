// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/include/fls/table/memory_table.hpp
// ────────────────────────────────────────────────────────
#ifndef FLS_TABLE_MEMORY_TABLE_HPP
#define FLS_TABLE_MEMORY_TABLE_HPP

#include "fls/api/api.hpp"
#include "fls/cfg/cfg.hpp"
#include "fls/common/alias.hpp"
#include "fls/footer/operator_token_generated.h"
#include "fls/std/filesystem.hpp"
#include "fls/std/string.hpp"
#include "fls/std/vector.hpp"
#include <cstdint>
#include <span>
#include <variant>

namespace fastlanes {

class Connection;

using MemoryColumnData = std::variant<std::span<const int8_t>,
                                      std::span<const int16_t>,
                                      std::span<const int32_t>,
                                      std::span<const int64_t>,
                                      std::span<const uint8_t>,
                                      std::span<const uint16_t>,
                                      std::span<const uint32_t>,
                                      std::span<const uint64_t>,
                                      std::span<const float>,
                                      std::span<const double>>;

struct MemoryColumn {
	string           name;
	MemoryColumnData data;
};

struct MemoryTable {
	std::span<const MemoryColumn> columns;
};

struct MemoryTableOptions {
	n_t  n_vectors_per_rowgroup = CFG::N_VEC_PER_RG;
	bool force_schema           = false;
	vector<OperatorToken> forced_schema {};
};

FLS_API void load_memory_table(Connection& connection, const MemoryTable& table, const MemoryTableOptions& options = {});

FLS_API void write_memory_table_to_fls(const MemoryTable&             table,
                                       const path&                   output_path,
                                       const MemoryTableOptions&     options = {});

} // namespace fastlanes

#endif // FLS_TABLE_MEMORY_TABLE_HPP
