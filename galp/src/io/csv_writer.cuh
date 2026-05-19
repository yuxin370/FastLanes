// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/io/csv_writer.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_IO_TO_CSV_CUH
#define ENGINE_IO_TO_CSV_CUH

#include "engine/table/table.cuh"
#include "core/expression.cuh"
#include <filesystem>
#include <optional>
#include <ostream>

namespace galp::io {

void read_table_to_csv(const std::filesystem::path&              fls_path,
                       std::ostream&                             out,
                       bool                                      write_header,
                       const galp::execution::TableDecompressionConfig& table_cfg = {},
                       const std::optional<size_t>&              rowgroup  = std::nullopt);

} // namespace galp::io

#endif // ENGINE_IO_TO_CSV_CUH
