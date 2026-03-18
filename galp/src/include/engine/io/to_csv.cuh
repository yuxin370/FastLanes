// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/io/to_csv.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_IO_TO_CSV_CUH
#define ENGINE_IO_TO_CSV_CUH

#include "engine/execution/table.cuh"
#include "engine/expression.cuh"
#include <filesystem>
#include <optional>
#include <ostream>

namespace io {

void read_table_to_csv(const std::filesystem::path&              fls_path,
                       std::ostream&                             out,
                       bool                                      write_header,
                       const dispatch::TableDecompressionConfig& table_cfg = {},
                       const std::optional<size_t>&              rowgroup  = std::nullopt);

} // namespace io

#endif // ENGINE_IO_TO_CSV_CUH
