// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/io/to_csv.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_IO_TO_CSV_CUH
#define ENGINE_IO_TO_CSV_CUH

#include "engine/execution/common.cuh"
#include "engine/execution/rowgroup.cuh"
#include "engine/expression.cuh"
#include "engine/reader.cuh"
#include <ostream>

namespace io {

void read_rowgroups_to_csv(reader::reader&         rdr,
                           std::ostream&           out,
                           size_t                  start_rowgroup,
                           size_t                  end_rowgroup,
                           bool                    write_header,
                           const dispatch::Config& decode_cfg);

} // namespace io

#endif // ENGINE_IO_TO_CSV_CUH
