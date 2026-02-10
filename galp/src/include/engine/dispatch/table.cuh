// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/dispatch/table.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_DISPATCH_TABLE_CUH
#define ENGINE_DISPATCH_TABLE_CUH

#include "engine/dispatch/common.cuh"
#include <optional>
#include <vector>

namespace dispatch::table {

struct TableBatches {
	using HostBatches   = typename dispatch::BatchSetFromList<dispatch::SupportedTypes>::type;
	using DeviceBatches = typename dispatch::DeviceBatchSetFromList<dispatch::SupportedTypes>::type;
	HostBatches                                    host_batches;
	DeviceBatches                                  device_batches;
	std::vector<dispatch::WorkItemAny>             work_items;
	std::optional<GPUArray<dispatch::WorkItemAny>> d_items;
};

// Append expressions (per-rowgroup) into table batches. Returns elapsed ms.
double append_expressions(TableBatches& table_batches, const std::vector<expr::Expression>& expressions);

// Finalize device buffers (d_exprs + d_items). Returns elapsed ms.
double finalize_batches(TableBatches& table_batches);

// Run mega-kernel on the whole table. Returns kernel ms.
double run_kernel(TableBatches& table_batches, uint32_t samples, size_t* out_grid = nullptr);

// Free device-side batches.
void free_batches(TableBatches& table_batches);

} // namespace dispatch::table

#endif // ENGINE_DISPATCH_TABLE_CUH
