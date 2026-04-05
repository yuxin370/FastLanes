// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/execution/rowgroup.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_ROWGROUP_CUH
#define ENGINE_EXECUTION_ROWGROUP_CUH

#include "engine/execution/common.cuh"
#include "engine/reader.cuh"
#include "flsgpu/structs.cuh"

namespace dispatch {

inline void free_rowgroup(reader::Rowgroup& rowgroup) {
	for (auto& col : rowgroup.columns) {
		if (col.alias_of.has_value()) {
			// Alias columns (e.g. EXP_EQUAL) share storage with source columns.
			// Skip freeing to avoid double-free.
			continue;
		}
		if (rowgroup.backing_storage && col.host_owned_by_backing) {
			// Zero-copy-backed columns are released via rowgroup.backing_storage lifetime.
			continue;
		}
		std::visit([](auto& host_col) { flsgpu::host::free_column(host_col); }, col.host);
	}
	rowgroup.columns.clear();
	if (rowgroup.backing_storage) {
		rowgroup.backing_storage.reset();
	}
}

RowgroupData decompress_rowgroup(std::vector<expr::Expression>& expressions, const ExecutionConfig& cfg = {});
RowgroupData decompress_rowgroup(const std::vector<expr::Expression>& expressions, const ExecutionConfig& cfg = {});

} // namespace dispatch

#endif // ENGINE_EXECUTION_ROWGROUP_CUH
