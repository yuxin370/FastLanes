// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/operators/rowgroup.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_ROWGROUP_CUH
#define ENGINE_EXECUTION_ROWGROUP_CUH

#include "codecs/encodings/all.cuh"
#include "core/data/model.cuh"
#include "engine/config.cuh"
#include "format/reader.cuh"

namespace galp::execution {

inline void free_rowgroup(galp::format::Rowgroup& rowgroup) {
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
		std::visit([](auto& host_col) { galp::codec::host::free_column(host_col); }, col.host);
	}
	rowgroup.columns.clear();
	if (rowgroup.backing_storage) {
		rowgroup.backing_storage.reset();
	}
}

RowgroupData decompress_rowgroup(std::vector<galp::expression::Expression>& expressions, const ExecutionConfig& cfg = {});
RowgroupData decompress_rowgroup(const std::vector<galp::expression::Expression>& expressions, const ExecutionConfig& cfg = {});

} // namespace galp::execution

#endif // ENGINE_EXECUTION_ROWGROUP_CUH
