// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// fastlens/src/engine/operators/rowgroup.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_ROWGROUP_CUH
#define ENGINE_EXECUTION_ROWGROUP_CUH

#include "codecs/encodings/all.cuh"
#include "core/data/model.cuh"
#include "core/expression.cuh"
#include "engine/config.cuh"
#include "format/reader.cuh"

namespace fastlens::execution {

inline void free_rowgroup(fastlens::format::Rowgroup& rowgroup) {
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
		std::visit([](auto& host_col) { fastlens::codec::host::free_column(host_col); }, col.host);
	}
	rowgroup.columns.clear();
	if (rowgroup.backing_storage) {
		rowgroup.backing_storage.reset();
	}
	rowgroup.materialized_column_indices.clear();
	rowgroup.packed_device_payload.reset();
}

RowgroupData decompress_rowgroup(std::vector<fastlens::expression::Expression>& expressions, const ExecutionConfig& cfg = {});
RowgroupData decompress_rowgroup(const std::vector<fastlens::expression::Expression>& expressions, const ExecutionConfig& cfg = {});

} // namespace fastlens::execution

#endif // ENGINE_EXECUTION_ROWGROUP_CUH
