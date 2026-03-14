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
		std::visit([](auto& host_col) { flsgpu::host::free_column(host_col); }, col.host);
	}
}

struct RowgroupExecuteResult {
	std::optional<RowgroupData>    materialized;
	std::optional<BenchmarkResult> benchmark;
};

RowgroupExecuteResult execute_rowgroup(std::vector<expr::Expression>& expressions,
                                       const Config&                  cfg  = {},
                                       const ExecuteMode              mode = ExecuteMode::Materialize);
RowgroupExecuteResult execute_rowgroup(const std::vector<expr::Expression>& expressions,
                                       const Config&                        cfg  = {},
                                       const ExecuteMode                    mode = ExecuteMode::Materialize);
RowgroupData          decompress_rowgroup(std::vector<expr::Expression>& expressions, const Config& cfg = {});
RowgroupData          decompress_rowgroup(const std::vector<expr::Expression>& expressions, const Config& cfg = {});
BenchmarkResult       benchmark_rowgroup(std::vector<expr::Expression>& expressions, const Config& cfg = {});
BenchmarkResult       benchmark_rowgroup(const std::vector<expr::Expression>& expressions, const Config& cfg = {});

} // namespace dispatch

#endif // ENGINE_EXECUTION_ROWGROUP_CUH
