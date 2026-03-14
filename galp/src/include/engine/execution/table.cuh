// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/execution/table.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_TABLE_CUH
#define ENGINE_EXECUTION_TABLE_CUH

#include "engine/execution/common.cuh"
#include "engine/execution/rowgroup.cuh"
#include "engine/reader.cuh"
#include <filesystem>
#include <optional>
#include <utility>
#include <vector>

namespace dispatch {

// Unified benchmark workset used by both rowgroup and table orchestrators.
struct BenchmarkWorkset {
	using HostBatches   = typename dispatch::BatchSetFromList<dispatch::SupportedTypes>::type;
	using DeviceBatches = typename dispatch::DeviceBatchSetFromList<dispatch::SupportedTypes>::type;
	HostBatches                                    host_batches;
	DeviceBatches                                  device_batches;
	std::vector<dispatch::WorkItemAny>             work_items;
	std::optional<GPUArray<dispatch::WorkItemAny>> d_items;
	bool                                           freq_prefetch_all_branchless = false;
	bool                                           freq_hybrid_patcher          = false;
	float                                          freq_branchless_threshold     = 6.0f;
};

struct TableBenchmarkConfig {
	uint32_t              samples                      = 1;
	bool                  mega_kernel                  = true;
	bool                  gpu_dispatch_kernel          = false; // true: single mixed-type kernel per sample
	bool                  write_out                    = false; // true: write decompressed values to global output buffers
	bool                  freq_prefetch_all_branchless = false;
	bool                  freq_hybrid_patcher          = false;
	float                 freq_branchless_threshold    = 6.0f;
	std::optional<size_t> rowgroup;
};

struct TableBenchmarkResult {
	double end_to_end_ms = 0.0; // excludes teardown
	double kernel_ms     = 0.0;
	double setup_ms      = 0.0;
	double h2d_ms        = 0.0;
	double teardown_ms   = 0.0;

	size_t   total_launches    = 0;
	size_t   total_launch_grid = 0;
	size_t   total_columns     = 0;
	size_t   total_items       = 0;
	size_t   total_bytes       = 0;
	size_t   total_rgs         = 0;
	uint32_t samples           = 1;
};

// Append one rowgroup's expressions into a workset. Returns elapsed ms.
double append_expressions(BenchmarkWorkset&              workset,
                          std::vector<expr::Expression>& expressions,
                          size_t*                        out_total_bytes = nullptr,
                          size_t*                        out_n_exprs     = nullptr);

// Prepare dispatch-side device buffers (d_exprs + d_items). Returns elapsed ms.
double prepare_dispatch_buffers(BenchmarkWorkset& workset);

// Run kernels on the whole workset. Returns kernel ms.
// out_grid reports grid size per kernel launch.
// out_launches reports total kernel launches across all samples.
double run_kernel(BenchmarkWorkset& workset,
                  uint32_t          samples,
                  bool              gpu_dispatch_kernel = false,
                  bool              write_out           = false,
                  size_t*           out_grid            = nullptr,
                  size_t*           out_launches        = nullptr);

// Free device-side batches.
void free_batches(BenchmarkWorkset& workset);

// Benchmark one table (or a single rowgroup) with either mega-kernel or per-rowgroup mode.
TableBenchmarkResult benchmark_table(const std::filesystem::path& fls_path, const TableBenchmarkConfig& cfg = {});

template <typename RowgroupPredicate, typename RowgroupCallback>
inline TableData decompress_table(const std::filesystem::path& fls_path,
                                  const Config&                cfg,
                                  RowgroupPredicate&&          should_decompress,
                                  RowgroupCallback&&           on_rowgroup) {
	reader::reader rdr(fls_path);
	const size_t   n_rowgroups = rdr.rowgroup_count();
	TableData      table_data;

	for (size_t rg_idx = 0; rg_idx < n_rowgroups; ++rg_idx) {
		if (!std::forward<RowgroupPredicate>(should_decompress)(rg_idx)) {
			continue;
		}
		auto rowgroup    = rdr.read_rowgroup(rg_idx);
		auto expressions = expr::assemble(rowgroup);

		++table_data.rowgroups;
		table_data.total_columns += expressions.size();
		auto  exec_result = dispatch::execute_rowgroup(expressions, cfg, dispatch::ExecuteMode::Materialize);
		auto& result      = exec_result.materialized.value();
		std::forward<RowgroupCallback>(on_rowgroup)(rg_idx, rowgroup, expressions, result);
		free_rowgroup(rowgroup);
	}

	return table_data;
}

} // namespace dispatch

#endif // ENGINE_EXECUTION_TABLE_CUH
