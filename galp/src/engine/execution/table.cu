// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/execution/table.cu
// ────────────────────────────────────────────────────────
#include "engine/execution/internal/table_pipeline.cuh"
#include "engine/execution/table.cuh"

namespace dispatch {

TableData decompress_table(const std::filesystem::path& fls_path, const TableDecompressionConfig& cfg) {
	return decompress_table(
	    fls_path,
	    cfg,
	    [](size_t) { return true; },
	    [](size_t, reader::Rowgroup&, const std::vector<expr::Expression>&, const RowgroupData&) {});
}

TableData decompress_table(const std::filesystem::path&    fls_path,
                           const TableDecompressionConfig& cfg,
                           const std::optional<size_t>&    rowgroup) {
	runtime::TableExecutionRequest request {};
	request.config              = cfg;
	request.samples             = 1;
	request.rowgroup            = rowgroup;
	request.materialize_results = true;
	request.warmup_first_run    = false;
	runtime::NoopTableExecutionObserver observer {};

	return runtime::execute_table_pipeline(
	    fls_path,
	    request,
	    [](size_t) { return true; },
	    [](size_t, reader::Rowgroup&, const std::vector<expr::Expression>&, const RowgroupData*) {},
	    observer);
}

TableData decompress_table(const std::filesystem::path&    fls_path,
                           const TableDecompressionConfig& cfg,
                           const TableRowgroupPredicate&   should_decompress,
                           const TableRowgroupCallback&    on_rowgroup) {
	runtime::TableExecutionRequest request {};
	request.config              = cfg;
	request.samples             = 1;
	request.materialize_results = true;
	request.warmup_first_run    = false;
	runtime::NoopTableExecutionObserver observer {};

	return runtime::execute_table_pipeline(
	    fls_path,
	    request,
	    should_decompress,
	    [&](const size_t                        rowgroup_index,
	        reader::Rowgroup&                   rowgroup,
	        const std::vector<expr::Expression>& expressions,
	        const RowgroupData*                materialized) {
		    if (materialized == nullptr) {
			    throw std::runtime_error("table pipeline did not materialize rowgroup output");
		    }
		    on_rowgroup(rowgroup_index, rowgroup, expressions, *materialized);
	    },
	    observer);
}

} // namespace dispatch
