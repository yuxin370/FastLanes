// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/execution/table.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_TABLE_CUH
#define ENGINE_EXECUTION_TABLE_CUH

#include "engine/data/model.cuh"
#include "engine/execution/common.cuh"
#include "engine/expression.cuh"
#include "engine/reader.cuh"
#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <vector>

namespace dispatch {

enum class TableDecompressionScope {
	PerRowgroup,
	WholeTable,
};

struct TableDecompressionConfig {
	ExecutionConfig         execution                = {};
	TableDecompressionScope scope                    = TableDecompressionScope::PerRowgroup;
	bool                    enable_rowgroup_prefetch = true;
	size_t                  prefetch_depth           = 4;
	// Zero selects a rowgroup-count based default in the table pipeline. The
	// current warm path favors more fused workers on large tables after
	// owner-affine buffer reuse, while small tables still avoid extra CPU/cache
	// contention.
	size_t prefetch_workers = 0;
	// Zero derives a byte budget from prefetch_depth * max rowgroup storage
	// bytes. Non-zero caps compressed bytes reserved by fused prefetch workers.
	size_t max_prefetch_storage_bytes   = 0;
	size_t streaming_target_work_items  = 1u << 18;
	size_t streaming_target_rowgroups   = 1;
};

using TableRowgroupPredicate = std::function<bool(size_t)>;
using TableRowgroupCallback =
    std::function<void(size_t, reader::Rowgroup&, const std::vector<expr::Expression>&, const RowgroupData&)>;

TableData decompress_table(const std::filesystem::path& fls_path, const TableDecompressionConfig& cfg = {});
TableData decompress_table(const std::filesystem::path&    fls_path,
                           const TableDecompressionConfig& cfg,
                           const std::optional<size_t>&    rowgroup);
TableData decompress_table(const std::filesystem::path&    fls_path,
                           const TableDecompressionConfig& cfg,
                           const TableRowgroupPredicate&   should_decompress,
                           const TableRowgroupCallback&    on_rowgroup);

} // namespace dispatch

#endif // ENGINE_EXECUTION_TABLE_CUH
