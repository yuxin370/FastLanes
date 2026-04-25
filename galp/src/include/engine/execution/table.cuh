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
	ExecutionConfig         execution           = {};
	TableDecompressionScope scope               = TableDecompressionScope::PerRowgroup;
	bool                    use_zero_copy_parse = true;
	bool                    enable_streaming            = true;
	bool                    enable_rowgroup_prefetch    = true;
	size_t                  prefetch_depth              = 2;
	size_t                  prefetch_workers            = 2;
	size_t                  streaming_target_work_items = 1u << 18;
	size_t                  streaming_target_rowgroups  = 8;
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
