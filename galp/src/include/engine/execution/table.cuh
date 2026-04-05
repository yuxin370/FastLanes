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
