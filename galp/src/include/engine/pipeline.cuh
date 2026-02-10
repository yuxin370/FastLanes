// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/pipeline.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_PIPELINE_CUH
#define ENGINE_PIPELINE_CUH

#include "engine/dispatch/column.cuh"
#include "engine/dispatch/rowgroup.cuh"
#include "engine/expression.cuh"
#include "engine/reader.cuh"
#include "flsgpu/structs.cuh"
#include <utility>

namespace pipeline {

inline void free_rowgroup(reader::Rowgroup& rowgroup) {
	for (auto& col : rowgroup.columns) {
		std::visit([](auto& host_col) { flsgpu::host::free_column(host_col); }, col.host);
	}
}

struct TableDecompressResult {
	size_t                                          rowgroups     = 0;
	size_t                                          total_columns = 0;
	std::vector<size_t>                             rowgroup_indices;
	std::vector<dispatch::RowgroupDecompressResult> results;
};

inline dispatch::DecompressResult decompress_column(const std::filesystem::path& fls_path,
                                                    const size_t                 rowgroup_idx,
                                                    const size_t                 column_idx,
                                                    const dispatch::Config&      cfg = {}) {
	reader::reader rdr(fls_path);
	auto           rowgroup    = rdr.read_rowgroup(rowgroup_idx);
	auto           expressions = expr::assemble(rowgroup);

	if (column_idx >= expressions.size()) {
		free_rowgroup(rowgroup);
		throw std::out_of_range("column_idx out of range");
	}

	auto result = dispatch::decompress(expressions[column_idx], cfg);
	free_rowgroup(rowgroup);
	return result;
}

template <typename RowgroupPredicate, typename RowgroupCallback>
inline TableDecompressResult decompress_table(const std::filesystem::path& fls_path,
                                              const dispatch::Config&      cfg,
                                              RowgroupPredicate&&          should_decompress,
                                              RowgroupCallback&&           on_rowgroup,
                                              const bool                   collect_results = false) {
	reader::reader        rdr(fls_path);
	const size_t          n_rowgroups = rdr.rowgroup_count();
	TableDecompressResult table_result;
	if (collect_results) {
		table_result.rowgroup_indices.reserve(n_rowgroups);
		table_result.results.reserve(n_rowgroups);
	}

	for (size_t rg_idx = 0; rg_idx < n_rowgroups; ++rg_idx) {
		if (!std::forward<RowgroupPredicate>(should_decompress)(rg_idx)) {
			continue;
		}
		auto rowgroup    = rdr.read_rowgroup(rg_idx);
		auto expressions = expr::assemble(rowgroup);

		++table_result.rowgroups;
		table_result.total_columns += expressions.size();
		auto result = dispatch::decompress_rowgroup(expressions, cfg);
		std::forward<RowgroupCallback>(on_rowgroup)(rg_idx, rowgroup, expressions, result);
		if (collect_results) {
			table_result.rowgroup_indices.push_back(rg_idx);
			table_result.results.push_back(std::move(result));
		}
		free_rowgroup(rowgroup);
	}

	return table_result;
}

template <typename RowgroupCallback>
inline TableDecompressResult decompress_table(const std::filesystem::path& fls_path,
                                              const dispatch::Config&      cfg,
                                              RowgroupCallback&&           on_rowgroup,
                                              const bool                   collect_results = false) {
	return decompress_table(
	    fls_path, cfg, [](size_t) { return true; }, std::forward<RowgroupCallback>(on_rowgroup), collect_results);
}

inline TableDecompressResult decompress_table(const std::filesystem::path& fls_path,
                                              const dispatch::Config&      cfg             = {},
                                              const bool                   collect_results = false) {
	return decompress_table(
	    fls_path,
	    cfg,
	    [](size_t) { return true; },
	    [](size_t, reader::Rowgroup&, const std::vector<expr::Expression>&, const dispatch::RowgroupDecompressResult&) {
	    },
	    collect_results);
}

} // namespace pipeline

#endif // ENGINE_PIPELINE_CUH
