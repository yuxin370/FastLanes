// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/io/to_csv.cu
// ────────────────────────────────────────────────────────
#include "engine/io/to_csv.cuh"
#include "engine/reader.cuh"
#include <cstdint>
#include <filesystem>
#include <string>
#include <type_traits>
#include <vector>

namespace galp::io {
namespace {

template <typename PtrT>
void write_cell(std::ostream& out, const PtrT& ptr, size_t row) {
	using T = typename PtrT::element_type;
	if constexpr (std::is_integral_v<T> && sizeof(T) == 1) {
		out << static_cast<int>(ptr[row]);
	} else {
		out << static_cast<int64_t>(ptr[row]);
	}
}

void write_row(std::ostream&                        out,
               const std::vector<size_t>&           value_indices,
               const galp::execution::RowgroupData& rowgroup_data,
               const size_t                         row) {
	for (size_t ci = 0; ci < value_indices.size(); ++ci) {
		if (ci > 0) {
			out << "|";
		}
		const auto  col_idx = value_indices[ci];
		const auto& opt     = rowgroup_data.columns[col_idx];
		if (!opt.has_value()) {
			continue;
		}
		std::visit([&](const auto& ptr) { write_cell(out, ptr, row); }, opt->values);
	}
	out << "\n";
}

size_t resolve_value_index(const std::vector<galp::expression::Expression>& expressions, const size_t idx) {
	size_t cur = idx;
	for (size_t step = 0; step < expressions.size(); ++step) {
		if (cur >= expressions.size()) {
			break;
		}
		const auto& e = expressions[cur];
		if (!e.column || !e.column->skip_decompress || !e.column->alias_of.has_value()) {
			break;
		}
		const size_t next = *e.column->alias_of;
		if (next >= expressions.size() || next == cur) {
			break;
		}
		cur = next;
	}
	return cur;
}

} // namespace

void read_table_to_csv(const std::filesystem::path&                     fls_path,
                       std::ostream&                                    out,
                       const bool                                       write_header,
                       const galp::execution::TableDecompressionConfig& table_cfg,
                       const std::optional<size_t>&                     rowgroup) {
	if (rowgroup.has_value()) {
		galp::format::FlsReader rdr(fls_path);
		if (*rowgroup >= rdr.rowgroup_count()) {
			throw std::out_of_range("rowgroup index out of range");
		}
	}

	bool header_written = false;
	galp::execution::decompress_table(
	    fls_path,
	    table_cfg,
	    [rowgroup](const size_t rg_idx) { return !rowgroup.has_value() || *rowgroup == rg_idx; },
	    [&](size_t,
	        galp::execution::Rowgroup&                       rowgroup,
	        const std::vector<galp::expression::Expression>& expressions,
	        const galp::execution::RowgroupData&             result) {
		    std::vector<size_t>      value_indices;
		    std::vector<std::string> col_names;
		    for (size_t i = 0; i < expressions.size(); ++i) {
			    const auto& expression = expressions[i];
			    if (!expression.column) {
				    continue;
			    }
			    value_indices.push_back(resolve_value_index(expressions, i));
			    auto name = expression.column->name;
			    if (name.empty()) {
				    name = "col_" + std::to_string(i);
			    }
			    col_names.push_back(std::move(name));
		    }

		    if (write_header && !header_written) {
			    for (size_t i = 0; i < col_names.size(); ++i) {
				    if (i > 0) {
					    out << ",";
				    }
				    out << col_names[i];
			    }
			    out << "\n";
			    header_written = true;
		    }

		    for (size_t row = 0; row < rowgroup.n_tuples; ++row) {
			    write_row(out, value_indices, result, row);
		    }
	    });
}

} // namespace galp::io
