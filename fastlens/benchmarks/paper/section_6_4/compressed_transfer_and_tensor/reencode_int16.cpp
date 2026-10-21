// Matched compression control: preserve coefficient columns and rowgroup boundaries.
#include "fls/connection.hpp"
#include "fls/json/nlohmann/json.hpp"
#include "fls/reader/table_reader.hpp"
#include "fls/table/memory_table.hpp"
#include "fls/table/table.hpp"
#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {
using Columns = std::array<std::vector<int16_t>, 64>;

Columns coefficients(const fastlanes::Table& table, std::vector<fastlanes::n_t>& row_counts) {
	Columns columns;
	for (const auto& rowgroup : table.m_rowgroups) {
		if (rowgroup->internal_rowgroup.size() != columns.size()) {
			throw std::runtime_error("expected exactly 64 DCT coefficient columns");
		}
		row_counts.push_back(rowgroup->n_tup);
		for (size_t index = 0; index < columns.size(); ++index) {
			const auto& source = rowgroup->internal_rowgroup[index];
			if (const auto* values = std::get_if<fastlanes::up<fastlanes::col_i16>>(&source)) {
				columns[index].insert(columns[index].end(),
				                      (*values)->data.begin(),
				                      (*values)->data.begin() + static_cast<ptrdiff_t>(rowgroup->n_tup));
			} else if (const auto* narrow = std::get_if<fastlanes::up<fastlanes::col_i08>>(&source)) {
				columns[index].insert(columns[index].end(),
				                      (*narrow)->data.begin(),
				                      (*narrow)->data.begin() + static_cast<ptrdiff_t>(rowgroup->n_tup));
			} else {
				throw std::runtime_error("DCT source column must be signed int8 or int16");
			}
		}
	}
	return columns;
}
} // namespace

int main(int argc, char** argv) {
	if (argc != 3) {
		throw std::runtime_error("usage: fastlens_raw_dct_control SOURCE.fls DESTINATION.fls");
	}
	const std::filesystem::path             source(argv[1]);
	const std::filesystem::path             destination(argv[2]);
	fastlanes::Connection                   reader;
	const auto                              started = std::chrono::steady_clock::now();
	auto                                    table   = reader.read_fls(source)->materialize();
	std::vector<fastlanes::n_t>             row_counts;
	const auto                              data = coefficients(*table, row_counts);
	std::array<fastlanes::MemoryColumn, 64> columns;
	for (size_t index = 0; index < columns.size(); ++index) {
		columns[index].name = table->GetRowgroup(0).m_descriptor.m_column_descriptors[index]->name;
		columns[index].data = std::span<const int16_t>(data[index]);
	}
	table.reset();
	fastlanes::MemoryTableOptions options;
	options.n_vectors_per_rowgroup = 128;
	options.rowgroup_n_tuples      = std::span<const fastlanes::n_t>(row_counts);
	options.force_schema           = true;
	options.forced_schema.assign(columns.size(), fastlanes::OperatorToken::EXP_UNCOMPRESSED_I16);
	fastlanes::Connection writer;
	writer.read_memory(fastlanes::MemoryTable {std::span<const fastlanes::MemoryColumn>(columns)}, options);
	writer.inline_footer().to_fls(destination);
	const auto                  encoded  = std::chrono::steady_clock::now();
	auto                        restored = reader.read_fls(destination)->materialize();
	std::vector<fastlanes::n_t> restored_counts;
	if (coefficients(*restored, restored_counts) != data || restored_counts != row_counts) {
		throw std::runtime_error("raw-int16 control changed coefficients or rowgroup boundaries");
	}
	const auto finished = std::chrono::steady_clock::now();
	std::cout << nlohmann::json({{"source", source.string()},
	                             {"destination", destination.string()},
	                             {"source_bytes", std::filesystem::file_size(source)},
	                             {"raw_int16_bytes", std::filesystem::file_size(destination)},
	                             {"rowgroups", row_counts.size()},
	                             {"rows", data[0].size()},
	                             {"conversion_s", std::chrono::duration<double>(encoded - started).count()},
	                             {"verification_s", std::chrono::duration<double>(finished - encoded).count()},
	                             {"coefficients_and_rowgroups_exact", true}})
	                 .dump()
	          << '\n';
}
