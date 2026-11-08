// CPU preparation for the paper's matched compressed/raw-int16 transfer control.
#include "fls/connection.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/footer/table_descriptor.hpp"
#include "fls/io/file.hpp"
#include "fls/json/nlohmann/json.hpp"
#include "fls/table/memory_table.hpp"
#include "jpeg/jpeg_dct_shard_reader.hpp"
#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
fastlanes::TableDescriptorHandle descriptor(const std::filesystem::path& path) {
	fastlanes::File       file(path);
	fastlanes::FileFooter footer {};
	if (!fastlanes::FileFooter::Load(footer, file).success) {
		throw std::runtime_error("cannot read FLS footer: " + path.string());
	}
	return fastlanes::TableDescriptorHandle::FromFileSlice(
	    file, footer.table_descriptor_offset, footer.table_descriptor_size, true);
}
} // namespace

int main(int argc, char** argv) try {
	if (argc == 2 && std::string_view(argv[1]) == "--help") {
		std::cout << "Usage: fastlens_raw_dct_control SOURCE.fls NEW_OUTPUT.fls\n";
		return 0;
	}
	if (argc != 3) {
		throw std::runtime_error("expected SOURCE.fls NEW_OUTPUT.fls");
	}
	const std::filesystem::path source(argv[1]), output(argv[2]);
	if (std::filesystem::exists(output)) {
		throw std::runtime_error("refusing to replace output: " + output.string());
	}
	const auto  input  = descriptor(source);
	const auto* groups = input->m_rowgroup_descriptors();
	if (groups->size() == 0U) {
		throw std::runtime_error("source has no rowgroups");
	}
	std::array<std::vector<int16_t>, 64>                       columns;
	std::vector<fastlanes::n_t>                                rows;
	fastlanes::n_t                                             capacity = 0U;
	fastlens::jpeg::detail::JpegDctSelectedVectorProfileReader reader(source);
	for (uint32_t group = 0U; group < groups->size(); ++group) {
		const auto* record = groups->Get(group);
		if (record->m_column_descriptors()->size() != columns.size()) {
			throw std::runtime_error("raw control requires exactly 64 int16 DCT columns");
		}
		const auto start = columns.front().size();
		reader.AppendFullRowgroup(group, columns);
		rows.push_back(record->m_n_tuples());
		capacity = std::max(capacity, record->m_n_vec());
		for (auto& column : columns) {
			column.resize(start + static_cast<size_t>(record->m_n_tuples()));
		}
	}
	std::array<fastlanes::MemoryColumn, 64> inputs;
	for (size_t column = 0U; column < columns.size(); ++column) {
		inputs[column].name =
		    groups->Get(0U)->m_column_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(column))->name()->str();
		inputs[column].data = std::span<const int16_t>(columns[column]);
	}
	fastlanes::MemoryTableOptions options;
	options.n_vectors_per_rowgroup = capacity;
	options.rowgroup_n_tuples      = rows;
	options.force_schema           = true;
	options.forced_schema.assign(columns.size(), fastlanes::OperatorToken::EXP_UNCOMPRESSED_I16);
	fastlanes::Connection writer;
	fastlanes::load_memory_table(writer, {inputs}, options);
	writer.inline_footer();
	writer.to_fls(output);
	const auto encoded = descriptor(output);
	if (encoded->m_rowgroup_descriptors()->size() != groups->size()) {
		throw std::runtime_error("raw control changed rowgroup count");
	}
	fastlens::jpeg::detail::JpegDctSelectedVectorProfileReader check(output);
	size_t                                                     offset = 0U;
	for (uint32_t group = 0U; group < groups->size(); ++group) {
		const auto* before = groups->Get(group);
		const auto* after  = encoded->m_rowgroup_descriptors()->Get(group);
		if (before->m_n_tuples() != after->m_n_tuples() || before->m_n_vec() != after->m_n_vec()) {
			throw std::runtime_error("raw control changed rowgroup geometry");
		}
		std::array<std::vector<int16_t>, 64> decoded;
		check.AppendFullRowgroup(group, decoded);
		const auto count = static_cast<size_t>(rows[group]);
		for (size_t column = 0U; column < columns.size(); ++column) {
			if (!std::equal(
			        decoded[column].begin(), decoded[column].begin() + count, columns[column].begin() + offset)) {
				throw std::runtime_error("raw control changed DCT coefficients");
			}
		}
		offset += count;
	}
	std::cout << nlohmann::json({{"source", source.string()},
	                             {"destination", output.string()},
	                             {"coefficients_and_rowgroups_exact", true},
	                             {"rowgroups", rows.size()}})
	                 .dump()
	          << '\n';
	return 0;
} catch (const std::exception& error) {
	std::cerr << "raw DCT control: " << error.what() << '\n';
	return 1;
}
