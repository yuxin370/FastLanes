#include <cstdint>
#include <filesystem>
#include <galp/galp.hpp>
#include <iostream>

int main(int argc, char** argv) {
	if (argc != 2) {
		std::cerr << "usage: public_api_reader <data.fls>\n";
		return 2;
	}

	galp::Reader            reader(std::filesystem::path {argv[1]});
	galp::DecompressOptions options {};
	options.write_output = true;

	const galp::Table table = reader.decompress(options);
	std::cout << "rowgroups=" << table.rowgroup_count() << " columns=" << table.total_columns() << "\n";
	for (size_t rg_idx = 0; rg_idx < table.rowgroup_count(); ++rg_idx) {
		const auto rowgroup = table.rowgroup(rg_idx);
		std::cout << "rowgroup[" << rg_idx << "] columns=" << rowgroup.column_count() << "\n";
	}

	if (table.rowgroup_count() == 0 || table.rowgroup(0).column_count() == 0) {
		return 0;
	}

	const auto column = table.rowgroup(0).column(0);
	std::cout << "first_column name=" << column.name() << " values=" << column.size() << "\n";
	if (column.type() == galp::DataType::I8) {
		const auto values = column.values<int8_t>();
		if (!values.empty()) {
			std::cout << "first_value=" << static_cast<int>(values.front()) << "\n";
		}
	} else if (column.type() == galp::DataType::I16) {
		const auto values = column.values<int16_t>();
		if (!values.empty()) {
			std::cout << "first_value=" << values.front() << "\n";
		}
	}

	return 0;
}
