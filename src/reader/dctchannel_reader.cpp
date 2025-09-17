// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/reader/dctchannel_reader.cpp
// ────────────────────────────────────────────────────────
#include "fls/reader/dctchannel_reader.hpp"
#include "fls/cfg/cfg.hpp"
#include "fls/common/alias.hpp"
#include "fls/common/assert.hpp"
#include "fls/connection.hpp"
#include "fls/csv/csv-parser/parser.hpp"
#include "fls/footer/column_descriptor.hpp"
#include "fls/io/file.hpp"
#include "fls/json/fls_json.hpp"
#include "fls/json/nlohmann/json.hpp"
#include "fls/std/filesystem.hpp"
#include "fls/std/string.hpp"
#include "fls/table/attribute.hpp"
#include "fls/table/rowgroup.hpp"
#include "fls/table/table.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream> // for std::ifstream
#include <stdexcept>
#include <utility> // for std::move

namespace fastlanes {

up<Table> DctChannelReader::Read(const ProcessedDCTChannel& channel, const Connection& connection) {
	auto table = make_unique<Table>(connection);

    // construct RowgroupDescriptorT ---
    auto rowgroup_descriptor = make_unique<RowgroupDescriptorT>();

	// construct 64 rowgroup_descriptor
	for (int i = 0; i < 64; ++i) {
		auto col = make_unique<ColumnDescriptorT>();
		col->data_type = fastlanes::DataType::INT32; // 可改为 INT64 等
		col->idx = i;
		col->name = "col_" + std::to_string(i); // 命名如 col_0, col_1, ..., col_63
		rowgroup_descriptor->m_column_descriptors.push_back(std::move(col));
	}

    // // construct first rowgroup_descriptor (zero_count)
    // auto col1 = make_unique<ColumnDescriptorT>();
    // col1->data_type = fastlanes::DataType::INT32; // 或 INT64, 根据你的需求
    // col1->idx = 0;
    // col1->name = "zero_count";
    // rowgroup_descriptor->m_column_descriptors.push_back(std::move(col1));

    // // construct second rowgroup_descriptor  (nonzero_count)
    // auto col2 = make_unique<ColumnDescriptorT>();
    // col2->data_type = fastlanes::DataType::INT32; // 或 INT64, 根据你的需求
    // col2->idx = 1;
    // col2->name = "nonzero_count";
    // rowgroup_descriptor->m_column_descriptors.push_back(std::move(col2));

    // set index
    set_index(rowgroup_descriptor->m_column_descriptors);

	n_t  n_tup {0};
	auto cur_rowgroup = make_unique<Rowgroup>(*rowgroup_descriptor, connection);
	[[maybe_unused]] const auto n_cols = cur_rowgroup->ColCount();
	FLS_ASSERT_EQUALITY(2, n_cols)
	for(auto& tuple : channel.raw_blocks){
	// for(auto& tuple : channel.mixed_run_encoding_pattern){
		for(int i = 0; i < 64; i ++){
			col_pt& physical_column = cur_rowgroup->internal_rowgroup[i];
			Attribute::Ingest(physical_column, std::to_string(tuple.data[i]), *cur_rowgroup->m_descriptor.m_column_descriptors[i]);
		}
		// col_pt& physical_column = cur_rowgroup->internal_rowgroup[0];
		// Attribute::Ingest(physical_column, std::to_string(tuple.zero_count), *cur_rowgroup->m_descriptor.m_column_descriptors[0]);
		// col_pt&  physical_column_non = cur_rowgroup->internal_rowgroup[1];
		// Attribute::Ingest(physical_column_non, std::to_string(tuple.nonzero_count), *cur_rowgroup->m_descriptor.m_column_descriptors[1]);
		
		n_tup = n_tup + 1;
		if (n_tup == cur_rowgroup->capacity) {
			cur_rowgroup->n_tup = n_tup;
			table->m_rowgroups.push_back(std::move(cur_rowgroup));
			cur_rowgroup = make_unique<Rowgroup>(*rowgroup_descriptor, connection);
			n_tup        = 0;
		}
	}

	if (n_tup != 0) {
		const n_t leftover = n_tup % CFG::VEC_SZ;
		if (leftover != 0) {
			n_t how_many_to_fill = CFG::VEC_SZ - leftover;
			cur_rowgroup->FillMissingValues(how_many_to_fill);
		}
		cur_rowgroup->n_tup = n_tup;
		table->m_rowgroups.push_back(std::move(cur_rowgroup));
	}

	return table;
};
} // namespace fastlanes
