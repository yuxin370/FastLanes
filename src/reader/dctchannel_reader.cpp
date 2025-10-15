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

up<Table> DctChannelReader::Read(const std::vector<ChannelDCT>& channels, const Connection& connection) {
	auto table = make_unique<Table>(connection);

	// construct RowgroupDescriptorT ---
	auto rowgroup_descriptor = make_unique<RowgroupDescriptorT>();

	// construct 64 rowgroup_descriptor
	for (int i = 0; i < 64; ++i) {
		auto col       = make_unique<ColumnDescriptorT>();
		col->data_type = fastlanes::DataType::INT32; // 可改为 INT64 等
		col->idx       = i;
		col->name      = "col_" + std::to_string(i); // 命名如 col_0, col_1, ..., col_63
		rowgroup_descriptor->m_column_descriptors.push_back(std::move(col));
	}

	// set index
	set_index(rowgroup_descriptor->m_column_descriptors);

	n_t  n_tup {0};
	auto cur_rowgroup = make_unique<Rowgroup>(*rowgroup_descriptor, connection);
	for (auto& channel : channels) {
		for (auto& tuple : channel.blocks) {
			// for(auto& tuple : channel.mixed_run_encoding_pattern){
			for (int i = 0; i < 64; i++) {
				col_pt& physical_column = cur_rowgroup->internal_rowgroup[i];
				Attribute::Ingest(physical_column,
				                  std::to_string(tuple.data[i]),
				                  *cur_rowgroup->m_descriptor.m_column_descriptors[i]);
			}

			n_tup = n_tup + 1;
			if (n_tup == cur_rowgroup->capacity) {
				cur_rowgroup->n_tup = n_tup;
				table->m_rowgroups.push_back(std::move(cur_rowgroup));
				cur_rowgroup = make_unique<Rowgroup>(*rowgroup_descriptor, connection);
				n_tup        = 0;
			}
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

// Helper functions for creating struct schema
static up<ColumnDescriptorT> make_typed_child(const string& name, DataType dt, const uint64_t idx) {
	auto child          = make_unique<ColumnDescriptorT>();
	child->name         = name;
	child->data_type    = dt;
	child->idx          = idx;
	child->max          = make_unique<BinaryValueT>(); //  max
	child->encoding_rpn = make_unique<RPNT>();         // encoding_rpn
	return child;
}

up<ColumnDescriptorT> make_dct_channel_struct_column_descriptor(const string& name) {

	std::vector<up<ColumnDescriptorT>> children;
	children.push_back(make_typed_child("dc_values", DataType::INT32, 0));
	children.push_back(make_typed_child("ac_values", DataType::INT32, 1));
	children.push_back(make_typed_child("mix_run_nonzero_values", DataType::INT32, 2));
	children.push_back(make_typed_child("mix_run_pattern", DataType::INT32, 3));

	auto dct_channel_struct          = make_unique<ColumnDescriptorT>();
	dct_channel_struct->name         = name;
	dct_channel_struct->data_type    = DataType::STRUCT;
	dct_channel_struct->children     = std::move(children);
	dct_channel_struct->max          = make_unique<BinaryValueT>(); //  max
	dct_channel_struct->encoding_rpn = make_unique<RPNT>();         //  encoding_rpn

	// for (n_t i = 0; i < dct_channel_struct->children.size(); i++) {
	//     dct_channel_struct->children[i]->idx = i;
	// }

	return dct_channel_struct;
}

// Helper functions for accessing columns
template <typename PT>
static up<TypedCol<PT>>& as_typed(col_pt& c) {
	return std::get<up<TypedCol<PT>>>(c);
}

void ingest_dct_channel_into_struct(up<Rowgroup>& cur_rowgroup, const ProcessedDCTChannel& channel) {

	auto& variant_elem = cur_rowgroup->internal_rowgroup[0];

	if (auto* struct_ptr = std::get_if<std::unique_ptr<fastlanes::Struct>>(&variant_elem)) {
		if (*struct_ptr && !(*struct_ptr)->internal_rowgroup.empty()) {
			// 1. DC values - append using standard Attribute::Ingest with error checking
			auto& physical_column0 = (*struct_ptr)->internal_rowgroup[0];
			for (auto v : channel.DC_values) {
				try {
					Attribute::Ingest(physical_column0,
					                  std::to_string(v),
					                  *cur_rowgroup->m_descriptor.m_column_descriptors[0]->children[0]);
				} catch (const std::exception& e) {
					throw std::runtime_error("Failed to ingest DC value: " + std::string(e.what()));
				}
			}

			// 2. AC values - append using standard Attribute::Ingest with error checking
			auto& physical_column1 = (*struct_ptr)->internal_rowgroup[1];
			for (auto v : channel.AC_values) {
				// auto& physical_column = cur_rowgroup->internal_rowgroup[1];
				try {
					Attribute::Ingest(physical_column1,
					                  std::to_string(v),
					                  *cur_rowgroup->m_descriptor.m_column_descriptors[0]->children[1]);
				} catch (const std::exception& e) {
					throw std::runtime_error("Failed to ingest AC value: " + std::string(e.what()));
				}
			}

			// 3. mix_run_nonzero_values - append using standard Attribute::Ingest with error checking
			auto& physical_column2 = (*struct_ptr)->internal_rowgroup[2];
			for (auto v : channel.mix_run_nonzero_values) {
				// auto& physical_column = cur_rowgroup->internal_rowgroup[2];
				try {
					Attribute::Ingest(physical_column2,
					                  std::to_string(v),
					                  *cur_rowgroup->m_descriptor.m_column_descriptors[0]->children[2]);
				} catch (const std::exception& e) {
					throw std::runtime_error("Failed to ingest mix run nonzero value: " + std::string(e.what()));
				}
			}

			// 4. mix_run_pattern - append using standard Attribute::Ingest with error checking
			auto& physical_column3 = (*struct_ptr)->internal_rowgroup[3];
			for (auto& pattern : channel.mix_run_pattern) {
				// auto& physical_column = cur_rowgroup->internal_rowgroup[3];
				try {
					// Append both zero_count and nonzero_count to the pattern column
					Attribute::Ingest(physical_column3,
					                  std::to_string(pattern.zero_count),
					                  *cur_rowgroup->m_descriptor.m_column_descriptors[0]->children[3]);
					Attribute::Ingest(physical_column3,
					                  std::to_string(pattern.nonzero_count),
					                  *cur_rowgroup->m_descriptor.m_column_descriptors[0]->children[3]);
				} catch (const std::exception& e) {
					throw std::runtime_error("Failed to ingest mix run pattern: " + std::string(e.what()));
				}
			}
		}
	}
}

up<Table> DctChannelReader::Read(const ProcessedDCTChannel& channel, const int tag, const Connection& connection) {
	auto table = make_unique<Table>(connection);

	switch (tag) {
	case 1:
		// --------------------------
		// 1. DC values → single column
		// --------------------------
		{
			auto rowgroup_descriptor = make_unique<RowgroupDescriptorT>();

			auto col       = make_unique<ColumnDescriptorT>();
			col->data_type = fastlanes::DataType::INT32;
			col->idx       = 0;
			col->name      = "DC";
			rowgroup_descriptor->m_column_descriptors.push_back(std::move(col));
			set_index(rowgroup_descriptor->m_column_descriptors);

			auto cur_rowgroup = make_unique<Rowgroup>(*rowgroup_descriptor, connection);
			n_t  n_tup {0};

			for (auto v : channel.DC_values) {
				col_pt& physical_column = cur_rowgroup->internal_rowgroup[0];
				Attribute::Ingest(
				    physical_column, std::to_string(v), *cur_rowgroup->m_descriptor.m_column_descriptors[0]);
				n_tup++;
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
		}
		break;
	// --------------------------
	// 2. AC values → single column
	// --------------------------
	case 2: {
		auto rowgroup_descriptor = make_unique<RowgroupDescriptorT>();

		auto col       = make_unique<ColumnDescriptorT>();
		col->data_type = fastlanes::DataType::INT32;
		col->idx       = 0;
		col->name      = "AC";
		rowgroup_descriptor->m_column_descriptors.push_back(std::move(col));
		set_index(rowgroup_descriptor->m_column_descriptors);

		auto cur_rowgroup = make_unique<Rowgroup>(*rowgroup_descriptor, connection);
		n_t  n_tup {0};

		for (auto v : channel.AC_values) {
			col_pt& physical_column = cur_rowgroup->internal_rowgroup[0];
			Attribute::Ingest(physical_column, std::to_string(v), *cur_rowgroup->m_descriptor.m_column_descriptors[0]);
			n_tup++;
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
	} break;

	// --------------------------
	// 3. mix_run_nonzero_values → single column
	// --------------------------
	case 3: {
		auto rowgroup_descriptor = make_unique<RowgroupDescriptorT>();

		auto col       = make_unique<ColumnDescriptorT>();
		col->data_type = fastlanes::DataType::INT32;
		col->idx       = 0;
		col->name      = "mix_nonzero";
		rowgroup_descriptor->m_column_descriptors.push_back(std::move(col));
		set_index(rowgroup_descriptor->m_column_descriptors);

		auto cur_rowgroup = make_unique<Rowgroup>(*rowgroup_descriptor, connection);
		n_t  n_tup {0};

		for (auto v : channel.mix_run_nonzero_values) {
			col_pt& physical_column = cur_rowgroup->internal_rowgroup[0];
			Attribute::Ingest(physical_column, std::to_string(v), *cur_rowgroup->m_descriptor.m_column_descriptors[0]);
			n_tup++;
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
	} break;
	// --------------------------
	// 4. mix_run_pattern → two column (zero_count, nonzero_count)
	// --------------------------
	case 4: {
		auto rowgroup_descriptor = make_unique<RowgroupDescriptorT>();

		auto col1       = make_unique<ColumnDescriptorT>();
		col1->data_type = fastlanes::DataType::INT32;
		col1->idx       = 0;
		col1->name      = "zero_count";
		rowgroup_descriptor->m_column_descriptors.push_back(std::move(col1));

		auto col2       = make_unique<ColumnDescriptorT>();
		col2->data_type = fastlanes::DataType::INT32;
		col2->idx       = 1;
		col2->name      = "nonzero_count";
		rowgroup_descriptor->m_column_descriptors.push_back(std::move(col2));

		set_index(rowgroup_descriptor->m_column_descriptors);

		auto cur_rowgroup = make_unique<Rowgroup>(*rowgroup_descriptor, connection);
		n_t  n_tup {0};

		for (auto& p : channel.mix_run_pattern) {
			{
				col_pt& physical_column = cur_rowgroup->internal_rowgroup[0];
				Attribute::Ingest(
				    physical_column, std::to_string(p.zero_count), *cur_rowgroup->m_descriptor.m_column_descriptors[0]);
			}
			{
				col_pt& physical_column = cur_rowgroup->internal_rowgroup[1];
				Attribute::Ingest(physical_column,
				                  std::to_string(p.nonzero_count),
				                  *cur_rowgroup->m_descriptor.m_column_descriptors[1]);
			}
			n_tup++;
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
	} break;
	// --------------------------
	// DCT Channel as Struct → single columnstruct，include all the data
	// --------------------------
	case 5: // New case for struct-based storage
	{
		auto rowgroup_descriptor = make_unique<RowgroupDescriptorT>();
		auto struct_col          = make_dct_channel_struct_column_descriptor("dct_channel");
		struct_col->idx          = 0;
		struct_col->max          = make_unique<BinaryValueT>();
		struct_col->encoding_rpn = make_unique<RPNT>();

		rowgroup_descriptor->m_column_descriptors.push_back(std::move(struct_col));
		set_index(rowgroup_descriptor->m_column_descriptors);

		auto cur_rowgroup = make_unique<Rowgroup>(*rowgroup_descriptor, connection);
		if (!cur_rowgroup) {
			throw std::runtime_error("Failed to create rowgroup");
		}

		cur_rowgroup->n_tup = 0;

		ingest_dct_channel_into_struct(cur_rowgroup, channel);

		cur_rowgroup->n_tup = 1; // One struct row per channel

		const n_t leftover = cur_rowgroup->n_tup % CFG::VEC_SZ;
		if (leftover != 0) {
			n_t how_many_to_fill = CFG::VEC_SZ - leftover;
			cur_rowgroup->FillMissingValues(how_many_to_fill);
		}

		table->m_rowgroups.push_back(std::move(cur_rowgroup));
	} break;
	default:
		break;
	}
	return table;
}

// // New function to support multiple images in one table
// up<Table> DctChannelReader::ReadMultipleChannels(const std::vector<ProcessedDCTChannel>& channels, const Connection&
// connection)
// {
//     auto table = make_unique<Table>(connection);

//     // Create struct column descriptor for the entire table
//     auto rowgroup_descriptor = make_unique<RowgroupDescriptorT>();
//     auto struct_col = make_dct_channel_struct_column_descriptor("dct_channel");
//     struct_col->idx = 0;
//     rowgroup_descriptor->m_column_descriptors.push_back(std::move(struct_col));
//     set_index(rowgroup_descriptor->m_column_descriptors);

//     auto cur_rowgroup = make_unique<Rowgroup>(*rowgroup_descriptor, connection);
//     n_t n_tup {0};

//     // Process each channel as a separate row in the same table
//     for (const auto& channel : channels) {
//         // Ingest the entire channel as a single struct row
//         col_pt& physical_column = cur_rowgroup->internal_rowgroup[0];
//         ingest_dct_channel_into_struct(physical_column, channel);

//         n_tup++; // One struct row per channel

//         // Check if we need to create a new rowgroup
//         if (n_tup == cur_rowgroup->capacity) {
//             cur_rowgroup->n_tup = n_tup;
//             table->m_rowgroups.push_back(std::move(cur_rowgroup));
//             cur_rowgroup = make_unique<Rowgroup>(*rowgroup_descriptor, connection);
//             n_tup = 0;
//         }
//     }

//     // Handle the last rowgroup if it has data
//     if (n_tup != 0) {
//         const n_t leftover = n_tup % CFG::VEC_SZ;
//         if (leftover != 0) {
//             n_t how_many_to_fill = CFG::VEC_SZ - leftover;
//             cur_rowgroup->FillMissingValues(how_many_to_fill);
//         }
//         cur_rowgroup->n_tup = n_tup;
//         table->m_rowgroups.push_back(std::move(cur_rowgroup));
//     }

//     return table;
// }

// Alternative function to add a single channel to an existing table
// void DctChannelReader::AddChannelToTable(Table& table, const ProcessedDCTChannel& channel, const Connection&
// connection)
// {
//     // If table is empty, we need to create the schema first
//     if (table.m_rowgroups.empty()) {
//         auto rowgroup_descriptor = make_unique<RowgroupDescriptorT>();
//         auto struct_col = make_dct_channel_struct_column_descriptor("dct_channel");
//         struct_col->idx = 0;
//         rowgroup_descriptor->m_column_descriptors.push_back(std::move(struct_col));
//         set_index(rowgroup_descriptor->m_column_descriptors);

//         auto new_rowgroup = make_unique<Rowgroup>(*rowgroup_descriptor, connection);

//         // Ingest the channel data
//         col_pt& physical_column = new_rowgroup->internal_rowgroup[0];
//         ingest_dct_channel_into_struct(physical_column, channel);

//         new_rowgroup->n_tup = 1;
//         table.m_rowgroups.push_back(std::move(new_rowgroup));
//     } else {
//         // Table already has data, append to the last rowgroup if it has space
//         auto& last_rowgroup = table.m_rowgroups.back();

//         if (last_rowgroup->n_tup < last_rowgroup->capacity) {
//             // Add to existing rowgroup
//             col_pt& physical_column = last_rowgroup->internal_rowgroup[0];
//             ingest_dct_channel_into_struct(physical_column, channel);
//             last_rowgroup->n_tup++;
//         } else {
//             // Create new rowgroup
//             auto rowgroup_descriptor = make_unique<RowgroupDescriptorT>();
//             auto struct_col = make_dct_channel_struct_column_descriptor("dct_channel");
//             struct_col->idx = 0;
//             rowgroup_descriptor->m_column_descriptors.push_back(std::move(struct_col));
//             set_index(rowgroup_descriptor->m_column_descriptors);

//             auto new_rowgroup = make_unique<Rowgroup>(*rowgroup_descriptor, connection);

//             // Ingest the channel data
//             col_pt& physical_column = new_rowgroup->internal_rowgroup[0];
//             ingest_dct_channel_into_struct(physical_column, channel);

//             new_rowgroup->n_tup = 1;
//             table.m_rowgroups.push_back(std::move(new_rowgroup));
//         }
//     }
// }

} // namespace fastlanes
