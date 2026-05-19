// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/table/memory_table.cpp
// ────────────────────────────────────────────────────────
#include "fls/table/memory_table.hpp"
#include "fls/common/alias.hpp"
#include "fls/connection.hpp"
#include "fls/expression/data_type.hpp"
#include "fls/footer/column_descriptor_generated.h"
#include "fls/footer/rowgroup_descriptor_generated.h"
#include "fls/footer/table_descriptor.hpp"
#include "fls/table/rowgroup.hpp"
#include "fls/table/table.hpp"
#include <algorithm>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace fastlanes {

namespace {

template <typename T>
constexpr DataType data_type_for();

template <>
constexpr DataType data_type_for<int8_t>() {
	return DataType::INT8;
}

template <>
constexpr DataType data_type_for<int16_t>() {
	return DataType::INT16;
}

template <>
constexpr DataType data_type_for<int32_t>() {
	return DataType::INT32;
}

template <>
constexpr DataType data_type_for<int64_t>() {
	return DataType::INT64;
}

template <>
constexpr DataType data_type_for<uint8_t>() {
	return DataType::UINT8;
}

template <>
constexpr DataType data_type_for<uint16_t>() {
	return DataType::UINT16;
}

template <>
constexpr DataType data_type_for<uint32_t>() {
	return DataType::UINT32;
}

template <>
constexpr DataType data_type_for<uint64_t>() {
	return DataType::UINT64;
}

template <>
constexpr DataType data_type_for<float>() {
	return DataType::FLOAT;
}

template <>
constexpr DataType data_type_for<double>() {
	return DataType::DOUBLE;
}

size_t column_size(const MemoryColumn& column) {
	return std::visit([](const auto span) { return span.size(); }, column.data);
}

DataType column_data_type(const MemoryColumn& column) {
	return std::visit([](const auto span) {
		using SpanT = decltype(span);
		using T     = std::remove_cv_t<typename SpanT::element_type>;
		return data_type_for<T>();
	}, column.data);
}

void validate_columns(std::span<const MemoryColumn> columns) {
	if (columns.empty()) {
		throw std::runtime_error("load_memory_table requires at least one column");
	}

	const auto n_rows = column_size(columns.front());
	for (size_t col_idx = 0; col_idx < columns.size(); ++col_idx) {
		const auto& column = columns[col_idx];
		if (column.name.empty()) {
			std::ostringstream msg;
			msg << "MemoryColumn " << col_idx << " has an empty name";
			throw std::runtime_error(msg.str());
		}
		const auto size = column_size(column);
		if (size != n_rows) {
			std::ostringstream msg;
			msg << "MemoryColumn '" << column.name << "' has " << size << " values; expected " << n_rows;
			throw std::runtime_error(msg.str());
		}
	}
}

void validate_default_cast_safety(std::span<const MemoryColumn> columns, const MemoryTableOptions& options) {
	if (options.force_schema) {
		return;
	}
	constexpr auto kMaxSafeUint64ForDefaultCast = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
	for (const auto& column : columns) {
		const auto* values = std::get_if<std::span<const uint64_t>>(&column.data);
		if (values == nullptr) {
			continue;
		}
		const auto unsafe = std::find_if(values->begin(), values->end(), [](const uint64_t value) {
			return value > kMaxSafeUint64ForDefaultCast;
		});
		if (unsafe != values->end()) {
			std::ostringstream msg;
			msg << "MemoryColumn '" << column.name
			    << "' contains UINT64 values above INT64_MAX; default memory-table casting cannot preserve them. "
			       "Use MemoryTableOptions::force_schema with a UINT64 schema, or keep UINT64 values <= INT64_MAX.";
			throw std::runtime_error(msg.str());
		}
	}
}

std::unique_ptr<ColumnDescriptorT> make_column_descriptor(const MemoryColumn& column, const n_t idx) {
	auto descriptor       = std::make_unique<ColumnDescriptorT>();
	descriptor->idx      = idx;
	descriptor->name     = column.name;
	descriptor->data_type = column_data_type(column);
	descriptor->encoding_rpn = std::make_unique<RPNT>();
	descriptor->max          = std::make_unique<BinaryValueT>();
	return descriptor;
}

std::unique_ptr<RowgroupDescriptorT> make_rowgroup_descriptor(std::span<const MemoryColumn> columns,
                                                             const n_t                     n_tuples) {
	auto descriptor        = std::make_unique<RowgroupDescriptorT>();
	descriptor->m_n_tuples = n_tuples;
	descriptor->m_n_vec    = (n_tuples + CFG::VEC_SZ - 1) / CFG::VEC_SZ;
	descriptor->m_size     = columns.size();
	descriptor->m_column_descriptors.reserve(columns.size());
	for (n_t col_idx = 0; col_idx < columns.size(); ++col_idx) {
		descriptor->m_column_descriptors.push_back(make_column_descriptor(columns[col_idx], col_idx));
	}
	return descriptor;
}

template <typename T>
void assign_typed_column(Rowgroup& rowgroup, const size_t col_idx, std::span<const T> values, const size_t padded_size) {
	auto* typed = std::get_if<up<TypedCol<T>>>(&rowgroup.internal_rowgroup[col_idx]);
	if (typed == nullptr || !*typed) {
		std::ostringstream msg;
		msg << "MemoryColumn " << col_idx << " could not be mapped to the requested FastLanes physical type";
		throw std::runtime_error(msg.str());
	}
	(*typed)->data.assign(values.begin(), values.end());
	if ((*typed)->data.empty() && padded_size != 0) {
		throw std::runtime_error("memory table cannot pad an empty rowgroup");
	}
	if ((*typed)->data.size() < padded_size) {
		(*typed)->data.resize(padded_size, (*typed)->data.back());
	}
}

void assign_column_slice(Rowgroup&       rowgroup,
                         const size_t    col_idx,
                         const MemoryColumn& column,
                         const size_t    offset,
                         const size_t    n,
                         const size_t    padded_size) {
	std::visit(
	    [&](const auto span) {
		    using SpanT = decltype(span);
		    using T     = std::remove_cv_t<typename SpanT::element_type>;
		    assign_typed_column<T>(rowgroup, col_idx, span.subspan(offset, n), padded_size);
	    },
	    column.data);
}

} // namespace

class MemoryTableLoader {
public:
	static void load(Connection& connection, const MemoryTable& table, const MemoryTableOptions& options) {
		const auto columns = table.columns;
		validate_columns(columns);
		if (options.n_vectors_per_rowgroup == 0) {
			throw std::runtime_error("MemoryTableOptions::n_vectors_per_rowgroup must be greater than zero");
		}

		const auto rowgroup_capacity = options.n_vectors_per_rowgroup * CFG::VEC_SZ;
		const auto n_rows            = column_size(columns.front());

			connection.reset();
			connection.m_config->n_vector_per_rowgroup = options.n_vectors_per_rowgroup;
			connection.clear_forced_schema_state();
			if (options.force_schema) {
			if (options.forced_schema.size() != columns.size()) {
				std::ostringstream msg;
				msg << "MemoryTableOptions::forced_schema has " << options.forced_schema.size()
				    << " entries; expected one token per column (" << columns.size() << ")";
				throw std::runtime_error(msg.str());
			}
			connection.m_config->is_forced_schema = true;
			connection.m_config->forced_schema    = options.forced_schema;
		}
		validate_default_cast_safety(columns, options);

		connection.m_table = std::make_unique<Table>(connection);
		for (size_t offset = 0; offset < n_rows; offset += rowgroup_capacity) {
			const auto n_this = std::min(rowgroup_capacity, n_rows - offset);
			auto       desc   = make_rowgroup_descriptor(columns, n_this);
			auto       rg     = std::make_unique<Rowgroup>(*desc, connection);
			const auto padded_size = rg->m_descriptor.m_n_vec * CFG::VEC_SZ;

			for (size_t col_idx = 0; col_idx < columns.size(); ++col_idx) {
				assign_column_slice(*rg, col_idx, columns[col_idx], offset, n_this, padded_size);
			}
			connection.m_table->m_rowgroups.push_back(std::move(rg));
		}
	}
};

void load_memory_table(Connection& connection, const MemoryTable& table, const MemoryTableOptions& options) {
	MemoryTableLoader::load(connection, table, options);
}

void write_memory_table_to_fls(const MemoryTable&             table,
                               const path&                   output_path,
                               const MemoryTableOptions&     options) {
	Connection connection;
	load_memory_table(connection, table, options);
	connection.to_fls(output_path);
}

} // namespace fastlanes
