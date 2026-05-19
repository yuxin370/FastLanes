#include "engine/table/table_options.cuh"
#include "format/reader.cuh"
#include "galp/reader.hpp"
#include "galp/table.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace galp {

struct Table::Impl {
	struct Column {
		std::string                                                                         name;
		DataType                                                                            type = DataType::I8;
		size_t                                                                              size = 0;
		std::variant<std::monostate, std::shared_ptr<int8_t[]>, std::shared_ptr<int16_t[]>> values;
	};

	struct Rowgroup {
		std::vector<Column> columns;
	};

	static void append_column(Rowgroup& dst, const galp::execution::MaterializedColumn& src, size_t logical_row_count);

	size_t                rowgroup_total = 0;
	size_t                total_columns  = 0;
	std::vector<size_t>   column_counts;
	std::vector<Rowgroup> rowgroups;
};

namespace {

DataType to_public_data_type(const galp::format::DataType type) {
	switch (type) {
	case galp::format::DataType::I8:
		return DataType::I8;
	case galp::format::DataType::I16:
		return DataType::I16;
	case galp::format::DataType::U32:
		return DataType::U32;
	case galp::format::DataType::U64:
		return DataType::U64;
	case galp::format::DataType::F32:
		return DataType::F32;
	case galp::format::DataType::F64:
		return DataType::F64;
	default:
		throw std::runtime_error("unknown GALP materialized column type");
	}
}

} // namespace

void Table::Impl::append_column(Rowgroup&                                  dst,
                                const galp::execution::MaterializedColumn& src,
                                const size_t                               logical_row_count) {
	if (logical_row_count > src.meta.value_count) {
		throw std::runtime_error("GALP public table logical row count exceeds materialized value count");
	}
	Column column {};
	column.name = src.meta.column_name;
	column.type = to_public_data_type(src.meta.value_type);
	column.size = logical_row_count;
	std::visit(
	    [&](const auto& values) {
		    using Ptr = std::decay_t<decltype(values)>;
		    if constexpr (std::is_same_v<Ptr, std::shared_ptr<int8_t[]>> ||
		                  std::is_same_v<Ptr, std::shared_ptr<int16_t[]>>) {
			    column.values = values;
		    } else {
			    throw std::runtime_error("unsupported GALP materialized value storage");
		    }
	    },
	    src.values);
	dst.columns.push_back(std::move(column));
}

Table::Table() noexcept = default;

Table::Table(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
}

Table::~Table() = default;

Table::Table(Table&&) noexcept = default;

Table& Table::operator=(Table&&) noexcept = default;

size_t Table::rowgroup_count() const noexcept {
	return impl_ == nullptr ? 0 : impl_->rowgroup_total;
}

size_t Table::total_columns() const noexcept {
	return impl_ == nullptr ? 0 : impl_->total_columns;
}

size_t Table::rowgroup_column_count(const size_t rowgroup_index) const {
	if (impl_ == nullptr || rowgroup_index >= impl_->column_counts.size()) {
		throw std::out_of_range("galp::Table rowgroup index out of range");
	}
	return impl_->column_counts[rowgroup_index];
}

RowgroupView Table::rowgroup(const size_t rowgroup_index) const {
	if (impl_ == nullptr || rowgroup_index >= impl_->rowgroup_total) {
		throw std::out_of_range("galp::Table rowgroup index out of range");
	}
	return RowgroupView(impl_.get(), rowgroup_index);
}

const std::vector<size_t>& Table::rowgroup_column_counts() const noexcept {
	static const std::vector<size_t> empty_counts;
	return impl_ == nullptr ? empty_counts : impl_->column_counts;
}

bool Table::empty() const noexcept {
	return rowgroup_count() == 0 || total_columns() == 0;
}

Table decompress_table(const std::filesystem::path& fls_path, const DecompressOptions& options) {
	const auto cfg  = galp::execution::make_table_decompression_config(options);
	auto       impl = std::make_unique<Table::Impl>();

	if (options.write_output) {
		const auto data = galp::execution::decompress_table(
		    fls_path,
		    cfg,
		    [](size_t) { return true; },
		    [&](const size_t,
		        galp::format::Rowgroup& rowgroup_in,
		        const std::vector<galp::expression::Expression>&,
		        const galp::execution::RowgroupData& rowgroup_data) {
			    Table::Impl::Rowgroup rowgroup {};
			    rowgroup.columns.reserve(rowgroup_data.columns.size());
			    for (const auto& column : rowgroup_data.columns) {
				    if (!column.has_value()) {
					    throw std::runtime_error("GALP table materialization produced an empty column slot");
				    }
				    Table::Impl::append_column(rowgroup, *column, rowgroup_in.n_tuples);
			    }
			    impl->rowgroups.push_back(std::move(rowgroup));
		    });
		impl->rowgroup_total = data.rowgroups;
		impl->total_columns  = data.total_columns;
		impl->column_counts  = data.column_counts;
	} else {
		const auto data      = galp::execution::decompress_table(fls_path, cfg);
		impl->rowgroup_total = data.rowgroups;
		impl->total_columns  = data.total_columns;
		impl->column_counts  = data.column_counts;
	}
	return Table(std::move(impl));
}

ColumnView::ColumnView(const void* impl, const size_t rowgroup_index, const size_t column_index) noexcept
    : impl_(impl)
    , rowgroup_index_(rowgroup_index)
    , column_index_(column_index) {
}

const std::string& ColumnView::name() const {
	const auto* impl = static_cast<const Table::Impl*>(impl_);
	if (impl == nullptr || rowgroup_index_ >= impl->rowgroups.size() ||
	    column_index_ >= impl->rowgroups[rowgroup_index_].columns.size()) {
		throw std::out_of_range("galp::ColumnView column index out of range");
	}
	return impl->rowgroups[rowgroup_index_].columns[column_index_].name;
}

DataType ColumnView::type() const {
	const auto* impl = static_cast<const Table::Impl*>(impl_);
	if (impl == nullptr || rowgroup_index_ >= impl->rowgroups.size() ||
	    column_index_ >= impl->rowgroups[rowgroup_index_].columns.size()) {
		throw std::out_of_range("galp::ColumnView column index out of range");
	}
	return impl->rowgroups[rowgroup_index_].columns[column_index_].type;
}

size_t ColumnView::size() const {
	const auto* impl = static_cast<const Table::Impl*>(impl_);
	if (impl == nullptr || rowgroup_index_ >= impl->rowgroups.size() ||
	    column_index_ >= impl->rowgroups[rowgroup_index_].columns.size()) {
		throw std::out_of_range("galp::ColumnView column index out of range");
	}
	return impl->rowgroups[rowgroup_index_].columns[column_index_].size;
}

bool ColumnView::empty() const {
	return size() == 0;
}

ColumnView::UntypedSpan ColumnView::values_untyped(const DataType expected_type, const size_t element_size) const {
	const auto* impl = static_cast<const Table::Impl*>(impl_);
	if (impl == nullptr || rowgroup_index_ >= impl->rowgroups.size() ||
	    column_index_ >= impl->rowgroups[rowgroup_index_].columns.size()) {
		throw std::out_of_range("galp::ColumnView column index out of range");
	}
	const auto& column = impl->rowgroups[rowgroup_index_].columns[column_index_];
	if (column.type != expected_type) {
		throw std::bad_variant_access();
	}
	return std::visit(
	    [&](const auto& values) -> UntypedSpan {
		    using Ptr = std::decay_t<decltype(values)>;
		    if constexpr (std::is_same_v<Ptr, std::shared_ptr<int8_t[]>> ||
		                  std::is_same_v<Ptr, std::shared_ptr<int16_t[]>>) {
			    if (sizeof(std::remove_extent_t<typename Ptr::element_type>) != element_size) {
				    throw std::bad_variant_access();
			    }
			    return {values.get(), column.size};
		    } else {
			    throw std::runtime_error("galp::ColumnView::values<T>() requires DecompressOptions::write_output=true");
		    }
	    },
	    column.values);
}

RowgroupView::RowgroupView(const void* impl, const size_t rowgroup_index) noexcept
    : impl_(impl)
    , rowgroup_index_(rowgroup_index) {
}

size_t RowgroupView::column_count() const {
	const auto* impl = static_cast<const Table::Impl*>(impl_);
	if (impl == nullptr || rowgroup_index_ >= impl->rowgroup_total) {
		throw std::out_of_range("galp::RowgroupView rowgroup index out of range");
	}
	if (rowgroup_index_ < impl->rowgroups.size()) {
		return impl->rowgroups[rowgroup_index_].columns.size();
	}
	return impl->column_counts[rowgroup_index_];
}

ColumnView RowgroupView::column(const size_t column_index) const {
	const auto* impl = static_cast<const Table::Impl*>(impl_);
	if (impl == nullptr || rowgroup_index_ >= impl->rowgroup_total) {
		throw std::out_of_range("galp::RowgroupView rowgroup index out of range");
	}
	if (rowgroup_index_ >= impl->rowgroups.size()) {
		throw std::runtime_error("galp::RowgroupView::column() requires DecompressOptions::write_output=true");
	}
	if (column_index >= impl->rowgroups[rowgroup_index_].columns.size()) {
		throw std::out_of_range("galp::RowgroupView column index out of range");
	}
	return ColumnView(impl_, rowgroup_index_, column_index);
}

bool RowgroupView::empty() const {
	return column_count() == 0;
}

Reader::Reader(std::filesystem::path fls_path)
    : fls_path_(std::move(fls_path)) {
}

Reader::~Reader() = default;

Reader::Reader(Reader&&) noexcept = default;

Reader& Reader::operator=(Reader&&) noexcept = default;

size_t Reader::rowgroup_count() const {
	galp::format::FlsReader internal_reader(fls_path_);
	return internal_reader.rowgroup_count();
}

Table Reader::decompress(const DecompressOptions& options) const {
	return decompress_table(fls_path_, options);
}

const std::filesystem::path& Reader::path() const noexcept {
	return fls_path_;
}

} // namespace galp
