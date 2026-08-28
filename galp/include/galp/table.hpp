#ifndef GALP_TABLE_HPP
#define GALP_TABLE_HPP

#include "galp/options.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace galp {

class ColumnView {
public:
	ColumnView() noexcept = default;

	[[nodiscard]] const std::string& name() const;
	[[nodiscard]] DataType           type() const;
	[[nodiscard]] size_t             size() const;
	[[nodiscard]] bool               empty() const;

	template <typename T>
	requires(std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>)
	[[nodiscard]] std::span<const T> values() const {
		const auto untyped = values_untyped(data_type_for<T>(), sizeof(T));
		return {static_cast<const T*>(untyped.data), untyped.size};
	}

private:
	struct UntypedSpan {
		const void* data = nullptr;
		size_t      size = 0;
	};

	ColumnView(const void* impl, size_t rowgroup_index, size_t column_index) noexcept;

	template <typename T>
	static constexpr DataType data_type_for() {
		if constexpr (std::is_same_v<T, int8_t>) {
			return DataType::I8;
		} else {
			static_assert(std::is_same_v<T, int16_t>);
			return DataType::I16;
		}
	}

	[[nodiscard]] UntypedSpan values_untyped(DataType expected_type, size_t element_size) const;

	const void* impl_           = nullptr;
	size_t      rowgroup_index_ = 0;
	size_t      column_index_   = 0;

	friend class RowgroupView;
};

class RowgroupView {
public:
	RowgroupView() noexcept = default;

	[[nodiscard]] size_t     column_count() const;
	[[nodiscard]] ColumnView column(size_t column_index) const;
	[[nodiscard]] bool       empty() const;

private:
	RowgroupView(const void* impl, size_t rowgroup_index) noexcept;

	const void* impl_           = nullptr;
	size_t      rowgroup_index_ = 0;

	friend class Table;
};

class Table {
public:
	Table() noexcept;
	~Table();

	Table(Table&&) noexcept;
	Table& operator=(Table&&) noexcept;

	Table(const Table&)            = delete;
	Table& operator=(const Table&) = delete;

	[[nodiscard]] size_t                     rowgroup_count() const noexcept;
	[[nodiscard]] size_t                     total_columns() const noexcept;
	[[nodiscard]] size_t                     rowgroup_column_count(size_t rowgroup_index) const;
	[[nodiscard]] RowgroupView               rowgroup(size_t rowgroup_index) const;
	[[nodiscard]] const std::vector<size_t>& rowgroup_column_counts() const noexcept;
	[[nodiscard]] bool                       empty() const noexcept;

private:
	struct Impl;

	explicit Table(std::unique_ptr<Impl> impl) noexcept;

	std::unique_ptr<Impl> impl_;

	friend class ColumnView;
	friend class RowgroupView;
	friend Table decompress_table(const std::filesystem::path& fls_path, const DecompressOptions& options);
};

Table decompress_table(const std::filesystem::path& fls_path, const DecompressOptions& options = {});

} // namespace galp

#endif // GALP_TABLE_HPP
