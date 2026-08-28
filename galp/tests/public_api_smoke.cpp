#include <filesystem>
#include <galp/stable.hpp>
#include <concepts>
#include <stdexcept>
#include <string>
#include <type_traits>

template <typename T>
concept GalpColumnValue = requires(const galp::ColumnView& column) {
	{ column.template values<T>() } -> std::same_as<std::span<const T>>;
};

int main() {
	static_assert(GALP_STABLE_API == 1);
	static_assert(!std::is_copy_constructible_v<galp::Reader>);
	static_assert(!std::is_copy_assignable_v<galp::Reader>);
	static_assert(std::is_move_constructible_v<galp::Reader>);
	static_assert(!std::is_copy_constructible_v<galp::Table>);
	static_assert(!std::is_copy_assignable_v<galp::Table>);
	static_assert(std::is_move_constructible_v<galp::Table>);
	static_assert(GalpColumnValue<int8_t>);
	static_assert(GalpColumnValue<int16_t>);
	static_assert(!GalpColumnValue<uint32_t>);
	static_assert(!GalpColumnValue<uint64_t>);
	static_assert(!GalpColumnValue<float>);
	static_assert(!GalpColumnValue<double>);

	galp::DecompressOptions options {};
	options.scope                   = galp::TableDecompressionScope::PerRowgroup;
	options.write_output            = false;
	options.advanced.prefetch_depth = 2;

	galp::Table table;
	if (!table.empty()) {
		return 1;
	}
	if (table.rowgroup_count() != 0 || table.total_columns() != 0) {
		return 2;
	}
	if (!table.rowgroup_column_counts().empty()) {
		return 3;
	}
	galp::RowgroupView rowgroup;
	galp::ColumnView   column;
	(void)rowgroup;
	(void)column;

	const auto missing_path = std::filesystem::path {"__galp_public_api_smoke_missing__"} / "data.fls";
	if (std::filesystem::exists(missing_path)) {
		return 4;
	}

	galp::Reader reader(missing_path);
	if (reader.path().empty()) {
		return 5;
	}

	try {
		(void)reader.decompress(options);
	} catch (const galp::UnsupportedFormatError&) { return 6; } catch (const std::exception& e) {
		const std::string message = e.what();
		if (message.find("table materialization requires execution.write_out=true") != std::string::npos) {
			return 7;
		}
	}

	return 0;
}
