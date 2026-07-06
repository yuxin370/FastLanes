#include <filesystem>
#include <galp/galp.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>

int main() {
	static_assert(!std::is_copy_constructible_v<galp::Reader>);
	static_assert(!std::is_copy_assignable_v<galp::Reader>);
	static_assert(std::is_move_constructible_v<galp::Reader>);
	static_assert(!std::is_copy_constructible_v<galp::Table>);
	static_assert(!std::is_copy_assignable_v<galp::Table>);
	static_assert(std::is_move_constructible_v<galp::Table>);

#if GALP_WITH_JPEG_DCT
	static_assert(!std::is_copy_constructible_v<galp::jpeg::DirectDctBatch>);
	static_assert(!std::is_copy_assignable_v<galp::jpeg::DirectDctBatch>);
	static_assert(std::is_move_constructible_v<galp::jpeg::DirectDctBatch>);

	galp::jpeg::DirectDctBatch direct_dct_batch;
	const auto                 direct_dct_tensor = direct_dct_batch.tensor();
	if (direct_dct_tensor.data != nullptr || direct_dct_tensor.rows() != 0 || direct_dct_tensor.columns() != 64 ||
	    direct_dct_tensor.strides[0] != 64 || direct_dct_tensor.strides[1] != 1 ||
	    direct_dct_tensor.dtype != galp::jpeg::DirectDctTensorDataType::kInt16 ||
	    direct_dct_tensor.device != galp::jpeg::DirectDctTensorDevice::kCuda) {
		return 8;
	}
#endif

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
