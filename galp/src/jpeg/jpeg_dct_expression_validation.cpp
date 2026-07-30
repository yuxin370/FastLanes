#include "jpeg/jpeg_dct_expression_validation.hpp"
#include "core/operator_capabilities.hpp"
#include "fls/expression/rpn.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/file/file_header.hpp"
#include "fls/footer/table_descriptor.hpp"
#include "fls/io/file.hpp"
#include "jpeg/jpeg_dct_plan_types.hpp"
#include <sstream>
#include <stdexcept>
#include <string>

namespace galp::jpeg::detail {

void validate_jpeg_dct_fls_gpu_expressions(const std::filesystem::path& fls_path, const uint32_t shard_id) {
	fastlanes::File       file(fls_path);
	fastlanes::FileHeader header {};
	fastlanes::FileFooter footer {};
	fastlanes::FileHeader::Load(header, file);
	fastlanes::FileFooter::Load(footer, file);
	if (!header.settings.inline_footer) {
		throw std::runtime_error("JPEG DCT GPU expression validation requires an inline footer: " + fls_path.string());
	}
	const auto descriptor = fastlanes::TableDescriptorHandle::FromFileSlice(
	    file, footer.table_descriptor_offset, footer.table_descriptor_size, true);
	const auto* table = descriptor.Get();
	if (table == nullptr || table->m_rowgroup_descriptors() == nullptr) {
		throw std::runtime_error("missing JPEG DCT table/rowgroup descriptor while validating staged shard " +
		                         std::to_string(shard_id));
	}

	const auto fail = [&](const size_t       rowgroup_index,
	                      const size_t       column_index,
	                      const std::string& token,
	                      const std::string& reason) -> void {
		std::ostringstream message;
		message << "JPEG DCT GPU expression validation failed: shard=" << shard_id << " rowgroup=" << rowgroup_index
		        << " column=" << column_index << " coefficient=" << column_index << " token=" << token << ": "
		        << reason;
		throw std::runtime_error(message.str());
	};

	for (size_t rowgroup_index = 0; rowgroup_index < table->m_rowgroup_descriptors()->size(); ++rowgroup_index) {
		const auto* rowgroup =
		    table->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rowgroup_index));
		if (rowgroup == nullptr || rowgroup->m_column_descriptors() == nullptr) {
			fail(rowgroup_index, 0, "<missing>", "missing rowgroup/column descriptor");
		}
		if (rowgroup->m_column_descriptors()->size() != kJpegDctCoefficientCount) {
			fail(rowgroup_index,
			     rowgroup->m_column_descriptors()->size(),
			     "<missing>",
			     "expected exactly 64 JPEG DCT coefficient columns");
		}
		for (size_t column_index = 0; column_index < rowgroup->m_column_descriptors()->size(); ++column_index) {
			const auto* column =
			    rowgroup->m_column_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(column_index));
			if (column == nullptr || column->encoding_rpn() == nullptr ||
			    column->encoding_rpn()->operator_tokens() == nullptr ||
			    column->encoding_rpn()->operator_tokens()->size() != 1U) {
				fail(rowgroup_index, column_index, "<missing>", "expected exactly one root operator token");
			}
			const auto token = column->encoding_rpn()->operator_tokens()->Get(0);
			if (!galp::expression::is_supported_token(token)) {
				fail(rowgroup_index,
				     column_index,
				     fastlanes::token_to_string(token),
				     "operator is not supported by the GALP GPU runtime");
			}
		}
	}
}

} // namespace galp::jpeg::detail
