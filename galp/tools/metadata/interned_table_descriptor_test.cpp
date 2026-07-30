#include "fls/footer/column_descriptor_generated.h"
#include "fls/footer/footer_generated.h"
#include "fls/footer/rowgroup_descriptor_generated.h"
#include "fls/footer/rpn_generated.h"
#include "fls/footer/table_descriptor.hpp"
#include "interned_table_descriptor.hpp"
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

flatbuffers::Offset<fastlanes::ColumnDescriptor> make_column(flatbuffers::FlatBufferBuilder& builder,
                                                             const uint64_t                  column_offset) {
	const std::vector<fastlanes::OperatorToken> operators {fastlanes::OperatorToken::EXP_UNCOMPRESSED_I16};
	const std::vector<uint64_t>                 operands {1024U};
	const auto                                  rpn  = fastlanes::CreateRPNDirect(builder, &operators, &operands);
	const auto                                  name = builder.CreateString("coefficient_00");
	const std::vector<uint8_t>                  maximum_bytes {0x2aU, 0x00U};
	const auto                                  maximum = fastlanes::CreateBinaryValueDirect(builder, &maximum_bytes);

	std::vector<flatbuffers::Offset<fastlanes::ExpressionResult>> expressions;
	expressions.push_back(
	    fastlanes::CreateExpressionResult(builder, fastlanes::OperatorToken::EXP_UNCOMPRESSED_I16, 1024U));
	const auto expression_vector = builder.CreateVector(expressions);

	std::vector<flatbuffers::Offset<fastlanes::SegmentDescriptor>> segments;
	segments.push_back(
	    fastlanes::CreateSegmentDescriptor(builder, 0U, 128U, 128U, 2048U, fastlanes::EntryPointType::UINT8));
	const auto segment_vector = builder.CreateVector(segments);
	return fastlanes::CreateColumnDescriptor(builder,
	                                         fastlanes::DataType::INT16,
	                                         rpn,
	                                         0U,
	                                         name,
	                                         {},
	                                         maximum,
	                                         column_offset,
	                                         2176U,
	                                         expression_vector,
	                                         segment_vector,
	                                         0U,
	                                         {});
}

std::vector<uint8_t> make_legacy_descriptor() {
	flatbuffers::FlatBufferBuilder                                  builder;
	std::vector<flatbuffers::Offset<fastlanes::RowgroupDescriptor>> rowgroups;
	for (uint64_t rowgroup = 0; rowgroup < 4U; ++rowgroup) {
		std::vector<flatbuffers::Offset<fastlanes::ColumnDescriptor>> columns;
		columns.push_back(make_column(builder, 0U));
		columns.push_back(make_column(builder, 2176U));
		const auto column_vector = builder.CreateVector(columns);
		rowgroups.push_back(
		    fastlanes::CreateRowgroupDescriptor(builder, 1U, column_vector, 4352U, 24U + rowgroup * 4352U, 1024U));
	}
	const auto rowgroup_vector = builder.CreateVector(rowgroups);
	const auto root            = fastlanes::CreateTableDescriptor(builder, rowgroup_vector, 17432U);
	fastlanes::FinishTableDescriptorBuffer(builder, root);
	auto buffer = builder.Release();
	return {buffer.data(), buffer.data() + buffer.size()};
}

int fail(const std::string& message) {
	std::cerr << "galp_metadata_interning_test: " << message << '\n';
	return 1;
}

} // namespace

int main() {
	try {
		auto legacy         = make_legacy_descriptor();
		auto handle         = fastlanes::TableDescriptorHandle::FromBytes(legacy, true);
		auto packed         = galp::metadata::pack_interned_table_descriptor(*handle, handle.size());
		auto compact_handle = fastlanes::TableDescriptorHandle::FromBytes(packed.bytes, true);
		if (!compact_handle || compact_handle->m_table_binary_size() != handle->m_table_binary_size()) {
			return fail("round-trip descriptor root differs");
		}
		if (packed.bytes.size() >= handle.size()) {
			return fail("repeated fixture descriptor did not shrink");
		}
		if (packed.stats.strings.references != 8U || packed.stats.strings.emitted != 1U ||
		    packed.stats.rpns.references != 8U || packed.stats.rpns.emitted != 1U ||
		    packed.stats.expression_results.references != 8U || packed.stats.expression_results.emitted != 1U ||
		    packed.stats.segment_descriptors.references != 8U || packed.stats.segment_descriptors.emitted != 1U ||
		    packed.stats.binary_values.references != 8U || packed.stats.binary_values.emitted != 1U) {
			return fail("interning counters disagree with repeated fixture");
		}
		const auto* rowgroups = compact_handle->m_rowgroup_descriptors();
		if (rowgroups == nullptr || rowgroups->size() != 4U ||
		    rowgroups->Get(3)->m_column_descriptors()->Get(1)->column_offset() != 2176U) {
			return fail("random rowgroup/column lookup changed");
		}
		auto corrupt  = packed.bytes;
		corrupt[0]    = 0xffU;
		corrupt[1]    = 0xffU;
		corrupt[2]    = 0xffU;
		corrupt[3]    = 0x7fU;
		bool rejected = false;
		try {
			static_cast<void>(fastlanes::TableDescriptorHandle::FromBytes(std::move(corrupt), true));
		} catch (const std::exception&) { rejected = true; }
		if (!rejected) {
			return fail("corrupt root offset was not rejected");
		}
		bool oversized_rejected = false;
		try {
			static_cast<void>(galp::metadata::pack_interned_table_descriptor(
			    *handle, static_cast<std::size_t>(std::numeric_limits<flatbuffers::uoffset_t>::max()) + 1U));
		} catch (const std::exception&) { oversized_rejected = true; }
		if (!oversized_rejected) {
			return fail("descriptor beyond the FlatBuffers offset limit was not rejected");
		}
		return 0;
	} catch (const std::exception& error) { return fail(error.what()); }
}
