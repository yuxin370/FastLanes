#include "fls/connection.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/file/file_header.hpp"
#include "fls/footer/table_descriptor.hpp"
#include "fls/footer/table_descriptor_generated.h"
#include "fls/io/file.hpp"
#include "fls/table/memory_table.hpp"
#include "format/compact_descriptor_v3.hpp"
#include "format/compact_shard_reader.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class TemporaryDirectory {
public:
	TemporaryDirectory() {
		const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
		path_             = std::filesystem::temp_directory_path() / ("galp_compact_v3_" + std::to_string(suffix));
		std::filesystem::create_directories(path_);
	}
	~TemporaryDirectory() {
		std::error_code ignored;
		std::filesystem::remove_all(path_, ignored);
	}
	[[nodiscard]] const std::filesystem::path& path() const noexcept {
		return path_;
	}

private:
	std::filesystem::path path_;
};

void write_standard_vector_rowgroups(const std::filesystem::path& path) {
	std::array<std::vector<int16_t>, 2> values;
	for (size_t row = 0U; row < 1500U; ++row) {
		const size_t rank = row % 61U;
		values[0].push_back(static_cast<int16_t>(static_cast<int>(rank) - 30));
		values[1].push_back(static_cast<int16_t>(static_cast<int>((rank * 37U) % 61U) * 41 - 15000));
	}
	std::array<fastlanes::MemoryColumn, 2> columns;
	columns[0].name = "dct_zz_00";
	columns[0].data = std::span<const int16_t>(values[0]);
	columns[1].name = "dct_zz_01";
	columns[1].data = std::span<const int16_t>(values[1]);
	const fastlanes::MemoryTable        table {std::span<const fastlanes::MemoryColumn>(columns)};
	const std::array<fastlanes::n_t, 2> rowgroup_rows {{1024U, 476U}};
	fastlanes::MemoryTableOptions       options;
	options.n_vectors_per_rowgroup = 1U;
	options.rowgroup_n_tuples      = std::span<const fastlanes::n_t>(rowgroup_rows);
	fastlanes::Connection connection;
	fastlanes::load_memory_table(connection, table, options);
	connection.inline_footer();
	connection.to_fls(path);
}

void write_standard_zero_payload_rowgroup(const std::filesystem::path& path) {
	std::array<std::vector<int16_t>, 64>    values;
	std::array<fastlanes::MemoryColumn, 64> columns;
	for (size_t column = 0U; column < columns.size(); ++column) {
		values[column].assign(1024U, static_cast<int16_t>(column));
		columns[column].name = "dct_zz_" + std::to_string(column);
		columns[column].data = std::span<const int16_t>(values[column]);
	}
	const fastlanes::MemoryTable        table {std::span<const fastlanes::MemoryColumn>(columns)};
	const std::array<fastlanes::n_t, 1> rowgroup_rows {{1024U}};
	fastlanes::MemoryTableOptions       options;
	options.n_vectors_per_rowgroup = 1U;
	options.rowgroup_n_tuples      = std::span<const fastlanes::n_t>(rowgroup_rows);
	fastlanes::Connection connection;
	fastlanes::load_memory_table(connection, table, options);
	connection.inline_footer();
	connection.to_fls(path);
}

void replace_zero_payload_operator(const std::filesystem::path& path,
	                               const fastlanes::OperatorToken token) {
	fastlanes::File       file(path);
	fastlanes::FileHeader header {};
	fastlanes::FileFooter footer {};
	if (!fastlanes::FileHeader::Load(header, file).success || !fastlanes::FileFooter::Load(footer, file).success) {
		throw std::runtime_error("failed to load zero-payload fixture header/footer");
	}
	auto descriptor = fastlanes::TableDescriptorHandle::FromFileSlice(
	    file, footer.table_descriptor_offset, footer.table_descriptor_size, true);
	auto native = descriptor.Unpack();
	if (native == nullptr || native->m_rowgroup_descriptors.size() != 1U ||
	    native->m_rowgroup_descriptors.front() == nullptr ||
	    native->m_rowgroup_descriptors.front()->m_column_descriptors.empty() ||
	    native->m_rowgroup_descriptors.front()->m_column_descriptors.front() == nullptr ||
	    native->m_rowgroup_descriptors.front()->m_column_descriptors.front()->encoding_rpn == nullptr) {
		throw std::runtime_error("zero-payload fixture has an unexpected descriptor shape");
	}
	auto& rpn = *native->m_rowgroup_descriptors.front()->m_column_descriptors.front()->encoding_rpn;
	rpn.operator_tokens.assign(1U, token);
	rpn.operand_tokens.clear();
	const auto rewritten = fastlanes::TableDescriptorHandle::FromNative(*native);
	footer.table_descriptor_offset = sizeof(fastlanes::FileHeader);
	footer.table_descriptor_size   = rewritten.size();
	std::ofstream output(path, std::ios::binary | std::ios::trunc);
	if (!output) {
		throw std::runtime_error("failed to rewrite zero-payload fixture");
	}
	output.write(reinterpret_cast<const char*>(&header), sizeof(header));
	output.write(reinterpret_cast<const char*>(rewritten.data()), static_cast<std::streamsize>(rewritten.size()));
	output.write(reinterpret_cast<const char*>(&footer), sizeof(footer));
	if (!output) {
		throw std::runtime_error("failed to finalize rewritten zero-payload fixture");
	}
}

fastlanes::TableDescriptorHandle load_standard_descriptor(const std::filesystem::path& path,
                                                          fastlanes::FileFooter&       footer) {
	fastlanes::File file(path);
	EXPECT_TRUE(fastlanes::FileFooter::Load(footer, file).success);
	return fastlanes::TableDescriptorHandle::FromFileSlice(
	    file, footer.table_descriptor_offset, footer.table_descriptor_size, true);
}

void clear_expression_candidate_sizes(fastlanes::ColumnDescriptorT& column) {
	for (auto& expression : column.expr_space) {
		ASSERT_NE(expression, nullptr);
		expression->size = 0U;
	}
	for (auto& child : column.children) {
		ASSERT_NE(child, nullptr);
		clear_expression_candidate_sizes(*child);
	}
}

void expect_column_equal(const fastlanes::ColumnDescriptor& expected, const fastlanes::ColumnDescriptor& actual) {
	std::unique_ptr<fastlanes::ColumnDescriptorT> expected_native(expected.UnPack());
	std::unique_ptr<fastlanes::ColumnDescriptorT> actual_native(actual.UnPack());
	ASSERT_NE(expected_native, nullptr);
	ASSERT_NE(actual_native, nullptr);
	clear_expression_candidate_sizes(*expected_native);
	flatbuffers::FlatBufferBuilder expected_builder;
	flatbuffers::FlatBufferBuilder actual_builder;
	const auto expected_root = fastlanes::ColumnDescriptor::Pack(expected_builder, expected_native.get());
	const auto actual_root   = fastlanes::ColumnDescriptor::Pack(actual_builder, actual_native.get());
	fastlanes::FinishColumnDescriptorBuffer(expected_builder, expected_root);
	fastlanes::FinishColumnDescriptorBuffer(actual_builder, actual_root);
	EXPECT_EQ(std::vector<uint8_t>(expected_builder.GetBufferPointer(),
	                               expected_builder.GetBufferPointer() + expected_builder.GetSize()),
	          std::vector<uint8_t>(actual_builder.GetBufferPointer(),
	                               actual_builder.GetBufferPointer() + actual_builder.GetSize()));
	EXPECT_EQ(expected.data_type(), actual.data_type());
	EXPECT_EQ(expected.idx(), actual.idx());
	ASSERT_NE(expected.name(), nullptr);
	ASSERT_NE(actual.name(), nullptr);
	EXPECT_EQ(expected.name()->string_view(), actual.name()->string_view());
	EXPECT_EQ(expected.column_offset(), actual.column_offset());
	EXPECT_EQ(expected.total_size(), actual.total_size());
	EXPECT_EQ(expected.n_null(), actual.n_null());
	ASSERT_NE(expected.encoding_rpn(), nullptr);
	ASSERT_NE(actual.encoding_rpn(), nullptr);
	ASSERT_NE(expected.encoding_rpn()->operator_tokens(), nullptr);
	ASSERT_NE(actual.encoding_rpn()->operator_tokens(), nullptr);
	ASSERT_EQ(expected.encoding_rpn()->operator_tokens()->size(), actual.encoding_rpn()->operator_tokens()->size());
	for (size_t index = 0U; index < expected.encoding_rpn()->operator_tokens()->size(); ++index) {
		EXPECT_EQ(expected.encoding_rpn()->operator_tokens()->Get(static_cast<flatbuffers::uoffset_t>(index)),
		          actual.encoding_rpn()->operator_tokens()->Get(static_cast<flatbuffers::uoffset_t>(index)));
	}
	const auto* expected_operands = expected.encoding_rpn()->operand_tokens();
	const auto* actual_operands   = actual.encoding_rpn()->operand_tokens();
	ASSERT_EQ(expected_operands == nullptr, actual_operands == nullptr);
	if (expected_operands != nullptr) {
		ASSERT_EQ(expected_operands->size(), actual_operands->size());
		for (size_t index = 0U; index < expected_operands->size(); ++index) {
			EXPECT_EQ(expected_operands->Get(static_cast<flatbuffers::uoffset_t>(index)),
			          actual_operands->Get(static_cast<flatbuffers::uoffset_t>(index)));
		}
	}
	ASSERT_NE(expected.max(), nullptr);
	ASSERT_NE(actual.max(), nullptr);
	ASSERT_NE(expected.max()->binary_data(), nullptr);
	ASSERT_NE(actual.max()->binary_data(), nullptr);
	ASSERT_EQ(expected.max()->binary_data()->size(), actual.max()->binary_data()->size());
	for (size_t index = 0U; index < expected.max()->binary_data()->size(); ++index) {
		EXPECT_EQ(expected.max()->binary_data()->Get(static_cast<flatbuffers::uoffset_t>(index)),
		          actual.max()->binary_data()->Get(static_cast<flatbuffers::uoffset_t>(index)));
	}
	const auto* expected_expressions = expected.expr_space();
	const auto* actual_expressions   = actual.expr_space();
	ASSERT_EQ(expected_expressions == nullptr, actual_expressions == nullptr);
	if (expected_expressions != nullptr) {
		ASSERT_EQ(expected_expressions->size(), actual_expressions->size());
		for (size_t index = 0U; index < expected_expressions->size(); ++index) {
			const auto* left  = expected_expressions->Get(static_cast<flatbuffers::uoffset_t>(index));
			const auto* right = actual_expressions->Get(static_cast<flatbuffers::uoffset_t>(index));
			ASSERT_NE(left, nullptr);
			ASSERT_NE(right, nullptr);
			EXPECT_EQ(left->operator_token(), right->operator_token());
			EXPECT_EQ(right->size(), 0U);
		}
	}
	const auto* expected_segments = expected.segment_descriptors();
	const auto* actual_segments   = actual.segment_descriptors();
	ASSERT_EQ(expected_segments == nullptr, actual_segments == nullptr);
	if (expected_segments != nullptr) {
		ASSERT_EQ(expected_segments->size(), actual_segments->size());
		for (size_t index = 0U; index < expected_segments->size(); ++index) {
			const auto* left  = expected_segments->Get(static_cast<flatbuffers::uoffset_t>(index));
			const auto* right = actual_segments->Get(static_cast<flatbuffers::uoffset_t>(index));
			ASSERT_NE(left, nullptr);
			ASSERT_NE(right, nullptr);
			EXPECT_EQ(left->entrypoint_offset(), right->entrypoint_offset());
			EXPECT_EQ(left->entrypoint_size(), right->entrypoint_size());
			EXPECT_EQ(left->data_offset(), right->data_offset());
			EXPECT_EQ(left->data_size(), right->data_size());
			EXPECT_EQ(left->entry_point_t(), right->entry_point_t());
		}
	}
}

void expect_payload_prefix_equal(const std::filesystem::path& expected,
                                 const std::filesystem::path& actual,
                                 const uint64_t               bytes) {
	std::ifstream left(expected, std::ios::binary);
	std::ifstream right(actual, std::ios::binary);
	ASSERT_TRUE(left);
	ASSERT_TRUE(right);
	std::vector<char> left_bytes(static_cast<size_t>(bytes));
	std::vector<char> right_bytes(static_cast<size_t>(bytes));
	left.read(left_bytes.data(), static_cast<std::streamsize>(left_bytes.size()));
	right.read(right_bytes.data(), static_cast<std::streamsize>(right_bytes.size()));
	ASSERT_EQ(left.gcount(), static_cast<std::streamsize>(left_bytes.size()));
	ASSERT_EQ(right.gcount(), static_cast<std::streamsize>(right_bytes.size()));
	EXPECT_EQ(left_bytes, right_bytes);
}

uint64_t crc64_ecma(const std::span<const uint8_t> bytes) {
	constexpr uint64_t polynomial = UINT64_C(0x42f0e1eba9ea3693);
	uint64_t           crc        = 0U;
	for (const auto byte : bytes) {
		crc ^= static_cast<uint64_t>(byte) << 56U;
		for (unsigned bit = 0U; bit < 8U; ++bit) {
			crc = (crc & (UINT64_C(1) << 63U)) != 0U ? (crc << 1U) ^ polynomial : crc << 1U;
		}
	}
	return crc;
}

uint64_t read_le_u64(const std::span<const uint8_t> bytes, const size_t offset) {
	EXPECT_LE(offset + sizeof(uint64_t), bytes.size());
	uint64_t value = 0U;
	for (size_t byte = 0U; byte < sizeof(uint64_t); ++byte) {
		value |= static_cast<uint64_t>(bytes[offset + byte]) << (byte * 8U);
	}
	return value;
}

void write_le_u64(const std::span<uint8_t> bytes, const size_t offset, const uint64_t value) {
	ASSERT_LE(offset + sizeof(uint64_t), bytes.size());
	for (size_t byte = 0U; byte < sizeof(uint64_t); ++byte) {
		bytes[offset + byte] = static_cast<uint8_t>(value >> (byte * 8U));
	}
}

} // namespace

TEST(CompactDescriptorV3, AcceptsZeroPayloadConstantVectorRowgroup) {
	TemporaryDirectory temp;
	const auto         standard = temp.path() / "constant-standard.fls";
	const auto         compact  = temp.path() / "constant-compact.fls";
	write_standard_zero_payload_rowgroup(standard);

	fastlanes::FileFooter standard_footer {};
	const auto            standard_descriptor = load_standard_descriptor(standard, standard_footer);
	ASSERT_NE(standard_descriptor.Get(), nullptr);
	ASSERT_NE(standard_descriptor->m_rowgroup_descriptors(), nullptr);
	ASSERT_EQ(standard_descriptor->m_rowgroup_descriptors()->size(), 1U);
	const auto* standard_rowgroup = standard_descriptor->m_rowgroup_descriptors()->Get(0U);
	ASSERT_NE(standard_rowgroup, nullptr);
	ASSERT_EQ(standard_rowgroup->m_size(), 0U);
	ASSERT_EQ(standard_footer.table_descriptor_offset, sizeof(fastlanes::FileHeader));

	galp::format::CompactV3BuildOptions options;
	options.vector_size   = 1024U;
	options.spatial_order = 3U;
	galp::format::CompactV3ImageInput image;
	image.first_rowgroup     = 0U;
	image.rowgroup_count     = 1U;
	image.real_row_count     = 1024U;
	image.first_physical_row = 0U;
	image.components.push_back({0U, 32U, 32U, 32U, 32U, 0U, 0U});
	options.images.push_back(image);

	const auto report = galp::format::compact_standard_fls_to_v3(standard, compact, options);
	EXPECT_EQ(report.payload_bytes, 0U);
	auto parsed = galp::format::CompactDescriptorV3::Open(compact);
	ASSERT_EQ(parsed.rowgroup_count(), 1U);
	EXPECT_EQ(parsed.rowgroup(0U).payload_size, 0U);
	for (size_t column = 0U; column < parsed.column_count(); ++column) {
		EXPECT_EQ(parsed.coefficient_range(0U, column).size, 0U);
	}

	const auto audit = galp::format::verify_compact_v3_payload(compact);
	EXPECT_TRUE(audit.exact());
	galp::format::CompactShardReader reader(compact);
	const auto                       plan = reader.Plan({0U}, {0U});
	EXPECT_EQ(plan.stats().logical_bytes, 0U);
	EXPECT_EQ(plan.stats().read_bytes, 0U);
	EXPECT_TRUE(plan.ranges().empty());
	const auto read = reader.Read(plan);
	ASSERT_EQ(read.rowgroups.size(), 1U);
	EXPECT_TRUE(read.rowgroups.front().bytes.empty());
}

TEST(CompactDescriptorV3, MappingTelemetryTracksRetainedLifetime) {
	TemporaryDirectory temp;
	const auto         standard = temp.path() / "mapping-standard.fls";
	const auto         compact  = temp.path() / "mapping-compact.fls";
	write_standard_vector_rowgroups(standard);

	galp::format::CompactV3BuildOptions options;
	options.vector_size   = 1024U;
	options.spatial_order = 3U;
	(void)galp::format::compact_standard_fls_to_v3(standard, compact, options);

	const auto before = galp::format::compact_descriptor_v3_mapping_stats();
	size_t     descriptor_bytes = 0U;
	{
		auto descriptor = galp::format::CompactDescriptorV3::Open(compact);
		descriptor_bytes = descriptor.descriptor_bytes();
		ASSERT_GT(descriptor_bytes, 0U);
		const auto live = galp::format::compact_descriptor_v3_mapping_stats();
		EXPECT_EQ(live.current_mapping_count, before.current_mapping_count + 1U);
		EXPECT_EQ(live.map_count, before.map_count + 1U);
		EXPECT_GT(live.current_mapped_bytes, before.current_mapped_bytes);
		EXPECT_GE(live.peak_mapping_count, live.current_mapping_count);
		EXPECT_GE(live.peak_mapped_bytes, live.current_mapped_bytes);
		descriptor.release_resident_pages();
		EXPECT_EQ(descriptor.rowgroup_count(), 2U);
	}
	const auto after = galp::format::compact_descriptor_v3_mapping_stats();
	EXPECT_EQ(after.current_mapping_count, before.current_mapping_count);
	EXPECT_EQ(after.current_mapped_bytes, before.current_mapped_bytes);
	EXPECT_EQ(after.unmap_count, before.unmap_count + 1U);
}

TEST(CompactDescriptorV3, RejectsZeroPayloadWithPayloadDependentOperator) {
	TemporaryDirectory temp;
	const auto         standard = temp.path() / "malformed-standard.fls";
	const auto         compact  = temp.path() / "malformed-compact.fls";
	write_standard_zero_payload_rowgroup(standard);
	replace_zero_payload_operator(standard, fastlanes::OperatorToken::EXP_UNCOMPRESSED_I16);

	galp::format::CompactV3BuildOptions options;
	options.vector_size   = 1024U;
	options.spatial_order = 3U;
	try {
		(void)galp::format::compact_standard_fls_to_v3(standard, compact, options);
		FAIL() << "payload-dependent zero-payload rowgroup was accepted";
	} catch (const std::runtime_error& error) {
		EXPECT_NE(std::string(error.what()).find("payload-dependent operator"), std::string::npos);
	}
	EXPECT_FALSE(std::filesystem::exists(compact));
}

TEST(CompactDescriptorV3, CompactsParsesAndExpandsWithoutChangingPayload) {
	TemporaryDirectory temp;
	const auto         standard = temp.path() / "standard.fls";
	const auto         compact  = temp.path() / "compact.fls";
	const auto         expanded = temp.path() / "expanded.fls";
	write_standard_vector_rowgroups(standard);

	fastlanes::FileFooter standard_footer {};
	const auto            standard_descriptor = load_standard_descriptor(standard, standard_footer);
	ASSERT_NE(standard_descriptor.Get(), nullptr);
	ASSERT_NE(standard_descriptor->m_rowgroup_descriptors(), nullptr);
	ASSERT_EQ(standard_descriptor->m_rowgroup_descriptors()->size(), 2U);

	galp::format::CompactV3BuildOptions options;
	options.vector_size   = 1024U;
	options.spatial_order = 3U;
	galp::format::CompactV3ImageInput image;
	image.first_rowgroup     = 0U;
	image.rowgroup_count     = 2U;
	image.real_row_count     = 1500U;
	image.first_physical_row = 0U;
	image.components.push_back({0U, 50U, 30U, 50U, 30U, 0U, 0U});
	options.images.push_back(image);
	const auto compact_report = galp::format::compact_standard_fls_to_v3(standard, compact, options);
	EXPECT_TRUE(galp::format::is_compact_v3_fls(compact));
	EXPECT_EQ(compact_report.payload_bytes, standard_footer.table_descriptor_offset - sizeof(fastlanes::FileHeader));
	EXPECT_LT(compact_report.compact_descriptor_bytes, compact_report.source_descriptor_bytes);

	auto parsed = galp::format::CompactDescriptorV3::Open(compact);
	EXPECT_EQ(parsed.vector_size(), 1024U);
	EXPECT_EQ(parsed.spatial_order(), 3U);
	EXPECT_EQ(parsed.rowgroup_count(), 2U);
	EXPECT_EQ(parsed.column_count(), 2U);
	EXPECT_EQ(parsed.image_count(), 1U);
	EXPECT_GT(parsed.schema_count(), 0U);
	const auto payload_audit = galp::format::verify_compact_v3_payload(compact);
	EXPECT_TRUE(payload_audit.exact());
	EXPECT_EQ(payload_audit.expected_payload_crc64, compact_report.payload_crc64);
	EXPECT_EQ(payload_audit.actual_payload_crc64, compact_report.payload_crc64);
	EXPECT_EQ(payload_audit.actual_rowgroup_crc64.size(), parsed.rowgroup_count());
	EXPECT_TRUE(payload_audit.rowgroup_crc_mismatches.empty());
	const auto parsed_image = parsed.image(0U);
	EXPECT_EQ(parsed_image.first_rowgroup, 0U);
	EXPECT_EQ(parsed_image.rowgroup_count, 2U);
	EXPECT_EQ(parsed_image.real_row_count, 1500U);
	EXPECT_EQ(parsed_image.component_count, 1U);
	const auto parsed_component = parsed.component(0U);
	EXPECT_EQ(parsed_component.width_in_blocks, 50U);
	EXPECT_EQ(parsed_component.height_in_blocks, 30U);

	for (size_t rowgroup_index = 0U; rowgroup_index < parsed.rowgroup_count(); ++rowgroup_index) {
		const auto* expected =
		    standard_descriptor->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rowgroup_index));
		const auto actual = parsed.unpack_rowgroup(rowgroup_index);
		ASSERT_NE(expected, nullptr);
		ASSERT_NE(actual, nullptr);
		EXPECT_EQ(expected->m_n_vec(), actual->m_n_vec);
		EXPECT_EQ(expected->m_size(), actual->m_size);
		EXPECT_EQ(expected->m_offset(), actual->m_offset);
		EXPECT_EQ(expected->m_n_tuples(), actual->m_n_tuples);
		ASSERT_NE(expected->m_column_descriptors(), nullptr);
		ASSERT_EQ(expected->m_column_descriptors()->size(), actual->m_column_descriptors.size());
		flatbuffers::FlatBufferBuilder builder;
		const auto                     root = fastlanes::RowgroupDescriptor::Pack(builder, actual.get());
		fastlanes::FinishRowgroupDescriptorBuffer(builder, root);
		const auto* actual_view = fastlanes::GetRowgroupDescriptor(builder.GetBufferPointer());
		for (size_t column = 0U; column < actual->m_column_descriptors.size(); ++column) {
			expect_column_equal(*expected->m_column_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(column)),
			                    *actual_view->m_column_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(column)));
			const auto range = parsed.coefficient_range(rowgroup_index, column);
			EXPECT_EQ(range.offset, actual->m_column_descriptors[column]->column_offset);
			EXPECT_EQ(range.size, actual->m_column_descriptors[column]->total_size);
		}
	}
	const auto corrupted = temp.path() / "compact-corrupted.fls";
	std::filesystem::copy_file(compact, corrupted);
	{
		std::fstream bytes(corrupted, std::ios::binary | std::ios::in | std::ios::out);
		ASSERT_TRUE(bytes);
		const auto first_rowgroup = parsed.rowgroup(0U);
		bytes.seekg(static_cast<std::streamoff>(first_rowgroup.payload_offset));
		char value = 0;
		bytes.read(&value, 1);
		ASSERT_EQ(bytes.gcount(), 1);
		value ^= 0x01;
		bytes.seekp(static_cast<std::streamoff>(first_rowgroup.payload_offset));
		bytes.write(&value, 1);
	}
	const auto corrupted_audit = galp::format::verify_compact_v3_payload(corrupted);
	EXPECT_FALSE(corrupted_audit.exact());
	ASSERT_EQ(corrupted_audit.rowgroup_crc_mismatches.size(), 1U);
	EXPECT_EQ(corrupted_audit.rowgroup_crc_mismatches.front(), 0U);
	const auto corrupted_descriptor = temp.path() / "compact-descriptor-corrupted.fls";
	std::filesystem::copy_file(compact, corrupted_descriptor);
	fastlanes::FileFooter compact_footer {};
	{
		fastlanes::File file(corrupted_descriptor);
		ASSERT_TRUE(fastlanes::FileFooter::Load(compact_footer, file).success);
	}
	{
		std::fstream bytes(corrupted_descriptor, std::ios::binary | std::ios::in | std::ios::out);
		ASSERT_TRUE(bytes);
		const auto offset = compact_footer.table_descriptor_offset + 200U;
		bytes.seekg(static_cast<std::streamoff>(offset));
		char value = 0;
		bytes.read(&value, 1);
		ASSERT_EQ(bytes.gcount(), 1);
		value ^= 0x01;
		bytes.seekp(static_cast<std::streamoff>(offset));
		bytes.write(&value, 1);
	}
	EXPECT_THROW((void)galp::format::CompactDescriptorV3::Open(corrupted_descriptor), std::runtime_error);

	const auto structurally_corrupted_descriptor = temp.path() / "compact-structurally-corrupted.fls";
	std::filesystem::copy_file(compact, structurally_corrupted_descriptor);
	{
		constexpr size_t     descriptor_checksum_offset = 168U;
		constexpr size_t     rowgroup_section_offset    = 120U;
		constexpr size_t     rowgroup_page_size_offset  = 24U;
		std::vector<uint8_t> descriptor(static_cast<size_t>(compact_footer.table_descriptor_size));
		std::fstream         bytes(structurally_corrupted_descriptor, std::ios::binary | std::ios::in | std::ios::out);
		ASSERT_TRUE(bytes);
		bytes.seekg(static_cast<std::streamoff>(compact_footer.table_descriptor_offset));
		bytes.read(reinterpret_cast<char*>(descriptor.data()), static_cast<std::streamsize>(descriptor.size()));
		ASSERT_EQ(bytes.gcount(), static_cast<std::streamsize>(descriptor.size()));
		const auto rowgroup_offset = read_le_u64(descriptor, rowgroup_section_offset);
		ASSERT_LE(rowgroup_offset + rowgroup_page_size_offset + sizeof(uint32_t), descriptor.size());
		std::fill_n(descriptor.begin() + static_cast<size_t>(rowgroup_offset) + rowgroup_page_size_offset,
		            sizeof(uint32_t),
		            uint8_t {0U});
		write_le_u64(descriptor, descriptor_checksum_offset, 0U);
		write_le_u64(descriptor, descriptor_checksum_offset, crc64_ecma(descriptor));
		bytes.seekp(static_cast<std::streamoff>(compact_footer.table_descriptor_offset));
		bytes.write(reinterpret_cast<const char*>(descriptor.data()), static_cast<std::streamsize>(descriptor.size()));
		ASSERT_TRUE(bytes);
	}
	try {
		(void)galp::format::CompactDescriptorV3::Open(structurally_corrupted_descriptor);
		FAIL() << "a structurally invalid descriptor with a valid CRC64 was accepted";
	} catch (const std::runtime_error& error) {
		EXPECT_NE(std::string(error.what()).find("dense non-empty partition"), std::string::npos);
	}

	galp::format::CompactShardReader compact_reader(compact);
	const auto                       full_all    = compact_reader.Plan({0U, 1U}, {0U, 1U});
	const auto                       crop_all    = compact_reader.Plan({0U}, {0U, 1U});
	const auto                       full_prefix = compact_reader.Plan({0U, 1U}, {0U});
	const auto                       crop_prefix = compact_reader.Plan({0U}, {0U});
	const auto                       mapped_only = compact_reader.Plan({0U}, {1U});
	EXPECT_EQ(full_all.stats().selected_vector_ratio, 1.0);
	EXPECT_EQ(full_all.stats().selected_coefficient_ratio, 1.0);
	EXPECT_EQ(crop_all.stats().selected_vector_ratio, 0.5);
	EXPECT_EQ(full_prefix.stats().selected_coefficient_ratio, 0.5);
	EXPECT_EQ(crop_prefix.stats().selected_vector_ratio, 0.5);
	EXPECT_EQ(crop_prefix.stats().selected_coefficient_ratio, 0.5);
	EXPECT_LT(crop_all.stats().logical_bytes, full_all.stats().logical_bytes);
	EXPECT_LT(full_prefix.stats().logical_bytes, full_all.stats().logical_bytes);
	EXPECT_LE(crop_prefix.stats().logical_bytes, crop_all.stats().logical_bytes);
	EXPECT_LE(crop_prefix.stats().logical_bytes, full_prefix.stats().logical_bytes);
	const auto dependency_rowgroup = parsed.unpack_rowgroup(0U);
	ASSERT_EQ(dependency_rowgroup->m_column_descriptors.size(), 2U);
	ASSERT_NE(dependency_rowgroup->m_column_descriptors[1], nullptr);
	ASSERT_NE(dependency_rowgroup->m_column_descriptors[1]->encoding_rpn, nullptr);
	ASSERT_FALSE(dependency_rowgroup->m_column_descriptors[1]->encoding_rpn->operator_tokens.empty());
	EXPECT_EQ(dependency_rowgroup->m_column_descriptors[1]->encoding_rpn->operator_tokens.front(),
	          fastlanes::OperatorToken::EXP_DICT_I16_U08);
	EXPECT_EQ(mapped_only.stats().selected_coefficient_count, 1U);
	EXPECT_GE(mapped_only.stats().range_count, 2U);
	EXPECT_GE(mapped_only.stats().logical_bytes,
	          static_cast<uint64_t>(parsed.coefficient_range(0U, 0U).size) + parsed.coefficient_range(0U, 1U).size);
	const auto prefix_read = compact_reader.Read(crop_prefix);
	ASSERT_EQ(prefix_read.rowgroups.size(), 1U);
	EXPECT_EQ(prefix_read.io.bytes_read, crop_prefix.stats().read_bytes);
	EXPECT_EQ(prefix_read.io.pread_count, crop_prefix.stats().coalesced_run_count);
	for (const auto& range : crop_prefix.ranges()) {
		std::ifstream source(compact, std::ios::binary);
		ASSERT_TRUE(source);
		std::vector<std::byte> expected(range.size);
		source.seekg(static_cast<std::streamoff>(range.file_offset));
		source.read(reinterpret_cast<char*>(expected.data()), static_cast<std::streamsize>(expected.size()));
		ASSERT_EQ(source.gcount(), static_cast<std::streamsize>(expected.size()));
		ASSERT_LE(static_cast<size_t>(range.backing_offset) + range.size, prefix_read.rowgroups[0].bytes.size());
		EXPECT_TRUE(std::equal(
		    expected.begin(), expected.end(), prefix_read.rowgroups[0].bytes.begin() + range.backing_offset));
	}

	const auto expand_report = galp::format::expand_compact_fls_v3(compact, expanded);
	EXPECT_FALSE(galp::format::is_compact_v3_fls(expanded));
	EXPECT_EQ(expand_report.payload_crc64, compact_report.payload_crc64);
	expect_payload_prefix_equal(standard, compact, standard_footer.table_descriptor_offset);
	expect_payload_prefix_equal(standard, expanded, standard_footer.table_descriptor_offset);

	fastlanes::FileFooter expanded_footer {};
	const auto            expanded_descriptor = load_standard_descriptor(expanded, expanded_footer);
	ASSERT_NE(expanded_descriptor.Get(), nullptr);
	ASSERT_NE(expanded_descriptor->m_rowgroup_descriptors(), nullptr);
	ASSERT_EQ(expanded_descriptor->m_rowgroup_descriptors()->size(), 2U);
	for (size_t rowgroup_index = 0U; rowgroup_index < 2U; ++rowgroup_index) {
		const auto* expected =
		    standard_descriptor->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rowgroup_index));
		const auto* actual =
		    expanded_descriptor->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rowgroup_index));
		ASSERT_NE(expected, nullptr);
		ASSERT_NE(actual, nullptr);
		ASSERT_EQ(expected->m_column_descriptors()->size(), actual->m_column_descriptors()->size());
		for (size_t column = 0U; column < expected->m_column_descriptors()->size(); ++column) {
			expect_column_equal(*expected->m_column_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(column)),
			                    *actual->m_column_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(column)));
		}
	}
}
