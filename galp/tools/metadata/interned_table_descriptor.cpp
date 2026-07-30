#include "interned_table_descriptor.hpp"
#include "fls/footer/column_descriptor_generated.h"
#include "fls/footer/decimal_type_generated.h"
#include "fls/footer/footer_generated.h"
#include "fls/footer/rowgroup_descriptor_generated.h"
#include "fls/footer/rpn_generated.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace galp::metadata {
namespace {

constexpr std::size_t kInitialBuilderLimit = 64U * 1024U * 1024U;
constexpr std::size_t kMaxColumnDepth      = 128U;

void append_u8(std::string& output, const uint8_t value) {
	output.push_back(static_cast<char>(value));
}

void append_u16(std::string& output, const uint16_t value) {
	for (unsigned shift = 0; shift < 16; shift += 8) {
		append_u8(output, static_cast<uint8_t>((value >> shift) & 0xffU));
	}
}

void append_u64(std::string& output, const uint64_t value) {
	for (unsigned shift = 0; shift < 64; shift += 8) {
		append_u8(output, static_cast<uint8_t>((value >> shift) & 0xffU));
	}
}

void append_blob(std::string& output, const std::string& value) {
	append_u64(output, static_cast<uint64_t>(value.size()));
	output.append(value);
}

template <typename T>
using Offset = flatbuffers::Offset<T>;

using ExpressionVector = flatbuffers::Vector<Offset<fastlanes::ExpressionResult>>;
using SegmentVector    = flatbuffers::Vector<Offset<fastlanes::SegmentDescriptor>>;
using ChildVector      = flatbuffers::Vector<Offset<fastlanes::ColumnDescriptor>>;

class InterningPacker {
public:
	explicit InterningPacker(const std::size_t original_descriptor_size)
	    : builder_(std::max<std::size_t>(1024U, std::min(original_descriptor_size, kInitialBuilderLimit))) {
	}

	InternedDescriptorResult pack(const fastlanes::TableDescriptor& table) {
		std::vector<Offset<fastlanes::RowgroupDescriptor>> rowgroup_offsets;
		const auto*                                        rowgroups = table.m_rowgroup_descriptors();
		if (rowgroups != nullptr) {
			rowgroup_offsets.reserve(rowgroups->size());
			for (const auto* rowgroup : *rowgroups) {
				if (rowgroup == nullptr) {
					throw std::runtime_error("table descriptor contains a null rowgroup");
				}
				rowgroup_offsets.push_back(pack_rowgroup(*rowgroup));
			}
		}

		Offset<flatbuffers::Vector<Offset<fastlanes::RowgroupDescriptor>>> rowgroup_vector;
		if (rowgroups != nullptr) {
			rowgroup_vector = builder_.CreateVector(rowgroup_offsets);
		}
		const auto root = fastlanes::CreateTableDescriptor(builder_, rowgroup_vector, table.m_table_binary_size());
		fastlanes::FinishTableDescriptorBuffer(builder_, root);
		auto detached = builder_.Release();

		InternedDescriptorResult result;
		result.stats = stats_;
		result.bytes.assign(detached.data(), detached.data() + detached.size());
		return result;
	}

private:
	Offset<fastlanes::RowgroupDescriptor> pack_rowgroup(const fastlanes::RowgroupDescriptor& rowgroup) {
		++stats_.rowgroups;
		std::vector<Offset<fastlanes::ColumnDescriptor>> column_offsets;
		const auto*                                      columns = rowgroup.m_column_descriptors();
		if (columns != nullptr) {
			column_offsets.reserve(columns->size());
			for (const auto* column : *columns) {
				if (column == nullptr) {
					throw std::runtime_error("rowgroup descriptor contains a null column");
				}
				column_offsets.push_back(pack_column(*column, 1U).first);
			}
		}
		Offset<flatbuffers::Vector<Offset<fastlanes::ColumnDescriptor>>> column_vector;
		if (columns != nullptr) {
			column_vector = builder_.CreateVector(column_offsets);
		}
		return fastlanes::CreateRowgroupDescriptor(
		    builder_, rowgroup.m_n_vec(), column_vector, rowgroup.m_size(), rowgroup.m_offset(), rowgroup.m_n_tuples());
	}

	std::pair<Offset<fastlanes::ColumnDescriptor>, std::string> pack_column(const fastlanes::ColumnDescriptor& column,
	                                                                        const std::size_t                  depth) {
		if (depth > kMaxColumnDepth) {
			throw std::runtime_error("column descriptor nesting exceeds safety limit");
		}
		++stats_.columns;
		std::string key;
		append_u8(key, static_cast<uint8_t>(column.data_type()));
		append_u64(key, column.idx());

		Offset<flatbuffers::String> name;
		append_u8(key, column.name() == nullptr ? 0U : 1U);
		if (column.name() != nullptr) {
			++stats_.strings.references;
			const std::string value(column.name()->c_str(), column.name()->size());
			append_blob(key, value);
			const auto before = builder_.GetSize();
			name              = builder_.CreateSharedString(value);
			if (builder_.GetSize() != before) {
				++stats_.strings.emitted;
			}
		}

		const auto rpn = pack_rpn(column.encoding_rpn(), key);

		std::vector<Offset<fastlanes::ColumnDescriptor>> child_offsets;
		std::string                                      child_vector_key;
		const auto*                                      children = column.children();
		append_u8(key, children == nullptr ? 0U : 1U);
		if (children != nullptr) {
			append_u64(child_vector_key, children->size());
			child_offsets.reserve(children->size());
			for (const auto* child : *children) {
				if (child == nullptr) {
					throw std::runtime_error("column descriptor contains a null child");
				}
				auto packed_child = pack_column(*child, depth + 1U);
				child_offsets.push_back(packed_child.first);
				append_blob(child_vector_key, packed_child.second);
			}
			append_blob(key, child_vector_key);
		}
		const auto child_vector = pack_child_vector(children, child_offsets, child_vector_key);

		const auto maximum = pack_binary(column.max(), key);
		append_u64(key, column.column_offset());
		append_u64(key, column.total_size());
		const auto expressions = pack_expressions(column.expr_space(), key);
		const auto segments    = pack_segments(column.segment_descriptors(), key);
		append_u64(key, column.n_null());
		const auto decimal = pack_decimal(column.fix_me_decimal_type(), key);

		const auto result = fastlanes::CreateColumnDescriptor(builder_,
		                                                      column.data_type(),
		                                                      rpn,
		                                                      column.idx(),
		                                                      name,
		                                                      child_vector,
		                                                      maximum,
		                                                      column.column_offset(),
		                                                      column.total_size(),
		                                                      expressions,
		                                                      segments,
		                                                      column.n_null(),
		                                                      decimal);
		return {result, std::move(key)};
	}

	Offset<fastlanes::RPN> pack_rpn(const fastlanes::RPN* rpn, std::string& column_key) {
		append_u8(column_key, rpn == nullptr ? 0U : 1U);
		if (rpn == nullptr) {
			return {};
		}
		++stats_.rpns.references;
		std::string key;
		const auto* operators = rpn->operator_tokens();
		append_u8(key, operators == nullptr ? 0U : 1U);
		if (operators != nullptr) {
			append_u64(key, operators->size());
			for (const auto token : *operators) {
				append_u16(key, static_cast<uint16_t>(token));
			}
		}
		const auto* operands = rpn->operand_tokens();
		append_u8(key, operands == nullptr ? 0U : 1U);
		if (operands != nullptr) {
			append_u64(key, operands->size());
			for (const auto operand : *operands) {
				append_u64(key, operand);
			}
		}
		append_blob(column_key, key);
		if (const auto found = rpn_cache_.find(key); found != rpn_cache_.end()) {
			return found->second;
		}
		Offset<flatbuffers::Vector<fastlanes::OperatorToken>> operator_vector;
		Offset<flatbuffers::Vector<uint64_t>>                 operand_vector;
		if (operators != nullptr) {
			operator_vector = builder_.CreateVector(operators->data(), operators->size());
		}
		if (operands != nullptr) {
			operand_vector = builder_.CreateVector(operands->data(), operands->size());
		}
		const auto offset = fastlanes::CreateRPN(builder_, operator_vector, operand_vector);
		rpn_cache_.emplace(std::move(key), offset);
		++stats_.rpns.emitted;
		return offset;
	}

	Offset<ChildVector> pack_child_vector(const flatbuffers::Vector<Offset<fastlanes::ColumnDescriptor>>* source,
	                                      const std::vector<Offset<fastlanes::ColumnDescriptor>>&         offsets,
	                                      const std::string&                                              key) {
		if (source == nullptr) {
			return {};
		}
		++stats_.child_vectors.references;
		if (const auto found = child_vector_cache_.find(key); found != child_vector_cache_.end()) {
			return found->second;
		}
		const auto result = builder_.CreateVector(offsets);
		child_vector_cache_.emplace(key, result);
		++stats_.child_vectors.emitted;
		return result;
	}

	Offset<fastlanes::BinaryValue> pack_binary(const fastlanes::BinaryValue* value, std::string& column_key) {
		append_u8(column_key, value == nullptr ? 0U : 1U);
		if (value == nullptr) {
			return {};
		}
		++stats_.binary_values.references;
		std::string key;
		const auto* bytes = value->binary_data();
		append_u8(key, bytes == nullptr ? 0U : 1U);
		if (bytes != nullptr) {
			append_u64(key, bytes->size());
			key.append(reinterpret_cast<const char*>(bytes->data()), bytes->size());
		}
		append_blob(column_key, key);
		if (const auto found = binary_cache_.find(key); found != binary_cache_.end()) {
			return found->second;
		}
		Offset<flatbuffers::Vector<uint8_t>> byte_vector;
		if (bytes != nullptr) {
			byte_vector = builder_.CreateVector(bytes->data(), bytes->size());
		}
		const auto result = fastlanes::CreateBinaryValue(builder_, byte_vector);
		binary_cache_.emplace(std::move(key), result);
		++stats_.binary_values.emitted;
		return result;
	}

	Offset<ExpressionVector>
	pack_expressions(const flatbuffers::Vector<Offset<fastlanes::ExpressionResult>>* expressions,
	                 std::string&                                                    column_key) {
		append_u8(column_key, expressions == nullptr ? 0U : 1U);
		if (expressions == nullptr) {
			return {};
		}
		++stats_.expression_vectors.references;
		std::vector<Offset<fastlanes::ExpressionResult>> offsets;
		std::string                                      vector_key;
		append_u64(vector_key, expressions->size());
		offsets.reserve(expressions->size());
		for (const auto* expression : *expressions) {
			if (expression == nullptr) {
				throw std::runtime_error("column descriptor contains a null expression result");
			}
			++stats_.expression_results.references;
			std::string key;
			append_u16(key, static_cast<uint16_t>(expression->operator_token()));
			append_u64(key, expression->size());
			append_blob(vector_key, key);
			if (const auto found = expression_cache_.find(key); found != expression_cache_.end()) {
				offsets.push_back(found->second);
			} else {
				const auto result =
				    fastlanes::CreateExpressionResult(builder_, expression->operator_token(), expression->size());
				expression_cache_.emplace(std::move(key), result);
				offsets.push_back(result);
				++stats_.expression_results.emitted;
			}
		}
		append_blob(column_key, vector_key);
		if (const auto found = expression_vector_cache_.find(vector_key); found != expression_vector_cache_.end()) {
			return found->second;
		}
		const auto result = builder_.CreateVector(offsets);
		expression_vector_cache_.emplace(std::move(vector_key), result);
		++stats_.expression_vectors.emitted;
		return result;
	}

	Offset<SegmentVector> pack_segments(const flatbuffers::Vector<Offset<fastlanes::SegmentDescriptor>>* segments,
	                                    std::string&                                                     column_key) {
		append_u8(column_key, segments == nullptr ? 0U : 1U);
		if (segments == nullptr) {
			return {};
		}
		++stats_.segment_vectors.references;
		std::vector<Offset<fastlanes::SegmentDescriptor>> offsets;
		std::string                                       vector_key;
		append_u64(vector_key, segments->size());
		offsets.reserve(segments->size());
		for (const auto* segment : *segments) {
			if (segment == nullptr) {
				throw std::runtime_error("column descriptor contains a null segment descriptor");
			}
			++stats_.segment_descriptors.references;
			std::string key;
			append_u64(key, segment->entrypoint_offset());
			append_u64(key, segment->entrypoint_size());
			append_u64(key, segment->data_offset());
			append_u64(key, segment->data_size());
			append_u8(key, static_cast<uint8_t>(segment->entry_point_t()));
			append_blob(vector_key, key);
			if (const auto found = segment_cache_.find(key); found != segment_cache_.end()) {
				offsets.push_back(found->second);
			} else {
				const auto result = fastlanes::CreateSegmentDescriptor(builder_,
				                                                       segment->entrypoint_offset(),
				                                                       segment->entrypoint_size(),
				                                                       segment->data_offset(),
				                                                       segment->data_size(),
				                                                       segment->entry_point_t());
				segment_cache_.emplace(std::move(key), result);
				offsets.push_back(result);
				++stats_.segment_descriptors.emitted;
			}
		}
		append_blob(column_key, vector_key);
		if (const auto found = segment_vector_cache_.find(vector_key); found != segment_vector_cache_.end()) {
			return found->second;
		}
		const auto result = builder_.CreateVector(offsets);
		segment_vector_cache_.emplace(std::move(vector_key), result);
		++stats_.segment_vectors.emitted;
		return result;
	}

	Offset<fastlanes::DecimalType> pack_decimal(const fastlanes::DecimalType* decimal, std::string& column_key) {
		append_u8(column_key, decimal == nullptr ? 0U : 1U);
		if (decimal == nullptr) {
			return {};
		}
		++stats_.decimals.references;
		std::string key;
		append_u64(key, decimal->precision());
		append_u64(key, decimal->scale());
		append_blob(column_key, key);
		if (const auto found = decimal_cache_.find(key); found != decimal_cache_.end()) {
			return found->second;
		}
		const auto result = fastlanes::CreateDecimalType(builder_, decimal->precision(), decimal->scale());
		decimal_cache_.emplace(std::move(key), result);
		++stats_.decimals.emitted;
		return result;
	}

private:
	flatbuffers::FlatBufferBuilder                                        builder_;
	InternedDescriptorStats                                               stats_;
	std::unordered_map<std::string, Offset<fastlanes::RPN>>               rpn_cache_;
	std::unordered_map<std::string, Offset<fastlanes::ExpressionResult>>  expression_cache_;
	std::unordered_map<std::string, Offset<ExpressionVector>>             expression_vector_cache_;
	std::unordered_map<std::string, Offset<fastlanes::SegmentDescriptor>> segment_cache_;
	std::unordered_map<std::string, Offset<SegmentVector>>                segment_vector_cache_;
	std::unordered_map<std::string, Offset<fastlanes::BinaryValue>>       binary_cache_;
	std::unordered_map<std::string, Offset<fastlanes::DecimalType>>       decimal_cache_;
	std::unordered_map<std::string, Offset<ChildVector>>                  child_vector_cache_;
};

} // namespace

InternedDescriptorResult pack_interned_table_descriptor(const fastlanes::TableDescriptor& table,
                                                        const std::size_t                 original_descriptor_size) {
	if (original_descriptor_size > std::numeric_limits<flatbuffers::uoffset_t>::max()) {
		throw std::runtime_error("table descriptor exceeds FlatBuffers 32-bit offset limit");
	}
	return InterningPacker(original_descriptor_size).pack(table);
}

} // namespace galp::metadata
