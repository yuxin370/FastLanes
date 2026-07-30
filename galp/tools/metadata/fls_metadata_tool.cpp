#include "fls/common/status.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/file/file_header.hpp"
#include "fls/footer/column_descriptor_generated.h"
#include "fls/footer/table_descriptor.hpp"
#include "fls/footer/table_descriptor_generated.h"
#include "fls/info.hpp"
#include "fls/io/file.hpp"
#include "fls/json/nlohmann/json.hpp"
#include "interned_table_descriptor.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using json  = nlohmann::json;
using Clock = std::chrono::steady_clock;

constexpr std::string_view kSchema                 = "galp_fls_metadata_breakdown_v1";
constexpr std::string_view kExternalDescriptorName = "table_descriptor.fbb";
constexpr std::size_t      kMaxDescriptorDepth     = 128;

struct Options {
	std::string           command;
	std::filesystem::path input;
	std::filesystem::path candidate;
	std::filesystem::path output;
	std::filesystem::path report;
	bool                  pretty = false;
};

[[noreturn]] void usage_error(const std::string& message) {
	throw std::invalid_argument(message + "\nRun galp_fls_metadata_tool --help for usage.");
}

void print_help() {
	std::cout << "Usage:\n"
	          << "  galp_fls_metadata_tool analyze --input FILE [--output JSON] [--pretty]\n"
	          << "  galp_fls_metadata_tool compact --input FILE --output FILE [--report JSON] [--pretty]\n\n"
	          << "  galp_fls_metadata_tool repack-legacy --input FILE --output FILE [--report JSON] [--pretty]\n"
	          << "  galp_fls_metadata_tool compare --input LEGACY --candidate COMPACT [--output JSON] [--pretty]\n\n"
	          << "The analyze command is read-only. It parses the actual FLS header/footer and\n"
	          << "TableDescriptor FlatBuffer, reports exact file regions, object counts,\n"
	          << "logical field bytes, and canonical descriptor repetition statistics.\n\n"
	          << "The compact command is an explicit experimental feature. It copies the FLS\n"
	          << "payload byte-for-byte and rewrites the current-schema inline descriptor with\n"
	          << "exact repeated FlatBuffer objects shared. Existing files and default writers\n"
	          << "are not modified.\n";
}

Options parse_options(const int argc, char** argv) {
	if (argc == 2 && std::string_view(argv[1]) == "--help") {
		print_help();
		std::exit(0);
	}
	if (argc < 2 || (std::string_view(argv[1]) != "analyze" && std::string_view(argv[1]) != "compact" &&
	                 std::string_view(argv[1]) != "repack-legacy" && std::string_view(argv[1]) != "compare")) {
		usage_error("expected the analyze, compact, repack-legacy, or compare command");
	}

	Options options;
	options.command = argv[1];
	for (int index = 2; index < argc; ++index) {
		const std::string_view argument(argv[index]);
		if (argument == "--input") {
			if (++index >= argc) {
				usage_error("--input requires a path");
			}
			options.input = argv[index];
		} else if (argument == "--output") {
			if (++index >= argc) {
				usage_error("--output requires a path");
			}
			options.output = argv[index];
		} else if (argument == "--candidate") {
			if (++index >= argc) {
				usage_error("--candidate requires a path");
			}
			options.candidate = argv[index];
		} else if (argument == "--pretty") {
			options.pretty = true;
		} else if (argument == "--report") {
			if (++index >= argc) {
				usage_error("--report requires a path");
			}
			options.report = argv[index];
		} else if (argument == "--help") {
			print_help();
			std::exit(0);
		} else {
			usage_error("unknown argument: " + std::string(argument));
		}
	}
	if (options.input.empty()) {
		usage_error("--input is required");
	}
	if ((options.command == "compact" || options.command == "repack-legacy") && options.output.empty()) {
		usage_error(options.command + " requires --output");
	}
	if (options.command == "compare" && options.candidate.empty()) {
		usage_error("compare requires --candidate");
	}
	if ((options.command == "analyze" || options.command == "compare") && !options.report.empty()) {
		usage_error("--report is only valid with compact or repack-legacy");
	}
	return options;
}

std::string hex_u64(const uint64_t value) {
	std::ostringstream stream;
	stream << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
	return stream.str();
}

void require_status(const fastlanes::Status status, const std::string_view context) {
	if (!status.success) {
		throw std::runtime_error(std::string(context) + ": " +
		                         std::string(fastlanes::Status::message_for(status.code)));
	}
}

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

void append_blob(std::string& output, const std::string_view value) {
	append_u64(output, static_cast<uint64_t>(value.size()));
	output.append(value.data(), value.size());
}

uint64_t uleb128_bytes(uint64_t value) {
	uint64_t bytes = 1;
	while (value >= 0x80U) {
		value >>= 7U;
		++bytes;
	}
	return bytes;
}

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
	return std::chrono::duration<double, std::milli>(end - start).count();
}

uint64_t peak_rss_bytes() {
	rusage usage {};
	if (::getrusage(RUSAGE_SELF, &usage) != 0) {
		return 0U;
	}
	return static_cast<uint64_t>(usage.ru_maxrss) * 1024U;
}

template <typename T>
double repetition_rate(const uint64_t count, const std::unordered_set<T>& unique) {
	if (count == 0) {
		return 0.0;
	}
	return 1.0 - static_cast<double>(unique.size()) / static_cast<double>(count);
}

struct Counts {
	uint64_t rowgroups               = 0;
	uint64_t root_columns            = 0;
	uint64_t recursive_columns       = 0;
	uint64_t rpn_tables              = 0;
	uint64_t rpn_operator_tokens     = 0;
	uint64_t rpn_operand_tokens      = 0;
	uint64_t expression_results      = 0;
	uint64_t segment_descriptors     = 0;
	uint64_t names                   = 0;
	uint64_t name_bytes              = 0;
	uint64_t max_tables              = 0;
	uint64_t max_value_bytes         = 0;
	uint64_t decimal_tables          = 0;
	uint64_t child_vectors           = 0;
	uint64_t max_depth               = 0;
	uint64_t rowgroup_scalar_bytes   = 0;
	uint64_t column_scalar_bytes     = 0;
	uint64_t rpn_scalar_bytes        = 0;
	uint64_t expression_scalar_bytes = 0;
	uint64_t segment_scalar_bytes    = 0;
	uint64_t decimal_scalar_bytes    = 0;
};

struct Uniques {
	std::unordered_set<std::string> names;
	std::unordered_set<std::string> rpns;
	std::unordered_set<std::string> expression_results;
	std::unordered_set<std::string> segment_descriptors;
	std::unordered_set<std::string> schemas;
	std::unordered_set<std::string> encoding_templates;
	std::unordered_set<std::string> exact_columns;
	std::unordered_set<std::string> max_values;
};

struct CompactV2Projection {
	std::unordered_map<std::string, uint64_t> schema_ids;
	std::unordered_map<std::string, uint64_t> rpn_ids;
	std::unordered_map<std::string, uint64_t> expression_ids;
	std::unordered_map<std::string, uint64_t> max_value_ids;
	uint64_t                                  schema_dictionary_bytes     = 0;
	uint64_t                                  rpn_dictionary_bytes        = 0;
	uint64_t                                  expression_dictionary_bytes = 0;
	uint64_t                                  max_dictionary_bytes        = 0;
	uint64_t                                  column_record_bytes         = 0;
	uint64_t                                  rowgroup_page_bytes         = 0;
	uint64_t                                  rowgroup_directory_bytes    = 0;
	uint64_t                                  max_rowgroup_page_bytes     = 0;
};

struct CanonicalColumn {
	std::string schema;
	std::string encoding_template;
	std::string exact;
};

struct Analyzer {
	Counts              counts;
	Uniques             unique;
	CompactV2Projection projection;

	static uint64_t dictionary_id(std::unordered_map<std::string, uint64_t>& dictionary,
	                              const std::string&                         key,
	                              const uint64_t                             encoded_entry_bytes,
	                              uint64_t&                                  dictionary_bytes) {
		const auto existing = dictionary.find(key);
		if (existing != dictionary.end()) {
			return existing->second;
		}
		const auto id = static_cast<uint64_t>(dictionary.size());
		dictionary.emplace(key, id);
		dictionary_bytes += encoded_entry_bytes;
		return id;
	}

	std::string rpn_key(const fastlanes::RPN* rpn) {
		std::string key;
		append_u8(key, rpn == nullptr ? 0U : 1U);
		if (rpn == nullptr) {
			return key;
		}
		++counts.rpn_tables;
		const auto* operators = rpn->operator_tokens();
		append_u64(key, operators == nullptr ? 0U : operators->size());
		if (operators != nullptr) {
			counts.rpn_operator_tokens += operators->size();
			counts.rpn_scalar_bytes += static_cast<uint64_t>(operators->size()) * sizeof(uint16_t);
			for (const auto token : *operators) {
				append_u16(key, static_cast<uint16_t>(token));
			}
		}
		const auto* operands = rpn->operand_tokens();
		append_u64(key, operands == nullptr ? 0U : operands->size());
		if (operands != nullptr) {
			counts.rpn_operand_tokens += operands->size();
			counts.rpn_scalar_bytes += static_cast<uint64_t>(operands->size()) * sizeof(uint64_t);
			for (const auto operand : *operands) {
				append_u64(key, operand);
			}
		}
		unique.rpns.insert(key);
		return key;
	}

	CanonicalColumn column(const fastlanes::ColumnDescriptor& descriptor, const std::size_t depth) {
		if (depth > kMaxDescriptorDepth) {
			throw std::runtime_error("column descriptor nesting exceeds safety limit");
		}
		counts.max_depth = std::max(counts.max_depth, static_cast<uint64_t>(depth));
		++counts.recursive_columns;
		counts.column_scalar_bytes += sizeof(uint8_t) + 4U * sizeof(uint64_t);

		CanonicalColumn result;
		uint64_t        projected_record_bytes = 1U; // optional-field flags
		append_u8(result.schema, static_cast<uint8_t>(descriptor.data_type()));
		append_u64(result.schema, descriptor.idx());

		const auto* name = descriptor.name();
		append_u8(result.schema, name == nullptr ? 0U : 1U);
		if (name != nullptr) {
			const std::string_view name_view(name->c_str(), name->size());
			append_blob(result.schema, name_view);
			++counts.names;
			counts.name_bytes += name->size();
			unique.names.emplace(name_view);
		}

		const auto* decimal = descriptor.fix_me_decimal_type();
		append_u8(result.schema, decimal == nullptr ? 0U : 1U);
		if (decimal != nullptr) {
			++counts.decimal_tables;
			counts.decimal_scalar_bytes += 2U * sizeof(uint64_t);
			append_u64(result.schema, decimal->precision());
			append_u64(result.schema, decimal->scale());
		}

		const auto* children = descriptor.children();
		append_u64(result.schema, children == nullptr ? 0U : children->size());
		std::vector<CanonicalColumn> child_keys;
		if (children != nullptr) {
			++counts.child_vectors;
			child_keys.reserve(children->size());
			for (const auto* child : *children) {
				if (child == nullptr) {
					throw std::runtime_error("column descriptor contains a null child");
				}
				auto child_key = column(*child, depth + 1U);
				append_blob(result.schema, child_key.schema);
				child_keys.push_back(std::move(child_key));
			}
		}

		result.encoding_template = result.schema;
		const auto rpn           = rpn_key(descriptor.encoding_rpn());
		append_blob(result.encoding_template, rpn);
		if (const auto* rpn_view = descriptor.encoding_rpn(); rpn_view != nullptr) {
			uint64_t encoded_bytes = 1U;
			if (const auto* operators = rpn_view->operator_tokens(); operators != nullptr) {
				encoded_bytes += uleb128_bytes(operators->size());
				for (const auto token : *operators) {
					encoded_bytes += uleb128_bytes(static_cast<uint16_t>(token));
				}
			}
			if (const auto* operands = rpn_view->operand_tokens(); operands != nullptr) {
				encoded_bytes += uleb128_bytes(operands->size());
				for (const auto operand : *operands) {
					encoded_bytes += uleb128_bytes(operand);
				}
			}
			const auto rpn_id = dictionary_id(projection.rpn_ids, rpn, encoded_bytes, projection.rpn_dictionary_bytes);
			projected_record_bytes += uleb128_bytes(rpn_id + 1U);
		} else {
			projected_record_bytes += 1U;
		}

		const auto* expressions = descriptor.expr_space();
		append_u64(result.encoding_template, expressions == nullptr ? 0U : expressions->size());
		std::string exact_expressions;
		append_u8(exact_expressions, expressions == nullptr ? 0U : 1U);
		if (expressions != nullptr) {
			projected_record_bytes += uleb128_bytes(expressions->size());
			for (const auto* expression : *expressions) {
				if (expression == nullptr) {
					throw std::runtime_error("column descriptor contains a null expression result");
				}
				++counts.expression_results;
				counts.expression_scalar_bytes += sizeof(uint16_t) + sizeof(uint64_t);
				std::string key;
				append_u16(key, static_cast<uint16_t>(expression->operator_token()));
				append_u64(key, expression->size());
				unique.expression_results.insert(key);
				const auto expression_id =
				    dictionary_id(projection.expression_ids,
				                  key,
				                  uleb128_bytes(static_cast<uint16_t>(expression->operator_token())) +
				                      uleb128_bytes(expression->size()),
				                  projection.expression_dictionary_bytes);
				projected_record_bytes += uleb128_bytes(expression_id);
				append_u16(result.encoding_template, static_cast<uint16_t>(expression->operator_token()));
				append_blob(exact_expressions, key);
			}
		} else {
			projected_record_bytes += 1U;
		}

		const auto* segments = descriptor.segment_descriptors();
		append_u64(result.encoding_template, segments == nullptr ? 0U : segments->size());
		std::string exact_segments;
		append_u8(exact_segments, segments == nullptr ? 0U : 1U);
		if (segments != nullptr) {
			projected_record_bytes += uleb128_bytes(segments->size());
			for (const auto* segment : *segments) {
				if (segment == nullptr) {
					throw std::runtime_error("column descriptor contains a null segment descriptor");
				}
				++counts.segment_descriptors;
				counts.segment_scalar_bytes += 4U * sizeof(uint64_t) + sizeof(uint8_t);
				std::string key;
				append_u64(key, segment->entrypoint_offset());
				append_u64(key, segment->entrypoint_size());
				append_u64(key, segment->data_offset());
				append_u64(key, segment->data_size());
				append_u8(key, static_cast<uint8_t>(segment->entry_point_t()));
				unique.segment_descriptors.insert(key);
				projected_record_bytes +=
				    uleb128_bytes(segment->entrypoint_offset()) + uleb128_bytes(segment->entrypoint_size()) +
				    uleb128_bytes(segment->data_offset()) + uleb128_bytes(segment->data_size()) + 1U;
				append_u8(result.encoding_template, static_cast<uint8_t>(segment->entry_point_t()));
				append_blob(exact_segments, key);
			}
		} else {
			projected_record_bytes += 1U;
		}

		for (const auto& child : child_keys) {
			append_blob(result.encoding_template, child.encoding_template);
		}

		result.exact = result.encoding_template;
		append_u64(result.exact, descriptor.column_offset());
		append_u64(result.exact, descriptor.total_size());
		append_u64(result.exact, descriptor.n_null());
		append_blob(result.exact, exact_expressions);
		append_blob(result.exact, exact_segments);

		const auto* maximum = descriptor.max();
		append_u8(result.exact, maximum == nullptr ? 0U : 1U);
		if (maximum != nullptr) {
			++counts.max_tables;
			const auto* bytes = maximum->binary_data();
			append_u8(result.exact, bytes == nullptr ? 0U : 1U);
			append_u64(result.exact, bytes == nullptr ? 0U : bytes->size());
			std::string max_key;
			append_u8(max_key, bytes == nullptr ? 0U : 1U);
			if (bytes != nullptr) {
				counts.max_value_bytes += bytes->size();
				result.exact.append(reinterpret_cast<const char*>(bytes->data()), bytes->size());
				max_key.append(reinterpret_cast<const char*>(bytes->data()), bytes->size());
			}
			unique.max_values.insert(max_key);
			const auto max_id = dictionary_id(projection.max_value_ids,
			                                  max_key,
			                                  1U + uleb128_bytes(bytes == nullptr ? 0U : bytes->size()) +
			                                      (bytes == nullptr ? 0U : bytes->size()),
			                                  projection.max_dictionary_bytes);
			projected_record_bytes += uleb128_bytes(max_id + 1U);
		} else {
			projected_record_bytes += 1U;
		}
		for (const auto& child : child_keys) {
			append_blob(result.exact, child.exact);
		}

		unique.schemas.insert(result.schema);
		unique.encoding_templates.insert(result.encoding_template);
		unique.exact_columns.insert(result.exact);
		const auto schema_id = dictionary_id(projection.schema_ids,
		                                     result.schema,
		                                     uleb128_bytes(result.schema.size()) + result.schema.size(),
		                                     projection.schema_dictionary_bytes);
		projected_record_bytes += uleb128_bytes(schema_id);
		projected_record_bytes += uleb128_bytes(descriptor.column_offset()) + uleb128_bytes(descriptor.total_size()) +
		                          uleb128_bytes(descriptor.n_null());
		projection.column_record_bytes += projected_record_bytes;
		return result;
	}
};

json analyze(const std::filesystem::path& input) {
	fastlanes::File file(input);
	const auto      file_size = file.Size();
	if (file_size < sizeof(fastlanes::FileHeader) + sizeof(fastlanes::FileFooter)) {
		throw std::runtime_error("FLS file is too small");
	}

	fastlanes::FileHeader header {};
	fastlanes::FileFooter footer {};
	require_status(fastlanes::FileHeader::Load(header, file), "read FLS header");
	require_status(fastlanes::FileFooter::Load(footer, file), "read FLS footer");
	if (header.magic_bytes != fastlanes::Info::get_magic_bytes() ||
	    footer.magic_bytes != fastlanes::Info::get_magic_bytes()) {
		throw std::runtime_error("FLS header/footer magic mismatch");
	}
	if (footer.table_descriptor_offset < sizeof(fastlanes::FileHeader) ||
	    footer.table_descriptor_offset > file_size - sizeof(fastlanes::FileFooter)) {
		throw std::runtime_error("FLS table descriptor offset is out of bounds");
	}

	const bool                       inline_descriptor = static_cast<bool>(header.settings.inline_footer);
	std::filesystem::path            descriptor_path;
	fastlanes::TableDescriptorHandle descriptor_handle;
	if (inline_descriptor) {
		if (footer.table_descriptor_size > file_size - sizeof(fastlanes::FileFooter) - footer.table_descriptor_offset) {
			throw std::runtime_error("inline FLS table descriptor range is out of bounds");
		}
		descriptor_path   = input;
		descriptor_handle = fastlanes::TableDescriptorHandle::FromFileSlice(
		    file, footer.table_descriptor_offset, footer.table_descriptor_size, true);
	} else {
		descriptor_path = input.parent_path() / kExternalDescriptorName;
		if (!std::filesystem::is_regular_file(descriptor_path)) {
			throw std::runtime_error("external table descriptor is missing: " + descriptor_path.string());
		}
		if (std::filesystem::file_size(descriptor_path) != footer.table_descriptor_size) {
			throw std::runtime_error("external table descriptor size disagrees with FLS footer");
		}
		descriptor_handle = fastlanes::TableDescriptorHandle::FromFile(descriptor_path, true);
	}

	const auto* table = descriptor_handle.Get();
	if (table == nullptr) {
		throw std::runtime_error("table descriptor root is null");
	}
	if (table->m_table_binary_size() != footer.table_descriptor_offset) {
		throw std::runtime_error("table descriptor payload boundary disagrees with FLS footer");
	}

	Analyzer    analyzer;
	const auto* rowgroups = table->m_rowgroup_descriptors();
	if (rowgroups != nullptr) {
		analyzer.counts.rowgroups             = rowgroups->size();
		analyzer.counts.rowgroup_scalar_bytes = static_cast<uint64_t>(rowgroups->size()) * 4U * sizeof(uint64_t);
		for (const auto* rowgroup : *rowgroups) {
			if (rowgroup == nullptr) {
				throw std::runtime_error("table descriptor contains a null rowgroup");
			}
			const auto* columns          = rowgroup->m_column_descriptors();
			const auto  projected_before = analyzer.projection.column_record_bytes;
			if (columns != nullptr) {
				analyzer.counts.root_columns += columns->size();
				for (const auto* column : *columns) {
					if (column == nullptr) {
						throw std::runtime_error("rowgroup descriptor contains a null column");
					}
					static_cast<void>(analyzer.column(*column, 1U));
				}
			}
			// Proposed V2 rowgroup page: 16-byte page header, a direct uint32
			// root-column offset index, compact records, CRC32C, and 8-byte padding.
			const uint64_t root_count = columns == nullptr ? 0U : columns->size();
			const uint64_t records    = analyzer.projection.column_record_bytes - projected_before;
			const uint64_t unaligned  = 16U + (root_count + 1U) * sizeof(uint32_t) + records + sizeof(uint32_t);
			const uint64_t page_bytes = (unaligned + 7U) & ~uint64_t {7U};
			analyzer.projection.rowgroup_page_bytes += page_bytes;
			analyzer.projection.max_rowgroup_page_bytes =
			    std::max(analyzer.projection.max_rowgroup_page_bytes, page_bytes);
		}
	}
	analyzer.projection.rowgroup_directory_bytes = analyzer.counts.rowgroups * 48U;

	const uint64_t inline_descriptor_end =
	    footer.table_descriptor_offset + (inline_descriptor ? footer.table_descriptor_size : 0U);
	const uint64_t expected_file_size      = inline_descriptor_end + sizeof(fastlanes::FileFooter);
	const uint64_t unclassified_file_bytes = file_size >= expected_file_size ? file_size - expected_file_size : 0U;
	const uint64_t logical_field_bytes     = sizeof(uint64_t) + analyzer.counts.rowgroup_scalar_bytes +
	                                     analyzer.counts.column_scalar_bytes + analyzer.counts.rpn_scalar_bytes +
	                                     analyzer.counts.expression_scalar_bytes +
	                                     analyzer.counts.segment_scalar_bytes + analyzer.counts.decimal_scalar_bytes +
	                                     analyzer.counts.name_bytes + analyzer.counts.max_value_bytes;
	// Estimated bytes for the explicitly documented V2 representation. This is
	// deliberately separate from measured FlatBuffer bytes and is not a benchmark.
	constexpr uint64_t kCompactHeaderBytes       = 64U;
	constexpr uint64_t kDictionaryDirectoryBytes = 4U * 24U;
	constexpr uint64_t kSectionChecksumsBytes    = 6U * sizeof(uint32_t);
	const uint64_t     compact_v2_projected_bytes =
	    kCompactHeaderBytes + kDictionaryDirectoryBytes + kSectionChecksumsBytes +
	    analyzer.projection.rowgroup_directory_bytes + analyzer.projection.rowgroup_page_bytes +
	    analyzer.projection.schema_dictionary_bytes + analyzer.projection.rpn_dictionary_bytes +
	    analyzer.projection.expression_dictionary_bytes + analyzer.projection.max_dictionary_bytes;

	json result;
	result["schema_version"]        = kSchema;
	result["classification_policy"] = {
	    {"exact_file_regions", "Verified by parsing FileHeader, FileFooter, and TableDescriptor boundaries"},
	    {"object_counts", "Verified by traversing the verified TableDescriptor FlatBuffer"},
	    {"logical_field_bytes",
	     "Lower bound from declared scalar/vector payload widths; excludes FlatBuffers overhead"},
	    {"canonical_repetition", "Exact semantic-key equality; not a serialized-byte attribution"},
	    {"compact_v2_projection", "Inferred byte count for a documented candidate encoding; not measured output"}};
	result["input"]            = std::filesystem::absolute(input).string();
	result["header"]           = {{"magic", hex_u64(header.magic_bytes)},
	                              {"version", hex_u64(header.version)},
	                              {"inline_table_descriptor", inline_descriptor}};
	result["file_layout"]      = {{"file_bytes", file_size},
	                              {"file_header_bytes", sizeof(fastlanes::FileHeader)},
	                              {"payload_offset", sizeof(fastlanes::FileHeader)},
	                              {"payload_bytes", footer.table_descriptor_offset - sizeof(fastlanes::FileHeader)},
	                              {"table_descriptor_storage", inline_descriptor ? "inline" : "external"},
	                              {"table_descriptor_path", std::filesystem::absolute(descriptor_path).string()},
	                              {"table_descriptor_offset", footer.table_descriptor_offset},
	                              {"table_descriptor_bytes", footer.table_descriptor_size},
	                              {"file_footer_bytes", sizeof(fastlanes::FileFooter)},
	                              {"unclassified_file_bytes", unclassified_file_bytes}};
	result["table_descriptor"] = {
	    {"declared_table_binary_size", table->m_table_binary_size()},
	    {"loaded_bytes", descriptor_handle.size()},
	    {"counts",
	     {{"rowgroups", analyzer.counts.rowgroups},
	      {"root_columns", analyzer.counts.root_columns},
	      {"recursive_columns", analyzer.counts.recursive_columns},
	      {"rpn_tables", analyzer.counts.rpn_tables},
	      {"rpn_operator_tokens", analyzer.counts.rpn_operator_tokens},
	      {"rpn_operand_tokens", analyzer.counts.rpn_operand_tokens},
	      {"expression_results", analyzer.counts.expression_results},
	      {"segment_descriptors", analyzer.counts.segment_descriptors},
	      {"names", analyzer.counts.names},
	      {"max_tables", analyzer.counts.max_tables},
	      {"decimal_tables", analyzer.counts.decimal_tables},
	      {"child_vectors", analyzer.counts.child_vectors},
	      {"max_column_depth", analyzer.counts.max_depth}}},
	    {"logical_field_bytes_lower_bound",
	     {{"table_level", sizeof(uint64_t)},
	      {"rowgroup_level", analyzer.counts.rowgroup_scalar_bytes},
	      {"column_level", analyzer.counts.column_scalar_bytes},
	      {"rpn_payload", analyzer.counts.rpn_scalar_bytes},
	      {"expression_result_payload", analyzer.counts.expression_scalar_bytes},
	      {"segment_descriptor_payload", analyzer.counts.segment_scalar_bytes},
	      {"decimal_payload", analyzer.counts.decimal_scalar_bytes},
	      {"string_payload", analyzer.counts.name_bytes},
	      {"max_value_payload", analyzer.counts.max_value_bytes},
	      {"total", logical_field_bytes}}},
	    {"canonical_uniqueness",
	     {{"names",
	       {{"count", analyzer.counts.names},
	        {"unique", analyzer.unique.names.size()},
	        {"repetition_rate", repetition_rate(analyzer.counts.names, analyzer.unique.names)}}},
	      {"rpn",
	       {{"count", analyzer.counts.rpn_tables},
	        {"unique", analyzer.unique.rpns.size()},
	        {"repetition_rate", repetition_rate(analyzer.counts.rpn_tables, analyzer.unique.rpns)}}},
	      {"expression_results",
	       {{"count", analyzer.counts.expression_results},
	        {"unique", analyzer.unique.expression_results.size()},
	        {"repetition_rate",
	         repetition_rate(analyzer.counts.expression_results, analyzer.unique.expression_results)}}},
	      {"segment_descriptors",
	       {{"count", analyzer.counts.segment_descriptors},
	        {"unique", analyzer.unique.segment_descriptors.size()},
	        {"repetition_rate",
	         repetition_rate(analyzer.counts.segment_descriptors, analyzer.unique.segment_descriptors)}}},
	      {"column_schema",
	       {{"count", analyzer.counts.recursive_columns},
	        {"unique", analyzer.unique.schemas.size()},
	        {"repetition_rate", repetition_rate(analyzer.counts.recursive_columns, analyzer.unique.schemas)}}},
	      {"column_encoding_template",
	       {{"count", analyzer.counts.recursive_columns},
	        {"unique", analyzer.unique.encoding_templates.size()},
	        {"repetition_rate",
	         repetition_rate(analyzer.counts.recursive_columns, analyzer.unique.encoding_templates)}}},
	      {"column_exact",
	       {{"count", analyzer.counts.recursive_columns},
	        {"unique", analyzer.unique.exact_columns.size()},
	        {"repetition_rate", repetition_rate(analyzer.counts.recursive_columns, analyzer.unique.exact_columns)}}},
	      {"max_values",
	       {{"count", analyzer.counts.max_tables},
	        {"unique", analyzer.unique.max_values.size()},
	        {"repetition_rate", repetition_rate(analyzer.counts.max_tables, analyzer.unique.max_values)}}}}},
	    {"compact_v2_projection",
	     {{"classification", "Inferred"},
	      {"measured", false},
	      {"encoding",
	       "64-bit top-level section offsets; uint32 rowgroup-local direct index; ULEB128 scalar fields and IDs; "
	       "schema/RPN/expression/max dictionaries; direct compact segment records; per-page CRC32C; 8-byte page "
	       "alignment"},
	      {"header_bytes", kCompactHeaderBytes},
	      {"dictionary_directory_bytes", kDictionaryDirectoryBytes},
	      {"section_checksums_bytes", kSectionChecksumsBytes},
	      {"rowgroup_directory_bytes", analyzer.projection.rowgroup_directory_bytes},
	      {"rowgroup_page_bytes", analyzer.projection.rowgroup_page_bytes},
	      {"max_rowgroup_page_bytes", analyzer.projection.max_rowgroup_page_bytes},
	      {"schema_dictionary_bytes", analyzer.projection.schema_dictionary_bytes},
	      {"rpn_dictionary_bytes", analyzer.projection.rpn_dictionary_bytes},
	      {"expression_dictionary_bytes", analyzer.projection.expression_dictionary_bytes},
	      {"max_dictionary_bytes", analyzer.projection.max_dictionary_bytes},
	      {"projected_bytes", compact_v2_projected_bytes},
	      {"projected_reduction_from_current_descriptor",
	       footer.table_descriptor_size == 0 ? 0.0
	                                         : 1.0 - static_cast<double>(compact_v2_projected_bytes) /
	                                                     static_cast<double>(footer.table_descriptor_size)},
	      {"overflow_policy",
	       "uint32 is rowgroup-page-relative only; writer rejects pages >= 4 GiB or emits a 64-bit-index feature "
	       "variant"}}}};
	return result;
}

template <typename Vector, typename Equal>
bool equal_optional_vector(const Vector* left, const Vector* right, Equal equal) {
	if ((left == nullptr) != (right == nullptr)) {
		return false;
	}
	if (left == nullptr) {
		return true;
	}
	if (left->size() != right->size()) {
		return false;
	}
	for (flatbuffers::uoffset_t index = 0; index < left->size(); ++index) {
		if (!equal(left->Get(index), right->Get(index))) {
			return false;
		}
	}
	return true;
}

template <typename Vector>
bool equal_scalar_vector(const Vector* left, const Vector* right) {
	return equal_optional_vector(
	    left, right, [](const auto left_value, const auto right_value) { return left_value == right_value; });
}

bool equivalent_column(const fastlanes::ColumnDescriptor* left,
                       const fastlanes::ColumnDescriptor* right,
                       const std::size_t                  depth) {
	if (left == nullptr || right == nullptr || depth > kMaxDescriptorDepth) {
		return left == right;
	}
	if (left->data_type() != right->data_type() || left->idx() != right->idx() ||
	    left->column_offset() != right->column_offset() || left->total_size() != right->total_size() ||
	    left->n_null() != right->n_null()) {
		return false;
	}
	const auto* left_name  = left->name();
	const auto* right_name = right->name();
	if ((left_name == nullptr) != (right_name == nullptr) ||
	    (left_name != nullptr && left_name->string_view() != right_name->string_view())) {
		return false;
	}
	const auto* left_rpn  = left->encoding_rpn();
	const auto* right_rpn = right->encoding_rpn();
	if ((left_rpn == nullptr) != (right_rpn == nullptr) ||
	    (left_rpn != nullptr && (!equal_scalar_vector(left_rpn->operator_tokens(), right_rpn->operator_tokens()) ||
	                             !equal_scalar_vector(left_rpn->operand_tokens(), right_rpn->operand_tokens())))) {
		return false;
	}
	const auto* left_max  = left->max();
	const auto* right_max = right->max();
	if ((left_max == nullptr) != (right_max == nullptr) ||
	    (left_max != nullptr && !equal_scalar_vector(left_max->binary_data(), right_max->binary_data()))) {
		return false;
	}
	const auto* left_decimal  = left->fix_me_decimal_type();
	const auto* right_decimal = right->fix_me_decimal_type();
	if ((left_decimal == nullptr) != (right_decimal == nullptr) ||
	    (left_decimal != nullptr && (left_decimal->precision() != right_decimal->precision() ||
	                                 left_decimal->scale() != right_decimal->scale()))) {
		return false;
	}
	if (!equal_optional_vector(
	        left->expr_space(), right->expr_space(), [](const auto* left_expression, const auto* right_expression) {
		        return left_expression != nullptr && right_expression != nullptr &&
		               left_expression->operator_token() == right_expression->operator_token() &&
		               left_expression->size() == right_expression->size();
	        })) {
		return false;
	}
	if (!equal_optional_vector(left->segment_descriptors(),
	                           right->segment_descriptors(),
	                           [](const auto* left_segment, const auto* right_segment) {
		                           return left_segment != nullptr && right_segment != nullptr &&
		                                  left_segment->entrypoint_offset() == right_segment->entrypoint_offset() &&
		                                  left_segment->entrypoint_size() == right_segment->entrypoint_size() &&
		                                  left_segment->data_offset() == right_segment->data_offset() &&
		                                  left_segment->data_size() == right_segment->data_size() &&
		                                  left_segment->entry_point_t() == right_segment->entry_point_t();
	                           })) {
		return false;
	}
	return equal_optional_vector(
	    left->children(), right->children(), [depth](const auto* left_child, const auto* right_child) {
		    return equivalent_column(left_child, right_child, depth + 1U);
	    });
}

bool equivalent_table(const fastlanes::TableDescriptor& left, const fastlanes::TableDescriptor& right) {
	if (left.m_table_binary_size() != right.m_table_binary_size()) {
		return false;
	}
	return equal_optional_vector(left.m_rowgroup_descriptors(),
	                             right.m_rowgroup_descriptors(),
	                             [](const auto* left_rowgroup, const auto* right_rowgroup) {
		                             if (left_rowgroup == nullptr || right_rowgroup == nullptr ||
		                                 left_rowgroup->m_n_vec() != right_rowgroup->m_n_vec() ||
		                                 left_rowgroup->m_size() != right_rowgroup->m_size() ||
		                                 left_rowgroup->m_offset() != right_rowgroup->m_offset() ||
		                                 left_rowgroup->m_n_tuples() != right_rowgroup->m_n_tuples()) {
			                             return false;
		                             }
		                             return equal_optional_vector(
		                                 left_rowgroup->m_column_descriptors(),
		                                 right_rowgroup->m_column_descriptors(),
		                                 [](const auto* left_column, const auto* right_column) {
			                                 return equivalent_column(left_column, right_column, 1U);
		                                 });
	                             });
}

struct LoadedInlineDescriptor {
	fastlanes::FileHeader            header;
	fastlanes::FileFooter            footer;
	fastlanes::TableDescriptorHandle descriptor;
	uint64_t                         file_size = 0;
};

LoadedInlineDescriptor load_inline_descriptor(const std::filesystem::path& path) {
	fastlanes::File        file(path);
	LoadedInlineDescriptor loaded;
	loaded.file_size = file.Size();
	require_status(fastlanes::FileHeader::Load(loaded.header, file), "read FLS header");
	require_status(fastlanes::FileFooter::Load(loaded.footer, file), "read FLS footer");
	if (loaded.header.magic_bytes != fastlanes::Info::get_magic_bytes() ||
	    loaded.footer.magic_bytes != fastlanes::Info::get_magic_bytes()) {
		throw std::runtime_error("FLS header/footer magic mismatch: " + path.string());
	}
	if (!static_cast<bool>(loaded.header.settings.inline_footer)) {
		throw std::runtime_error("equivalence check currently requires inline descriptors: " + path.string());
	}
	if (loaded.footer.table_descriptor_offset < sizeof(fastlanes::FileHeader) ||
	    loaded.footer.table_descriptor_offset > loaded.file_size - sizeof(fastlanes::FileFooter) ||
	    loaded.footer.table_descriptor_size >
	        loaded.file_size - sizeof(fastlanes::FileFooter) - loaded.footer.table_descriptor_offset) {
		throw std::runtime_error("inline descriptor range is out of bounds: " + path.string());
	}
	loaded.descriptor = fastlanes::TableDescriptorHandle::FromFileSlice(
	    file, loaded.footer.table_descriptor_offset, loaded.footer.table_descriptor_size, true);
	return loaded;
}

bool equal_file_prefixes(const std::filesystem::path& left, const std::filesystem::path& right, uint64_t bytes) {
	constexpr std::size_t kBufferBytes = 8U * 1024U * 1024U;
	std::ifstream         left_stream(left, std::ios::binary);
	std::ifstream         right_stream(right, std::ios::binary);
	if (!left_stream || !right_stream) {
		throw std::runtime_error("cannot open files for payload comparison");
	}
	std::vector<char> left_buffer(kBufferBytes);
	std::vector<char> right_buffer(kBufferBytes);
	while (bytes != 0) {
		const auto chunk = static_cast<std::streamsize>(std::min<uint64_t>(bytes, kBufferBytes));
		left_stream.read(left_buffer.data(), chunk);
		right_stream.read(right_buffer.data(), chunk);
		if (left_stream.gcount() != chunk || right_stream.gcount() != chunk) {
			throw std::runtime_error("short read during payload comparison");
		}
		if (!std::equal(left_buffer.begin(), left_buffer.begin() + chunk, right_buffer.begin())) {
			return false;
		}
		bytes -= static_cast<uint64_t>(chunk);
	}
	return true;
}

json compare_files(const Options& options) {
	const auto input     = std::filesystem::absolute(options.input);
	const auto candidate = std::filesystem::absolute(options.candidate);
	const auto left      = load_inline_descriptor(input);
	const auto right     = load_inline_descriptor(candidate);
	const bool headers_equal =
	    left.header.magic_bytes == right.header.magic_bytes && left.header.version == right.header.version &&
	    static_cast<bool>(left.header.settings.inline_footer) == static_cast<bool>(right.header.settings.inline_footer);
	const bool payload_boundary_equal = left.footer.table_descriptor_offset == right.footer.table_descriptor_offset;
	const bool payload_prefix_equal =
	    payload_boundary_equal && equal_file_prefixes(input, candidate, left.footer.table_descriptor_offset);
	const bool metadata_equal = equivalent_table(*left.descriptor, *right.descriptor);
	const bool ok             = headers_equal && payload_boundary_equal && payload_prefix_equal && metadata_equal;
	json       result;
	result["schema_version"] = "galp_fls_metadata_equivalence_v1";
	result["ok"]             = ok;
	result["input"]          = input.string();
	result["candidate"]      = candidate.string();
	result["checks"]         = {{"header_format_and_flags_equal", headers_equal},
	                            {"payload_boundary_equal", payload_boundary_equal},
	                            {"header_and_payload_prefix_byte_equal", payload_prefix_equal},
	                            {"decoded_metadata_semantically_equal", metadata_equal},
	                            {"current_flatbuffer_verifier_passed", true}};
	result["sizes"]          = {{"input_file_bytes", left.file_size},
	                            {"candidate_file_bytes", right.file_size},
	                            {"input_descriptor_bytes", left.footer.table_descriptor_size},
	                            {"candidate_descriptor_bytes", right.footer.table_descriptor_size}};
	if (!ok) {
		throw std::runtime_error("FLS equivalence check failed: " + result.dump());
	}
	return result;
}

json object_stats(const galp::metadata::InternedObjectStats& stats) {
	return {{"references", stats.references},
	        {"emitted", stats.emitted},
	        {"shared_references", stats.references - stats.emitted},
	        {"reuse_rate",
	         stats.references == 0 ? 0.0
	                               : 1.0 - static_cast<double>(stats.emitted) / static_cast<double>(stats.references)}};
}

void copy_prefix(std::ifstream& input, std::ofstream& output, uint64_t bytes) {
	constexpr std::size_t kCopyBufferBytes = 8U * 1024U * 1024U;
	std::vector<char>     buffer(kCopyBufferBytes);
	while (bytes != 0) {
		const auto chunk = static_cast<std::streamsize>(std::min<uint64_t>(bytes, buffer.size()));
		input.read(buffer.data(), chunk);
		if (input.gcount() != chunk) {
			throw std::runtime_error("failed to read FLS payload while copying");
		}
		output.write(buffer.data(), chunk);
		if (!output) {
			throw std::runtime_error("failed to write compacted FLS payload");
		}
		bytes -= static_cast<uint64_t>(chunk);
	}
}

json compact(const Options& options) {
	const auto total_start     = Clock::now();
	const auto input_absolute  = std::filesystem::absolute(options.input);
	const auto output_absolute = std::filesystem::absolute(options.output);
	if (input_absolute == output_absolute) {
		throw std::runtime_error("compact output must differ from input");
	}
	if (std::filesystem::exists(output_absolute)) {
		throw std::runtime_error("compact output already exists: " + output_absolute.string());
	}
	if (!output_absolute.parent_path().empty()) {
		std::filesystem::create_directories(output_absolute.parent_path());
	}

	const auto            load_start = Clock::now();
	fastlanes::File       input_file(input_absolute);
	const auto            input_size = input_file.Size();
	fastlanes::FileHeader header {};
	fastlanes::FileFooter footer {};
	require_status(fastlanes::FileHeader::Load(header, input_file), "read FLS header");
	require_status(fastlanes::FileFooter::Load(footer, input_file), "read FLS footer");
	if (header.magic_bytes != fastlanes::Info::get_magic_bytes() ||
	    footer.magic_bytes != fastlanes::Info::get_magic_bytes()) {
		throw std::runtime_error("FLS header/footer magic mismatch");
	}
	if (!static_cast<bool>(header.settings.inline_footer)) {
		throw std::runtime_error("prototype compact currently requires an inline table descriptor");
	}
	if (footer.table_descriptor_offset < sizeof(fastlanes::FileHeader) ||
	    footer.table_descriptor_offset > input_size - sizeof(fastlanes::FileFooter) ||
	    footer.table_descriptor_size > input_size - sizeof(fastlanes::FileFooter) - footer.table_descriptor_offset) {
		throw std::runtime_error("inline FLS table descriptor range is out of bounds");
	}
	const auto expected_input_size =
	    footer.table_descriptor_offset + footer.table_descriptor_size + sizeof(fastlanes::FileFooter);
	if (input_size != expected_input_size) {
		throw std::runtime_error("compact refuses files with unclassified trailing bytes");
	}

	auto descriptor = fastlanes::TableDescriptorHandle::FromFileSlice(
	    input_file, footer.table_descriptor_offset, footer.table_descriptor_size, true);
	if (!descriptor || descriptor->m_table_binary_size() != footer.table_descriptor_offset) {
		throw std::runtime_error("table descriptor payload boundary disagrees with FLS footer");
	}
	const auto                               load_end   = Clock::now();
	const auto                               pack_start = Clock::now();
	galp::metadata::InternedDescriptorResult packed;
	const bool                               use_interning = options.command == "compact";
	if (use_interning) {
		packed = galp::metadata::pack_interned_table_descriptor(*descriptor, descriptor.size());
	} else {
		auto native = descriptor.Unpack();
		if (!native) {
			throw std::runtime_error("failed to unpack legacy table descriptor");
		}
		flatbuffers::FlatBufferBuilder builder(
		    std::max<std::size_t>(1024U, std::min<std::size_t>(descriptor.size(), 64U * 1024U * 1024U)));
		const auto root = fastlanes::TableDescriptor::Pack(builder, native.get());
		fastlanes::FinishTableDescriptorBuffer(builder, root);
		auto bytes = builder.Release();
		packed.bytes.assign(bytes.data(), bytes.data() + bytes.size());
	}
	const auto pack_end = Clock::now();
	// The regular verifier is intentionally used here: compatibility is only claimed
	// if the existing reader's current-schema validation accepts shared objects.
	const auto verify_start = Clock::now();
	const auto verified     = fastlanes::TableDescriptorHandle::FromBytes(packed.bytes, true);
	if (!verified || verified->m_table_binary_size() != footer.table_descriptor_offset) {
		throw std::runtime_error("compacted table descriptor failed current-reader validation");
	}
	if (!equivalent_table(*descriptor, *verified)) {
		throw std::runtime_error("compacted table descriptor changed decoded metadata semantics");
	}
	const auto verify_end = Clock::now();

	const auto temporary = std::filesystem::path(output_absolute.string() + ".tmp");
	if (std::filesystem::exists(temporary)) {
		throw std::runtime_error("compact temporary output already exists: " + temporary.string());
	}
	const auto write_start = Clock::now();
	try {
		std::ifstream input(input_absolute, std::ios::binary);
		std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
		if (!input || !output) {
			throw std::runtime_error("cannot open compact input/output stream");
		}
		copy_prefix(input, output, footer.table_descriptor_offset);
		output.write(reinterpret_cast<const char*>(packed.bytes.data()),
		             static_cast<std::streamsize>(packed.bytes.size()));
		fastlanes::FileFooter compact_footer {footer.table_descriptor_offset, packed.bytes.size(), footer.magic_bytes};
		output.write(reinterpret_cast<const char*>(&compact_footer), sizeof(compact_footer));
		output.close();
		if (!output) {
			throw std::runtime_error("failed to finalize compacted FLS file");
		}
		std::filesystem::rename(temporary, output_absolute);
	} catch (...) {
		std::error_code ignored;
		std::filesystem::remove(temporary, ignored);
		throw;
	}
	const auto write_end = Clock::now();

	const auto  output_size = std::filesystem::file_size(output_absolute);
	const auto  total_end   = Clock::now();
	const auto& stats       = packed.stats;
	json        report;
	report["schema_version"] = use_interning ? "galp_fls_metadata_compaction_v1" : "galp_fls_metadata_legacy_repack_v1";
	report["prototype"]      = {
        {"enabled_by", "explicit " + options.command + " subcommand"},
        {"strategy", use_interning ? "current-schema exact-object interning" : "generated legacy unpack/pack"},
        {"default_writer_changed", false},
        {"format_version_changed", false},
        {"flatbuffer_schema_changed", false},
        {"payload_reencoded", false},
        {"existing_reader_verifier_passed", true},
        {"decoded_metadata_semantically_equal", true}};
	report["input"]   = {{"path", input_absolute.string()},
	                     {"file_bytes", input_size},
	                     {"payload_prefix_bytes", footer.table_descriptor_offset},
	                     {"descriptor_bytes", footer.table_descriptor_size}};
	report["output"]  = {{"path", output_absolute.string()},
	                     {"file_bytes", output_size},
	                     {"payload_prefix_bytes", footer.table_descriptor_offset},
	                     {"descriptor_bytes", packed.bytes.size()}};
	report["savings"] = {
	    {"descriptor_bytes", footer.table_descriptor_size - packed.bytes.size()},
	    {"descriptor_fraction",
	     1.0 - static_cast<double>(packed.bytes.size()) / static_cast<double>(footer.table_descriptor_size)},
	    {"file_bytes", input_size - output_size},
	    {"file_fraction", 1.0 - static_cast<double>(output_size) / static_cast<double>(input_size)}};
	report["writer_measurement"] = {
	    {"classification", "Verified by measurement"},
	    {"descriptor_load_and_verify_ms", elapsed_ms(load_start, load_end)},
	    {"descriptor_pack_ms", elapsed_ms(pack_start, pack_end)},
	    {"descriptor_pack_strategy",
	     use_interning ? "current-schema exact-object interning" : "generated legacy unpack/pack"},
	    {"candidate_verify_and_semantic_compare_ms", elapsed_ms(verify_start, verify_end)},
	    {"payload_copy_and_file_write_ms", elapsed_ms(write_start, write_end)},
	    {"total_ms", elapsed_ms(total_start, total_end)},
	    {"output_mib_per_second",
	     static_cast<double>(output_size) * 1000.0 / (1024.0 * 1024.0 * elapsed_ms(total_start, total_end))},
	    {"peak_rss_bytes", peak_rss_bytes()},
	    {"scope", "offline conversion including input descriptor load, packing, validation, payload copy, and rename"}};
	if (use_interning) {
		report["interning"] = {{"strings", object_stats(stats.strings)},
		                       {"rpn", object_stats(stats.rpns)},
		                       {"expression_results", object_stats(stats.expression_results)},
		                       {"expression_vectors", object_stats(stats.expression_vectors)},
		                       {"segment_descriptors", object_stats(stats.segment_descriptors)},
		                       {"segment_vectors", object_stats(stats.segment_vectors)},
		                       {"binary_values", object_stats(stats.binary_values)},
		                       {"decimals", object_stats(stats.decimals)},
		                       {"child_vectors", object_stats(stats.child_vectors)},
		                       {"columns", stats.columns},
		                       {"rowgroups", stats.rowgroups}};
	}
	return report;
}

void emit(const json& result, const Options& options) {
	const int   indent = options.pretty ? 2 : -1;
	const auto& destination =
	    (options.command == "compact" || options.command == "repack-legacy") ? options.report : options.output;
	if (destination.empty()) {
		std::cout << result.dump(indent) << '\n';
		return;
	}
	if (!destination.parent_path().empty()) {
		std::filesystem::create_directories(destination.parent_path());
	}
	std::ofstream output(destination);
	if (!output) {
		throw std::runtime_error("cannot open output file: " + destination.string());
	}
	output << result.dump(indent) << '\n';
	if (!output) {
		throw std::runtime_error("failed to write output file: " + destination.string());
	}
}

} // namespace

int main(const int argc, char** argv) {
	try {
		const auto options = parse_options(argc, argv);
		json       result;
		if (options.command == "analyze") {
			result = analyze(options.input);
		} else if (options.command == "compact" || options.command == "repack-legacy") {
			result = compact(options);
		} else {
			result = compare_files(options);
		}
		emit(result, options);
		return 0;
	} catch (const std::exception& error) {
		std::cerr << "galp_fls_metadata_tool: " << error.what() << '\n';
		return 1;
	}
}
