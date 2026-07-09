// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// examples/cpp_frequency_inspect.cpp
// ────────────────────────────────────────────────────────
#include "fls/common/alias.hpp"
#include "fls/cor/lyt/buf.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/file/file_header.hpp"
#include "fls/footer/datatype_generated.h"
#include "fls/footer/operator_token_generated.h"
#include "fls/footer/segment_descriptor.hpp"
#include "fls/footer/table_descriptor.hpp"
#include "fls/io/file.hpp"
#include "fls/io/io.hpp"
#include "fls/reader/column_view.hpp"
#include "fls/reader/rowgroup_view.hpp"
#include "fls/reader/segment.hpp"
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
	std::filesystem::path                input;
	std::optional<size_t>                rowgroup;
	std::optional<std::filesystem::path> csv_out;
	size_t                               max_samples = 8;
	bool                                 all_columns = false;
};

struct ExceptionSample {
	size_t      vec_idx = 0;
	uint16_t    pos     = 0;
	std::string value;
};

struct FrequencyStats {
	std::string                  frequent_value;
	uint64_t                     total_exceptions        = 0;
	uint64_t                     vectors_with_exceptions = 0;
	uint16_t                     max_exceptions_per_vec  = 0;
	std::vector<ExceptionSample> exception_samples;
};

struct CompressionStats {
	uint64_t compressed_bytes     = 0;
	double   bytes_per_value      = 0.0;
	double   bits_per_value       = 0.0;
	bool     has_per_value        = false;
	uint64_t raw_bytes_est        = 0;
	bool     has_raw_ratio        = false;
	double   ratio_raw_compressed = 0.0;
	double   ratio_compressed_raw = 0.0;
};

void print_usage(const char* prog) {
	std::cerr << "Usage:\n"
	          << "  " << prog << " <input.fls> [--rowgroup N] [--max-samples K] [--csv-out PATH] [--all-columns]\n"
	          << "\n"
	          << "Options:\n"
	          << "  --rowgroup N     Inspect one rowgroup only\n"
	          << "  --max-samples K  Max printed exception samples per frequency column (default: 8)\n"
	          << "  --csv-out PATH   Save stats to CSV\n"
	          << "  --all-columns    Export all columns (operator/token metadata + frequency stats when applicable)\n";
}

bool parse_args(const int argc, char** argv, Options& opt) {
	if (argc < 2) {
		return false;
	}

	for (int arg_idx = 1; arg_idx < argc; ++arg_idx) {
		const std::string_view arg = argv[arg_idx];
		if (arg == "--help" || arg == "-h") {
			return false;
		}
		if (arg == "--rowgroup" && arg_idx + 1 < argc) {
			opt.rowgroup = static_cast<size_t>(std::stoull(argv[++arg_idx]));
			continue;
		}
		if (arg == "--max-samples" && arg_idx + 1 < argc) {
			opt.max_samples = static_cast<size_t>(std::stoull(argv[++arg_idx]));
			continue;
		}
		if (arg == "--csv-out" && arg_idx + 1 < argc) {
			opt.csv_out = std::filesystem::path(argv[++arg_idx]);
			continue;
		}
		if (arg == "--all-columns") {
			opt.all_columns = true;
			continue;
		}
		if (opt.input.empty()) {
			opt.input = std::filesystem::path(arg);
			continue;
		}
		return false;
	}

	return !opt.input.empty();
}

std::string csv_escape(std::string_view input) {
	bool needs_quote = false;
	for (const char ch : input) {
		if (ch == ',' || ch == '"' || ch == '\n' || ch == '\r') {
			needs_quote = true;
			break;
		}
	}
	if (!needs_quote) {
		return std::string(input);
	}

	std::string out;
	out.reserve(input.size() + 2);
	out.push_back('"');
	for (const char ch : input) {
		if (ch == '"') {
			out.push_back('"');
		}
		out.push_back(ch);
	}
	out.push_back('"');
	return out;
}

size_t data_type_size(fastlanes::DataType dt);

CompressionStats compute_compression_stats(const fastlanes::ColumnDescriptor&   col_desc,
                                           const fastlanes::RowgroupDescriptor& rowgroup_desc,
                                           const fastlanes::DataType            dt) {
	CompressionStats out {};
	const uint64_t   n_tuples  = rowgroup_desc.m_n_tuples();
	const size_t     type_size = data_type_size(dt);

	out.compressed_bytes = col_desc.total_size();
	if (n_tuples > 0) {
		out.has_per_value   = true;
		out.bytes_per_value = static_cast<double>(out.compressed_bytes) / static_cast<double>(n_tuples);
		out.bits_per_value  = out.bytes_per_value * 8.0;
	}

	if (type_size == 0 || n_tuples == 0) {
		return out;
	}

	out.raw_bytes_est = n_tuples * type_size;
	if (out.compressed_bytes == 0) {
		return out;
	}

	out.has_raw_ratio        = true;
	const double raw_bytes   = static_cast<double>(out.raw_bytes_est);
	const double comp_bytes  = static_cast<double>(out.compressed_bytes);
	out.ratio_raw_compressed = raw_bytes / comp_bytes;
	out.ratio_compressed_raw = comp_bytes / raw_bytes;
	return out;
}

size_t data_type_size(const fastlanes::DataType dt) {
	using fastlanes::DataType;
	switch (dt) {
	case DataType::INT8:
	case DataType::UINT8:
	case DataType::BOOLEAN:
		return 1;
	case DataType::INT16:
	case DataType::UINT16:
		return 2;
	case DataType::INT32:
	case DataType::UINT32:
	case DataType::FLOAT:
	case DataType::DATE:
		return 4;
	case DataType::INT64:
	case DataType::UINT64:
	case DataType::DOUBLE:
	case DataType::TIMESTAMP:
		return 8;
	default:
		return 0;
	}
}

fastlanes::TableDescriptorHandle load_table_descriptor(const std::filesystem::path& file_path) {
	fastlanes::FileHeader header {};
	fastlanes::FileFooter footer {};

	fastlanes::FileHeader::Load(header, file_path);
	fastlanes::FileFooter::Load(footer, file_path);

	if (header.settings.inline_footer) {
		return fastlanes::TableDescriptorHandle::FromFileSlice(
		    file_path, footer.table_descriptor_offset, footer.table_descriptor_size, /*verify=*/true);
	}

	return fastlanes::TableDescriptorHandle::FromFile(file_path.parent_path() / "table_descriptor.fbb",
	                                                  /*verify=*/true);
}

bool is_frequency_token(const fastlanes::OperatorToken token) {
	using fastlanes::OperatorToken;
	switch (token) {
	case OperatorToken::EXP_FREQUENCY_DBL:
	case OperatorToken::EXP_FREQUENCY_FLT:
	case OperatorToken::EXP_FREQUENCY_I08:
	case OperatorToken::EXP_FREQUENCY_U08:
	case OperatorToken::EXP_FREQUENCY_I16:
	case OperatorToken::EXP_FREQUENCY_I32:
	case OperatorToken::EXP_FREQUENCY_I64:
	case OperatorToken::EXP_FREQUENCY_STR:
		return true;
	default:
		return false;
	}
}

std::string format_bytes_string(const uint8_t* bytes, const size_t len, const size_t max_len = 32) {
	std::ostringstream oss;
	oss << "\"";

	const size_t shown = std::min(len, max_len);
	for (size_t idx = 0; idx < shown; ++idx) {
		const unsigned char ch = bytes[idx];
		if (std::isprint(ch) && ch != '\\' && ch != '"') {
			oss << static_cast<char>(ch);
		} else {
			oss << "\\x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<uint32_t>(ch) << std::dec
			    << std::setfill(' ');
		}
	}
	if (len > shown) {
		oss << "...";
	}
	oss << "\"(" << len << "B)";
	return oss.str();
}

template <typename T>
std::string scalar_to_string(const T value) {
	if constexpr (std::is_same_v<T, int8_t>) {
		return std::to_string(static_cast<int32_t>(value));
	} else if constexpr (std::is_same_v<T, uint8_t>) {
		return std::to_string(static_cast<uint32_t>(value));
	} else if constexpr (std::is_floating_point_v<T>) {
		std::ostringstream oss;
		oss << std::setprecision(9) << value;
		return oss.str();
	} else {
		return std::to_string(value);
	}
}

uint64_t operand_at(const flatbuffers::Vector<uint64_t>& operands, const size_t idx) {
	return operands.Get(static_cast<flatbuffers::uoffset_t>(idx));
}

template <typename T>
FrequencyStats inspect_frequency_numeric(const fastlanes::ColumnView&         column_view,
                                         const flatbuffers::Vector<uint64_t>& operands,
                                         const size_t                         n_vecs,
                                         const size_t                         max_samples) {
	FrequencyStats out {};
	const size_t   base_idx = operands.size() - 1;

	auto frequent_value_seg = column_view.GetSegment(static_cast<fastlanes::n_t>(operand_at(operands, base_idx - 3)));
	auto exceptions_seg     = column_view.GetSegment(static_cast<fastlanes::n_t>(operand_at(operands, base_idx - 2)));
	auto positions_seg      = column_view.GetSegment(static_cast<fastlanes::n_t>(operand_at(operands, base_idx - 1)));
	auto count_seg          = column_view.GetSegment(static_cast<fastlanes::n_t>(operand_at(operands, base_idx - 0)));

	frequent_value_seg.PointTo(0);
	const auto* frequent_value_ptr = reinterpret_cast<const T*>(frequent_value_seg.data);
	out.frequent_value             = scalar_to_string(*frequent_value_ptr);

	for (size_t vec_idx = 0; vec_idx < n_vecs; ++vec_idx) {
		count_seg.PointTo(static_cast<fastlanes::n_t>(vec_idx));
		const auto* count_ptr = reinterpret_cast<const uint16_t*>(count_seg.data);
		const auto  count     = *count_ptr;

		out.total_exceptions += static_cast<uint64_t>(count);
		if (count > 0) {
			++out.vectors_with_exceptions;
		}
		out.max_exceptions_per_vec = std::max<uint16_t>(out.max_exceptions_per_vec, count);

		if (out.exception_samples.size() >= max_samples || count == 0) {
			continue;
		}

		exceptions_seg.PointTo(static_cast<fastlanes::n_t>(vec_idx));
		positions_seg.PointTo(static_cast<fastlanes::n_t>(vec_idx));
		const auto* exception_values_ptr = reinterpret_cast<const T*>(exceptions_seg.data);
		const auto* exception_pos_ptr    = reinterpret_cast<const uint16_t*>(positions_seg.data);

		const size_t remaining = max_samples - out.exception_samples.size();
		const size_t to_copy   = std::min(remaining, static_cast<size_t>(count));
		for (size_t sample_idx = 0; sample_idx < to_copy; ++sample_idx) {
			out.exception_samples.push_back(ExceptionSample {
			    vec_idx,
			    exception_pos_ptr[sample_idx],
			    scalar_to_string(exception_values_ptr[sample_idx]),
			});
		}
	}

	return out;
}

FrequencyStats inspect_frequency_string(const fastlanes::ColumnView&         column_view,
                                        const flatbuffers::Vector<uint64_t>& operands,
                                        const size_t                         n_vecs,
                                        const size_t                         max_samples) {
	FrequencyStats out {};
	const size_t   base_idx = operands.size() - 1;

	auto frequent_value_bytes_seg =
	    column_view.GetSegment(static_cast<fastlanes::n_t>(operand_at(operands, base_idx - 5)));
	auto frequent_value_size_seg =
	    column_view.GetSegment(static_cast<fastlanes::n_t>(operand_at(operands, base_idx - 4)));
	auto n_exceptions_seg = column_view.GetSegment(static_cast<fastlanes::n_t>(operand_at(operands, base_idx - 3)));
	auto exception_positions_seg =
	    column_view.GetSegment(static_cast<fastlanes::n_t>(operand_at(operands, base_idx - 2)));
	auto exception_values_bytes_seg =
	    column_view.GetSegment(static_cast<fastlanes::n_t>(operand_at(operands, base_idx - 1)));
	auto exception_values_offset_seg =
	    column_view.GetSegment(static_cast<fastlanes::n_t>(operand_at(operands, base_idx - 0)));

	frequent_value_bytes_seg.PointTo(0);
	frequent_value_size_seg.PointTo(0);
	const auto  frequent_len = *reinterpret_cast<const fastlanes::len_t*>(frequent_value_size_seg.data);
	const auto* frequent_ptr = reinterpret_cast<const uint8_t*>(frequent_value_bytes_seg.data);
	out.frequent_value       = format_bytes_string(frequent_ptr, static_cast<size_t>(frequent_len));

	for (size_t vec_idx = 0; vec_idx < n_vecs; ++vec_idx) {
		n_exceptions_seg.PointTo(static_cast<fastlanes::n_t>(vec_idx));
		const auto count = *reinterpret_cast<const uint16_t*>(n_exceptions_seg.data);

		out.total_exceptions += static_cast<uint64_t>(count);
		if (count > 0) {
			++out.vectors_with_exceptions;
		}
		out.max_exceptions_per_vec = std::max<uint16_t>(out.max_exceptions_per_vec, count);

		if (out.exception_samples.size() >= max_samples || count == 0) {
			continue;
		}

		exception_positions_seg.PointTo(static_cast<fastlanes::n_t>(vec_idx));
		exception_values_bytes_seg.PointTo(static_cast<fastlanes::n_t>(vec_idx));
		exception_values_offset_seg.PointTo(static_cast<fastlanes::n_t>(vec_idx));

		const auto* pos_ptr     = reinterpret_cast<const uint16_t*>(exception_positions_seg.data);
		const auto* value_bytes = reinterpret_cast<const uint8_t*>(exception_values_bytes_seg.data);
		const auto* value_offs  = reinterpret_cast<const fastlanes::ofs_t*>(exception_values_offset_seg.data);

		const size_t remaining = max_samples - out.exception_samples.size();
		const size_t to_copy   = std::min(remaining, static_cast<size_t>(count));
		for (size_t sample_idx = 0; sample_idx < to_copy; ++sample_idx) {
			const auto end_offset = static_cast<size_t>(value_offs[sample_idx]);
			const auto beg_offset = (sample_idx == 0) ? 0U : static_cast<size_t>(value_offs[sample_idx - 1]);
			const auto len        = (end_offset >= beg_offset) ? (end_offset - beg_offset) : 0U;
			out.exception_samples.push_back(ExceptionSample {
			    vec_idx,
			    pos_ptr[sample_idx],
			    format_bytes_string(value_bytes + beg_offset, len),
			});
		}
	}

	return out;
}

void print_compression_info(const CompressionStats& stats) {
	std::cout << "    compressed_bytes: " << stats.compressed_bytes << "\n";
	if (stats.has_per_value) {
		std::cout << "    bytes_per_value:  " << stats.bytes_per_value << "\n";
		std::cout << "    bits_per_value:   " << stats.bits_per_value << "\n";
	}
	if (stats.raw_bytes_est == 0) {
		std::cout << "    compression_ratio(raw/compressed): n/a\n";
		std::cout << "    compression_ratio(compressed/raw): n/a\n";
		return;
	}

	std::cout << "    raw_bytes_est:    " << stats.raw_bytes_est << "\n";
	if (!stats.has_raw_ratio) {
		std::cout << "    compression_ratio(raw/compressed): n/a\n";
		std::cout << "    compression_ratio(compressed/raw): 0\n";
		return;
	}
	std::cout << "    compression_ratio(raw/compressed): " << stats.ratio_raw_compressed << "\n";
	std::cout << "    compression_ratio(compressed/raw): " << stats.ratio_compressed_raw << "\n";
}

void print_frequency_stats(const FrequencyStats& stats, const size_t n_vecs) {
	std::cout << "    frequent_value:   " << stats.frequent_value << "\n";
	std::cout << "    exceptions_total: " << stats.total_exceptions << "\n";
	if (n_vecs > 0) {
		const double avg_exc_per_vec = static_cast<double>(stats.total_exceptions) / static_cast<double>(n_vecs);
		const double vec_with_exc_pct =
		    (100.0 * static_cast<double>(stats.vectors_with_exceptions) / static_cast<double>(n_vecs));
		std::cout << "    exceptions_avg_per_vec: " << avg_exc_per_vec << "\n";
		std::cout << "    exceptions_max_per_vec: " << stats.max_exceptions_per_vec << "\n";
		std::cout << "    vectors_with_exceptions: " << stats.vectors_with_exceptions << "/" << n_vecs << " ("
		          << vec_with_exc_pct << "%)\n";
	}
	if (stats.exception_samples.empty()) {
		std::cout << "    exception_samples: none\n";
		return;
	}
	std::cout << "    exception_samples:\n";
	for (const auto& sample : stats.exception_samples) {
		std::cout << "      - vec=" << sample.vec_idx << ", pos=" << sample.pos << ", value=" << sample.value << "\n";
	}
}

void write_raw_ratio_csv_fields(std::ostream& csv, const CompressionStats& comp_stats, const bool trailing_separator) {
	if (comp_stats.has_raw_ratio) {
		csv << comp_stats.ratio_raw_compressed << "," << comp_stats.ratio_compressed_raw;
	} else {
		csv << ",";
		if (comp_stats.raw_bytes_est > 0) {
			csv << "0";
		}
	}
	if (trailing_separator) {
		csv << ",";
	}
}

void write_csv_row(std::ostream&           csv,
                   const size_t            rowgroup_idx,
                   const size_t            column_idx,
                   const std::string_view  column_name,
                   const std::string_view  data_type_name,
                   const std::string_view  operator_name,
                   const size_t            n_vecs,
                   const uint64_t          n_tuples,
                   const FrequencyStats&   freq_stats,
                   const CompressionStats& comp_stats) {
	const double exc_avg_per_vec =
	    (n_vecs > 0) ? (static_cast<double>(freq_stats.total_exceptions) / static_cast<double>(n_vecs)) : 0.0;
	const double vec_with_exc_pct =
	    (n_vecs > 0) ? (100.0 * static_cast<double>(freq_stats.vectors_with_exceptions) / static_cast<double>(n_vecs))
	                 : 0.0;

	csv << rowgroup_idx << "," << column_idx << "," << csv_escape(column_name) << "," << data_type_name << ","
	    << operator_name << "," << n_vecs << "," << n_tuples << "," << csv_escape(freq_stats.frequent_value) << ","
	    << freq_stats.total_exceptions << "," << exc_avg_per_vec << "," << freq_stats.max_exceptions_per_vec << ","
	    << freq_stats.vectors_with_exceptions << "," << vec_with_exc_pct << "," << comp_stats.compressed_bytes << ",";

	if (comp_stats.has_per_value) {
		csv << comp_stats.bytes_per_value << "," << comp_stats.bits_per_value << ",";
	} else {
		csv << ",,";
	}

	if (comp_stats.raw_bytes_est > 0) {
		csv << comp_stats.raw_bytes_est << ",";
	} else {
		csv << ",";
	}

	write_raw_ratio_csv_fields(csv, comp_stats, /*trailing_separator=*/false);
	csv << "\n";
}

std::string operator_chain_to_string(const flatbuffers::Vector<fastlanes::OperatorToken>& op_tokens) {
	std::ostringstream oss;
	for (flatbuffers::uoffset_t i = 0; i < op_tokens.size(); ++i) {
		if (i > 0) {
			oss << ">";
		}
		const auto  tok  = op_tokens.Get(i);
		const char* name = fastlanes::EnumNameOperatorToken(tok);
		oss << (name ? name : "UNKNOWN");
	}
	return oss.str();
}

void write_csv_row_all(std::ostream&                        csv,
                       const size_t                         rowgroup_idx,
                       const size_t                         column_idx,
                       const std::string_view               column_name,
                       const std::string_view               data_type_name,
                       const std::string_view               last_operator_name,
                       const std::string_view               operator_chain,
                       const bool                           is_frequency,
                       const size_t                         n_vecs,
                       const uint64_t                       n_tuples,
                       const CompressionStats&              comp_stats,
                       const std::optional<FrequencyStats>& freq_stats) {
	csv << rowgroup_idx << "," << column_idx << "," << csv_escape(column_name) << "," << data_type_name << ","
	    << last_operator_name << "," << csv_escape(operator_chain) << "," << (is_frequency ? 1 : 0) << "," << n_vecs
	    << "," << n_tuples << "," << comp_stats.compressed_bytes << ",";

	if (comp_stats.has_per_value) {
		csv << comp_stats.bytes_per_value << "," << comp_stats.bits_per_value << ",";
	} else {
		csv << ",,";
	}
	if (comp_stats.raw_bytes_est > 0) {
		csv << comp_stats.raw_bytes_est << ",";
	} else {
		csv << ",";
	}
	write_raw_ratio_csv_fields(csv, comp_stats, /*trailing_separator=*/true);

	if (!freq_stats.has_value()) {
		csv << ",,,,,";
		csv << "\n";
		return;
	}

	const auto&  f = *freq_stats;
	const double exc_avg_per_vec =
	    (n_vecs > 0) ? (static_cast<double>(f.total_exceptions) / static_cast<double>(n_vecs)) : 0.0;
	const double vec_with_exc_pct =
	    (n_vecs > 0) ? (100.0 * static_cast<double>(f.vectors_with_exceptions) / static_cast<double>(n_vecs)) : 0.0;
	csv << csv_escape(f.frequent_value) << "," << f.total_exceptions << "," << exc_avg_per_vec << ","
	    << f.max_exceptions_per_vec << "," << f.vectors_with_exceptions << "," << vec_with_exc_pct << "\n";
}

} // namespace

int main(int argc, char** argv) {
	Options opt {};
	if (!parse_args(argc, argv, opt)) {
		print_usage(argv[0]);
		return 1;
	}

	try {
		const auto  td_handle = load_table_descriptor(opt.input);
		const auto* td        = td_handle.Get();
		if (!td) {
			throw std::runtime_error("failed to load table descriptor");
		}

		const auto*  rowgroups   = td->m_rowgroup_descriptors();
		const size_t n_rowgroups = rowgroups ? rowgroups->size() : 0U;
		if (n_rowgroups == 0) {
			throw std::runtime_error("no rowgroups found");
		}

		size_t rg_begin = 0;
		size_t rg_end   = n_rowgroups;
		if (opt.rowgroup.has_value()) {
			if (*opt.rowgroup >= n_rowgroups) {
				throw std::out_of_range("rowgroup index out of range");
			}
			rg_begin = *opt.rowgroup;
			rg_end   = rg_begin + 1;
		}

		std::unique_ptr<std::ofstream> csv_out_stream;
		if (opt.csv_out.has_value()) {
			csv_out_stream = std::make_unique<std::ofstream>(*opt.csv_out);
			if (!csv_out_stream->is_open()) {
				throw std::runtime_error("failed to open csv output file: " + opt.csv_out->string());
			}
			if (opt.all_columns) {
				(*csv_out_stream)
				    << "rowgroup,column_idx,column_name,data_type,last_operator,operator_chain,is_frequency,"
				    << "n_vec,n_tuples,compressed_bytes,bytes_per_value,bits_per_value,raw_bytes_est,"
				    << "compression_ratio_raw_over_compressed,compression_ratio_compressed_over_raw,"
				    << "frequent_value,exceptions_total,exceptions_avg_per_vec,exceptions_max_per_vec,"
				    << "vectors_with_exceptions,vectors_with_exceptions_pct\n";
			} else {
				(*csv_out_stream)
				    << "rowgroup,column_idx,column_name,data_type,operator,n_vec,n_tuples,frequent_value,"
				    << "exceptions_total,exceptions_avg_per_vec,exceptions_max_per_vec,vectors_with_exceptions,"
				    << "vectors_with_exceptions_pct,compressed_bytes,bytes_per_value,bits_per_value,raw_bytes_est,"
				    << "compression_ratio_raw_over_compressed,compression_ratio_compressed_over_raw\n";
			}
		}

		fastlanes::io input_io = fastlanes::make_unique<fastlanes::File>(opt.input);

		size_t total_freq_columns = 0;
		for (size_t rg_idx = rg_begin; rg_idx < rg_end; ++rg_idx) {
			const auto* rg_desc = rowgroups->Get(static_cast<flatbuffers::uoffset_t>(rg_idx));
			if (!rg_desc) {
				continue;
			}

			fastlanes::Buf rg_buf(rg_desc->m_size());
			fastlanes::IO::range_read(input_io, rg_buf, rg_desc->m_offset(), rg_desc->m_size());
			fastlanes::RowgroupView rowgroup_view(rg_buf.Span(), *rg_desc);

			const auto* column_descs = rg_desc->m_column_descriptors();
			if (!column_descs) {
				continue;
			}

			std::cout << "========== rowgroup " << rg_idx << " ==========\n";
			std::cout << "n_vec=" << rg_desc->m_n_vec() << ", n_tuples=" << rg_desc->m_n_tuples()
			          << ", n_columns=" << column_descs->size() << "\n";

			size_t rg_freq_columns = 0;
			for (size_t col_idx = 0; col_idx < column_descs->size(); ++col_idx) {
				const auto* col_desc = column_descs->Get(static_cast<flatbuffers::uoffset_t>(col_idx));
				if (!col_desc || !col_desc->encoding_rpn() || !col_desc->encoding_rpn()->operator_tokens() ||
				    !col_desc->encoding_rpn()->operand_tokens()) {
					continue;
				}

				const auto* op_tokens = col_desc->encoding_rpn()->operator_tokens();
				const auto* operands  = col_desc->encoding_rpn()->operand_tokens();
				if (op_tokens->size() == 0 || operands->size() == 0) {
					continue;
				}

				const auto last_token = op_tokens->Get(op_tokens->size() - 1);
				const bool is_freq    = is_frequency_token(last_token);
				if (!is_freq && !opt.all_columns) {
					continue;
				}

				if (is_freq) {
					++rg_freq_columns;
					++total_freq_columns;
				}

				const auto* column_name = col_desc->name() ? col_desc->name()->c_str() : "<unnamed>";
				const auto* dtype_name  = fastlanes::EnumNameDataType(col_desc->data_type());
				const auto* op_name     = fastlanes::EnumNameOperatorToken(last_token);
				const auto  op_chain    = operator_chain_to_string(*op_tokens);

				std::optional<FrequencyStats> stats;
				const auto&                   column_view = rowgroup_view[static_cast<fastlanes::n_t>(col_idx)];
				if (is_freq && last_token == fastlanes::OperatorToken::EXP_FREQUENCY_STR) {
					if (operands->size() < 6) {
						std::cout << "    warning: operand size < 6, skip parsing\n";
					} else {
						stats = inspect_frequency_string(
						    column_view, *operands, static_cast<size_t>(rg_desc->m_n_vec()), opt.max_samples);
					}
				} else if (is_freq) {
					if (operands->size() < 4) {
						std::cout << "    warning: operand size < 4, skip parsing\n";
					} else {
						switch (last_token) {
						case fastlanes::OperatorToken::EXP_FREQUENCY_DBL: {
							stats = inspect_frequency_numeric<double>(
							    column_view, *operands, static_cast<size_t>(rg_desc->m_n_vec()), opt.max_samples);
							break;
						}
						case fastlanes::OperatorToken::EXP_FREQUENCY_FLT: {
							stats = inspect_frequency_numeric<float>(
							    column_view, *operands, static_cast<size_t>(rg_desc->m_n_vec()), opt.max_samples);
							break;
						}
						case fastlanes::OperatorToken::EXP_FREQUENCY_I08: {
							stats = inspect_frequency_numeric<int8_t>(
							    column_view, *operands, static_cast<size_t>(rg_desc->m_n_vec()), opt.max_samples);
							break;
						}
						case fastlanes::OperatorToken::EXP_FREQUENCY_U08: {
							stats = inspect_frequency_numeric<uint8_t>(
							    column_view, *operands, static_cast<size_t>(rg_desc->m_n_vec()), opt.max_samples);
							break;
						}
						case fastlanes::OperatorToken::EXP_FREQUENCY_I16: {
							stats = inspect_frequency_numeric<int16_t>(
							    column_view, *operands, static_cast<size_t>(rg_desc->m_n_vec()), opt.max_samples);
							break;
						}
						case fastlanes::OperatorToken::EXP_FREQUENCY_I32: {
							stats = inspect_frequency_numeric<int32_t>(
							    column_view, *operands, static_cast<size_t>(rg_desc->m_n_vec()), opt.max_samples);
							break;
						}
						case fastlanes::OperatorToken::EXP_FREQUENCY_I64: {
							stats = inspect_frequency_numeric<int64_t>(
							    column_view, *operands, static_cast<size_t>(rg_desc->m_n_vec()), opt.max_samples);
							break;
						}
						default:
							std::cout << "    warning: unsupported frequency token\n";
							break;
						}
					}
				}

				const auto comp_stats = compute_compression_stats(*col_desc, *rg_desc, col_desc->data_type());
				if (is_freq) {
					std::cout << "\n";
					std::cout << "[column " << col_idx << "] name=\"" << column_name << "\"\n";
					std::cout << "    data_type:        " << dtype_name << "\n";
					std::cout << "    operator:         " << op_name << "\n";
					if (stats.has_value()) {
						print_frequency_stats(*stats, static_cast<size_t>(rg_desc->m_n_vec()));
					}
					print_compression_info(comp_stats);
				}

				if (csv_out_stream) {
					if (opt.all_columns) {
						write_csv_row_all(*csv_out_stream,
						                  rg_idx,
						                  col_idx,
						                  column_name ? column_name : "",
						                  dtype_name ? dtype_name : "",
						                  op_name ? op_name : "",
						                  op_chain,
						                  is_freq,
						                  static_cast<size_t>(rg_desc->m_n_vec()),
						                  rg_desc->m_n_tuples(),
						                  comp_stats,
						                  stats);
					} else if (stats.has_value()) {
						write_csv_row(*csv_out_stream,
						              rg_idx,
						              col_idx,
						              column_name,
						              dtype_name ? dtype_name : "",
						              op_name ? op_name : "",
						              static_cast<size_t>(rg_desc->m_n_vec()),
						              rg_desc->m_n_tuples(),
						              *stats,
						              comp_stats);
					}
				}
			}

			std::cout << "\n";
			std::cout << "rowgroup " << rg_idx << " frequency columns: " << rg_freq_columns << "\n\n";
		}

		std::cout << "========== summary ==========\n";
		std::cout << "rowgroups_scanned: " << (rg_end - rg_begin) << "\n";
		std::cout << "frequency_columns: " << total_freq_columns << "\n";
		return 0;
	} catch (const std::exception& ex) {
		std::cerr << "Error: " << ex.what() << "\n";
		return 1;
	}
}
