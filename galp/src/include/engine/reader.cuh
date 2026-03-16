// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/reader.cuh
// ────────────────────────────────────────────────────────
#ifndef FLS_READER_CUH
#define FLS_READER_CUH

#include "engine/data/model.cuh"
#include "fls/cor/lyt/buf.hpp"
#include "fls/expression/rpn.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/file/file_header.hpp"
#include "fls/footer/operator_token_generated.h"
#include "fls/footer/rowgroup_descriptor_generated.h"
#include "fls/footer/table_descriptor.hpp"
#include "fls/io/file.hpp"
#include "fls/io/io.hpp"
#include "fls/reader/column_view.hpp"
#include "fls/reader/rowgroup_view.hpp"
#include "fls/reader/segment.hpp"
#include "flsgpu/columns/all.cuh"
#include "flsgpu/flsgpu-api.cuh"
#include "flsgpu/utils.cuh"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <variant>
#include <vector>

namespace reader {

namespace detail {
inline fastlanes::TableDescriptorHandle load_table_descriptor(const std::filesystem::path& file_path) {
	fastlanes::FileHeader file_header {};
	fastlanes::FileFooter file_footer {};

	fastlanes::FileHeader::Load(file_header, file_path);
	fastlanes::FileFooter::Load(file_footer, file_path);

	if (file_header.settings.inline_footer) {
		return fastlanes::TableDescriptorHandle::FromFileSlice(
		    file_path, file_footer.table_descriptor_offset, file_footer.table_descriptor_size, /*verify=*/true);
	}

	const auto footer_path = file_path.parent_path() / "table_descriptor.fbb";
	return fastlanes::TableDescriptorHandle::FromFile(footer_path, /*verify=*/true);
}

} // namespace detail

using HostColumnVariant = dispatch::EncodedPayload;
using Column            = dispatch::Column;
using Rowgroup          = dispatch::Rowgroup;

class reader {
public:
	explicit reader(const std::filesystem::path& file_path)
	    : m_file_path(file_path)
	    , m_table_descriptor(detail::load_table_descriptor(file_path)) {
	}

	size_t rowgroup_count() const {
		const auto* td = m_table_descriptor.Get();
		if (!td) {
			throw std::runtime_error("TableDescriptor not loaded");
		}
		return static_cast<size_t>(td->m_rowgroup_descriptors()->size());
	}

	Rowgroup read_rowgroup(const size_t rowgroup_idx = 0) {
		const auto* td = m_table_descriptor.Get();
		if (!td) {
			throw std::runtime_error("TableDescriptor not loaded");
		}
		const auto n_rgs = td->m_rowgroup_descriptors()->size();
		if (rowgroup_idx >= n_rgs) {
			throw std::out_of_range("rowgroup_idx out of range");
		}

		const auto*  rg       = td->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rowgroup_idx));
		const size_t n_vecs   = static_cast<size_t>(rg->m_n_vec());
		const size_t n_values = n_vecs * consts::VALUES_PER_VECTOR;
		const size_t n_tuples = static_cast<size_t>(rg->m_n_tuples());

		// Read rowgroup bytes
		fastlanes::Buf buf(rg->m_size());
		fastlanes::io  io = fastlanes::make_unique<fastlanes::File>(m_file_path);
		fastlanes::IO::range_read(io, buf, rg->m_offset(), rg->m_size());
		fastlanes::RowgroupView rg_view(buf.Span(), *rg);

		// Build columns (with dependency resolution for DICT_I08_U08)
		const auto&                        col_descs = *rg->m_column_descriptors();
		std::vector<std::optional<Column>> built(col_descs.size());
		std::unordered_set<size_t>         in_progress;

		auto build_column = [&](auto&& self, size_t col_idx) -> Column& {
			if (col_idx >= built.size()) {
				throw std::out_of_range("column index out of range");
			}
			if (built[col_idx].has_value()) {
				return *built[col_idx];
			}
			if (in_progress.count(col_idx)) {
				throw std::runtime_error("cycle detected in column dependencies");
			}

			in_progress.insert(col_idx);

			const auto& col_desc = *col_descs.Get(static_cast<flatbuffers::uoffset_t>(col_idx));
			const auto* rpn      = col_desc.encoding_rpn();
			if (!rpn || !rpn->operator_tokens()) {
				throw std::runtime_error("missing encoding_rpn/operator_tokens");
			}
			const auto* ops = rpn->operator_tokens();
			if (ops->size() != 1) {
				std::ostringstream msg;
				msg << "only single-op expressions are supported in this reader; got ops=[";
				for (size_t i = 0; i < ops->size(); ++i) {
					if (i > 0) {
						msg << ", ";
					}
					msg << fastlanes::token_to_string(ops->Get(static_cast<flatbuffers::uoffset_t>(i)));
				}
				msg << "]";
				throw std::runtime_error(msg.str());
			}

			const auto op_token = ops->Get(0);
			Column     result;
			result.name          = col_desc.name() ? col_desc.name()->str() : std::string {};
			result.token         = op_token;
			auto& column_view    = rg_view[static_cast<fastlanes::n_t>(col_idx)];
			auto* operand_tokens = rpn->operand_tokens();

			columns::ParseContext ctx {column_view, col_desc, operand_tokens, n_values, n_vecs};

			switch (op_token) {
				using enum fastlanes::OperatorToken;
			case EXP_UNCOMPRESSED_I08: {
				auto parsed = columns::parse_uncompressed<int8_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_CONSTANT_I08: {
				auto parsed = columns::parse_constant<int8_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_FFOR_I08: {
				auto parsed = columns::parse_ffor<int8_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_FFOR_I16: {
				auto parsed = columns::parse_ffor<int16_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_FFOR_SLPATCH_I08: {
				auto parsed = columns::parse_slpatch<int8_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_FFOR_SLPATCH_I16: {
				auto parsed = columns::parse_slpatch<int16_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_FREQUENCY_I08: {
				auto parsed = columns::parse_frequency<int8_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_FREQUENCY_I16: {
				auto parsed = columns::parse_frequency<int16_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_CROSS_RLE_I08: {
				auto parsed = columns::parse_cross_rle<int8_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_DICT_I08_FFOR_SLPATCH_U08: {
				auto parsed = columns::parse_dict_slpatch<int8_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_DICT_I08_FFOR_U08: {
				auto parsed = columns::parse_dict_ffor<int8_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_DICT_I16_FFOR_U16: {
				auto parsed = columns::parse_dict_ffor<int16_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_DICT_I16_FFOR_U08: {
				auto parsed = columns::parse_dict_ffor<int16_t, uint8_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_DICT_I08_U08: {
				auto parsed = columns::parse_dict_ref<int8_t, uint8_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_DICT_I16_FFOR_SLPATCH_U16: {
				auto parsed = columns::parse_dict_slpatch<int16_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_DICT_I16_FFOR_SLPATCH_U08: {
				auto parsed = columns::parse_dict_slpatch<int16_t, uint8_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_RLE_I08_U16: {
				auto parsed = columns::parse_rle<int8_t, uint16_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_RLE_I16_U16: {
				auto parsed = columns::parse_rle<int16_t, uint16_t>(ctx);
				result.host = std::move(parsed.host);
				break;
			}
			case EXP_EQUAL: {
				if (!operand_tokens || operand_tokens->size() < 1) {
					throw std::runtime_error("EXP_EQUAL: missing operand tokens");
				}
				const auto src_col_idx = static_cast<size_t>(operand_tokens->Get(0));
				auto&      src_col     = self(self, src_col_idx);
				result.host            = src_col.host;
				result.skip_decompress = true;
				result.alias_of        = src_col_idx;
				break;
			}
			default: {
				std::ostringstream msg;
				msg << "unsupported operator token for this reader: " << fastlanes::token_to_string(op_token)
				    << " (col_index=" << col_idx
				    << ", name=" << (col_desc.name() ? col_desc.name()->str() : std::string("<unnamed>")) << ")";
				throw std::runtime_error(msg.str());
			}
			}

			built[col_idx] = std::move(result);
			in_progress.erase(col_idx);
			return *built[col_idx];
		};

		Rowgroup out {n_values, n_vecs, n_tuples, {}};
		out.columns.reserve(col_descs.size());
		for (size_t i = 0; i < col_descs.size(); ++i) {
			out.columns.push_back(build_column(build_column, i));
		}

		return out;
	}

	std::vector<Rowgroup> read_table() {
		const size_t          n_rgs = rowgroup_count();
		std::vector<Rowgroup> out;
		out.reserve(n_rgs);
		for (size_t rg_idx = 0; rg_idx < n_rgs; ++rg_idx) {
			out.emplace_back(read_rowgroup(rg_idx));
		}
		return out;
	}

private:
	std::filesystem::path            m_file_path;
	fastlanes::TableDescriptorHandle m_table_descriptor;
};

} // namespace reader

#endif // FLS_READER_CUH
