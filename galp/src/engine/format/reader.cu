// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/format/reader.cu
// ────────────────────────────────────────────────────────
#include "engine/format/zero_copy_materializer.cuh"
#include "engine/reader.cuh"
#include "fls/cor/lyt/buf.hpp"
#include "fls/expression/rpn.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/file/file_header.hpp"
#include "fls/io/file.hpp"
#include "galp/errors.hpp"
#include <chrono>
#include <flatbuffers/base.h>
#include <sstream>
#include <stdexcept>

namespace galp::format::detail {

fastlanes::TableDescriptorHandle load_table_descriptor(fastlanes::File& file, const std::filesystem::path& file_path) {
	fastlanes::FileHeader file_header {};
	fastlanes::FileFooter file_footer {};

	fastlanes::FileHeader::Load(file_header, file);
	fastlanes::FileFooter::Load(file_footer, file);

	if (file_header.settings.inline_footer) {
		return fastlanes::TableDescriptorHandle::FromFileSlice(
		    file, file_footer.table_descriptor_offset, file_footer.table_descriptor_size, /*verify=*/true);
	}

	const auto footer_path = file_path.parent_path() / "table_descriptor.fbb";
	return fastlanes::TableDescriptorHandle::FromFile(footer_path, /*verify=*/true);
}

fastlanes::TableDescriptorHandle load_table_descriptor(const std::filesystem::path& file_path) {
	fastlanes::File file(file_path);
	return load_table_descriptor(file, file_path);
}

} // namespace galp::format::detail

namespace galp::format {

FlsReader::FlsReader(const std::filesystem::path& file_path, const bool load_column_names)
    : m_file(std::make_shared<fastlanes::File>(file_path))
    , m_table_descriptor(
          std::make_shared<fastlanes::TableDescriptorHandle>(detail::load_table_descriptor(*m_file, file_path)))
    , m_load_column_names(load_column_names) {
	m_zero_copy_schema_plan = std::make_shared<ZeroCopySchemaPlan>(build_shared_zero_copy_schema_plan());
}

const fastlanes::TableDescriptor* FlsReader::table_descriptor() const {
	const auto* td = m_table_descriptor->Get();
	if (!td) {
		throw std::runtime_error("TableDescriptor not loaded");
	}
	return td;
}

size_t FlsReader::rowgroup_count() const {
	const auto* td = table_descriptor();
	return static_cast<size_t>(td->m_rowgroup_descriptors()->size());
}

size_t FlsReader::rowgroup_storage_bytes(const size_t rowgroup_idx) const {
	const auto* td = m_table_descriptor->Get();
	if (!td) {
		throw std::runtime_error("TableDescriptor not loaded");
	}
	const auto n_rgs = td->m_rowgroup_descriptors()->size();
	if (rowgroup_idx >= n_rgs) {
		throw std::out_of_range("rowgroup_idx out of range");
	}
	const auto* rg = td->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rowgroup_idx));
	return static_cast<size_t>(rg->m_size());
}

void FlsReader::read_rowgroup_bytes_into(const size_t        rowgroup_idx,
                                         std::byte* const    backing_data,
                                         const size_t        backing_capacity,
                                         ZeroCopyReadTiming* timing) {
	const auto* td = m_table_descriptor->Get();
	if (!td) {
		throw std::runtime_error("TableDescriptor not loaded");
	}
	const auto n_rgs = td->m_rowgroup_descriptors()->size();
	if (rowgroup_idx >= n_rgs) {
		throw std::out_of_range("rowgroup_idx out of range");
	}

	const auto*  rg       = td->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rowgroup_idx));
	const size_t rg_bytes = static_cast<size_t>(rg->m_size());
	if (backing_data == nullptr || backing_capacity < rg_bytes) {
		throw std::runtime_error("external rowgroup backing is null or too small");
	}

	const auto pread_start = std::chrono::steady_clock::now();
	m_file->ReadRangeUnchecked(backing_data, rg->m_offset(), rg->m_size());
	const auto pread_end = std::chrono::steady_clock::now();
	if (timing != nullptr) {
		timing->storage_bytes = rg_bytes;
		timing->pread_ms += std::chrono::duration<double, std::milli>(pread_end - pread_start).count();
		if (timing->pread_start == std::chrono::steady_clock::time_point {} || pread_start < timing->pread_start) {
			timing->pread_start = pread_start;
		}
		if (pread_end > timing->pread_end) {
			timing->pread_end = pread_end;
		}
	}
}

ZeroCopyRowgroup FlsReader::make_zero_copy_rowgroup_from_backing(const size_t          rowgroup_idx,
                                                                 std::shared_ptr<void> backing_owner,
                                                                 std::byte* const      backing_data,
                                                                 const size_t          backing_capacity,
                                                                 const bool            backing_is_pinned,
                                                                 ZeroCopyReadTiming*   timing) {
	const auto  setup_start = std::chrono::steady_clock::now();
	const auto* td          = m_table_descriptor->Get();
	if (!td) {
		throw std::runtime_error("TableDescriptor not loaded");
	}
	const auto n_rgs = td->m_rowgroup_descriptors()->size();
	if (rowgroup_idx >= n_rgs) {
		throw std::out_of_range("rowgroup_idx out of range");
	}

	const auto*  rg       = td->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rowgroup_idx));
	const size_t rg_bytes = static_cast<size_t>(rg->m_size());
	if (backing_data == nullptr || backing_capacity < rg_bytes) {
		throw std::runtime_error("external rowgroup backing is null or too small");
	}

	const size_t n_vecs   = static_cast<size_t>(rg->m_n_vec());
	const size_t n_values = n_vecs * galp::codec::consts::VALUES_PER_VECTOR;
	const size_t n_tuples = static_cast<size_t>(rg->m_n_tuples());

	auto        backing_span    = fastlanes::span<std::byte> {backing_data, rg_bytes};
	const auto& col_descs       = *rg->m_column_descriptors();
	const bool  use_schema_plan = m_zero_copy_schema_plan && m_zero_copy_schema_plan->enabled &&
	                             m_zero_copy_schema_plan->columns.size() == col_descs.size();

	std::shared_ptr<fastlanes::RowgroupView> view;
	if (!use_schema_plan) {
		view = std::make_shared<fastlanes::RowgroupView>(backing_span, *rg);
	}

	ZeroCopyRowgroup out {};
	out.rowgroup_index         = rowgroup_idx;
	out.n_values               = n_values;
	out.n_vecs                 = n_vecs;
	out.n_tuples               = n_tuples;
	out.table_descriptor_owner = m_table_descriptor;
	out.rowgroup_descriptor    = rg;
	out.backing_owner          = std::move(backing_owner);
	out.backing_span           = backing_span;
	out.backing_is_pinned      = backing_is_pinned;
	out.rowgroup_view          = view;

	const auto record_timing = [&](const std::chrono::steady_clock::time_point setup_end) {
		if (timing == nullptr) {
			return;
		}
		timing->storage_bytes = rg_bytes;
		timing->zero_copy_view_setup_ms += std::chrono::duration<double, std::milli>(setup_end - setup_start).count();
	};

	if (use_schema_plan) {
		out.schema_plan_owner = m_zero_copy_schema_plan;
		out.schema_plan       = out.schema_plan_owner.get();
		record_timing(std::chrono::steady_clock::now());
		return out;
	}

	out.columns.reserve(col_descs.size());
	for (size_t col_idx = 0; col_idx < col_descs.size(); ++col_idx) {
		const auto& col_desc = *col_descs.Get(static_cast<flatbuffers::uoffset_t>(col_idx));
		const auto* rpn      = col_desc.encoding_rpn();
		if (!rpn || !rpn->operator_tokens()) {
			throw std::runtime_error("missing encoding_rpn/operator_tokens");
		}
		const auto* ops = rpn->operator_tokens();
		if (ops->size() != 1) {
			std::ostringstream msg;
			msg << "only single-op expressions are supported in zero-copy reader; got ops=[";
			for (size_t i = 0; i < ops->size(); ++i) {
				if (i > 0) {
					msg << ", ";
				}
				msg << fastlanes::token_to_string(ops->Get(static_cast<flatbuffers::uoffset_t>(i)));
			}
			msg << "]";
			const std::string col_name =
			    (m_load_column_names && col_desc.name()) ? col_desc.name()->str() : std::string {};
			throw galp::UnsupportedFormatError(msg.str(), rowgroup_idx, col_idx, col_name);
		}

		ZeroCopyColumn col {};
		col.column_index      = col_idx;
		col.name              = (m_load_column_names && col_desc.name()) ? col_desc.name()->str() : std::string {};
		col.token             = ops->Get(0);
		col.column_descriptor = &col_desc;
		col.operand_tokens    = rpn->operand_tokens();
		col.column_view       = &(*view)[static_cast<fastlanes::n_t>(col_idx)];
		col.column_span       = backing_span;
		if (col.token == fastlanes::OperatorToken::EXP_EQUAL && col.operand_tokens && col.operand_tokens->size() >= 1) {
			col.skip_decompress = true;
			col.alias_of        = static_cast<size_t>(col.operand_tokens->Get(0));
		}
		out.columns.push_back(col);
	}

	record_timing(std::chrono::steady_clock::now());
	return out;
}

ZeroCopyRowgroup FlsReader::read_rowgroup_zero_copy_into(const size_t          rowgroup_idx,
                                                         std::shared_ptr<void> backing_owner,
                                                         std::byte* const      backing_data,
                                                         const size_t          backing_capacity,
                                                         const bool            backing_is_pinned,
                                                         ZeroCopyReadTiming*   timing) {
	read_rowgroup_bytes_into(rowgroup_idx, backing_data, backing_capacity, timing);
	return make_zero_copy_rowgroup_from_backing(
	    rowgroup_idx, std::move(backing_owner), backing_data, backing_capacity, backing_is_pinned, timing);
}

ZeroCopyRowgroup FlsReader::read_rowgroup_zero_copy(const size_t rowgroup_idx, ZeroCopyReadTiming* timing) {
	auto backing = std::make_shared<fastlanes::Buf>(rowgroup_storage_bytes(rowgroup_idx));
	return read_rowgroup_zero_copy_into(rowgroup_idx,
	                                    std::static_pointer_cast<void>(backing),
	                                    reinterpret_cast<std::byte*>(backing->mutable_data()),
	                                    backing->Capacity(),
	                                    /*backing_is_pinned=*/false,
	                                    timing);
}

Rowgroup FlsReader::materialize_zero_copy_rowgroup(ZeroCopyRowgroup zero_copy) const {
	return detail::materialize_zero_copy_rowgroup(std::move(zero_copy));
}

Rowgroup FlsReader::read_rowgroup_zero_copy_materialized(const size_t rowgroup_idx) {
	return materialize_zero_copy_rowgroup(read_rowgroup_zero_copy(rowgroup_idx));
}

Rowgroup FlsReader::read_rowgroup(const size_t rowgroup_idx) {
	return detail::make_owning_rowgroup(read_rowgroup_zero_copy_materialized(rowgroup_idx));
}

std::vector<Rowgroup> FlsReader::read_table() {
	const size_t          n_rgs = rowgroup_count();
	std::vector<Rowgroup> out;
	out.reserve(n_rgs);
	for (size_t rg_idx = 0; rg_idx < n_rgs; ++rg_idx) {
		out.emplace_back(read_rowgroup(rg_idx));
	}
	return out;
}

ZeroCopySchemaPlan FlsReader::build_shared_zero_copy_schema_plan() const {
	ZeroCopySchemaPlan plan {};
	const auto*        td = m_table_descriptor->Get();
	if (!td || !td->m_rowgroup_descriptors() || td->m_rowgroup_descriptors()->size() == 0) {
		return plan;
	}

	const auto* rowgroups      = td->m_rowgroup_descriptors();
	const auto* first_rowgroup = rowgroups->Get(static_cast<flatbuffers::uoffset_t>(0));
	if (first_rowgroup == nullptr) {
		return plan;
	}
	try {
		plan.columns     = detail::build_zero_copy_column_plan(*first_rowgroup, m_load_column_names);
		plan.build_order = detail::build_zero_copy_column_order(plan.columns);
	} catch (const std::exception&) {
		plan.columns.clear();
		plan.build_order.clear();
		return plan;
	}
	plan.enabled = true;
	for (size_t i = 1; i < rowgroups->size(); ++i) {
		const auto* rowgroup = rowgroups->Get(static_cast<flatbuffers::uoffset_t>(i));
		if (rowgroup == nullptr || !detail::rowgroup_matches_zero_copy_plan(*rowgroup, plan.columns)) {
			plan.enabled = false;
			plan.columns.clear();
			plan.build_order.clear();
			break;
		}
	}
	return plan;
}

} // namespace galp::format
