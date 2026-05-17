// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/storage/zero_copy_materializer.cu
// ────────────────────────────────────────────────────────
#include "storage/compression_column_builder.cuh"
#include "storage/zero_copy_materializer.cuh"
#include "fls/expression/rpn.hpp"
#include <flatbuffers/base.h>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <variant>

namespace galp::format::detail {

Rowgroup materialize_zero_copy_rowgroup(ZeroCopyRowgroup zero_copy) {
	auto storage           = std::make_shared<ZeroCopyHostStorage>();
	storage->backing_owner = zero_copy.backing_owner;
	storage->rowgroup_view = zero_copy.rowgroup_view;

	const auto*  schema_plan     = zero_copy.schema_plan;
	const bool   use_schema_plan = schema_plan != nullptr && schema_plan->enabled && schema_plan->columns.size() > 0;
	const size_t column_count    = use_schema_plan ? schema_plan->columns.size() : zero_copy.columns.size();

	Rowgroup out {zero_copy.n_values, zero_copy.n_vecs, zero_copy.n_tuples, {}};
	out.columns.resize(column_count);
	SmallBuildState build_state(column_count);

	const auto get_zero_copy_column = [&](const size_t col_idx) -> ZeroCopyColumn {
		if (use_schema_plan) {
			return make_zero_copy_column_from_plan(zero_copy, schema_plan->columns[col_idx]);
		}
		return zero_copy.columns[col_idx];
	};

	auto build_column = [&](auto&& self, size_t col_idx) -> Column& {
		if (col_idx >= out.columns.size()) {
			throw std::out_of_range("column index out of range");
		}
		if (build_state[col_idx] == 2) {
			return out.columns[col_idx];
		}
		if (build_state[col_idx] == 1) {
			throw std::runtime_error("cycle detected in column dependencies");
		}
		build_state[col_idx] = 1;

		const auto zcol = get_zero_copy_column(col_idx);
		Column     result {};
		result.name  = zero_copy_column_name(zcol);
		result.token = zcol.token;

		if (zcol.skip_decompress) {
			if (!zcol.alias_of.has_value()) {
				throw std::runtime_error("zero-copy alias column missing source");
			}
			const size_t src_col_idx = *zcol.alias_of;
			auto&        src_col     = self(self, src_col_idx);
			result.host = std::visit([](const auto& host_col) -> HostColumnVariant { return clone_column(host_col); },
			                         src_col.host);
			result.skip_decompress       = true;
			result.alias_of              = src_col_idx;
			result.host_owned_by_backing = false;
			result.backing_base          = nullptr;
			result.backing_bytes         = 0;
			result.backing_is_pinned     = false;
		} else {
			if (zcol.column_descriptor == nullptr) {
				std::ostringstream msg;
				msg << "zero-copy column metadata missing (col_index=" << zcol.column_index
				    << ", name=" << zero_copy_column_name(zcol) << ", token=" << fastlanes::token_to_string(zcol.token)
				    << ")";
				throw std::runtime_error(msg.str());
			}
			switch (zcol.token) {
				using enum fastlanes::OperatorToken;
			case EXP_UNCOMPRESSED_I08: {
				if (zero_copy_operand_count(zcol) < 1) {
					throw std::runtime_error("EXP_UNCOMPRESSED_I08: missing operand tokens");
				}
				const auto seg = zero_copy_segment(
				    zcol, static_cast<uint32_t>(zero_copy_operand(zcol, zero_copy_operand_count(zcol) - 1)));
				result.host                  = make_uncompressed_zero_copy<int8_t>(seg, zero_copy.n_values, *storage);
				result.host_owned_by_backing = true;
				break;
			}
			case EXP_CONSTANT_I08: {
				if (zcol.column_descriptor == nullptr || !zcol.column_descriptor->max()) {
					throw std::runtime_error("EXP_CONSTANT_I08: missing max value");
				}
				const auto* bin = zcol.column_descriptor->max()->binary_data();
				if (bin == nullptr || bin->size() != sizeof(int8_t)) {
					throw std::runtime_error("EXP_CONSTANT_I08: invalid constant size");
				}
				const auto value             = *reinterpret_cast<const int8_t*>(bin->data());
				result.host                  = galp::codec::host::CONSTANTColumn<int8_t> {zero_copy.n_values, value};
				result.host_owned_by_backing = false;
				break;
			}
			case EXP_FREQUENCY_I08: {
				if (zero_copy_operand_count(zcol) < 4) {
					throw std::runtime_error("EXP_FREQUENCY_I08: missing operand tokens");
				}
				const size_t base_idx = zero_copy_operand_count(zcol) - 1;
				const auto   seg_fv =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 3)));
				const auto seg_exc =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 2)));
				const auto seg_pos =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 1)));
				const auto seg_cnt =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 0)));
				result.host = make_frequency_zero_copy<int8_t>(
				    seg_fv, seg_exc, seg_pos, seg_cnt, zero_copy.n_values, zero_copy.n_vecs, *storage);
				result.host_owned_by_backing = true;
				break;
			}
			case EXP_FREQUENCY_I16: {
				if (zero_copy_operand_count(zcol) < 4) {
					throw std::runtime_error("EXP_FREQUENCY_I16: missing operand tokens");
				}
				const size_t base_idx = zero_copy_operand_count(zcol) - 1;
				const auto   seg_fv =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 3)));
				const auto seg_exc =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 2)));
				const auto seg_pos =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 1)));
				const auto seg_cnt =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 0)));
				result.host = make_frequency_zero_copy<int16_t>(
				    seg_fv, seg_exc, seg_pos, seg_cnt, zero_copy.n_values, zero_copy.n_vecs, *storage);
				result.host_owned_by_backing = true;
				break;
			}
			case EXP_FFOR_I08: {
				if (zero_copy_operand_count(zcol) < 3) {
					throw std::runtime_error("EXP_FFOR_I08: missing operand tokens");
				}
				const size_t base_idx = zero_copy_operand_count(zcol) - 1;
				const auto   seg_bitpacked =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 2)));
				const auto seg_bw =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 1)));
				const auto seg_base =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 0)));
				result.host = make_ffor_zero_copy<int8_t>(
				    seg_bitpacked, seg_bw, seg_base, zero_copy.n_values, zero_copy.n_vecs, *storage);
				result.host_owned_by_backing = true;
				break;
			}
			case EXP_FFOR_I16: {
				if (zero_copy_operand_count(zcol) < 3) {
					throw std::runtime_error("EXP_FFOR_I16: missing operand tokens");
				}
				const size_t base_idx = zero_copy_operand_count(zcol) - 1;
				const auto   seg_bitpacked =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 2)));
				const auto seg_bw =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 1)));
				const auto seg_base =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 0)));
				result.host = make_ffor_zero_copy<int16_t>(
				    seg_bitpacked, seg_bw, seg_base, zero_copy.n_values, zero_copy.n_vecs, *storage);
				result.host_owned_by_backing = true;
				break;
			}
			case EXP_FFOR_SLPATCH_I08: {
				if (zero_copy_operand_count(zcol) < 6) {
					throw std::runtime_error("EXP_FFOR_SLPATCH_I08: missing operand tokens");
				}
				const size_t base_idx = zero_copy_operand_count(zcol) - 1;
				const auto   seg_exc =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 5)));
				const auto seg_pos =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 4)));
				const auto seg_cnt =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 3)));
				const auto seg_bitpacked =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 2)));
				const auto seg_bw =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 1)));
				const auto seg_base =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 0)));
				result.host                  = make_slpatch_zero_copy<int8_t>(seg_exc,
                                                             seg_pos,
                                                             seg_cnt,
                                                             seg_bitpacked,
                                                             seg_bw,
                                                             seg_base,
                                                             zero_copy.n_values,
                                                             zero_copy.n_vecs,
                                                             *storage);
				result.host_owned_by_backing = true;
				break;
			}
			case EXP_FFOR_SLPATCH_I16: {
				if (zero_copy_operand_count(zcol) < 6) {
					throw std::runtime_error("EXP_FFOR_SLPATCH_I16: missing operand tokens");
				}
				const size_t base_idx = zero_copy_operand_count(zcol) - 1;
				const auto   seg_exc =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 5)));
				const auto seg_pos =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 4)));
				const auto seg_cnt =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 3)));
				const auto seg_bitpacked =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 2)));
				const auto seg_bw =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 1)));
				const auto seg_base =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 0)));
				result.host                  = make_slpatch_zero_copy<int16_t>(seg_exc,
                                                              seg_pos,
                                                              seg_cnt,
                                                              seg_bitpacked,
                                                              seg_bw,
                                                              seg_base,
                                                              zero_copy.n_values,
                                                              zero_copy.n_vecs,
                                                              *storage);
				result.host_owned_by_backing = true;
				break;
			}
			case EXP_DICT_I08_FFOR_SLPATCH_U08: {
				if (zero_copy_operand_count(zcol) < 7) {
					throw std::runtime_error("EXP_DICT_I08_FFOR_SLPATCH_U08: missing operand tokens");
				}
				const auto seg_keys      = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 0)));
				const auto seg_exc       = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 1)));
				const auto seg_pos       = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 2)));
				const auto seg_cnt       = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 3)));
				const auto seg_bitpacked = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 4)));
				const auto seg_bw        = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 5)));
				const auto seg_base      = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 6)));
				auto       index_slpatch = make_slpatch_zero_copy<uint8_t>(seg_exc,
                                                                     seg_pos,
                                                                     seg_cnt,
                                                                     seg_bitpacked,
                                                                     seg_bw,
                                                                     seg_base,
                                                                     zero_copy.n_values,
                                                                     zero_copy.n_vecs,
                                                                     *storage);
				using KeyT               = typename galp::codec::utils::same_width_uint<int8_t>::type;
				const size_t key_count   = seg_keys.data_span.size() / sizeof(KeyT);
				auto*        keys        = segment_ptr_or_copy<KeyT>(seg_keys, *storage);
				result.host              = galp::codec::host::DICTSLPATCHColumn<int8_t, uint8_t> {
                    std::move(index_slpatch), galp::codec::host::borrow_array(keys), key_count};
				result.host_owned_by_backing = true;
				break;
			}
			case EXP_DICT_I08_FFOR_U08: {
				if (zero_copy_operand_count(zcol) < 4) {
					throw std::runtime_error("EXP_DICT_I08_FFOR_U08: missing operand tokens");
				}
				const auto seg_keys      = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 0)));
				const auto seg_bitpacked = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 1)));
				const auto seg_bw        = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 2)));
				const auto seg_base      = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 3)));
				result.host              = make_dict_ffor_zero_copy<int8_t, uint8_t>(
                    seg_keys, seg_bitpacked, seg_bw, seg_base, zero_copy.n_values, zero_copy.n_vecs, *storage);
				result.host_owned_by_backing = true;
				break;
			}
			case EXP_DICT_I16_FFOR_U16: {
				if (zero_copy_operand_count(zcol) < 4) {
					throw std::runtime_error("EXP_DICT_I16_FFOR_U16: missing operand tokens");
				}
				const auto seg_keys      = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 0)));
				const auto seg_bitpacked = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 1)));
				const auto seg_bw        = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 2)));
				const auto seg_base      = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 3)));
				result.host              = make_dict_ffor_zero_copy<int16_t, uint16_t>(
                    seg_keys, seg_bitpacked, seg_bw, seg_base, zero_copy.n_values, zero_copy.n_vecs, *storage);
				result.host_owned_by_backing = true;
				break;
			}
			case EXP_DICT_I16_FFOR_U08: {
				if (zero_copy_operand_count(zcol) < 4) {
					throw std::runtime_error("EXP_DICT_I16_FFOR_U08: missing operand tokens");
				}
				const auto seg_keys      = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 0)));
				const auto seg_bitpacked = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 1)));
				const auto seg_bw        = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 2)));
				const auto seg_base      = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 3)));
				result.host              = make_dict_ffor_zero_copy<int16_t, uint8_t>(
                    seg_keys, seg_bitpacked, seg_bw, seg_base, zero_copy.n_values, zero_copy.n_vecs, *storage);
				result.host_owned_by_backing = true;
				break;
			}
			case EXP_DICT_I08_U08: {
				if (zero_copy_operand_count(zcol) < 2) {
					throw std::runtime_error("EXP_DICT_I08_U08: missing operand tokens");
				}
				const auto index_col_idx = static_cast<uint32_t>(zero_copy_operand(zcol, 0));
				const auto seg_keys      = zero_copy_segment(
                    zcol, static_cast<uint32_t>(zero_copy_operand(zcol, zero_copy_operand_count(zcol) - 1)));
				result.host =
				    make_dict_ref_zero_copy<int8_t, uint8_t>(seg_keys, index_col_idx, zero_copy.n_values, *storage);
				result.host_owned_by_backing = true;
				break;
			}
			case EXP_DICT_I16_FFOR_SLPATCH_U16: {
				if (zero_copy_operand_count(zcol) < 7) {
					throw std::runtime_error("EXP_DICT_I16_FFOR_SLPATCH_U16: missing operand tokens");
				}
				const auto seg_keys      = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 0)));
				const auto seg_exc       = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 1)));
				const auto seg_pos       = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 2)));
				const auto seg_cnt       = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 3)));
				const auto seg_bitpacked = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 4)));
				const auto seg_bw        = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 5)));
				const auto seg_base      = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 6)));
				auto       index_slpatch = make_slpatch_zero_copy<uint16_t>(seg_exc,
                                                                      seg_pos,
                                                                      seg_cnt,
                                                                      seg_bitpacked,
                                                                      seg_bw,
                                                                      seg_base,
                                                                      zero_copy.n_values,
                                                                      zero_copy.n_vecs,
                                                                      *storage);
				using KeyT               = typename galp::codec::utils::same_width_uint<int16_t>::type;
				const size_t key_count   = seg_keys.data_span.size() / sizeof(KeyT);
				auto*        keys        = segment_ptr_or_copy<KeyT>(seg_keys, *storage);
				result.host              = galp::codec::host::DICTSLPATCHColumn<int16_t, uint16_t> {
                    std::move(index_slpatch), galp::codec::host::borrow_array(keys), key_count};
				result.host_owned_by_backing = true;
				break;
			}
			case EXP_DICT_I16_FFOR_SLPATCH_U08: {
				if (zero_copy_operand_count(zcol) < 7) {
					throw std::runtime_error("EXP_DICT_I16_FFOR_SLPATCH_U08: missing operand tokens");
				}
				const auto seg_keys      = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 0)));
				const auto seg_exc       = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 1)));
				const auto seg_pos       = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 2)));
				const auto seg_cnt       = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 3)));
				const auto seg_bitpacked = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 4)));
				const auto seg_bw        = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 5)));
				const auto seg_base      = zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, 6)));
				auto       index_slpatch = make_slpatch_zero_copy<uint8_t>(seg_exc,
                                                                     seg_pos,
                                                                     seg_cnt,
                                                                     seg_bitpacked,
                                                                     seg_bw,
                                                                     seg_base,
                                                                     zero_copy.n_values,
                                                                     zero_copy.n_vecs,
                                                                     *storage);
				using KeyT               = typename galp::codec::utils::same_width_uint<int16_t>::type;
				const size_t key_count   = seg_keys.data_span.size() / sizeof(KeyT);
				auto*        keys        = segment_ptr_or_copy<KeyT>(seg_keys, *storage);
				result.host              = galp::codec::host::DICTSLPATCHColumn<int16_t, uint8_t> {
                    std::move(index_slpatch), galp::codec::host::borrow_array(keys), key_count};
				result.host_owned_by_backing = true;
				break;
			}
			case EXP_CROSS_RLE_I08: {
				if (zero_copy_operand_count(zcol) < 2) {
					throw std::runtime_error("EXP_CROSS_RLE_I08: missing operand tokens");
				}
				const size_t base_idx = zero_copy_operand_count(zcol) - 1;
				const auto   seg_vals =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 1)));
				const auto seg_lens =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 0)));
				result.host = make_cross_rle_zero_copy<int8_t>(
				    seg_vals, seg_lens, zero_copy.n_values, zero_copy.n_vecs, *storage);
				result.host_owned_by_backing = true;
				break;
			}
			case EXP_RLE_I08_U16: {
				if (zero_copy_operand_count(zcol) < 5) {
					throw std::runtime_error("EXP_RLE_I08_U16: missing operand tokens");
				}
				const size_t base_idx = zero_copy_operand_count(zcol) - 1;
				const auto   seg_vals =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 4)));
				const auto seg_rsum =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 3)));
				const auto seg_bitpacked =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 2)));
				const auto seg_bw =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 1)));
				const auto seg_base =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 0)));
				result.host                  = make_rle_zero_copy<int8_t, uint16_t>(seg_vals,
                                                                   seg_rsum,
                                                                   seg_bitpacked,
                                                                   seg_bw,
                                                                   seg_base,
                                                                   zero_copy.n_values,
                                                                   zero_copy.n_vecs,
                                                                   *storage);
				result.host_owned_by_backing = true;
				break;
			}
			case EXP_RLE_I16_U16: {
				if (zero_copy_operand_count(zcol) < 5) {
					throw std::runtime_error("EXP_RLE_I16_U16: missing operand tokens");
				}
				const size_t base_idx = zero_copy_operand_count(zcol) - 1;
				const auto   seg_vals =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 4)));
				const auto seg_rsum =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 3)));
				const auto seg_bitpacked =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 2)));
				const auto seg_bw =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 1)));
				const auto seg_base =
				    zero_copy_segment(zcol, static_cast<uint32_t>(zero_copy_operand(zcol, base_idx - 0)));
				result.host                  = make_rle_zero_copy<int16_t, uint16_t>(seg_vals,
                                                                    seg_rsum,
                                                                    seg_bitpacked,
                                                                    seg_bw,
                                                                    seg_base,
                                                                    zero_copy.n_values,
                                                                    zero_copy.n_vecs,
                                                                    *storage);
				result.host_owned_by_backing = true;
				break;
			}
			default:
				throw_unsupported_zero_copy_token(
				    zcol.token, zero_copy.rowgroup_index, zcol.column_index, zero_copy_column_name(zcol));
			}

			if (result.host_owned_by_backing && zero_copy.backing_is_pinned) {
				if (zcol.column_view != nullptr && !zcol.column_view->column_span.empty()) {
					result.backing_base  = zcol.column_view->column_span.data();
					result.backing_bytes = zcol.column_view->column_span.size();
				} else {
					result.backing_base  = zero_copy.backing_span.data();
					result.backing_bytes = zero_copy.backing_span.size();
				}
				result.backing_is_pinned = true;
			}
		}

		out.columns[col_idx] = std::move(result);
		build_state[col_idx] = 2;
		return out.columns[col_idx];
	};

	if (use_schema_plan) {
		for (const size_t i : schema_plan->build_order) {
			(void)build_column(build_column, i);
		}
	} else {
		for (size_t i = 0; i < zero_copy.columns.size(); ++i) {
			(void)build_column(build_column, i);
		}
	}
	out.backing_storage = std::move(storage);
	return out;
}

void release_transient_materialized_rowgroup(Rowgroup& rowgroup) {
	for (auto& col : rowgroup.columns) {
		if (col.alias_of.has_value()) {
			continue;
		}
		if (rowgroup.backing_storage && col.host_owned_by_backing) {
			continue;
		}
		std::visit([](auto& host_col) { galp::codec::host::free_column(host_col); }, col.host);
	}
	rowgroup.columns.clear();
	rowgroup.backing_storage.reset();
}

Rowgroup make_owning_rowgroup(Rowgroup rowgroup) {
	Rowgroup out {rowgroup.n_values, rowgroup.n_vecs, rowgroup.n_tuples, {}};
	out.columns.resize(rowgroup.columns.size());

	SmallBuildState build_state(rowgroup.columns.size());
	auto            clone_column_to_output = [&](auto&& self, const size_t col_idx) -> Column& {
        if (col_idx >= rowgroup.columns.size()) {
            throw std::out_of_range("column index out of range");
        }
        if (build_state[col_idx] == 2) {
            return out.columns[col_idx];
        }
        if (build_state[col_idx] == 1) {
            throw std::runtime_error("cycle detected in column dependencies");
        }
        build_state[col_idx] = 1;

        const auto& src = rowgroup.columns[col_idx];
        auto&       dst = out.columns[col_idx];
        dst.name        = src.name;
        dst.token       = src.token;

        if (src.alias_of.has_value()) {
            const size_t src_col_idx = *src.alias_of;
            auto&        src_col     = self(self, src_col_idx);
            dst.host = std::visit([](const auto& host_col) -> HostColumnVariant { return clone_column(host_col); },
                                  src_col.host);
            dst.skip_decompress = true;
            dst.alias_of        = src_col_idx;
        } else {
            dst.host =
                std::visit([](const auto& host_col) -> HostColumnVariant { return clone_column(host_col); }, src.host);
        }

        dst.host_owned_by_backing = false;
        dst.backing_base          = nullptr;
        dst.backing_bytes         = 0;
        dst.backing_is_pinned     = false;
        build_state[col_idx]      = 2;
        return dst;
	};

	for (size_t i = 0; i < rowgroup.columns.size(); ++i) {
		(void)clone_column_to_output(clone_column_to_output, i);
	}

	release_transient_materialized_rowgroup(rowgroup);
	return out;
}

} // namespace galp::format::detail
