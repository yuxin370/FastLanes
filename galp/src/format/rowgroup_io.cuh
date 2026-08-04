// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/format/rowgroup_io.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_ENGINE_FORMAT_ROWGROUP_IO_CUH
#define GALP_ENGINE_FORMAT_ROWGROUP_IO_CUH

#include "core/data/model.cuh"
#include "format/schema_plan.cuh"
#include "fls/footer/rowgroup_descriptor_generated.h"
#include "fls/footer/table_descriptor.hpp"
#include "fls/reader/column_view.hpp"
#include "fls/reader/rowgroup_view.hpp"
#include "fls/reader/segment.hpp"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace galp::format {

class CompactDescriptorV3;
struct CompactV3DirectColumn;
struct CompactV3DirectRowgroup;

using HostColumnVariant = galp::execution::EncodedPayload;
using Column            = galp::execution::Column;
using Rowgroup          = galp::execution::Rowgroup;

struct ZeroCopyColumn {
	size_t                               column_index = 0;
	std::string                          name;
	const std::string*                   name_ref          = nullptr;
	fastlanes::OperatorToken             token             = fastlanes::OperatorToken::INVALID;
	const fastlanes::ColumnDescriptor*   column_descriptor = nullptr;
	const flatbuffers::Vector<uint64_t>* operand_tokens    = nullptr;
	const std::vector<uint32_t>*         operand_ids       = nullptr;
	const fastlanes::ColumnView*         column_view       = nullptr;
	const CompactV3DirectRowgroup*       compact_rowgroup  = nullptr;
	const CompactV3DirectColumn*         compact_column    = nullptr;
	fastlanes::span<std::byte>           column_span;
	bool                                 skip_decompress = false;
	std::optional<size_t>                alias_of;
};

struct ZeroCopyRowgroup {
	size_t                                                  rowgroup_index = 0;
	size_t                                                  n_values       = 0;
	size_t                                                  n_vecs         = 0;
	size_t                                                  n_tuples       = 0;
	std::shared_ptr<const fastlanes::TableDescriptorHandle> table_descriptor_owner;
	// Compact v3 reconstructs one bounded RowgroupDescriptor FlatBuffer on
	// demand and owns those bytes without ever materializing a TableDescriptor.
	std::shared_ptr<const std::vector<uint8_t>>             rowgroup_descriptor_owner;
	std::shared_ptr<const CompactDescriptorV3>              compact_descriptor_owner;
	std::shared_ptr<const CompactV3DirectRowgroup>           compact_direct_owner;
	std::shared_ptr<const ZeroCopySchemaPlan>               schema_plan_owner;
	const fastlanes::RowgroupDescriptor*                    rowgroup_descriptor = nullptr;
	const ZeroCopySchemaPlan*                                           schema_plan         = nullptr;
	std::shared_ptr<void>                                               backing_owner;
	fastlanes::span<std::byte>                                          backing_span;
	// A batched compact-v3 read can place multiple aligned rowgroup spans in
	// one pinned allocation. Register the complete allocation with DeviceArena
	// so every encoded pointer resolves at its aligned relative offset while
	// the arena still issues one aggregate H2D copy (including alignment gaps).
	fastlanes::span<std::byte>                                          transfer_backing_span;
	bool                                                                backing_is_pinned = false;
	std::shared_ptr<fastlanes::RowgroupView>                            rowgroup_view;
	std::vector<ZeroCopyColumn>                                         columns;
	// Empty means that the complete rowgroup backing is present. Compact-v3
	// coefficient-range reads set this to the logical columns that may be
	// materialized; dependency columns are discovered recursively.
	std::vector<uint8_t>                                                materialized_column_indices;
	std::shared_ptr<const galp::execution::PackedRowgroupDevicePayload> packed_device_payload;
};

struct ZeroCopyReadTiming {
	size_t                                storage_bytes                    = 0;
	size_t                                logical_storage_bytes            = 0;
	size_t                                full_storage_bytes               = 0;
	size_t                                physical_page_bytes              = 0;
	size_t                                full_physical_page_bytes         = 0;
	size_t                                pread_count                      = 0;
	size_t                                preadv_count                     = 0;
	size_t                                coalesced_read_run_count         = 0;
	size_t                                selected_coefficient_count       = 0;
	size_t                                full_coefficient_count           = 0;
	bool                                  sparse_read_supported            = false;
	bool                                  used_sparse_read        = false;
	bool                                  used_coefficient_range_read = false;
	bool                                  used_pinned_backing     = false;
	bool                                  used_vector_bundle_read = false;
	bool                                  used_vector_bundle_envelope_read = false;
	std::string                           sparse_fallback_reason;
	double                                pread_ms                = 0.0;
	double                                zero_copy_view_setup_ms = 0.0;
	std::chrono::steady_clock::time_point pread_start {};
	std::chrono::steady_clock::time_point pread_end {};
};

size_t             zero_copy_operand_count(const ZeroCopyColumn& col);
uint32_t           zero_copy_operand(const ZeroCopyColumn& col, size_t idx);
const std::string& zero_copy_column_name(const ZeroCopyColumn& col);

[[noreturn]] void throw_unsupported_zero_copy_token(fastlanes::OperatorToken token,
                                                    size_t                   rowgroup_index,
                                                    size_t                   column_index,
                                                    const std::string&       column_name);

fastlanes::SegmentView zero_copy_segment(const ZeroCopyColumn& col, uint32_t segment_idx);

const uint8_t* zero_copy_maximum_data(const ZeroCopyColumn& col);
size_t         zero_copy_maximum_size(const ZeroCopyColumn& col);

ZeroCopyColumn make_zero_copy_column_from_plan(const ZeroCopyRowgroup& rowgroup, const ZeroCopyColumnPlan& plan_col);

} // namespace galp::format

#endif // GALP_ENGINE_FORMAT_ROWGROUP_IO_CUH
