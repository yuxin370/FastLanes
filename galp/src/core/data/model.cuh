// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/data/model.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_DATA_MODEL_CUH
#define ENGINE_DATA_MODEL_CUH

#include "core/data/value_store.cuh"
#include "core/types.cuh"
#include "fls/footer/operator_token_generated.h"
#include "codecs/encodings/all.cuh"
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace galp::execution {

// Runtime data model for decoded outputs and streaming decode granularity.
struct DecodeChunk {
	unsigned unpack_n_vectors = 1;
	unsigned unpack_n_values  = 1;

	constexpr unsigned values_per_step() const {
		return unpack_n_vectors * unpack_n_values;
	}
	constexpr bool supports_single_value_streaming() const {
		return values_per_step() == 1;
	}
};

struct ColumnMeta {
	size_t          column_index    = std::numeric_limits<size_t>::max();
	std::string     column_name     = {};
	size_t          value_count     = 0;
	galp::format::DataType value_type      = galp::format::DataType::I8;
	unsigned        values_per_step = 1;
};

using EncodedPayload = std::variant<galp::codec::host::BPColumn<int8_t>,
                                    galp::codec::host::FFORColumn<int8_t>,
                                    galp::codec::host::DICTREFColumn<int8_t, uint8_t>,
                                    galp::codec::host::DICTFFORColumn<int8_t>,
                                    galp::codec::host::DICTSLPATCHColumn<int8_t>,
                                    galp::codec::host::CONSTANTColumn<int8_t>,
	                                galp::codec::host::DELTAColumn<int8_t>,
                                    galp::codec::host::FREQColumn<int8_t>,
                                    galp::codec::host::SLPATCHColumn<int8_t>,
                                    galp::codec::host::CROSSRLEColumn<int8_t>,
                                    galp::codec::host::RLEColumn<int8_t, uint16_t>,
                                    galp::codec::host::RLESLPATCHColumn<int8_t, uint16_t>,
                                    galp::codec::host::BPColumn<int16_t>,
                                    galp::codec::host::FFORColumn<int16_t>,
                                    galp::codec::host::DICTREFColumn<int16_t, uint8_t>,
                                    galp::codec::host::DICTREFColumn<int16_t, uint16_t>,
                                    galp::codec::host::CONSTANTColumn<int16_t>,
	                                galp::codec::host::DELTAColumn<int16_t>,
                                    galp::codec::host::DICTFFORColumn<int16_t>,
                                    galp::codec::host::DICTFFORColumn<int16_t, uint8_t>,
                                    galp::codec::host::DICTSLPATCHColumn<int16_t>,
                                    galp::codec::host::DICTSLPATCHColumn<int16_t, uint8_t>,
                                    galp::codec::host::SLPATCHColumn<int16_t>,
                                    galp::codec::host::FREQColumn<int16_t>,
                                    galp::codec::host::CROSSRLEColumn<int16_t>,
                                    galp::codec::host::RLEColumn<int16_t, uint16_t>,
                                    galp::codec::host::RLESLPATCHColumn<int16_t, uint16_t>>;

struct Column {
	std::string              name;
	fastlanes::OperatorToken token;
	EncodedPayload           host;
	bool                     skip_decompress = false;
	std::optional<size_t>    alias_of;
	bool                     host_owned_by_backing = false;
	const std::byte*         backing_base          = nullptr;
	size_t                   backing_bytes         = 0;
	// Only set when backing_base points into CUDA-pinned memory. DeviceArena
	// may issue direct cudaMemcpyAsync from the backing region only when true;
	// otherwise the pageable copy silently falls back to internal staging.
	bool                     backing_is_pinned = false;
};

struct MaterializedColumn {
	ValueStore values;
	ColumnMeta meta;
};

struct PackedRowgroupScatterRange {
	size_t packed_offset  = 0;
	size_t logical_offset = 0;
	size_t size           = 0;
};

// A sparse vector-bundle read keeps its transfer payload in the same compact
// order as on disk. The GPU upload path scatters these ranges into an
// allocation-only logical rowgroup before launching the existing decoders.
struct PackedRowgroupDevicePayload {
	std::shared_ptr<void>                    packed_owner;
	const std::byte*                         packed_data   = nullptr;
	size_t                                   packed_bytes  = 0;
	size_t                                   packed_capacity_bytes = 0;
	const std::byte*                         logical_data  = nullptr;
	size_t                                   logical_bytes = 0;
	std::vector<PackedRowgroupScatterRange> ranges;
};

struct Rowgroup {
	size_t              n_values = 0;
	size_t              n_vecs   = 0;
	size_t              n_tuples = 0;
	std::vector<Column> columns;
	std::shared_ptr<void> backing_storage;
	size_t                backing_storage_capacity_bytes = 0;
	std::shared_ptr<void> transient_memory_accounting;
	// Empty means every column is materialized. A non-empty list identifies
	// the logical columns materialized by a coefficient-range read; the
	// columns vector retains its original indexing for aliases/dictionaries.
	std::vector<uint8_t> materialized_column_indices;
	std::shared_ptr<const PackedRowgroupDevicePayload> packed_device_payload;
};

struct Table {
	std::vector<Rowgroup> rowgroups;
};

struct RowgroupData {
	std::vector<std::optional<MaterializedColumn>> columns;
};

struct TableData {
	size_t              rowgroups     = 0;
	size_t              total_columns = 0;
	std::vector<size_t> column_counts;
};

} // namespace galp::execution

#endif // ENGINE_DATA_MODEL_CUH
