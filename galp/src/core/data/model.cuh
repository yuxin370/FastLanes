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
                                    galp::codec::host::FREQColumn<int8_t>,
                                    galp::codec::host::SLPATCHColumn<int8_t>,
                                    galp::codec::host::CROSSRLEColumn<int8_t>,
                                    galp::codec::host::RLEColumn<int8_t, uint16_t>,
                                    galp::codec::host::RLESLPATCHColumn<int8_t, uint16_t>,
                                    galp::codec::host::BPColumn<int16_t>,
                                    galp::codec::host::FFORColumn<int16_t>,
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

struct Rowgroup {
	size_t              n_values = 0;
	size_t              n_vecs   = 0;
	size_t              n_tuples = 0;
	std::vector<Column> columns;
	std::shared_ptr<void> backing_storage;
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
