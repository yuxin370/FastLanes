// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/data/model.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_DATA_MODEL_CUH
#define ENGINE_DATA_MODEL_CUH

#include "fls/footer/operator_token_generated.h"
#include "flsgpu/columns/all.cuh"
#include "engine/data/value-store.cuh"
#include "engine/types.cuh"
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace dispatch {

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
	types::DataType value_type      = types::DataType::I8;
	unsigned        values_per_step = 1;
};

using EncodedPayload = std::variant<flsgpu::host::BPColumn<int8_t>,
                                    flsgpu::host::FFORColumn<int8_t>,
                                    flsgpu::host::DICTFFORColumn<int8_t>,
                                    flsgpu::host::DICTSLPATCHColumn<int8_t>,
                                    flsgpu::host::CONSTANTColumn<int8_t>,
                                    flsgpu::host::FREQColumn<int8_t>,
                                    flsgpu::host::SLPATCHColumn<int8_t>,
                                    flsgpu::host::CROSSRLEColumn<int8_t>,
                                    flsgpu::host::RLEColumn<int8_t, uint16_t>,
                                    flsgpu::host::BPColumn<int16_t>,
                                    flsgpu::host::FFORColumn<int16_t>,
                                    flsgpu::host::DICTFFORColumn<int16_t>,
                                    flsgpu::host::DICTFFORColumn<int16_t, uint8_t>,
                                    flsgpu::host::DICTSLPATCHColumn<int16_t>,
                                    flsgpu::host::DICTSLPATCHColumn<int16_t, uint8_t>,
                                    flsgpu::host::SLPATCHColumn<int16_t>,
                                    flsgpu::host::FREQColumn<int16_t>,
                                    flsgpu::host::RLEColumn<int16_t, uint16_t>>;

struct Column {
	std::string              name;
	fastlanes::OperatorToken token;
	EncodedPayload           host;
	bool                     skip_decompress = false;
	std::optional<size_t>    alias_of;
};

struct MaterializedColumn {
	ValueStore values;
	ColumnMeta meta;
};

struct Rowgroup {
	size_t              n_values = 0;
	size_t              n_vecs   = 0;
	std::vector<Column> columns;
};

struct Table {
	std::vector<Rowgroup> rowgroups;
};

struct RowgroupData {
	std::vector<std::optional<MaterializedColumn>> columns;
};

struct TableData {
	size_t rowgroups     = 0;
	size_t total_columns = 0;
};

} // namespace dispatch

#endif // ENGINE_DATA_MODEL_CUH
