// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// fastlens/src/engine/materialization/types.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_RUNTIME_MATERIALIZE_TYPES_CUH
#define ENGINE_RUNTIME_MATERIALIZE_TYPES_CUH

#include "core/data/model.cuh"
#include "engine/workset/model.cuh"
#include <cstddef>
#include <memory>
#include <vector>

namespace fastlens::runtime {

using fastlens::execution::ExecutionConfig;
using fastlens::execution::MaterializedColumn;
using fastlens::execution::RowgroupData;
using fastlens::execution::ValueStore;

struct PendingMaterialize {
	struct Entry {
		size_t                 global_expr_index = 0;
		size_t                 n_values          = 0;
		size_t                 output_offset     = 0;
		size_t                 elem_size         = 0;
		fastlens::format::DataType value_type        = fastlens::format::DataType::I8;
	};

	std::shared_ptr<void>   pinned_owner;
	size_t                  total_bytes = 0;
	cudaStream_t            d2h_stream  = nullptr;
	fastlens::memory::CudaEvent d2h_event;
	std::vector<Entry>      entries;
	bool                    active = false;
};

} // namespace fastlens::runtime

#endif // ENGINE_RUNTIME_MATERIALIZE_TYPES_CUH
