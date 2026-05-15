// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/runtime/materialize/types.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_RUNTIME_MATERIALIZE_TYPES_CUH
#define ENGINE_RUNTIME_MATERIALIZE_TYPES_CUH

#include "engine/data/model.cuh"
#include "engine/runtime/workset/model.cuh"
#include <cstddef>
#include <memory>
#include <vector>

namespace galp::runtime {

using galp::execution::ExecutionConfig;
using galp::execution::MaterializedColumn;
using galp::execution::RowgroupData;
using galp::execution::ValueStore;

struct PendingMaterialize {
	struct Entry {
		size_t                 global_expr_index = 0;
		size_t                 n_values          = 0;
		size_t                 output_offset     = 0;
		size_t                 elem_size         = 0;
		galp::format::DataType value_type        = galp::format::DataType::I8;
	};

	std::shared_ptr<void>   pinned_owner;
	size_t                  total_bytes = 0;
	cudaStream_t            d2h_stream  = nullptr;
	galp::memory::CudaEvent d2h_event;
	std::vector<Entry>      entries;
	bool                    active = false;
};

} // namespace galp::runtime

#endif // ENGINE_RUNTIME_MATERIALIZE_TYPES_CUH
