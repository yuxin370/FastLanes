#include "engine/materialization/selected_vector_compactor.hpp"
#include "engine/materialization/selected_vector_compactor.cuh"

namespace galp::runtime {

void compact_selected_vectors(galp::execution::Rowgroup& rowgroup,
                              const std::vector<uint32_t>& selected_vectors) {
	detail::compact_selected_vectors(rowgroup, selected_vectors);
}

} // namespace galp::runtime
