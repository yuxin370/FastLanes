#ifndef GALP_ENGINE_MATERIALIZATION_SELECTED_VECTOR_COMPACTOR_HPP
#define GALP_ENGINE_MATERIALIZATION_SELECTED_VECTOR_COMPACTOR_HPP

#include "core/data/model.cuh"
#include <cstdint>
#include <vector>

namespace galp::runtime {

void compact_selected_vectors(galp::execution::Rowgroup& rowgroup,
                              const std::vector<uint32_t>& selected_vectors);

} // namespace galp::runtime

#endif
