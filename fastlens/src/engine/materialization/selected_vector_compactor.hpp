#ifndef FASTLENS_ENGINE_MATERIALIZATION_SELECTED_VECTOR_COMPACTOR_HPP
#define FASTLENS_ENGINE_MATERIALIZATION_SELECTED_VECTOR_COMPACTOR_HPP

#include "core/data/model.cuh"
#include <cstdint>
#include <vector>

namespace fastlens::runtime {

void compact_selected_vectors(fastlens::execution::Rowgroup& rowgroup,
                              const std::vector<uint32_t>& selected_vectors);

} // namespace fastlens::runtime

#endif
