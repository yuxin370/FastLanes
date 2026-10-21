#include "engine/materialization/selected_vector_compactor.hpp"
#include "engine/materialization/selected_vector_compactor.cuh"

namespace fastlens::runtime {

void compact_selected_vectors(fastlens::execution::Rowgroup& rowgroup,
                              const std::vector<uint32_t>& selected_vectors) {
	detail::compact_selected_vectors(rowgroup, selected_vectors);
}

} // namespace fastlens::runtime
