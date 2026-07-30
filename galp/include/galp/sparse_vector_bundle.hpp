#ifndef GALP_SPARSE_VECTOR_BUNDLE_HPP
#define GALP_SPARSE_VECTOR_BUNDLE_HPP

#include <filesystem>

namespace galp::format {

std::filesystem::path sparse_vector_bundle_path(const std::filesystem::path& fls_path);
void write_sparse_vector_bundle(const std::filesystem::path& fls_path,
                                const std::filesystem::path& bundle_path);

} // namespace galp::format

#endif // GALP_SPARSE_VECTOR_BUNDLE_HPP
