#ifndef FASTLENS_SPARSE_VECTOR_BUNDLE_HPP
#define FASTLENS_SPARSE_VECTOR_BUNDLE_HPP

#include <filesystem>

namespace fastlens::format {

std::filesystem::path sparse_vector_bundle_path(const std::filesystem::path& fls_path);
void write_sparse_vector_bundle(const std::filesystem::path& fls_path,
                                const std::filesystem::path& bundle_path);

} // namespace fastlens::format

#endif // FASTLENS_SPARSE_VECTOR_BUNDLE_HPP
