#ifndef FASTLENS_JPEG_DCT_METADATA_HPP
#define FASTLENS_JPEG_DCT_METADATA_HPP

#include "fastlens/jpeg_dct_format.hpp"
#include "fastlens/jpeg_dct_storage.hpp"
#include <filesystem>

namespace fastlens::jpeg::detail {

JpegDctDatasetMetadata read_jpeg_dct_metadata_file(const std::filesystem::path& path);
JpegDctShardManifest   read_jpeg_dct_shard_manifest_file(const std::filesystem::path& path);

} // namespace fastlens::jpeg::detail

#endif // FASTLENS_JPEG_DCT_METADATA_HPP
