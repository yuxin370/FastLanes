#ifndef GALP_JPEG_DCT_METADATA_HPP
#define GALP_JPEG_DCT_METADATA_HPP

#include "galp/jpeg_dct_format.hpp"
#include "galp/jpeg_dct_storage.hpp"
#include <filesystem>

namespace galp::jpeg::detail {

JpegDctDatasetMetadata read_jpeg_dct_metadata_file(const std::filesystem::path& path);
JpegDctShardManifest   read_jpeg_dct_shard_manifest_file(const std::filesystem::path& path);

} // namespace galp::jpeg::detail

#endif // GALP_JPEG_DCT_METADATA_HPP
