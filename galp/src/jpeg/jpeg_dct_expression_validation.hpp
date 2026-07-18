// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
#ifndef GALP_JPEG_DCT_EXPRESSION_VALIDATION_HPP
#define GALP_JPEG_DCT_EXPRESSION_VALIDATION_HPP

#include <cstdint>
#include <filesystem>

namespace galp::jpeg::detail {

// Validate the actual root expression stored for every coefficient column in
// an inline-footer JPEG-DCT shard. The implementation consumes the same
// operator capability table as the zero-copy materializer and dispatch path.
// It throws before a staged shard is committed when any expression cannot be
// decoded by the GPU runtime.
void validate_jpeg_dct_fls_gpu_expressions(const std::filesystem::path& fls_path, uint32_t shard_id);

} // namespace galp::jpeg::detail

#endif // GALP_JPEG_DCT_EXPRESSION_VALIDATION_HPP
