#ifndef GALP_JPEG_DCT_DECODE_HPP
#define GALP_JPEG_DCT_DECODE_HPP

#include "galp/jpeg_dct_format.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace galp::jpeg::detail {

using DecodedDctRow = std::array<int16_t, 64>;

struct DecodedComponent {
	JpegComponentMetadata      metadata;
	std::vector<DecodedDctRow> blocks;
	std::vector<size_t>        coord_to_block_index;
};

struct DecodedImage {
	JpegImageMetadata             metadata;
	std::vector<DecodedComponent> components;
};

struct ComponentSlot {
	uint32_t semantic_slot_id     = 0;
	size_t   component_index      = 0;
	int      component_id         = 0;
	uint32_t max_width_in_blocks  = 0;
	uint32_t max_height_in_blocks = 0;
};

// Internal build diagnostics used by the large-shard regression tests. A row
// slot represents one decoded 64-coefficient block or one populated table row.
struct JpegDctTableBuildStats {
	size_t expected_table_rows        = 0;
	size_t initial_decoded_rows       = 0;
	size_t released_decoded_rows      = 0;
	size_t remaining_decoded_rows     = 0;
	size_t peak_coefficient_row_slots = 0;
	size_t column_capacity_growths    = 0;
};

DecodedImage decode_jpeg_coefficients(const std::filesystem::path& path, const JpegDctReaderOptions& options);
DecodedImage decode_jpeg_layout(const std::filesystem::path& path, const JpegDctReaderOptions& options);

std::vector<DecodedImage> decode_jpeg_coefficients_parallel(const std::vector<std::filesystem::path>& paths,
                                                            const JpegDctReaderOptions&               options,
                                                            size_t                                    threads);
std::vector<DecodedImage> decode_jpeg_layouts_parallel(const std::vector<std::filesystem::path>& paths,
                                                       const JpegDctReaderOptions&               options,
                                                       size_t                                    threads);

void                       validate_supported_layout_options(const JpegDctReaderOptions& options);
void                       validate_decoded_dataset(const std::vector<DecodedImage>& images);
std::vector<ComponentSlot> normalize_component_slots(const std::vector<DecodedImage>& images);
const DecodedComponent*    find_component_for_slot(const DecodedImage& image, const ComponentSlot& slot);

JpegDctTable make_single_image_table(DecodedImage image, const JpegDctReaderOptions& options);
JpegDctTable make_dataset_table(std::vector<DecodedImage> images, const JpegDctReaderOptions& options);
JpegDctTable make_dataset_table(std::vector<DecodedImage>         images,
                                const JpegDctReaderOptions&       options,
                                const std::vector<ComponentSlot>* global_slots,
                                JpegDctPhysicalLayout             physical_layout,
                                JpegDctTableBuildStats*           build_stats = nullptr);

} // namespace galp::jpeg::detail

#endif // GALP_JPEG_DCT_DECODE_HPP
