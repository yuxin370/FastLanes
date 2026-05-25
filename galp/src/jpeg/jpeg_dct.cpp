#include "galp/jpeg_dct.hpp"
#include "fls/connection.hpp"
#include "fls/reader/rowgroup_reader.hpp"
#include "fls/table/memory_table.hpp"
#include "fls/table/rowgroup.hpp"
#include "jpeg/jpeg_dct_order.hpp"
#include <algorithm>
#include <array>
#include <csetjmp>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <jpeglib.h>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace galp::jpeg {

namespace {

using DctRow = std::array<int16_t, 64>;

struct DecodedComponent {
	JpegComponentMetadata metadata;
	std::vector<DctRow>   blocks;
	std::vector<size_t>   coord_to_block_index;
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

struct DatasetLayout {
	std::vector<ComponentSlot>                        slots;
	std::vector<std::vector<const DecodedComponent*>> image_slot_components;
};

struct FileCloser {
	void operator()(FILE* file) const {
		if (file != nullptr) {
			std::fclose(file);
		}
	}
};

using FilePtr = std::unique_ptr<FILE, FileCloser>;

struct JpegErrorManager {
	jpeg_error_mgr pub;
	jmp_buf        setjmp_buffer;
	char           message[JMSG_LENGTH_MAX] {};
	uint32_t       warning_count = 0;
};

extern "C" void jpeg_error_exit_bridge(j_common_ptr cinfo) {
	auto* err = reinterpret_cast<JpegErrorManager*>(cinfo->err);
	(*cinfo->err->format_message)(cinfo, err->message);
	longjmp(err->setjmp_buffer, 1);
}

extern "C" void jpeg_emit_message_bridge(j_common_ptr cinfo, int msg_level) {
	auto* err = reinterpret_cast<JpegErrorManager*>(cinfo->err);
	if (msg_level < 0) {
		++err->warning_count;
	}
}

FilePtr open_jpeg_file(const std::filesystem::path& path);

struct JpegDecoder {
	explicit JpegDecoder(std::filesystem::path input_path)
	    : path(std::move(input_path))
	    , file(open_jpeg_file(path)) {
		cinfo.err             = jpeg_std_error(&jerr.pub);
		jerr.pub.error_exit   = jpeg_error_exit_bridge;
		jerr.pub.emit_message = jpeg_emit_message_bridge;
		checked_void([&] { jpeg_create_decompress(&cinfo); });
		created = true;
	}

	JpegDecoder(const JpegDecoder&)            = delete;
	JpegDecoder& operator=(const JpegDecoder&) = delete;

	~JpegDecoder() {
		if (created) {
			jpeg_destroy_decompress(&cinfo);
		}
	}

	void set_stdio_src() {
		checked_void([&] { jpeg_stdio_src(&cinfo, file.get()); });
	}

	void save_metadata_markers() {
		checked_void([&] {
			jpeg_save_markers(&cinfo, JPEG_COM, 0xffff);
			for (int marker = 0; marker < 16; ++marker) {
				jpeg_save_markers(&cinfo, JPEG_APP0 + marker, 0xffff);
			}
		});
	}

	void read_header() {
		checked_void([&] { jpeg_read_header(&cinfo, TRUE); });
	}

	jvirt_barray_ptr* read_coefficients() {
		return checked_value([&] { return jpeg_read_coefficients(&cinfo); });
	}

	JBLOCKARRAY access_virt_barray(const jvirt_barray_ptr coefficient_array, const JDIMENSION start_row) {
		return checked_value([&] {
			return (*cinfo.mem->access_virt_barray)(
			    reinterpret_cast<j_common_ptr>(&cinfo), coefficient_array, start_row, 1, FALSE);
		});
	}

	void finish_decompress() {
		checked_void([&] { jpeg_finish_decompress(&cinfo); });
	}

	uint32_t warning_count() const {
		return jerr.warning_count;
	}

	jpeg_decompress_struct cinfo {};

private:
	[[noreturn]] void throw_error() const {
		throw std::runtime_error("libjpeg failed while reading '" + path.string() + "': " + jerr.message);
	}

	template <typename Fn>
	void checked_void(Fn&& fn) {
		if (setjmp(jerr.setjmp_buffer) != 0) {
			throw_error();
		}
		fn();
	}

	template <typename Fn>
	auto checked_value(Fn&& fn) -> decltype(fn()) {
		if (setjmp(jerr.setjmp_buffer) != 0) {
			throw_error();
		}
		return fn();
	}

	std::filesystem::path path;
	FilePtr               file;
	JpegErrorManager      jerr {};
	bool                  created = false;
};

FilePtr open_jpeg_file(const std::filesystem::path& path) {
	FILE* file = std::fopen(path.string().c_str(), "rb");
	if (file == nullptr) {
		throw std::runtime_error("failed to open JPEG file: " + path.string());
	}
	return FilePtr(file);
}

std::vector<size_t> selected_component_indices(const int num_components, const JpegDctReaderOptions& options) {
	if (options.component_mode == JpegComponentMode::kAllComponents) {
		std::vector<size_t> indices(static_cast<size_t>(num_components));
		for (size_t i = 0; i < indices.size(); ++i) {
			indices[i] = i;
		}
		return indices;
	}

	if (options.selected_component_index < 0 || options.selected_component_index >= num_components) {
		std::ostringstream msg;
		msg << "selected JPEG component index " << options.selected_component_index << " is outside [0, "
		    << num_components << ")";
		throw std::runtime_error(msg.str());
	}
	return {static_cast<size_t>(options.selected_component_index)};
}

DctRow block_to_row(const JCOEF* block, const bool use_zigzag_columns) {
	DctRow row {};
	for (size_t col = 0; col < row.size(); ++col) {
		const auto natural_idx = use_zigzag_columns ? detail::kZigzagColumnToNaturalIndex[col] : col;
		const auto value       = block[natural_idx];
		if (value < std::numeric_limits<int16_t>::min() || value > std::numeric_limits<int16_t>::max()) {
			throw std::runtime_error("JPEG DCT coefficient is outside int16_t range");
		}
		row[col] = static_cast<int16_t>(value);
	}
	return row;
}

std::vector<size_t> make_coord_to_index(const uint32_t                               width_in_blocks,
                                        const uint32_t                               height_in_blocks,
                                        const std::vector<detail::MortonBlockCoord>& order) {
	std::vector<size_t> coord_to_index(static_cast<size_t>(width_in_blocks) * height_in_blocks);
	for (size_t block_idx = 0; block_idx < order.size(); ++block_idx) {
		const auto& coord                                                        = order[block_idx];
		coord_to_index[static_cast<size_t>(coord.y) * width_in_blocks + coord.x] = block_idx;
	}
	return coord_to_index;
}

const JpegQuantTableMetadata* find_quant_table(const JpegImageMetadata& image, const int quant_tbl_no) {
	if (quant_tbl_no < 0) {
		return nullptr;
	}
	for (const auto& table : image.quant_tables) {
		if (table.table_id == static_cast<uint8_t>(quant_tbl_no)) {
			return &table;
		}
	}
	return nullptr;
}

uint64_t quant_table_fingerprint(const JpegImageMetadata& image, const int quant_tbl_no) {
	const auto* table = find_quant_table(image, quant_tbl_no);
	if (table == nullptr) {
		return 0;
	}
	uint64_t hash = 1469598103934665603ULL;
	for (const auto value : table->values) {
		hash ^= static_cast<uint64_t>(value & 0xffU);
		hash *= 1099511628211ULL;
		hash ^= static_cast<uint64_t>((value >> 8U) & 0xffU);
		hash *= 1099511628211ULL;
	}
	return hash;
}

DecodedImage decode_jpeg_coefficients(const std::filesystem::path& path, const JpegDctReaderOptions& options) {
	static_assert(sizeof(JCOEF) <= sizeof(int16_t), "JCOEF wider than int16_t requires explicit conversion policy");

	JpegDecoder decoder(path);
	decoder.set_stdio_src();
	if (options.capture_metadata_markers) {
		decoder.save_metadata_markers();
	}
	decoder.read_header();
	auto& cinfo = decoder.cinfo;
	// libjpeg-turbo exposes the libjpeg-compatible coefficient API. This path reads the compressed-domain,
	// quantized DCT coefficient arrays directly; it does not hand-roll JPEG entropy decode, IDCT, or RGB decode.
	auto** coefficient_arrays = decoder.read_coefficients();

	DecodedImage image;
	image.metadata.source_path      = path;
	image.metadata.image_width      = cinfo.image_width;
	image.metadata.image_height     = cinfo.image_height;
	image.metadata.data_precision   = static_cast<uint8_t>(cinfo.data_precision);
	image.metadata.jpeg_color_space = static_cast<int>(cinfo.jpeg_color_space);
	image.metadata.progressive      = jpeg_has_multiple_scans(&cinfo) != 0;
	for (int table_id = 0; table_id < NUM_QUANT_TBLS; ++table_id) {
		const auto* quant_table = cinfo.quant_tbl_ptrs[table_id];
		if (quant_table == nullptr) {
			continue;
		}
		JpegQuantTableMetadata metadata;
		metadata.table_id = static_cast<uint8_t>(table_id);
		for (size_t i = 0; i < metadata.values.size(); ++i) {
			metadata.values[i] = quant_table->quantval[i];
		}
		image.metadata.quant_tables.push_back(metadata);
	}
	for (jpeg_saved_marker_ptr marker = cinfo.marker_list; marker != nullptr; marker = marker->next) {
		JpegMarkerMetadata metadata;
		metadata.marker = static_cast<uint8_t>(marker->marker);
		metadata.payload.assign(marker->data, marker->data + marker->data_length);
		image.metadata.markers.push_back(std::move(metadata));
	}

	const auto indices = selected_component_indices(cinfo.num_components, options);
	image.components.reserve(indices.size());
	image.metadata.components.reserve(indices.size());

	for (const auto component_index : indices) {
		const auto& comp = cinfo.comp_info[component_index];

		DecodedComponent component;
		component.metadata.semantic_slot_id        = static_cast<uint32_t>(component_index);
		component.metadata.component_index         = component_index;
		component.metadata.local_component_index   = component_index;
		component.metadata.component_id            = comp.component_id;
		component.metadata.width_in_blocks         = comp.width_in_blocks;
		component.metadata.height_in_blocks        = comp.height_in_blocks;
		component.metadata.padded_width_in_blocks  = comp.width_in_blocks;
		component.metadata.padded_height_in_blocks = comp.height_in_blocks;
		component.metadata.h_samp_factor           = comp.h_samp_factor;
		component.metadata.v_samp_factor           = comp.v_samp_factor;
		component.metadata.quant_tbl_no            = comp.quant_tbl_no;
		component.metadata.quant_table_fingerprint = quant_table_fingerprint(image.metadata, comp.quant_tbl_no);

		const auto order =
		    detail::make_block_order(comp.width_in_blocks, comp.height_in_blocks, options.use_z_curve_block_order);
		component.blocks.resize(order.size());
		component.coord_to_block_index = make_coord_to_index(comp.width_in_blocks, comp.height_in_blocks, order);
		for (JDIMENSION y = 0; y < comp.height_in_blocks; ++y) {
			JBLOCKARRAY block_rows = decoder.access_virt_barray(coefficient_arrays[component_index], y);
			for (JDIMENSION x = 0; x < comp.width_in_blocks; ++x) {
				const auto block_index =
				    component.coord_to_block_index[static_cast<size_t>(y) * comp.width_in_blocks + x];
				component.blocks[block_index] = block_to_row(block_rows[0][x], options.use_zigzag_columns);
			}
		}

		image.metadata.components.push_back(component.metadata);
		image.components.push_back(std::move(component));
	}

	decoder.finish_decompress();
	image.metadata.warning_count = decoder.warning_count();
	return image;
}

DecodedImage decode_jpeg_layout(const std::filesystem::path& path, const JpegDctReaderOptions& options) {
	JpegDecoder decoder(path);
	decoder.set_stdio_src();
	decoder.read_header();
	auto& cinfo = decoder.cinfo;

	DecodedImage image;
	image.metadata.source_path      = path;
	image.metadata.image_width      = cinfo.image_width;
	image.metadata.image_height     = cinfo.image_height;
	image.metadata.data_precision   = static_cast<uint8_t>(cinfo.data_precision);
	image.metadata.jpeg_color_space = static_cast<int>(cinfo.jpeg_color_space);
	image.metadata.progressive      = jpeg_has_multiple_scans(&cinfo) != 0;

	const auto indices = selected_component_indices(cinfo.num_components, options);
	image.components.reserve(indices.size());
	image.metadata.components.reserve(indices.size());
	for (const auto component_index : indices) {
		const auto& comp = cinfo.comp_info[component_index];

		DecodedComponent component;
		component.metadata.semantic_slot_id        = static_cast<uint32_t>(component_index);
		component.metadata.component_index         = component_index;
		component.metadata.local_component_index   = component_index;
		component.metadata.component_id            = comp.component_id;
		component.metadata.width_in_blocks         = comp.width_in_blocks;
		component.metadata.height_in_blocks        = comp.height_in_blocks;
		component.metadata.padded_width_in_blocks  = comp.width_in_blocks;
		component.metadata.padded_height_in_blocks = comp.height_in_blocks;
		component.metadata.h_samp_factor           = comp.h_samp_factor;
		component.metadata.v_samp_factor           = comp.v_samp_factor;
		component.metadata.quant_tbl_no            = comp.quant_tbl_no;

		image.metadata.components.push_back(component.metadata);
		image.components.push_back(std::move(component));
	}
	image.metadata.warning_count = decoder.warning_count();
	return image;
}

void append_row(JpegDctTable& table, const DctRow& row, const bool is_padding) {
	for (size_t col = 0; col < row.size(); ++col) {
		table.columns[col].push_back(row[col]);
	}
	++table.row_count;
	if (is_padding) {
		++table.padding_row_count;
	} else {
		++table.real_row_count;
	}
}

uint32_t encoding_profile_id(std::vector<JpegEncodingProfileMetadata>& profiles,
                             const JpegImageMetadata&                  image,
                             const JpegComponentMetadata&              component);

void validate_supported_layout_options(const JpegDctReaderOptions& options) {
	if (options.compression_partition_policy != JpegCompressionPartitionPolicy::kBySemanticSlot) {
		throw std::runtime_error("JPEG DCT compression partition policy is not implemented yet; "
		                         "only kBySemanticSlot is currently supported");
	}
	if (options.coefficient_encoding != JpegDctCoefficientEncoding::kDense64FastLanes) {
		throw std::runtime_error("JPEG DCT coefficient encoding is not implemented yet; "
		                         "only kDense64FastLanes is currently supported");
	}
}

JpegDctTable make_single_image_table(DecodedImage image, const JpegDctReaderOptions& options) {
	validate_supported_layout_options(options);
	JpegDctTable table;
	for (size_t component_idx = 0; component_idx < image.metadata.components.size(); ++component_idx) {
		auto& component                 = image.metadata.components[component_idx];
		component.semantic_slot_id      = static_cast<uint32_t>(component_idx);
		component.local_component_index = component.component_index;
		component.encoding_profile_id =
		    encoding_profile_id(table.metadata.encoding_profiles, image.metadata, component);
		if (component_idx < image.components.size()) {
			image.components[component_idx].metadata = component;
		}
		table.metadata.semantic_components.push_back(component);
	}
	table.metadata.images.push_back(std::move(image.metadata));
	table.metadata.row_ordering                 = JpegDctRowOrdering::kSingleImageComponentMajorBlockMajor;
	table.metadata.validation_mode              = options.validation_mode;
	table.metadata.compression_partition_policy = options.compression_partition_policy;
	table.metadata.coefficient_encoding         = options.coefficient_encoding;
	table.metadata.zigzag_columns               = options.use_zigzag_columns;
	table.metadata.z_curve_block_order          = options.use_z_curve_block_order;
	table.metadata.image_count                  = 1;

	for (const auto& component : image.components) {
		const auto block_order = detail::make_block_order(
		    component.metadata.width_in_blocks, component.metadata.height_in_blocks, options.use_z_curve_block_order);
		for (size_t block_idx = 0; block_idx < component.blocks.size(); ++block_idx) {
			const auto&            row = component.blocks[block_idx];
			JpegDctBlockGroupIndex group;
			group.semantic_slot_id = static_cast<uint32_t>(component.metadata.semantic_slot_id);
			group.z_order_index    = static_cast<uint32_t>(block_idx);
			if (block_idx < block_order.size()) {
				group.block_x = block_order[block_idx].x;
				group.block_y = block_order[block_idx].y;
			}
			group.row_start = table.row_count;
			group.row_count = 1;
			table.metadata.block_group_index.push_back(group);
			append_row(table, row, false);
		}
	}
	table.block_group_count = table.metadata.block_group_index.size();
	return table;
}

bool same_semantic_slot(const JpegComponentMetadata& metadata, const ComponentSlot& slot) {
	return metadata.component_id == slot.component_id;
}

const DecodedComponent* find_component_for_slot(const DecodedImage& image, const ComponentSlot& slot) {
	for (const auto& component : image.components) {
		if (same_semantic_slot(component.metadata, slot)) {
			return &component;
		}
	}
	return nullptr;
}

std::vector<std::vector<const DecodedComponent*>> make_image_slot_components(const std::vector<DecodedImage>&  images,
                                                                             const std::vector<ComponentSlot>& slots) {
	std::vector<std::vector<const DecodedComponent*>> result;
	result.reserve(images.size());
	for (const auto& image : images) {
		auto& row = result.emplace_back(slots.size(), nullptr);
		for (size_t slot_idx = 0; slot_idx < slots.size(); ++slot_idx) {
			row[slot_idx] = find_component_for_slot(image, slots[slot_idx]);
		}
	}
	return result;
}

// Builds dataset-wide semantic component slots keyed by JPEG component_id. Sampling and quantization differences are
// represented by per-image component encoding profiles, not by splitting semantic slots.
std::vector<ComponentSlot> normalize_component_slots(const std::vector<DecodedImage>& images) {
	std::vector<ComponentSlot> slots;
	if (images.empty()) {
		return slots;
	}

	auto find_slot = [&](const JpegComponentMetadata& metadata) -> ComponentSlot* {
		for (auto& slot : slots) {
			if (same_semantic_slot(metadata, slot)) {
				return &slot;
			}
		}
		return nullptr;
	};

	for (const auto& image : images) {
		for (const auto& component : image.components) {
			auto* slot = find_slot(component.metadata);
			if (slot == nullptr) {
				ComponentSlot new_slot;
				new_slot.semantic_slot_id     = static_cast<uint32_t>(slots.size());
				new_slot.component_index      = slots.size();
				new_slot.component_id         = component.metadata.component_id;
				new_slot.max_width_in_blocks  = component.metadata.width_in_blocks;
				new_slot.max_height_in_blocks = component.metadata.height_in_blocks;
				slots.push_back(new_slot);
				continue;
			}

			slot->max_width_in_blocks  = std::max(slot->max_width_in_blocks, component.metadata.width_in_blocks);
			slot->max_height_in_blocks = std::max(slot->max_height_in_blocks, component.metadata.height_in_blocks);
		}
	}
	return slots;
}

uint32_t encoding_profile_id(std::vector<JpegEncodingProfileMetadata>& profiles,
                             const JpegImageMetadata&                  image,
                             const JpegComponentMetadata&              component) {
	const auto* table = find_quant_table(image, component.quant_tbl_no);
	for (const auto& profile : profiles) {
		if (profile.h_samp_factor == component.h_samp_factor && profile.v_samp_factor == component.v_samp_factor &&
		    profile.quant_tbl_no == component.quant_tbl_no &&
		    profile.quant_table_fingerprint == component.quant_table_fingerprint) {
			return profile.profile_id;
		}
	}

	JpegEncodingProfileMetadata profile;
	profile.profile_id              = static_cast<uint32_t>(profiles.size());
	profile.h_samp_factor           = component.h_samp_factor;
	profile.v_samp_factor           = component.v_samp_factor;
	profile.quant_tbl_no            = component.quant_tbl_no;
	profile.quant_table_fingerprint = component.quant_table_fingerprint;
	if (table != nullptr) {
		profile.quant_table_values = table->values;
	}
	profiles.push_back(profile);
	return profile.profile_id;
}

JpegImageMetadata normalized_image_metadata(const JpegImageMetadata&                    image_metadata,
                                            const std::vector<ComponentSlot>&           slots,
                                            const std::vector<const DecodedComponent*>& slot_components,
                                            std::vector<JpegEncodingProfileMetadata>&   profiles) {
	JpegImageMetadata out;
	out.source_path      = image_metadata.source_path;
	out.image_width      = image_metadata.image_width;
	out.image_height     = image_metadata.image_height;
	out.data_precision   = image_metadata.data_precision;
	out.jpeg_color_space = image_metadata.jpeg_color_space;
	out.progressive      = image_metadata.progressive;
	out.warning_count    = image_metadata.warning_count;
	out.quant_tables     = image_metadata.quant_tables;
	out.markers          = image_metadata.markers;
	out.components.reserve(slots.size());
	for (size_t slot_idx = 0; slot_idx < slots.size(); ++slot_idx) {
		const auto&           slot = slots[slot_idx];
		JpegComponentMetadata metadata;
		metadata.semantic_slot_id        = slot.semantic_slot_id;
		metadata.component_index         = slot.component_index;
		metadata.component_id            = slot.component_id;
		metadata.padded_width_in_blocks  = slot.max_width_in_blocks;
		metadata.padded_height_in_blocks = slot.max_height_in_blocks;

		if (const auto* component = slot_components[slot_idx]) {
			metadata.local_component_index   = component->metadata.local_component_index;
			metadata.component_id            = component->metadata.component_id;
			metadata.width_in_blocks         = component->metadata.width_in_blocks;
			metadata.height_in_blocks        = component->metadata.height_in_blocks;
			metadata.h_samp_factor           = component->metadata.h_samp_factor;
			metadata.v_samp_factor           = component->metadata.v_samp_factor;
			metadata.quant_tbl_no            = component->metadata.quant_tbl_no;
			metadata.quant_table_fingerprint = component->metadata.quant_table_fingerprint;
			metadata.encoding_profile_id     = encoding_profile_id(profiles, image_metadata, component->metadata);
			metadata.present                 = true;
		} else {
			metadata.width_in_blocks  = 0;
			metadata.height_in_blocks = 0;
			metadata.present          = false;
		}
		out.components.push_back(metadata);
	}
	return out;
}

std::vector<JpegComponentMetadata> dataset_component_metadata(const std::vector<ComponentSlot>& slots) {
	std::vector<JpegComponentMetadata> components;
	components.reserve(slots.size());
	for (const auto& slot : slots) {
		JpegComponentMetadata metadata;
		metadata.semantic_slot_id        = slot.semantic_slot_id;
		metadata.component_index         = slot.component_index;
		metadata.component_id            = slot.component_id;
		metadata.width_in_blocks         = slot.max_width_in_blocks;
		metadata.height_in_blocks        = slot.max_height_in_blocks;
		metadata.padded_width_in_blocks  = slot.max_width_in_blocks;
		metadata.padded_height_in_blocks = slot.max_height_in_blocks;
		metadata.present                 = true;
		components.push_back(metadata);
	}
	return components;
}

void validate_dataset(const std::vector<DecodedImage>& images, const JpegDctReaderOptions& options) {
	if (images.empty()) {
		throw std::runtime_error("JPEG DCT dataset requires at least one image");
	}
	if (options.validation_mode != JpegDatasetValidationMode::kRequireSameComponentGrids) {
		return;
	}

	const auto& reference = images.front();
	for (size_t image_idx = 1; image_idx < images.size(); ++image_idx) {
		const auto& image = images[image_idx];
		if (image.components.size() != reference.components.size()) {
			std::ostringstream msg;
			msg << "JPEG dataset image " << image_idx << " has " << image.components.size()
			    << " selected components; expected " << reference.components.size();
			throw std::runtime_error(msg.str());
		}
		for (size_t component_idx = 0; component_idx < reference.components.size(); ++component_idx) {
			const auto& expected = reference.components[component_idx].metadata;
			const auto& actual   = image.components[component_idx].metadata;
			if (actual.component_id != expected.component_id || actual.width_in_blocks != expected.width_in_blocks ||
			    actual.height_in_blocks != expected.height_in_blocks ||
			    actual.h_samp_factor != expected.h_samp_factor || actual.v_samp_factor != expected.v_samp_factor) {
				std::ostringstream msg;
				msg << "JPEG dataset image " << image_idx << " component " << component_idx
				    << " grid differs from image 0; expected component_id=" << expected.component_id
				    << " blocks=" << expected.width_in_blocks << "x" << expected.height_in_blocks
				    << " sampling=" << expected.h_samp_factor << "x" << expected.v_samp_factor
				    << ", got component_id=" << actual.component_id << " blocks=" << actual.width_in_blocks << "x"
				    << actual.height_in_blocks << " sampling=" << actual.h_samp_factor << "x" << actual.v_samp_factor;
				throw std::runtime_error(msg.str());
			}
		}
	}
}

const DctRow* find_component_block(const DecodedComponent& component, const uint32_t x, const uint32_t y) {
	if (x >= component.metadata.width_in_blocks || y >= component.metadata.height_in_blocks) {
		return nullptr;
	}
	const size_t flat_index = static_cast<size_t>(y) * component.metadata.width_in_blocks + x;
	if (flat_index >= component.coord_to_block_index.size()) {
		return nullptr;
	}
	const size_t block_index = component.coord_to_block_index[flat_index];
	if (block_index >= component.blocks.size()) {
		return nullptr;
	}
	return &component.blocks[block_index];
}

DatasetLayout make_dataset_layout(const std::vector<DecodedImage>& images) {
	DatasetLayout layout;
	layout.slots                 = normalize_component_slots(images);
	layout.image_slot_components = make_image_slot_components(images, layout.slots);
	return layout;
}

DatasetLayout make_dataset_layout(const std::vector<DecodedImage>& images, const std::vector<ComponentSlot>& slots) {
	DatasetLayout layout;
	layout.slots                 = slots;
	layout.image_slot_components = make_image_slot_components(images, layout.slots);
	return layout;
}

JpegDctTable make_dataset_table(std::vector<DecodedImage>         images,
                                const JpegDctReaderOptions&       options,
                                const std::vector<ComponentSlot>* global_slots) {
	validate_supported_layout_options(options);
	if (global_slots == nullptr) {
		validate_dataset(images, options);
	}
	const auto layout =
	    global_slots == nullptr ? make_dataset_layout(images) : make_dataset_layout(images, *global_slots);

	JpegDctTable table;
	table.metadata.row_ordering                 = JpegDctRowOrdering::kDatasetComponentMajorBlockMajorImageMinor;
	table.metadata.validation_mode              = options.validation_mode;
	table.metadata.compression_partition_policy = options.compression_partition_policy;
	table.metadata.coefficient_encoding         = options.coefficient_encoding;
	table.metadata.zigzag_columns               = options.use_zigzag_columns;
	table.metadata.z_curve_block_order          = options.use_z_curve_block_order;
	table.metadata.image_count                  = images.size();

	table.metadata.semantic_components = dataset_component_metadata(layout.slots);
	table.metadata.images.reserve(images.size());
	for (size_t image_idx = 0; image_idx < images.size(); ++image_idx) {
		table.metadata.images.push_back(normalized_image_metadata(images[image_idx].metadata,
		                                                          layout.slots,
		                                                          layout.image_slot_components[image_idx],
		                                                          table.metadata.encoding_profiles));
	}

	const DctRow padding_row {};
	for (size_t slot_idx = 0; slot_idx < layout.slots.size(); ++slot_idx) {
		const auto& slot        = layout.slots[slot_idx];
		const auto  block_order = detail::make_block_order(
            slot.max_width_in_blocks, slot.max_height_in_blocks, options.use_z_curve_block_order);
		for (size_t z_order_index = 0; z_order_index < block_order.size(); ++z_order_index) {
			const auto&            block_coord = block_order[z_order_index];
			JpegDctBlockGroupIndex group;
			group.semantic_slot_id = slot.semantic_slot_id;
			group.z_order_index    = static_cast<uint32_t>(z_order_index);
			group.block_x          = block_coord.x;
			group.block_y          = block_coord.y;
			group.row_start        = table.row_count;
			for (const auto& slot_components : layout.image_slot_components) {
				const auto* component = slot_components[slot_idx];
				const auto* row =
				    component == nullptr ? nullptr : find_component_block(*component, block_coord.x, block_coord.y);
				if (row != nullptr) {
					append_row(table, *row, false);
					++group.row_count;
				} else if (options.validation_mode == JpegDatasetValidationMode::kPadToMaxComponentGrids) {
					append_row(table, padding_row, true);
					++group.row_count;
				}
			}
			if (group.row_count != 0) {
				table.metadata.block_group_index.push_back(group);
			}
		}
	}
	table.block_group_count = table.metadata.block_group_index.size();
	return table;
}

JpegDctTable make_dataset_table(std::vector<DecodedImage> images, const JpegDctReaderOptions& options) {
	return make_dataset_table(std::move(images), options, nullptr);
}

int row_ordering_id(const JpegDctRowOrdering ordering) {
	switch (ordering) {
	case JpegDctRowOrdering::kSingleImageComponentMajorBlockMajor:
		return 0;
	case JpegDctRowOrdering::kDatasetComponentMajorBlockMajorImageMinor:
		return 1;
	}
	return -1;
}

int validation_mode_id(const JpegDatasetValidationMode mode) {
	switch (mode) {
	case JpegDatasetValidationMode::kRequireSameComponentGrids:
		return 0;
	case JpegDatasetValidationMode::kPadToMaxComponentGrids:
		return 1;
	case JpegDatasetValidationMode::kRaggedBlockMajor:
		return 2;
	}
	return -1;
}

uint16_t metadata_profile_id(const JpegMetadataProfile profile) {
	switch (profile) {
	case JpegMetadataProfile::kDctDatasetOnly:
		return 0;
	case JpegMetadataProfile::kReconstructableJpeg:
		return 1;
	case JpegMetadataProfile::kPreserveOriginalMarkers:
		return 2;
	}
	return 0;
}

struct BinaryWriter {
	explicit BinaryWriter(std::ostream& output)
	    : out(output) {
	}

	void u8(const uint8_t value) {
		out.put(static_cast<char>(value));
	}

	void u16(const uint16_t value) {
		u8(static_cast<uint8_t>(value & 0xffU));
		u8(static_cast<uint8_t>((value >> 8U) & 0xffU));
	}

	void u32(const uint32_t value) {
		for (unsigned shift = 0; shift < 32; shift += 8) {
			u8(static_cast<uint8_t>((value >> shift) & 0xffU));
		}
	}

	void u64(const uint64_t value) {
		for (unsigned shift = 0; shift < 64; shift += 8) {
			u8(static_cast<uint8_t>((value >> shift) & 0xffU));
		}
	}

	void i32(const int32_t value) {
		u32(static_cast<uint32_t>(value));
	}

	std::ostream& out;
};

struct BinaryReader {
	explicit BinaryReader(std::vector<uint8_t> input)
	    : data(std::move(input)) {
	}

	bool eof() const {
		return pos == data.size();
	}

	size_t remaining() const {
		return data.size() - pos;
	}

	uint8_t u8() {
		require(1);
		return data[pos++];
	}

	uint16_t u16() {
		uint16_t value = 0;
		for (unsigned shift = 0; shift < 16; shift += 8) {
			value |= static_cast<uint16_t>(u8()) << shift;
		}
		return value;
	}

	uint32_t u32() {
		uint32_t value = 0;
		for (unsigned shift = 0; shift < 32; shift += 8) {
			value |= static_cast<uint32_t>(u8()) << shift;
		}
		return value;
	}

	uint64_t u64() {
		uint64_t value = 0;
		for (unsigned shift = 0; shift < 64; shift += 8) {
			value |= static_cast<uint64_t>(u8()) << shift;
		}
		return value;
	}

	int32_t i32() {
		return static_cast<int32_t>(u32());
	}

	std::vector<uint8_t> bytes(const size_t n) {
		require(n);
		std::vector<uint8_t> out(data.begin() + static_cast<std::ptrdiff_t>(pos),
		                         data.begin() + static_cast<std::ptrdiff_t>(pos + n));
		pos += n;
		return out;
	}

	std::string string() {
		const auto n = u32();
		require(n);
		std::string out(reinterpret_cast<const char*>(data.data() + pos), n);
		pos += n;
		return out;
	}

private:
	void require(const size_t n) const {
		if (n > remaining()) {
			throw std::runtime_error("truncated JPEG DCT binary metadata");
		}
	}

	std::vector<uint8_t> data;
	size_t               pos = 0;
};

enum class MetadataSection : uint16_t {
	kComponentGrid            = 1,
	kPerImageGrid             = 2,
	kReconstructableImageInfo = 3,
	kOriginalMarkers          = 4,
	kEncodingProfiles         = 5,
	kBlockGroupIndex          = 6,
};

template <typename Fn>
std::vector<uint8_t> build_section(Fn&& fn) {
	std::ostringstream payload(std::ios::binary);
	BinaryWriter       writer(payload);
	fn(writer);
	const auto str = payload.str();
	return std::vector<uint8_t>(str.begin(), str.end());
}

void write_section(BinaryWriter& writer, const MetadataSection section_id, const std::vector<uint8_t>& payload) {
	writer.u16(static_cast<uint16_t>(section_id));
	writer.u64(static_cast<uint64_t>(payload.size()));
	writer.out.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
}

std::string dct_column_name(const size_t col) {
	std::ostringstream out;
	out << "dct_zz_" << std::setw(2) << std::setfill('0') << col;
	return out.str();
}

std::vector<uint64_t> make_block_group_aligned_rowgroups(JpegDctDatasetMetadata& metadata,
                                                         const size_t            row_count,
                                                         const uint32_t          rowgroup_vectors) {
	if (rowgroup_vectors == 0) {
		throw std::runtime_error("JPEG DCT shard rowgroup_vectors must be greater than zero");
	}
	const uint64_t        target_rows = static_cast<uint64_t>(rowgroup_vectors) * fastlanes::CFG::VEC_SZ;
	std::vector<uint64_t> rowgroups;
	uint64_t              current_rowgroup_rows = 0;

	for (auto& group : metadata.block_group_index) {
		if (group.row_count == 0) {
			continue;
		}
		if (current_rowgroup_rows != 0 && current_rowgroup_rows + group.row_count > target_rows) {
			rowgroups.push_back(current_rowgroup_rows);
			current_rowgroup_rows = 0;
		}
		group.fls_rowgroup_index    = static_cast<uint32_t>(rowgroups.size());
		group.row_start_in_rowgroup = static_cast<uint32_t>(current_rowgroup_rows);
		current_rowgroup_rows += group.row_count;
	}
	if (current_rowgroup_rows != 0) {
		rowgroups.push_back(current_rowgroup_rows);
	}

	uint64_t sum = 0;
	for (const auto n_tuples : rowgroups) {
		sum += n_tuples;
	}
	if (sum != row_count) {
		std::ostringstream msg;
		msg << "JPEG DCT rowgroup layout covers " << sum << " rows; expected " << row_count;
		throw std::runtime_error(msg.str());
	}
	return rowgroups;
}

bool layout_component_has_block(const DecodedComponent* component, const uint32_t block_x, const uint32_t block_y) {
	return component != nullptr && block_x < component->metadata.width_in_blocks &&
	       block_y < component->metadata.height_in_blocks;
}

size_t estimate_shard_rowgroup_count(const std::vector<DecodedImage>&  layout_images,
                                     const std::vector<ComponentSlot>& global_slots,
                                     const size_t                      first_image,
                                     const size_t                      image_count,
                                     const JpegDctReaderOptions&       options,
                                     const uint32_t                    rowgroup_vectors) {
	const uint64_t target_rows           = static_cast<uint64_t>(rowgroup_vectors) * fastlanes::CFG::VEC_SZ;
	size_t         rowgroup_count        = 0;
	uint64_t       current_rowgroup_rows = 0;

	for (const auto& slot : global_slots) {
		const auto block_order = detail::make_block_order(
		    slot.max_width_in_blocks, slot.max_height_in_blocks, options.use_z_curve_block_order);
		for (const auto& block_coord : block_order) {
			uint64_t group_row_count = 0;
			for (size_t image_idx = first_image; image_idx < first_image + image_count; ++image_idx) {
				const auto* component = find_component_for_slot(layout_images[image_idx], slot);
				if (layout_component_has_block(component, block_coord.x, block_coord.y)) {
					++group_row_count;
				} else if (options.validation_mode == JpegDatasetValidationMode::kPadToMaxComponentGrids) {
					++group_row_count;
				}
			}
			if (group_row_count == 0) {
				continue;
			}
			if (current_rowgroup_rows != 0 && current_rowgroup_rows + group_row_count > target_rows) {
				++rowgroup_count;
				current_rowgroup_rows = 0;
			}
			current_rowgroup_rows += group_row_count;
		}
	}
	if (current_rowgroup_rows != 0) {
		++rowgroup_count;
	}
	return rowgroup_count;
}

size_t choose_shard_image_count(const std::vector<DecodedImage>&  layout_images,
                                const std::vector<ComponentSlot>& global_slots,
                                const size_t                      first_image,
                                const size_t                      max_image_count,
                                const JpegDctReaderOptions&       options,
                                const JpegDctShardOptions&        shard_options) {
	if (estimate_shard_rowgroup_count(
	        layout_images, global_slots, first_image, max_image_count, options, shard_options.rowgroup_vectors) <=
	    shard_options.rowgroups_per_shard) {
		return max_image_count;
	}

	size_t lo = 1;
	size_t hi = max_image_count;
	while (lo < hi) {
		const auto mid = lo + (hi - lo + 1) / 2;
		if (estimate_shard_rowgroup_count(
		        layout_images, global_slots, first_image, mid, options, shard_options.rowgroup_vectors) <=
		    shard_options.rowgroups_per_shard) {
			lo = mid;
		} else {
			hi = mid - 1;
		}
	}

	if (estimate_shard_rowgroup_count(
	        layout_images, global_slots, first_image, lo, options, shard_options.rowgroup_vectors) >
	    shard_options.rowgroups_per_shard) {
		std::ostringstream msg;
		msg << "JPEG DCT single-image shard at global image index " << first_image << " requires more than "
		    << shard_options.rowgroups_per_shard << " rowgroups; increase --rowgroups-per-shard or --rowgroup-vectors";
		throw std::runtime_error(msg.str());
	}
	return lo;
}

void write_jpeg_dct_fls_data(const JpegDctTable&                  table,
                             const std::filesystem::path&         fls_output_path,
                             const fastlanes::MemoryTableOptions& options       = {},
                             const bool                           inline_footer = false) {
	std::array<fastlanes::MemoryColumn, 64> columns;
	for (size_t col = 0; col < columns.size(); ++col) {
		columns[col].name = dct_column_name(col);
		columns[col].data = std::span<const int16_t>(table.columns[col].data(), table.columns[col].size());
	}

	const fastlanes::MemoryTable memory_table {
	    std::span<const fastlanes::MemoryColumn>(columns.data(), columns.size())};
	fastlanes::Connection connection;
	fastlanes::load_memory_table(connection, memory_table, options);
	if (inline_footer) {
		connection.inline_footer();
	}
	connection.to_fls(fls_output_path);
}

std::string shard_file_name(const uint32_t shard_id, const std::string_view suffix) {
	std::ostringstream out;
	out << "shard_" << std::setw(6) << std::setfill('0') << shard_id << suffix;
	return out.str();
}

JpegDctShardOptions effective_shard_options(JpegDctShardOptions options) {
	constexpr size_t   kBalancedShardImages        = 8192;
	constexpr uint32_t kBalancedRowgroupVectors    = 128;
	constexpr uint32_t kBalancedRowgroupsPerShard  = 256;
	size_t             preset_shard_images        = kBalancedShardImages;
	uint32_t           preset_rowgroup_vectors    = kBalancedRowgroupVectors;
	uint32_t           preset_rowgroups_per_shard = kBalancedRowgroupsPerShard;

	switch (options.preset) {
	case JpegDctShardPreset::kCropLatency:
		preset_shard_images     = 4096;
		preset_rowgroup_vectors = 64;
		break;
	case JpegDctShardPreset::kBalanced:
		break;
	case JpegDctShardPreset::kThroughput:
		preset_rowgroup_vectors = 256;
		break;
	}

	if (!options.shard_images_specified && options.shard_images == kBalancedShardImages) {
		options.shard_images = preset_shard_images;
	}
	if (!options.rowgroup_vectors_specified && options.rowgroup_vectors == kBalancedRowgroupVectors) {
		options.rowgroup_vectors = preset_rowgroup_vectors;
	}
	if (!options.rowgroups_per_shard_specified && options.rowgroups_per_shard == kBalancedRowgroupsPerShard) {
		options.rowgroups_per_shard = preset_rowgroups_per_shard;
	}
	return options;
}

std::vector<uint8_t> read_binary_file(const std::filesystem::path& path) {
	std::ifstream in(path, std::ios::binary);
	if (!in) {
		throw std::runtime_error("failed to open binary file: " + path.string());
	}
	return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void expect_magic(BinaryReader& reader, const std::array<uint8_t, 8>& expected, const char* label) {
	for (const auto byte : expected) {
		if (reader.u8() != byte) {
			throw std::runtime_error(std::string("invalid ") + label + " magic");
		}
	}
}

JpegDatasetValidationMode validation_mode_from_id(const uint16_t id) {
	switch (id) {
	case 0:
		return JpegDatasetValidationMode::kRequireSameComponentGrids;
	case 1:
		return JpegDatasetValidationMode::kPadToMaxComponentGrids;
	case 2:
		return JpegDatasetValidationMode::kRaggedBlockMajor;
	default:
		throw std::runtime_error("unknown JPEG DCT validation mode id in metadata");
	}
}

JpegDctRowOrdering row_ordering_from_id(const uint16_t id) {
	switch (id) {
	case 0:
		return JpegDctRowOrdering::kSingleImageComponentMajorBlockMajor;
	case 1:
		return JpegDctRowOrdering::kDatasetComponentMajorBlockMajorImageMinor;
	default:
		throw std::runtime_error("unknown JPEG DCT row ordering id in metadata");
	}
}

JpegCompressionPartitionPolicy partition_policy_from_id(const uint16_t id) {
	switch (id) {
	case 0:
		return JpegCompressionPartitionPolicy::kBySemanticSlot;
	case 1:
		return JpegCompressionPartitionPolicy::kByEncodingProfile;
	default:
		throw std::runtime_error("unknown JPEG DCT compression partition policy id in metadata");
	}
}

JpegDctCoefficientEncoding coefficient_encoding_from_id(const uint16_t id) {
	switch (id) {
	case 0:
		return JpegDctCoefficientEncoding::kDense64FastLanes;
	case 1:
		return JpegDctCoefficientEncoding::kExpCrossRleI16;
	case 2:
		return JpegDctCoefficientEncoding::kDcDeltaDense64FastLanes;
	case 3:
		return JpegDctCoefficientEncoding::kDcDeltaAcSparseRle;
	case 4:
		return JpegDctCoefficientEncoding::kJpegLikeRunLength;
	default:
		throw std::runtime_error("unknown JPEG DCT coefficient encoding id in metadata");
	}
}

JpegDctShardManifest read_jpeg_dct_shard_manifest_file(const std::filesystem::path& path) {
	BinaryReader reader(read_binary_file(path));
	expect_magic(reader, {'G', 'J', 'D', 'C', 'T', 'S', 'H', '1'}, "JPEG DCT shard manifest");

	JpegDctShardManifest manifest;
	manifest.version             = reader.u32();
	manifest.policy              = validation_mode_from_id(reader.u16());
	manifest.rowgroup_vectors    = reader.u32();
	manifest.rowgroups_per_shard = reader.u32();
	manifest.image_count         = reader.u64();
	const auto shard_count       = reader.u32();
	manifest.shards.reserve(shard_count);
	for (uint32_t shard_idx = 0; shard_idx < shard_count; ++shard_idx) {
		JpegDctShardManifestEntry entry;
		entry.shard_id                 = reader.u32();
		entry.first_global_image_index = reader.u64();
		entry.image_count              = reader.u32();
		entry.real_row_count           = reader.u64();
		entry.padding_row_count        = reader.u64();
		entry.physical_row_count       = reader.u64();
		entry.rowgroup_count           = reader.u32();
		entry.block_group_count        = reader.u32();
		entry.fls_file_size            = reader.u64();
		entry.metadata_file_size       = reader.u64();
		entry.fls_file_name            = reader.string();
		entry.metadata_file_name       = reader.string();
		manifest.shards.push_back(std::move(entry));
	}
	return manifest;
}

template <typename ColT>
const ColT* typed_column_ptr(const fastlanes::col_pt& column) {
	const auto* holder = std::get_if<fastlanes::up<ColT>>(&column);
	if (holder == nullptr || !*holder) {
		return nullptr;
	}
	return holder->get();
}

template <typename T>
int16_t checked_dct_value(const T value) {
	return static_cast<int16_t>(value);
}

int16_t coefficient_value(const fastlanes::col_pt& column, const size_t row_idx) {
	if (const auto* col = typed_column_ptr<fastlanes::col_i08>(column)) {
		return checked_dct_value(col->data.at(row_idx));
	}
	if (const auto* col = typed_column_ptr<fastlanes::col_i16>(column)) {
		return checked_dct_value(col->data.at(row_idx));
	}
	if (const auto* col = typed_column_ptr<fastlanes::col_i32>(column)) {
		return checked_dct_value(col->data.at(row_idx));
	}
	if (const auto* col = typed_column_ptr<fastlanes::col_i64>(column)) {
		return checked_dct_value(col->data.at(row_idx));
	}
	if (const auto* col = typed_column_ptr<fastlanes::u08_col_t>(column)) {
		return checked_dct_value(col->data.at(row_idx));
	}
	if (const auto* col = typed_column_ptr<fastlanes::u16_col_t>(column)) {
		return checked_dct_value(col->data.at(row_idx));
	}
	if (const auto* col = typed_column_ptr<fastlanes::u32_col_t>(column)) {
		return checked_dct_value(col->data.at(row_idx));
	}
	if (const auto* col = typed_column_ptr<fastlanes::u64_col_t>(column)) {
		return checked_dct_value(col->data.at(row_idx));
	}
	throw std::runtime_error("JPEG DCT FLS column materialized to an unsupported type");
}

} // namespace

JpegDctTable read_jpeg_dct_file(const std::filesystem::path& path, const JpegDctReaderOptions& options) {
	return make_single_image_table(decode_jpeg_coefficients(path, options), options);
}

JpegDctTable read_jpeg_dct_dataset(const std::vector<std::filesystem::path>& paths,
                                   const JpegDctReaderOptions&               options) {
	std::vector<DecodedImage> images;
	images.reserve(paths.size());
	for (const auto& path : paths) {
		images.push_back(decode_jpeg_coefficients(path, options));
	}
	return make_dataset_table(std::move(images), options);
}

const std::vector<JpegComponentMetadata>* legacy_metadata_components(const JpegDctDatasetMetadata& metadata) {
	if (!metadata.images.empty()) {
		return &metadata.images.front().components;
	}
	if (!metadata.semantic_components.empty()) {
		return &metadata.semantic_components;
	}
	return nullptr;
}

void write_jpeg_dct_metadata(const JpegDctDatasetMetadata& metadata, const std::filesystem::path& output_path) {
	std::ofstream out(output_path, std::ios::binary);
	if (!out) {
		throw std::runtime_error("failed to open JPEG DCT metadata output: " + output_path.string());
	}

	const auto* components      = legacy_metadata_components(metadata);
	const auto  component_count = components == nullptr ? 0 : components->size();

	BinaryWriter  writer(out);
	const uint8_t magic[8] {'G', 'J', 'D', 'C', 'T', 'M', 'D', '1'};
	out.write(reinterpret_cast<const char*>(magic), sizeof(magic));
	writer.u16(1);
	writer.u16(static_cast<uint16_t>(row_ordering_id(metadata.row_ordering)));
	writer.u16(static_cast<uint16_t>(validation_mode_id(metadata.validation_mode)));
	writer.u16(static_cast<uint16_t>((metadata.zigzag_columns ? 1U : 0U) | (metadata.z_curve_block_order ? 2U : 0U)));
	writer.u64(static_cast<uint64_t>(metadata.image_count));
	writer.u32(static_cast<uint32_t>(component_count));

	for (size_t component_idx = 0; component_idx < component_count; ++component_idx) {
		const auto& component = (*components)[component_idx];
		writer.u32(static_cast<uint32_t>(component.component_index));
		writer.i32(static_cast<int32_t>(component.component_id));
		writer.u32(component.width_in_blocks);
		writer.u32(component.height_in_blocks);
		writer.u32(component.padded_width_in_blocks);
		writer.u32(component.padded_height_in_blocks);
		writer.i32(static_cast<int32_t>(component.h_samp_factor));
		writer.i32(static_cast<int32_t>(component.v_samp_factor));
	}

	uint8_t has_image_records = 0;
	if (metadata.validation_mode != JpegDatasetValidationMode::kRequireSameComponentGrids &&
	    metadata.row_ordering == JpegDctRowOrdering::kDatasetComponentMajorBlockMajorImageMinor) {
		has_image_records = 1;
	}
	writer.u8(has_image_records);

	if (has_image_records != 0) {
		for (const auto& image : metadata.images) {
			for (const auto& component : image.components) {
				writer.u8(component.present ? 1 : 0);
				writer.u32(component.width_in_blocks);
				writer.u32(component.height_in_blocks);
			}
		}
	}
}

void write_jpeg_dct_metadata(const JpegDctDatasetMetadata&       metadata,
                             const std::filesystem::path&        output_path,
                             const JpegDctMetadataWriterOptions& options) {
	std::ofstream out(output_path, std::ios::binary);
	if (!out) {
		throw std::runtime_error("failed to open JPEG DCT metadata output: " + output_path.string());
	}

	const auto* components = metadata.semantic_components.empty() ? nullptr : &metadata.semantic_components;
	if (components == nullptr && !metadata.images.empty()) {
		components = &metadata.images.front().components;
	}
	const auto component_count = components == nullptr ? 0 : components->size();

	BinaryWriter  writer(out);
	const uint8_t magic[8] {'G', 'J', 'D', 'C', 'T', 'M', 'D', '3'};
	out.write(reinterpret_cast<const char*>(magic), sizeof(magic));
	const uint16_t version         = 3;
	const uint16_t row_ordering    = static_cast<uint16_t>(row_ordering_id(metadata.row_ordering));
	const uint16_t validation_mode = static_cast<uint16_t>(validation_mode_id(metadata.validation_mode));
	const uint16_t profile         = metadata_profile_id(options.profile);
	const uint16_t layout_flags =
	    static_cast<uint16_t>((metadata.zigzag_columns ? 1U : 0U) | (metadata.z_curve_block_order ? 2U : 0U));
	const uint64_t image_count         = static_cast<uint64_t>(metadata.image_count);
	const uint32_t n_components        = static_cast<uint32_t>(component_count);
	const uint32_t n_encoding_profiles = static_cast<uint32_t>(metadata.encoding_profiles.size());
	writer.u16(version);
	writer.u16(profile);
	writer.u16(row_ordering);
	writer.u16(validation_mode);
	writer.u16(layout_flags);
	writer.u64(image_count);
	writer.u32(n_components);
	writer.u32(n_encoding_profiles);
	writer.u16(static_cast<uint16_t>(metadata.compression_partition_policy));
	writer.u16(static_cast<uint16_t>(metadata.coefficient_encoding));

	const auto component_section = build_section([&](BinaryWriter& section) {
		for (size_t component_idx = 0; component_idx < component_count; ++component_idx) {
			const auto& component = (*components)[component_idx];
			section.u32(component.semantic_slot_id);
			section.u32(static_cast<uint32_t>(component.component_index));
			section.i32(static_cast<int32_t>(component.component_id));
			section.u32(component.width_in_blocks);
			section.u32(component.height_in_blocks);
			section.u32(component.padded_width_in_blocks);
			section.u32(component.padded_height_in_blocks);
		}
	});
	write_section(writer, MetadataSection::kComponentGrid, component_section);

	const auto encoding_profile_section = build_section([&](BinaryWriter& section) {
		for (const auto& encoding_profile : metadata.encoding_profiles) {
			section.u32(encoding_profile.profile_id);
			section.i32(encoding_profile.h_samp_factor);
			section.i32(encoding_profile.v_samp_factor);
			section.i32(encoding_profile.quant_tbl_no);
			section.u64(encoding_profile.quant_table_fingerprint);
			for (const auto value : encoding_profile.quant_table_values) {
				section.u16(value);
			}
		}
	});
	write_section(writer, MetadataSection::kEncodingProfiles, encoding_profile_section);

	const auto block_group_index_section = build_section([&](BinaryWriter& section) {
		for (const auto& group : metadata.block_group_index) {
			section.u32(group.semantic_slot_id);
			section.u32(group.z_order_index);
			section.u32(group.block_x);
			section.u32(group.block_y);
			section.u64(group.row_start);
			section.u32(group.row_count);
			section.u32(group.fls_rowgroup_index);
			section.u32(group.row_start_in_rowgroup);
		}
	});
	write_section(writer, MetadataSection::kBlockGroupIndex, block_group_index_section);

	const size_t first_image_idx   = 0;
	bool         has_image_records = !metadata.images.empty();
	if (has_image_records) {
		const auto per_image_section = build_section([&](BinaryWriter& section) {
			for (size_t image_idx = first_image_idx; image_idx < metadata.images.size(); ++image_idx) {
				const auto& image = metadata.images[image_idx];
				section.u32(image.image_width);
				section.u32(image.image_height);
				section.u8(image.data_precision);
				section.i32(image.jpeg_color_space);
				section.u32(image.warning_count);
				for (const auto& component : image.components) {
					section.u8(component.present ? 1 : 0);
					section.u32(component.semantic_slot_id);
					section.u32(static_cast<uint32_t>(component.local_component_index));
					section.i32(component.component_id);
					section.u32(component.width_in_blocks);
					section.u32(component.height_in_blocks);
					section.u32(component.encoding_profile_id);
				}
			}
		});
		write_section(writer, MetadataSection::kPerImageGrid, per_image_section);
	}

	if (options.profile == JpegMetadataProfile::kReconstructableJpeg ||
	    options.profile == JpegMetadataProfile::kPreserveOriginalMarkers) {
		const auto reconstruction_section = build_section([&](BinaryWriter& section) {
			section.u64(static_cast<uint64_t>(metadata.images.size() - first_image_idx));
			for (size_t image_idx = first_image_idx; image_idx < metadata.images.size(); ++image_idx) {
				const auto& image = metadata.images[image_idx];
				section.u32(image.image_width);
				section.u32(image.image_height);
				section.u8(image.data_precision);
				section.i32(image.jpeg_color_space);
				section.u32(static_cast<uint32_t>(image.quant_tables.size()));
				for (const auto& table : image.quant_tables) {
					section.u8(table.table_id);
					for (const auto value : table.values) {
						section.u16(value);
					}
				}
			}
		});
		write_section(writer, MetadataSection::kReconstructableImageInfo, reconstruction_section);
	}

	if (options.profile == JpegMetadataProfile::kPreserveOriginalMarkers) {
		const auto marker_section = build_section([&](BinaryWriter& section) {
			section.u64(static_cast<uint64_t>(metadata.images.size() - first_image_idx));
			for (size_t image_idx = first_image_idx; image_idx < metadata.images.size(); ++image_idx) {
				const auto& image = metadata.images[image_idx];
				section.u32(static_cast<uint32_t>(image.markers.size()));
				for (const auto& marker : image.markers) {
					section.u8(marker.marker);
					section.u64(static_cast<uint64_t>(marker.payload.size()));
					section.out.write(reinterpret_cast<const char*>(marker.payload.data()),
					                  static_cast<std::streamsize>(marker.payload.size()));
				}
			}
		});
		write_section(writer, MetadataSection::kOriginalMarkers, marker_section);
	}
}

void write_jpeg_dct_shard_manifest(const JpegDctShardManifest& manifest, const std::filesystem::path& output_path) {
	std::ofstream out(output_path, std::ios::binary);
	if (!out) {
		throw std::runtime_error("failed to open JPEG DCT shard manifest output: " + output_path.string());
	}

	BinaryWriter  writer(out);
	const uint8_t magic[8] {'G', 'J', 'D', 'C', 'T', 'S', 'H', '1'};
	out.write(reinterpret_cast<const char*>(magic), sizeof(magic));
	writer.u32(manifest.version);
	writer.u16(static_cast<uint16_t>(validation_mode_id(manifest.policy)));
	writer.u32(manifest.rowgroup_vectors);
	writer.u32(manifest.rowgroups_per_shard);
	writer.u64(manifest.image_count);
	writer.u32(static_cast<uint32_t>(manifest.shards.size()));

	const auto write_string = [&](const std::string& value) {
		writer.u32(static_cast<uint32_t>(value.size()));
		out.write(value.data(), static_cast<std::streamsize>(value.size()));
	};

	for (const auto& shard : manifest.shards) {
		writer.u32(shard.shard_id);
		writer.u64(shard.first_global_image_index);
		writer.u32(shard.image_count);
		writer.u64(shard.real_row_count);
		writer.u64(shard.padding_row_count);
		writer.u64(shard.physical_row_count);
		writer.u32(shard.rowgroup_count);
		writer.u32(shard.block_group_count);
		writer.u64(shard.fls_file_size);
		writer.u64(shard.metadata_file_size);
		write_string(shard.fls_file_name);
		write_string(shard.metadata_file_name);
	}
}

void compress_jpeg_dct_to_fls(const JpegDctTable&                 table,
                              const std::filesystem::path&        fls_output_path,
                              const std::filesystem::path&        metadata_output_path,
                              const JpegDctMetadataWriterOptions& metadata_options) {
	write_jpeg_dct_fls_data(table, fls_output_path);
	write_jpeg_dct_metadata(table.metadata, metadata_output_path, metadata_options);
}

void compress_jpeg_dct_to_fls(const JpegDctTable&          table,
                              const std::filesystem::path& fls_output_path,
                              const std::filesystem::path& metadata_output_path) {
	write_jpeg_dct_fls_data(table, fls_output_path);
	write_jpeg_dct_metadata(table.metadata, metadata_output_path);
}

void compress_jpeg_dct_file_to_fls(const std::filesystem::path&        jpeg_path,
                                   const std::filesystem::path&        fls_output_path,
                                   const std::filesystem::path&        metadata_output_path,
                                   const JpegDctReaderOptions&         options,
                                   const JpegDctMetadataWriterOptions& metadata_options) {
	auto read_options = options;
	if (metadata_options.profile == JpegMetadataProfile::kPreserveOriginalMarkers) {
		read_options.capture_metadata_markers = true;
	}
	compress_jpeg_dct_to_fls(
	    read_jpeg_dct_file(jpeg_path, read_options), fls_output_path, metadata_output_path, metadata_options);
}

void compress_jpeg_dct_file_to_fls(const std::filesystem::path& jpeg_path,
                                   const std::filesystem::path& fls_output_path,
                                   const std::filesystem::path& metadata_output_path,
                                   const JpegDctReaderOptions&  options) {
	compress_jpeg_dct_to_fls(read_jpeg_dct_file(jpeg_path, options), fls_output_path, metadata_output_path);
}

void compress_jpeg_dct_dataset_to_fls(const std::vector<std::filesystem::path>& jpeg_paths,
                                      const std::filesystem::path&              fls_output_path,
                                      const std::filesystem::path&              metadata_output_path,
                                      const JpegDctReaderOptions&               options,
                                      const JpegDctMetadataWriterOptions&       metadata_options) {
	auto read_options = options;
	if (metadata_options.profile == JpegMetadataProfile::kPreserveOriginalMarkers) {
		read_options.capture_metadata_markers = true;
	}
	compress_jpeg_dct_to_fls(
	    read_jpeg_dct_dataset(jpeg_paths, read_options), fls_output_path, metadata_output_path, metadata_options);
}

void compress_jpeg_dct_dataset_to_fls(const std::vector<std::filesystem::path>& jpeg_paths,
                                      const std::filesystem::path&              fls_output_path,
                                      const std::filesystem::path&              metadata_output_path,
                                      const JpegDctReaderOptions&               options) {
	compress_jpeg_dct_to_fls(read_jpeg_dct_dataset(jpeg_paths, options), fls_output_path, metadata_output_path);
}

JpegDctShardManifest compress_jpeg_dct_dataset_to_sharded_fls(const std::vector<std::filesystem::path>& jpeg_paths,
                                                              const std::filesystem::path&              output_dir,
                                                              const JpegDctReaderOptions&               options,
                                                              const JpegDctShardOptions&                shard_options,
                                                              const JpegDctMetadataWriterOptions& metadata_options) {
	const auto effective_options = effective_shard_options(shard_options);
	if (jpeg_paths.empty()) {
		throw std::runtime_error("JPEG DCT sharded dataset requires at least one image");
	}
	if (effective_options.shard_images == 0) {
		throw std::runtime_error("JPEG DCT shard_images must be greater than zero");
	}
	if (effective_options.rowgroup_vectors == 0) {
		throw std::runtime_error("JPEG DCT rowgroup_vectors must be greater than zero");
	}
	if (effective_options.rowgroups_per_shard == 0) {
		throw std::runtime_error("JPEG DCT rowgroups_per_shard must be greater than zero");
	}

	std::filesystem::create_directories(output_dir);

	auto read_options = options;
	if (metadata_options.profile == JpegMetadataProfile::kPreserveOriginalMarkers) {
		read_options.capture_metadata_markers = true;
	}
	std::vector<DecodedImage> global_layout_images;
	global_layout_images.reserve(jpeg_paths.size());
	for (const auto& path : jpeg_paths) {
		global_layout_images.push_back(decode_jpeg_layout(path, read_options));
	}
	validate_supported_layout_options(read_options);
	validate_dataset(global_layout_images, read_options);
	const auto global_slots = normalize_component_slots(global_layout_images);

	JpegDctShardManifest manifest;
	manifest.version             = 1;
	manifest.policy              = read_options.validation_mode;
	manifest.rowgroup_vectors    = effective_options.rowgroup_vectors;
	manifest.rowgroups_per_shard = effective_options.rowgroups_per_shard;
	manifest.image_count         = jpeg_paths.size();

	for (size_t first_image = 0, shard_id = 0; first_image < jpeg_paths.size(); ++shard_id) {
		const auto max_shard_image_count = std::min(effective_options.shard_images, jpeg_paths.size() - first_image);
		const auto shard_image_count =
		    choose_shard_image_count(global_layout_images,
		                             global_slots,
		                             first_image,
		                             max_shard_image_count,
		                             read_options,
		                             effective_options);
		const auto                         shard_begin = jpeg_paths.begin() + static_cast<std::ptrdiff_t>(first_image);
		const auto                         shard_end   = shard_begin + static_cast<std::ptrdiff_t>(shard_image_count);
		std::vector<std::filesystem::path> shard_paths(shard_begin, shard_end);

		std::vector<DecodedImage> shard_images;
		shard_images.reserve(shard_paths.size());
		for (const auto& path : shard_paths) {
			shard_images.push_back(decode_jpeg_coefficients(path, read_options));
		}
		auto table = make_dataset_table(std::move(shard_images), read_options, &global_slots);
		table.rowgroup_n_tuples =
		    make_block_group_aligned_rowgroups(table.metadata, table.row_count, effective_options.rowgroup_vectors);
		if (table.rowgroup_n_tuples.size() > effective_options.rowgroups_per_shard) {
			std::ostringstream msg;
			msg << "JPEG DCT shard produced " << table.rowgroup_n_tuples.size()
			    << " rowgroups after layout estimation selected " << shard_image_count
			    << " images; configured maximum is " << effective_options.rowgroups_per_shard;
			throw std::runtime_error(msg.str());
		}

		const auto fls_name      = shard_file_name(static_cast<uint32_t>(shard_id), ".fls");
		const auto metadata_name = shard_file_name(static_cast<uint32_t>(shard_id), ".meta.bin");
		const auto fls_path      = output_dir / fls_name;
		const auto metadata_path = output_dir / metadata_name;

		fastlanes::MemoryTableOptions memory_options;
		memory_options.n_vectors_per_rowgroup = effective_options.rowgroup_vectors;
		memory_options.rowgroup_n_tuples =
		    std::span<const fastlanes::n_t>(table.rowgroup_n_tuples.data(), table.rowgroup_n_tuples.size());
		write_jpeg_dct_fls_data(table, fls_path, memory_options, true);
		write_jpeg_dct_metadata(table.metadata, metadata_path, metadata_options);

		JpegDctShardManifestEntry entry;
		entry.shard_id                 = static_cast<uint32_t>(shard_id);
		entry.first_global_image_index = first_image;
		entry.image_count              = static_cast<uint32_t>(shard_image_count);
		entry.real_row_count           = table.real_row_count;
		entry.padding_row_count        = table.padding_row_count;
		entry.physical_row_count       = table.row_count;
		entry.rowgroup_count           = static_cast<uint32_t>(table.rowgroup_n_tuples.size());
		entry.block_group_count        = static_cast<uint32_t>(table.block_group_count);
		entry.fls_file_size            = std::filesystem::file_size(fls_path);
		entry.metadata_file_size       = std::filesystem::file_size(metadata_path);
		entry.fls_file_name            = fls_name;
		entry.metadata_file_name       = metadata_name;
		manifest.shards.push_back(std::move(entry));
		first_image += shard_image_count;
	}

	write_jpeg_dct_shard_manifest(manifest, output_dir / "manifest.bin");
	return manifest;
}

struct JpegDctShardDatasetReader::Impl {
	struct ShardState {
		JpegDctShardManifestEntry entry;
		std::filesystem::path     fls_path;
		std::filesystem::path     metadata_path;
		JpegDctDatasetMetadata    metadata;
	};

	explicit Impl(const std::filesystem::path& manifest_path)
	    : root_dir(manifest_path.parent_path())
	    , manifest(read_jpeg_dct_shard_manifest_file(manifest_path)) {
		shards.reserve(manifest.shards.size());
		for (const auto& entry : manifest.shards) {
			ShardState state;
			state.entry         = entry;
			state.fls_path      = root_dir / entry.fls_file_name;
			state.metadata_path = root_dir / entry.metadata_file_name;
			state.metadata      = read_metadata(state.metadata_path);
			shards.push_back(std::move(state));
		}
	}

	static JpegDctDatasetMetadata read_metadata(const std::filesystem::path& path) {
		BinaryReader reader(read_binary_file(path));
		expect_magic(reader, {'G', 'J', 'D', 'C', 'T', 'M', 'D', '3'}, "JPEG DCT metadata");

		JpegDctDatasetMetadata      metadata;
		[[maybe_unused]] const auto version         = reader.u16();
		[[maybe_unused]] const auto profile         = reader.u16();
		metadata.row_ordering                       = row_ordering_from_id(reader.u16());
		metadata.validation_mode                    = validation_mode_from_id(reader.u16());
		const auto layout_flags                     = reader.u16();
		metadata.zigzag_columns                     = (layout_flags & 1U) != 0;
		metadata.z_curve_block_order                = (layout_flags & 2U) != 0;
		metadata.image_count                        = reader.u64();
		const auto                  component_count = reader.u32();
		[[maybe_unused]] const auto profile_count   = reader.u32();
		metadata.compression_partition_policy       = partition_policy_from_id(reader.u16());
		metadata.coefficient_encoding               = coefficient_encoding_from_id(reader.u16());

		while (!reader.eof()) {
			const auto section_id   = static_cast<MetadataSection>(reader.u16());
			const auto payload_size = reader.u64();
			if (payload_size > std::numeric_limits<size_t>::max()) {
				throw std::runtime_error("JPEG DCT metadata section is too large");
			}
			BinaryReader section(reader.bytes(static_cast<size_t>(payload_size)));
			switch (section_id) {
			case MetadataSection::kComponentGrid:
				while (!section.eof()) {
					JpegComponentMetadata component;
					component.semantic_slot_id        = section.u32();
					component.component_index         = section.u32();
					component.component_id            = section.i32();
					component.width_in_blocks         = section.u32();
					component.height_in_blocks        = section.u32();
					component.padded_width_in_blocks  = section.u32();
					component.padded_height_in_blocks = section.u32();
					metadata.semantic_components.push_back(component);
				}
				break;
			case MetadataSection::kPerImageGrid:
				metadata.images.reserve(metadata.image_count);
				for (uint64_t image_idx = 0; image_idx < metadata.image_count; ++image_idx) {
					JpegImageMetadata image;
					image.image_width      = section.u32();
					image.image_height     = section.u32();
					image.data_precision   = section.u8();
					image.jpeg_color_space = section.i32();
					image.warning_count    = section.u32();
					image.components.reserve(component_count);
					for (uint32_t component_idx = 0; component_idx < component_count; ++component_idx) {
						JpegComponentMetadata component;
						component.present               = section.u8() != 0;
						component.semantic_slot_id      = section.u32();
						component.local_component_index = section.u32();
						component.component_id          = section.i32();
						component.width_in_blocks       = section.u32();
						component.height_in_blocks      = section.u32();
						component.encoding_profile_id   = section.u32();
						image.components.push_back(component);
					}
					metadata.images.push_back(std::move(image));
				}
				break;
			case MetadataSection::kBlockGroupIndex:
				while (!section.eof()) {
					JpegDctBlockGroupIndex group;
					group.semantic_slot_id = section.u32();
					group.z_order_index    = section.u32();
					group.block_x          = section.u32();
					group.block_y          = section.u32();
					group.row_start        = section.u64();
					group.row_count        = section.u32();
					if (section.remaining() >= 8) {
						group.fls_rowgroup_index    = section.u32();
						group.row_start_in_rowgroup = section.u32();
					}
					metadata.block_group_index.push_back(group);
				}
				break;
			case MetadataSection::kEncodingProfiles:
			case MetadataSection::kReconstructableImageInfo:
			case MetadataSection::kOriginalMarkers:
				break;
			}
		}
		return metadata;
	}

	const ShardState& shard_for_global_image(const uint32_t global_image_index) const {
		for (const auto& shard : shards) {
			const auto first = shard.entry.first_global_image_index;
			const auto last  = first + shard.entry.image_count;
			if (global_image_index >= first && global_image_index < last) {
				return shard;
			}
		}
		throw std::runtime_error("JPEG DCT global image index is outside the shard manifest");
	}

	const ShardState& shard_by_id(const uint32_t shard_id) const {
		for (const auto& shard : shards) {
			if (shard.entry.shard_id == shard_id) {
				return shard;
			}
		}
		throw std::runtime_error("JPEG DCT shard id is outside the shard manifest");
	}

	static const JpegDctBlockGroupIndex& find_group(const JpegDctDatasetMetadata& metadata,
	                                                const uint32_t                semantic_slot_id,
	                                                const uint32_t                block_x,
	                                                const uint32_t                block_y) {
		const auto* group = find_group_or_null(metadata, semantic_slot_id, block_x, block_y);
		if (group == nullptr) {
			throw std::runtime_error("JPEG DCT block group was not found in shard metadata");
		}
		return *group;
	}

	static const JpegDctBlockGroupIndex* find_group_or_null(const JpegDctDatasetMetadata& metadata,
	                                                        const uint32_t                semantic_slot_id,
	                                                        const uint32_t                block_x,
	                                                        const uint32_t                block_y) {
		for (const auto& group : metadata.block_group_index) {
			if (group.semantic_slot_id == semantic_slot_id && group.block_x == block_x && group.block_y == block_y) {
				return &group;
			}
		}
		return nullptr;
	}

	static bool image_has_block(const JpegImageMetadata& image,
	                            const uint32_t           semantic_slot_id,
	                            const uint32_t           block_x,
	                            const uint32_t           block_y) {
		for (const auto& component : image.components) {
			if (component.semantic_slot_id == semantic_slot_id) {
				return component.present && block_x < component.width_in_blocks && block_y < component.height_in_blocks;
			}
		}
		return false;
	}

	std::filesystem::path   root_dir;
	JpegDctShardManifest    manifest;
	std::vector<ShardState> shards;
};

JpegDctShardDatasetReader::JpegDctShardDatasetReader(const std::filesystem::path& manifest_path)
    : impl_(std::make_unique<Impl>(manifest_path)) {
}

JpegDctShardDatasetReader::~JpegDctShardDatasetReader() = default;

JpegDctShardDatasetReader::JpegDctShardDatasetReader(JpegDctShardDatasetReader&&) noexcept = default;

JpegDctShardDatasetReader& JpegDctShardDatasetReader::operator=(JpegDctShardDatasetReader&&) noexcept = default;

JpegDctRowRef JpegDctShardDatasetReader::LocateRow(const uint32_t global_image_index,
                                                   const uint32_t semantic_slot_id,
                                                   const uint32_t block_x,
                                                   const uint32_t block_y) {
	const auto& shard             = impl_->shard_for_global_image(global_image_index);
	const auto  local_image_index = static_cast<uint32_t>(global_image_index - shard.entry.first_global_image_index);
	if (local_image_index >= shard.metadata.images.size()) {
		throw std::runtime_error("JPEG DCT shard metadata does not contain the requested local image");
	}

	JpegDctRowRef ref;
	ref.shard_id              = shard.entry.shard_id;
	ref.local_image_index     = local_image_index;
	ref.semantic_slot_id      = semantic_slot_id;
	ref.block_x               = block_x;
	ref.block_y               = block_y;

	const auto has_target =
	    Impl::image_has_block(shard.metadata.images[local_image_index], semantic_slot_id, block_x, block_y);
	const auto* group = Impl::find_group_or_null(shard.metadata, semantic_slot_id, block_x, block_y);
	if (group == nullptr) {
		if (shard.metadata.validation_mode == JpegDatasetValidationMode::kRaggedBlockMajor && !has_target) {
			return ref;
		}
		throw std::runtime_error("JPEG DCT block group was not found in shard metadata");
	}
	ref.fls_rowgroup_index    = group->fls_rowgroup_index;
	ref.row_start_in_rowgroup = group->row_start_in_rowgroup;

	if (shard.metadata.validation_mode == JpegDatasetValidationMode::kRaggedBlockMajor) {
		uint32_t rank = 0;
		for (uint32_t image_idx = 0; image_idx < local_image_index; ++image_idx) {
			if (Impl::image_has_block(shard.metadata.images[image_idx], semantic_slot_id, block_x, block_y)) {
				++rank;
			}
		}
		ref.row_offset_in_block_group = rank;
		ref.present                   = has_target;
	} else {
		ref.row_offset_in_block_group = local_image_index;
		ref.present =
		    has_target || shard.metadata.validation_mode == JpegDatasetValidationMode::kPadToMaxComponentGrids;
	}
	ref.physical_row_index = group->row_start + ref.row_offset_in_block_group;
	return ref;
}

JpegDctBlockGroup JpegDctShardDatasetReader::ReadBlockGroup(const uint32_t shard_id,
                                                            const uint32_t semantic_slot_id,
                                                            const uint32_t block_x,
                                                            const uint32_t block_y) {
	const auto& shard = impl_->shard_by_id(shard_id);
	const auto& group = Impl::find_group(shard.metadata, semantic_slot_id, block_x, block_y);

	fastlanes::Connection connection;
	auto                  table_reader    = connection.read_fls(shard.fls_path);
	auto                  rowgroup_reader = table_reader->get_rowgroup_reader(group.fls_rowgroup_index);
	auto                  rowgroup        = rowgroup_reader->materialize();
	if (rowgroup->internal_rowgroup.size() < 64) {
		throw std::runtime_error("JPEG DCT FLS shard has fewer than 64 coefficient columns");
	}

	JpegDctBlockGroup block_group;
	block_group.index = group;
	block_group.rows.reserve(group.row_count);
	for (uint32_t row_idx = 0; row_idx < group.row_count; ++row_idx) {
		const auto            materialized_row_idx = static_cast<size_t>(group.row_start_in_rowgroup) + row_idx;
		JpegDctCoefficientRow row {};
		for (size_t col_idx = 0; col_idx < row.size(); ++col_idx) {
			row[col_idx] = coefficient_value(rowgroup->internal_rowgroup[col_idx], materialized_row_idx);
		}
		block_group.rows.push_back(row);
	}
	return block_group;
}

MaterializedJpegDctImage JpegDctShardDatasetReader::MaterializeImageDct(const uint32_t global_image_index) {
	const auto& shard = impl_->shard_for_global_image(global_image_index);
	fastlanes::Connection connection;
	auto                  table_reader = connection.read_fls(shard.fls_path);
	std::unordered_map<uint32_t, fastlanes::up<fastlanes::Rowgroup>> rowgroups;

	const auto materialized_rowgroup = [&](const uint32_t rowgroup_index) -> fastlanes::Rowgroup& {
		auto [it, inserted] = rowgroups.try_emplace(rowgroup_index);
		if (inserted) {
			auto rowgroup_reader = table_reader->get_rowgroup_reader(rowgroup_index);
			it->second           = rowgroup_reader->materialize();
			if (it->second->internal_rowgroup.size() < 64) {
				throw std::runtime_error("JPEG DCT FLS shard has fewer than 64 coefficient columns");
			}
		}
		return *it->second;
	};

	MaterializedJpegDctImage image;
	image.global_image_index = global_image_index;
	for (const auto& group : shard.metadata.block_group_index) {
		auto ref = LocateRow(global_image_index, group.semantic_slot_id, group.block_x, group.block_y);
		if (!ref.present || ref.row_offset_in_block_group >= group.row_count) {
			continue;
		}
		const auto& rowgroup             = materialized_rowgroup(ref.fls_rowgroup_index);
			const auto materialized_row_idx =
			    static_cast<size_t>(ref.row_start_in_rowgroup) + ref.row_offset_in_block_group;
		MaterializedJpegDctBlock block;
		block.semantic_slot_id = group.semantic_slot_id;
		block.block_x          = group.block_x;
		block.block_y          = group.block_y;
		for (size_t col_idx = 0; col_idx < block.coefficients.size(); ++col_idx) {
			block.coefficients[col_idx] = coefficient_value(rowgroup.internal_rowgroup[col_idx], materialized_row_idx);
		}
		image.blocks.push_back(block);
	}
	return image;
}

} // namespace galp::jpeg
