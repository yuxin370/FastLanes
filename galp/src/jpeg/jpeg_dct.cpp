#include "galp/jpeg_dct.hpp"
#include "fls/connection.hpp"
#include "fls/reader/rowgroup_reader.hpp"
#include "fls/table/memory_table.hpp"
#include "fls/table/rowgroup.hpp"
#include "jpeg/jpeg_dct_device.cuh"
#include "jpeg/jpeg_dct_order.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csetjmp>
#include <cstdio>
#include <deque>
#include <future>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <jpeglib.h>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace galp::jpeg {

bool parse_jpeg_dct_coefficient_selection(const std::string_view spec, JpegDctCoefficientSelection& selection) {
	JpegDctCoefficientSelection parsed;
	if (spec == "all") {
		selection = std::move(parsed);
		return true;
	}

	const auto parse_coeff = [](const std::string_view token, uint8_t& out) -> bool {
		if (token.empty()) {
			return false;
		}
		try {
			size_t     parsed_chars = 0;
			const auto value        = std::stoull(std::string(token), &parsed_chars);
			if (parsed_chars != token.size() || value >= detail::kJpegDctCoefficientCount) {
				return false;
			}
			out = static_cast<uint8_t>(value);
			return true;
		} catch (...) { return false; }
	};

	constexpr std::string_view first_prefix = "first:";
	constexpr std::string_view list_prefix  = "list:";
	if (spec.substr(0, first_prefix.size()) == first_prefix) {
		size_t     count    = 0;
		const auto count_sv = spec.substr(first_prefix.size());
		try {
			size_t parsed_chars = 0;
			count               = std::stoull(std::string(count_sv), &parsed_chars);
			if (parsed_chars != count_sv.size()) {
				return false;
			}
		} catch (...) { return false; }
		if (count == 0 || count > detail::kJpegDctCoefficientCount) {
			return false;
		}
		parsed.coefficients.reserve(count);
		for (size_t coeff = 0; coeff < count; ++coeff) {
			parsed.coefficients.push_back(static_cast<uint8_t>(coeff));
		}
		selection = std::move(parsed);
		return true;
	}

	if (spec.substr(0, list_prefix.size()) == list_prefix) {
		auto list = spec.substr(list_prefix.size());
		if (list.empty()) {
			return false;
		}
		while (!list.empty()) {
			const auto comma = list.find(',');
			const auto token = comma == std::string_view::npos ? list : list.substr(0, comma);
			uint8_t    coeff = 0;
			if (!parse_coeff(token, coeff)) {
				return false;
			}
			if (std::find(parsed.coefficients.begin(), parsed.coefficients.end(), coeff) != parsed.coefficients.end()) {
				return false;
			}
			parsed.coefficients.push_back(coeff);
			if (comma == std::string_view::npos) {
				break;
			}
			list = list.substr(comma + 1);
		}
		selection = std::move(parsed);
		return true;
	}

	return false;
}

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

template <typename Fn>
std::vector<DecodedImage> decode_jpeg_images_parallel(const std::vector<std::filesystem::path>& paths,
                                                      const JpegDctReaderOptions&               options,
                                                      const size_t                              requested_threads,
                                                      Fn&&                                      decode) {
	if (paths.empty()) {
		return {};
	}
	const size_t thread_count = std::max<size_t>(1, std::min(requested_threads, paths.size()));
	if (thread_count == 1) {
		std::vector<DecodedImage> images;
		images.reserve(paths.size());
		for (const auto& path : paths) {
			images.push_back(decode(path, options));
		}
		return images;
	}

	std::vector<DecodedImage> images(paths.size());
	std::vector<std::future<void>> futures;
	futures.reserve(thread_count);
	for (size_t worker = 0; worker < thread_count; ++worker) {
		futures.push_back(std::async(std::launch::async, [&, worker] {
			for (size_t index = worker; index < paths.size(); index += thread_count) {
				images[index] = decode(paths[index], options);
			}
		}));
	}
	for (auto& future : futures) {
		future.get();
	}
	return images;
}

void append_row(JpegDctTable& table, const DctRow& row) {
	for (size_t col = 0; col < row.size(); ++col) {
		table.columns[col].push_back(row[col]);
	}
	++table.row_count;
	++table.real_row_count;
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
			append_row(table, row);
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

void validate_dataset(const std::vector<DecodedImage>& images, const JpegDctReaderOptions& /*options*/) {
	if (images.empty()) {
		throw std::runtime_error("JPEG DCT dataset requires at least one image");
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
					append_row(table, *row);
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

// Metadata and shard manifests retain a reserved validation-mode slot for backward compatibility. New files always
// write the ragged id, and readers ignore older strict/padded ids because ragged is now the only supported layout.
constexpr uint16_t kRaggedValidationModeOnDiskId = 2;

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

// Precomputed data for shard-boundary sizing. Building the block orders and per-image slot grid dimensions once (rather
// than re-deriving them for every binary-search probe) turns shard sizing from O(slots * global_max_grid * images) per
// probe into O(images + slots * global_max_grid). For large heterogeneous datasets (e.g. ImageNet, whose largest image
// inflates the global max grid) the old cost dominated the whole shard writer and looked like a hang.
struct ShardSizingContext {
	std::vector<std::vector<detail::MortonBlockCoord>> block_orders; // [slot]
	std::vector<std::vector<uint32_t>>                 slot_width;   // [slot][image] (0 when component absent)
	std::vector<std::vector<uint32_t>>                 slot_height;  // [slot][image] (0 when component absent)
	std::vector<uint32_t>                              max_width;    // [slot]
	std::vector<uint32_t>                              max_height;   // [slot]
	uint64_t                                           target_rows = 0;
};

ShardSizingContext build_shard_sizing_context(const std::vector<DecodedImage>&  layout_images,
                                              const std::vector<ComponentSlot>& global_slots,
                                              const JpegDctReaderOptions&       options,
                                              const uint32_t                    rowgroup_vectors) {
	ShardSizingContext ctx;
	ctx.target_rows  = static_cast<uint64_t>(rowgroup_vectors) * fastlanes::CFG::VEC_SZ;
	const size_t n_slots = global_slots.size();
	ctx.block_orders.resize(n_slots);
	ctx.slot_width.assign(n_slots, std::vector<uint32_t>(layout_images.size(), 0));
	ctx.slot_height.assign(n_slots, std::vector<uint32_t>(layout_images.size(), 0));
	ctx.max_width.resize(n_slots);
	ctx.max_height.resize(n_slots);
	for (size_t slot_idx = 0; slot_idx < n_slots; ++slot_idx) {
		const auto& slot        = global_slots[slot_idx];
		ctx.max_width[slot_idx]  = slot.max_width_in_blocks;
		ctx.max_height[slot_idx] = slot.max_height_in_blocks;
		ctx.block_orders[slot_idx] =
		    detail::make_block_order(slot.max_width_in_blocks, slot.max_height_in_blocks, options.use_z_curve_block_order);
		for (size_t image_idx = 0; image_idx < layout_images.size(); ++image_idx) {
			const auto* component = find_component_for_slot(layout_images[image_idx], slot);
			if (component != nullptr) {
				ctx.slot_width[slot_idx][image_idx]  = component->metadata.width_in_blocks;
				ctx.slot_height[slot_idx][image_idx] = component->metadata.height_in_blocks;
			}
		}
	}
	return ctx;
}

// Mirrors the greedy packing in make_block_group_aligned_rowgroups. For each slot the per-block-group row count is the
// number of images in [first_image, first_image + image_count) whose component reaches that block. We obtain those
// counts in O(1) via a 2D suffix histogram over (width_in_blocks, height_in_blocks): group_row_count(x, y) is the number
// of images with width > x and height > y. `scratch` is reused across probes to avoid per-call allocation.
size_t estimate_shard_rowgroup_count(const ShardSizingContext& ctx,
                                     const size_t              first_image,
                                     const size_t              image_count,
                                     std::vector<uint64_t>&    scratch) {
	const uint64_t target_rows           = ctx.target_rows;
	size_t         rowgroup_count        = 0;
	uint64_t       current_rowgroup_rows = 0;

	const auto pack_group = [&](const uint64_t group_row_count) {
		if (group_row_count == 0) {
			return;
		}
		if (current_rowgroup_rows != 0 && current_rowgroup_rows + group_row_count > target_rows) {
			++rowgroup_count;
			current_rowgroup_rows = 0;
		}
		current_rowgroup_rows += group_row_count;
	};

	for (size_t slot_idx = 0; slot_idx < ctx.block_orders.size(); ++slot_idx) {
		const uint32_t max_w  = ctx.max_width[slot_idx];
		const uint32_t max_h  = ctx.max_height[slot_idx];
		const size_t   stride = static_cast<size_t>(max_h) + 2;
		scratch.assign((static_cast<size_t>(max_w) + 2) * stride, 0);
		const auto& widths  = ctx.slot_width[slot_idx];
		const auto& heights = ctx.slot_height[slot_idx];
		for (size_t image_idx = first_image; image_idx < first_image + image_count; ++image_idx) {
			const uint32_t w = widths[image_idx];
			const uint32_t h = heights[image_idx];
			if (w == 0 || h == 0) {
				continue;
			}
			const size_t cw = w > max_w ? max_w : w;
			const size_t ch = h > max_h ? max_h : h;
			++scratch[cw * stride + ch];
		}
		// Suffix sum so that scratch[x][y] becomes the number of images with width >= x and height >= y.
		for (size_t x = static_cast<size_t>(max_w) + 1; x-- > 0;) {
			for (size_t y = static_cast<size_t>(max_h) + 1; y-- > 0;) {
				scratch[x * stride + y] += scratch[(x + 1) * stride + y] + scratch[x * stride + (y + 1)] -
				                           scratch[(x + 1) * stride + (y + 1)];
			}
		}
		for (const auto& block_coord : ctx.block_orders[slot_idx]) {
			// group_row_count(x, y) = images with width > x and height > y = suffix at (x + 1, y + 1).
			pack_group(scratch[(static_cast<size_t>(block_coord.x) + 1) * stride + (block_coord.y + 1)]);
		}
	}
	if (current_rowgroup_rows != 0) {
		++rowgroup_count;
	}
	return rowgroup_count;
}

size_t choose_shard_image_count(const ShardSizingContext&  ctx,
                                const size_t               first_image,
                                const size_t               max_image_count,
                                const JpegDctShardOptions& shard_options,
                                std::vector<uint64_t>&     scratch) {
	if (estimate_shard_rowgroup_count(ctx, first_image, max_image_count, scratch) <=
	    shard_options.rowgroups_per_shard) {
		return max_image_count;
	}

	size_t lo = 1;
	size_t hi = max_image_count;
	while (lo < hi) {
		const auto mid = lo + (hi - lo + 1) / 2;
		if (estimate_shard_rowgroup_count(ctx, first_image, mid, scratch) <= shard_options.rowgroups_per_shard) {
			lo = mid;
		} else {
			hi = mid - 1;
		}
	}

	if (estimate_shard_rowgroup_count(ctx, first_image, lo, scratch) > shard_options.rowgroups_per_shard) {
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
	constexpr size_t   kBalancedShardImages       = 8192;
	constexpr uint32_t kBalancedRowgroupVectors   = 128;
	constexpr uint32_t kBalancedRowgroupsPerShard = 256;
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
	static_cast<void>(reader.u16()); // reserved validation-mode id (always ragged)
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
	writer.u16(kRaggedValidationModeOnDiskId);
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
	if (metadata.row_ordering == JpegDctRowOrdering::kDatasetComponentMajorBlockMajorImageMinor) {
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
	const uint16_t validation_mode = kRaggedValidationModeOnDiskId;
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
	writer.u16(kRaggedValidationModeOnDiskId);
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
	if (effective_options.threads == 0) {
		throw std::runtime_error("JPEG DCT shard threads must be greater than zero");
	}
	if (effective_options.shard_workers == 0) {
		throw std::runtime_error("JPEG DCT shard workers must be greater than zero");
	}

	std::filesystem::create_directories(output_dir);

	auto read_options = options;
	if (metadata_options.profile == JpegMetadataProfile::kPreserveOriginalMarkers) {
		read_options.capture_metadata_markers = true;
	}
	auto global_layout_images =
	    decode_jpeg_images_parallel(jpeg_paths, read_options, effective_options.threads, decode_jpeg_layout);
	validate_supported_layout_options(read_options);
	validate_dataset(global_layout_images, read_options);
	const auto global_slots = normalize_component_slots(global_layout_images);

	JpegDctShardManifest manifest;
	manifest.version             = 1;
	manifest.rowgroup_vectors    = effective_options.rowgroup_vectors;
	manifest.rowgroups_per_shard = effective_options.rowgroups_per_shard;
	manifest.image_count         = jpeg_paths.size();

	struct ShardWorkItem {
		size_t first_image = 0;
		size_t image_count = 0;
	};
	std::vector<ShardWorkItem> shard_work_items;
	const auto sizing_ctx = build_shard_sizing_context(
	    global_layout_images, global_slots, read_options, effective_options.rowgroup_vectors);
	std::vector<uint64_t> sizing_scratch;
	for (size_t first_image = 0; first_image < jpeg_paths.size();) {
		const auto max_shard_image_count = std::min(effective_options.shard_images, jpeg_paths.size() - first_image);
		const auto shard_image_count =
		    choose_shard_image_count(sizing_ctx, first_image, max_shard_image_count, effective_options, sizing_scratch);
		shard_work_items.push_back(ShardWorkItem {first_image, shard_image_count});
		first_image += shard_image_count;
	}

	std::vector<JpegDctShardManifestEntry> shard_entries(shard_work_items.size());
	const auto process_shard = [&](const size_t shard_id) {
		const auto& work = shard_work_items[shard_id];
		const auto  first_image = work.first_image;
		const auto  shard_image_count = work.image_count;
		const auto                         shard_begin = jpeg_paths.begin() + static_cast<std::ptrdiff_t>(first_image);
		const auto                         shard_end   = shard_begin + static_cast<std::ptrdiff_t>(shard_image_count);
		std::vector<std::filesystem::path> shard_paths(shard_begin, shard_end);

		auto shard_images =
		    decode_jpeg_images_parallel(shard_paths, read_options, effective_options.threads, decode_jpeg_coefficients);
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
		shard_entries[shard_id] = std::move(entry);
	};

	const size_t shard_worker_count = std::max<size_t>(1, std::min(effective_options.shard_workers, shard_work_items.size()));
	if (shard_worker_count == 1) {
		for (size_t shard_id = 0; shard_id < shard_work_items.size(); ++shard_id) {
			process_shard(shard_id);
		}
	} else {
		std::vector<std::future<void>> futures;
		futures.reserve(shard_worker_count);
		for (size_t worker = 0; worker < shard_worker_count; ++worker) {
			futures.push_back(std::async(std::launch::async, [&, worker] {
				for (size_t shard_id = worker; shard_id < shard_work_items.size(); shard_id += shard_worker_count) {
					process_shard(shard_id);
				}
			}));
		}
		for (auto& future : futures) {
			future.get();
		}
	}

	manifest.shards = std::move(shard_entries);
	write_jpeg_dct_shard_manifest(manifest, output_dir / "manifest.bin");
	return manifest;
}

struct JpegDctShardDatasetReader::Impl {
	struct BlockGroupKey {
		uint32_t semantic_slot_id = 0;
		uint32_t block_x          = 0;
		uint32_t block_y          = 0;

		bool operator==(const BlockGroupKey& other) const noexcept {
			return semantic_slot_id == other.semantic_slot_id && block_x == other.block_x && block_y == other.block_y;
		}
	};

	struct BlockGroupKeyHash {
		size_t operator()(const BlockGroupKey& key) const noexcept {
			uint64_t h = static_cast<uint64_t>(key.semantic_slot_id) * 0x9e3779b185ebca87ULL;
			h ^= static_cast<uint64_t>(key.block_x) + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
			h ^= static_cast<uint64_t>(key.block_y) + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
			return static_cast<size_t>(h);
		}
	};

	struct RankCursorKey {
		uint32_t shard_id    = 0;
		size_t   group_index = 0;

		bool operator==(const RankCursorKey& other) const noexcept {
			return shard_id == other.shard_id && group_index == other.group_index;
		}
	};

	struct RankCursorKeyHash {
		size_t operator()(const RankCursorKey& key) const noexcept {
			auto h = static_cast<uint64_t>(key.shard_id) * 0x9e3779b185ebca87ULL;
			h ^= static_cast<uint64_t>(key.group_index) + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
			return static_cast<size_t>(h);
		}
	};

	struct RankCursor {
		uint32_t next_local_image    = 0;
		uint32_t present_before_next = 0;
	};

	struct ShardState {
		JpegDctShardManifestEntry                                    entry;
		std::filesystem::path                                        fls_path;
		std::filesystem::path                                        metadata_path;
		JpegDctDatasetMetadata                                       metadata;
		std::vector<uint64_t>                                        rowgroup_n_tuples;
		std::unordered_map<BlockGroupKey, size_t, BlockGroupKeyHash> block_group_lookup;
	};

	explicit Impl(const std::filesystem::path& manifest_path)
	    : root_dir(manifest_path.parent_path())
	    , manifest(read_jpeg_dct_shard_manifest_file(manifest_path)) {
		shards.reserve(manifest.shards.size());
		for (const auto& entry : manifest.shards) {
			ShardState state;
			state.entry             = entry;
			state.fls_path          = root_dir / entry.fls_file_name;
			state.metadata_path     = root_dir / entry.metadata_file_name;
			state.metadata          = read_metadata(state.metadata_path);
			state.rowgroup_n_tuples = derive_rowgroup_n_tuples(state.metadata, entry.rowgroup_count);
			state.block_group_lookup.reserve(state.metadata.block_group_index.size());
			for (size_t group_idx = 0; group_idx < state.metadata.block_group_index.size(); ++group_idx) {
				const auto& group = state.metadata.block_group_index[group_idx];
				state.block_group_lookup.emplace(BlockGroupKey {group.semantic_slot_id, group.block_x, group.block_y},
				                                 group_idx);
			}
			shards.push_back(std::move(state));
		}
	}

	static std::vector<uint64_t> derive_rowgroup_n_tuples(const JpegDctDatasetMetadata& metadata,
	                                                      const uint32_t                rowgroup_count) {
		std::vector<uint64_t> rowgroup_n_tuples(rowgroup_count, 0);
		for (const auto& group : metadata.block_group_index) {
			if (group.fls_rowgroup_index >= rowgroup_n_tuples.size()) {
				throw std::runtime_error("JPEG DCT metadata rowgroup index exceeds shard manifest rowgroup count");
			}
			const uint64_t row_end =
			    static_cast<uint64_t>(group.row_start_in_rowgroup) + static_cast<uint64_t>(group.row_count);
			rowgroup_n_tuples[group.fls_rowgroup_index] =
			    std::max(rowgroup_n_tuples[group.fls_rowgroup_index], row_end);
		}
		return rowgroup_n_tuples;
	}

	static size_t row_count_to_vector_count(const uint64_t row_count) {
		return static_cast<size_t>((row_count + fastlanes::CFG::VEC_SZ - 1U) / fastlanes::CFG::VEC_SZ);
	}

	static JpegDctDatasetMetadata read_metadata(const std::filesystem::path& path) {
		BinaryReader reader(read_binary_file(path));
		expect_magic(reader, {'G', 'J', 'D', 'C', 'T', 'M', 'D', '3'}, "JPEG DCT metadata");

		JpegDctDatasetMetadata      metadata;
		[[maybe_unused]] const auto version         = reader.u16();
		[[maybe_unused]] const auto profile         = reader.u16();
		metadata.row_ordering                       = row_ordering_from_id(reader.u16());
		static_cast<void>(reader.u16()); // reserved validation-mode id (always ragged)
		const auto layout_flags                     = reader.u16();
		metadata.zigzag_columns                     = (layout_flags & 1U) != 0;
		metadata.z_curve_block_order                = (layout_flags & 2U) != 0;
		metadata.image_count                        = reader.u64();
		const auto                  component_count = reader.u32();
		const auto                  profile_count   = reader.u32();
		metadata.compression_partition_policy       = partition_policy_from_id(reader.u16());
		metadata.coefficient_encoding               = coefficient_encoding_from_id(reader.u16());
		metadata.encoding_profiles.reserve(profile_count);

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
				while (!section.eof()) {
					JpegEncodingProfileMetadata encoding_profile;
					encoding_profile.profile_id              = section.u32();
					encoding_profile.h_samp_factor           = section.i32();
					encoding_profile.v_samp_factor           = section.i32();
					encoding_profile.quant_tbl_no            = section.i32();
					encoding_profile.quant_table_fingerprint = section.u64();
					for (auto& value : encoding_profile.quant_table_values) {
						value = section.u16();
					}
					metadata.encoding_profiles.push_back(encoding_profile);
				}
				break;
			case MetadataSection::kReconstructableImageInfo:
			{
				const auto reconstruct_image_count = section.u64();
				if (reconstruct_image_count != metadata.image_count) {
					throw std::runtime_error("JPEG DCT reconstructable metadata image count mismatch");
				}
				if (metadata.images.empty()) {
					metadata.images.resize(static_cast<size_t>(metadata.image_count));
				}
				for (uint64_t image_idx = 0; image_idx < reconstruct_image_count; ++image_idx) {
					auto& image = metadata.images.at(static_cast<size_t>(image_idx));
					image.image_width      = section.u32();
					image.image_height     = section.u32();
					image.data_precision   = section.u8();
					image.jpeg_color_space = section.i32();
					const auto quant_table_count = section.u32();
					image.quant_tables.clear();
					image.quant_tables.reserve(quant_table_count);
					for (uint32_t table_idx = 0; table_idx < quant_table_count; ++table_idx) {
						JpegQuantTableMetadata table;
						table.table_id = section.u8();
						for (auto& value : table.values) {
							value = section.u16();
						}
						image.quant_tables.push_back(table);
					}
				}
				break;
			}
			case MetadataSection::kOriginalMarkers:
				break;
			}
		}
		for (auto& image : metadata.images) {
			for (auto& component : image.components) {
				if (component.encoding_profile_id == std::numeric_limits<uint32_t>::max()) {
					continue;
				}
				const auto found =
				    std::find_if(metadata.encoding_profiles.begin(),
				                 metadata.encoding_profiles.end(),
				                 [&component](const JpegEncodingProfileMetadata& profile) {
					                 return profile.profile_id == component.encoding_profile_id;
				                 });
				if (found == metadata.encoding_profiles.end()) {
					continue;
					}
					component.h_samp_factor           = found->h_samp_factor;
					component.v_samp_factor           = found->v_samp_factor;
					component.quant_tbl_no            = found->quant_tbl_no;
					component.quant_table_fingerprint = found->quant_table_fingerprint;
					if (found->quant_tbl_no >= 0 && found->quant_tbl_no <= std::numeric_limits<uint8_t>::max()) {
						const auto table_id = static_cast<uint8_t>(found->quant_tbl_no);
						const auto table_found =
						    std::find_if(image.quant_tables.begin(),
						                 image.quant_tables.end(),
						                 [table_id](const JpegQuantTableMetadata& table) {
							                 return table.table_id == table_id;
						                 });
						if (table_found == image.quant_tables.end()) {
							JpegQuantTableMetadata table;
							table.table_id = table_id;
							table.values   = found->quant_table_values;
							image.quant_tables.push_back(table);
						}
					}
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

	static const JpegDctBlockGroupIndex* find_group_or_null(const ShardState& shard,
	                                                        const uint32_t    semantic_slot_id,
	                                                        const uint32_t    block_x,
	                                                        const uint32_t    block_y) {
		const auto it = shard.block_group_lookup.find(BlockGroupKey {semantic_slot_id, block_x, block_y});
		if (it == shard.block_group_lookup.end()) {
			return nullptr;
		}
		return &shard.metadata.block_group_index[it->second];
	}

	static size_t group_index_in_shard(const ShardState& shard, const JpegDctBlockGroupIndex& group) {
		return static_cast<size_t>(&group - shard.metadata.block_group_index.data());
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

	static uint32_t ceil_mul_div_u32(const uint32_t lhs, const uint32_t rhs, const uint32_t divisor) {
		if (divisor == 0) {
			throw std::runtime_error("JPEG DCT crop planning encountered a zero image dimension");
		}
		const uint64_t product = static_cast<uint64_t>(lhs) * static_cast<uint64_t>(rhs);
		return static_cast<uint32_t>((product + divisor - 1U) / divisor);
	}

	static uint32_t floor_mul_div_u32(const uint32_t lhs, const uint32_t rhs, const uint32_t divisor) {
		if (divisor == 0) {
			throw std::runtime_error("JPEG DCT crop planning encountered a zero image dimension");
		}
		return static_cast<uint32_t>((static_cast<uint64_t>(lhs) * static_cast<uint64_t>(rhs)) / divisor);
	}

	static JpegDctCropBox effective_crop_box(const JpegImageMetadata& image, JpegDctCropBox crop) {
		if (image.image_width == 0 || image.image_height == 0) {
			throw std::runtime_error("JPEG DCT device crop planning requires per-image dimensions");
		}
		if (crop.width == 0 || crop.height == 0) {
			crop.x      = 0;
			crop.y      = 0;
			crop.width  = image.image_width;
			crop.height = image.image_height;
			return crop;
		}
		if (crop.x >= image.image_width || crop.y >= image.image_height) {
			throw std::out_of_range("JPEG DCT crop starts outside the source image");
		}
		if (crop.width > image.image_width - crop.x) {
			crop.width = image.image_width - crop.x;
		}
		if (crop.height > image.image_height - crop.y) {
			crop.height = image.image_height - crop.y;
		}
		return crop;
	}

	static uint32_t closest_aligned_crop_extent(const uint32_t               source_extent,
	                                            const uint32_t               output_extent,
	                                            const uint32_t               reference_extent,
	                                            const std::vector<uint32_t>& preferred_small_extents) {
		if (source_extent == 0 || output_extent == 0 || reference_extent == 0) {
			return 0;
		}
		const auto target = static_cast<uint32_t>(std::nearbyint(
		    (static_cast<double>(source_extent) * static_cast<double>(output_extent)) /
		    static_cast<double>(reference_extent)));
		if (target <= output_extent && !preferred_small_extents.empty()) {
			uint32_t best      = output_extent;
			uint32_t best_diff = std::numeric_limits<uint32_t>::max();
			for (const auto choice : preferred_small_extents) {
				if (choice == 0 || choice > output_extent) {
					continue;
				}
				const auto diff = choice > target ? choice - target : target - choice;
				if (diff < best_diff) {
					best      = choice;
					best_diff = diff;
				}
			}
			return best;
		}
		auto closest = static_cast<uint32_t>(
		    std::nearbyint(static_cast<double>(target) / static_cast<double>(output_extent)) * output_extent);
		if (closest > source_extent) {
			closest = closest > output_extent ? closest - output_extent : output_extent;
		}
		return std::max<uint32_t>(1U, closest);
	}

	static void validate_grid_transform_spec(const JpegDctGridTransformSpec& spec) {
		if (spec.y_output_width_blocks == 0 || spec.y_output_height_blocks == 0 ||
		    spec.cbcr_output_width_blocks == 0 || spec.cbcr_output_height_blocks == 0 ||
		    spec.crop_reference_width_blocks == 0 || spec.crop_reference_height_blocks == 0) {
			throw std::runtime_error("transformed DCT grid requires non-zero output and crop-reference geometry");
		}
		if (spec.crop_origin_alignment_blocks == 0 || spec.chroma_crop_scale_x == 0 ||
		    spec.chroma_crop_scale_y == 0) {
			throw std::runtime_error("transformed DCT grid requires positive crop alignment and chroma scale");
		}
		if (spec.clamp_min < std::numeric_limits<int16_t>::min() ||
		    spec.clamp_max > std::numeric_limits<int16_t>::max() || spec.clamp_min > spec.clamp_max) {
			throw std::runtime_error("transformed DCT grid clamp range must be ordered and fit int16");
		}
		if (!spec.dequantize) {
			throw std::runtime_error("transformed DCT grid executor currently requires dequantize=true");
		}
		if (!spec.require_all_coefficients) {
			throw std::runtime_error("transformed DCT grid executor currently requires all 64 DCT coefficients");
		}
		for (const auto& ratio : spec.allowed_chroma_sampling_ratios) {
			if (ratio.horizontal_numerator == 0 || ratio.horizontal_denominator == 0 ||
			    ratio.vertical_numerator == 0 || ratio.vertical_denominator == 0) {
				throw std::runtime_error("transformed DCT grid sampling ratios must be positive");
			}
		}
	}

	static bool sampling_ratio_allowed(const JpegComponentMetadata&   reference,
	                                  const JpegComponentMetadata&   component,
	                                  const JpegDctGridTransformSpec& spec) {
		return std::any_of(spec.allowed_chroma_sampling_ratios.begin(),
		                   spec.allowed_chroma_sampling_ratios.end(),
		                   [&](const JpegDctSamplingRatio& ratio) {
			                   return static_cast<uint32_t>(component.h_samp_factor) *
			                                  ratio.horizontal_denominator ==
			                              static_cast<uint32_t>(reference.h_samp_factor) *
			                                  ratio.horizontal_numerator &&
			                          static_cast<uint32_t>(component.v_samp_factor) *
			                                  ratio.vertical_denominator ==
			                              static_cast<uint32_t>(reference.v_samp_factor) *
			                                  ratio.vertical_numerator;
		                   });
	}

	static int32_t floor_div_i32(const int32_t value, const int32_t divisor) {
		if (divisor <= 0) {
			throw std::invalid_argument("floor_div_i32 requires a positive divisor");
		}
		int32_t       quotient = value / divisor;
		const int32_t rem      = value % divisor;
		if (rem != 0 && ((rem < 0) != (divisor < 0))) {
			--quotient;
		}
		return quotient;
	}

	struct DctResizeAxisWeight {
		uint32_t out_block = 0;
		uint32_t out_coeff = 0;
		uint32_t in_block  = 0;
		uint32_t in_coeff  = 0;
		float    weight    = 0.0F;
	};

	struct DctResizeCacheCounters {
		size_t conversion_hits       = 0;
		size_t conversion_misses     = 0;
		size_t resize_weight_hits    = 0;
		size_t resize_weight_misses  = 0;
		double resize_weight_build_ms = 0.0;
	};

	static DctResizeCacheCounters& dct_resize_cache_counters() {
		thread_local DctResizeCacheCounters counters;
		return counters;
	}

	static DctResizeCacheCounters dct_resize_cache_counter_delta(const DctResizeCacheCounters& before,
	                                                             const DctResizeCacheCounters& after) {
		return DctResizeCacheCounters {
		    after.conversion_hits - before.conversion_hits,
		    after.conversion_misses - before.conversion_misses,
		    after.resize_weight_hits - before.resize_weight_hits,
		    after.resize_weight_misses - before.resize_weight_misses,
		    after.resize_weight_build_ms - before.resize_weight_build_ms};
	}

	static std::vector<float> dct_conversion_matrix_uncached(const uint32_t mult) {
		const uint32_t n = 8U * mult;
		std::vector<float> large(static_cast<size_t>(n) * n);
		std::vector<float> small(64);
		constexpr double   pi = 3.141592653589793238462643383279502884;
		const auto         basis = [](const uint32_t rows, const uint32_t u, const uint32_t x) {
			const double scale = u == 0 ? std::sqrt(1.0 / static_cast<double>(rows))
			                            : std::sqrt(2.0 / static_cast<double>(rows));
			return static_cast<float>(
			    scale * std::cos((static_cast<double>(u) * (static_cast<double>(x) + 0.5) * pi) /
			                     static_cast<double>(rows)));
		};
		for (uint32_t u = 0; u < n; ++u) {
			for (uint32_t x = 0; x < n; ++x) {
				large[static_cast<size_t>(u) * n + x] = basis(n, u, x);
			}
		}
		for (uint32_t u = 0; u < 8U; ++u) {
			for (uint32_t x = 0; x < 8U; ++x) {
				small[static_cast<size_t>(u) * 8U + x] = basis(8U, u, x);
			}
		}
		std::vector<float> conversion(static_cast<size_t>(n) * n, 0.0F);
		for (uint32_t out = 0; out < n; ++out) {
			for (uint32_t block = 0; block < mult; ++block) {
				for (uint32_t in = 0; in < 8U; ++in) {
					float sum = 0.0F;
					for (uint32_t x = 0; x < 8U; ++x) {
						sum += large[static_cast<size_t>(out) * n + block * 8U + x] *
						       small[static_cast<size_t>(in) * 8U + x];
					}
					conversion[static_cast<size_t>(out) * n + block * 8U + in] = sum;
				}
			}
		}
		return conversion;
	}

	static const std::vector<float>& dct_conversion_matrix(const uint32_t mult) {
		struct Entry {
			uint32_t           mult = 0;
			std::vector<float> values;
		};
		// References to two different factors are used together while composing
		// a rational resize matrix. deque preserves existing element addresses
		// when another factor is appended; a vector reallocation left the first
		// reference dangling and made cold mixed-ratio planning nondeterministic.
		thread_local std::deque<Entry> cache;
		auto& counters = dct_resize_cache_counters();
		for (auto& entry : cache) {
			if (entry.mult == mult) {
				++counters.conversion_hits;
				return entry.values;
			}
		}
		++counters.conversion_misses;
		cache.push_back(Entry {mult, dct_conversion_matrix_uncached(mult)});
		return cache.back().values;
	}

	static std::vector<DctResizeAxisWeight> dct_resize_axis_weights_uncached(const uint32_t source_blocks,
	                                                                         const uint32_t output_blocks) {
		std::vector<DctResizeAxisWeight> weights;
		if (source_blocks == output_blocks) {
			weights.reserve(static_cast<size_t>(output_blocks) * 8U);
			for (uint32_t block = 0; block < output_blocks; ++block) {
				for (uint32_t coeff = 0; coeff < 8U; ++coeff) {
					weights.push_back(DctResizeAxisWeight {block, coeff, block, coeff, 1.0F});
				}
			}
			return weights;
		}
		if (source_blocks > output_blocks && source_blocks % output_blocks == 0) {
			const uint32_t mult = source_blocks / output_blocks;
			const auto     conv = dct_conversion_matrix(mult);
			const float    norm = 1.0F / std::sqrt(static_cast<float>(mult));
			weights.reserve(static_cast<size_t>(output_blocks) * 8U * mult * 8U);
			for (uint32_t out_block = 0; out_block < output_blocks; ++out_block) {
				for (uint32_t out_coeff = 0; out_coeff < 8U; ++out_coeff) {
					for (uint32_t in_subblock = 0; in_subblock < mult; ++in_subblock) {
						for (uint32_t in_coeff = 0; in_coeff < 8U; ++in_coeff) {
							const auto idx = static_cast<size_t>(out_coeff) * (mult * 8U) + in_subblock * 8U + in_coeff;
							weights.push_back(DctResizeAxisWeight {
							    out_block,
							    out_coeff,
							    out_block * mult + in_subblock,
							    in_coeff,
							    conv[idx] * norm});
						}
					}
				}
			}
			return weights;
		}
		if (output_blocks > source_blocks && output_blocks % source_blocks == 0) {
			const uint32_t mult = output_blocks / source_blocks;
			const auto     conv = dct_conversion_matrix(mult);
			const float    norm = std::sqrt(static_cast<float>(mult));
			weights.reserve(static_cast<size_t>(source_blocks) * 8U * mult * 8U);
			for (uint32_t in_block = 0; in_block < source_blocks; ++in_block) {
				for (uint32_t out_subblock = 0; out_subblock < mult; ++out_subblock) {
					for (uint32_t out_coeff = 0; out_coeff < 8U; ++out_coeff) {
						for (uint32_t in_coeff = 0; in_coeff < 8U; ++in_coeff) {
							const uint32_t out_block = in_block * mult + out_subblock;
							const auto idx = static_cast<size_t>(in_coeff) * (mult * 8U) + out_subblock * 8U + out_coeff;
							weights.push_back(DctResizeAxisWeight {out_block, out_coeff, in_block, in_coeff, conv[idx] * norm});
						}
					}
				}
			}
			return weights;
		}
		throw std::runtime_error("transformed DCT grid exact resize currently requires integer up/downsample factors");
	}

	static std::vector<DctResizeAxisWeight> dct_resize_axis_weights(const uint32_t source_blocks,
	                                                                 const uint32_t output_blocks) {
		constexpr size_t kCapacity = 32;
		struct Entry {
			uint32_t                         source_blocks = 0;
			uint32_t                         output_blocks = 0;
			std::vector<DctResizeAxisWeight> values;
		};
		thread_local std::vector<Entry> cache;
		auto& counters = dct_resize_cache_counters();
		for (auto& entry : cache) {
			if (entry.source_blocks == source_blocks && entry.output_blocks == output_blocks) {
				++counters.resize_weight_hits;
				return entry.values;
			}
		}
		++counters.resize_weight_misses;
		const auto build_start = std::chrono::steady_clock::now();
		auto       values      = dct_resize_axis_weights_uncached(source_blocks, output_blocks);
		const auto build_end   = std::chrono::steady_clock::now();
		counters.resize_weight_build_ms += std::chrono::duration<double, std::milli>(build_end - build_start).count();
		if (cache.size() >= kCapacity) {
			cache.erase(cache.begin());
		}
		cache.push_back(Entry {source_blocks, output_blocks, std::move(values)});
		return cache.back().values;
	}

	static std::array<float, 64> transformed_dct_grid_axis_matrix(const uint16_t up_factor,
	                                                         const uint16_t down_factor,
	                                                         const uint16_t source_block,
	                                                         const uint16_t output_block) {
		if (up_factor == 0U || down_factor == 0U) {
			throw std::runtime_error("transformed DCT grid resize requires positive rational factors");
		}
		const auto& up_conversion   = dct_conversion_matrix(up_factor);
		const auto& down_conversion = dct_conversion_matrix(down_factor);
		const auto axis_weight = [](const std::vector<float>& conversion,
		                            const uint16_t            factor,
		                            const bool                upsample,
		                            const uint16_t            subblock,
		                            const uint8_t             out_coeff,
		                            const uint8_t             in_coeff) {
			if (factor == 1U) {
				return out_coeff == in_coeff ? 1.0F : 0.0F;
			}
			const auto stride = static_cast<size_t>(factor) * 8U;
			if (upsample) {
				return conversion[static_cast<size_t>(in_coeff) * stride +
				                  static_cast<size_t>(subblock) * 8U + out_coeff] *
				       std::sqrt(static_cast<float>(factor));
			}
			return conversion[static_cast<size_t>(out_coeff) * stride +
			                  static_cast<size_t>(subblock) * 8U + in_coeff] /
			       std::sqrt(static_cast<float>(factor));
		};

		std::array<float, 64> matrix {};
		const auto up_start = static_cast<uint32_t>(source_block) * up_factor;
		for (uint8_t out_coeff = 0; out_coeff < 8U; ++out_coeff) {
			for (uint8_t in_coeff = 0; in_coeff < 8U; ++in_coeff) {
				float sum = 0.0F;
				for (uint16_t up_subblock = 0; up_subblock < up_factor; ++up_subblock) {
					const auto upsampled_block = up_start + up_subblock;
					if (upsampled_block / down_factor != output_block) {
						continue;
					}
					const auto down_subblock = static_cast<uint16_t>(upsampled_block % down_factor);
					for (uint8_t mid_coeff = 0; mid_coeff < 8U; ++mid_coeff) {
						sum += axis_weight(
						           up_conversion, up_factor, true, up_subblock, mid_coeff, in_coeff) *
						       axis_weight(
						           down_conversion, down_factor, false, down_subblock, out_coeff, mid_coeff);
					}
				}
				matrix[static_cast<size_t>(out_coeff) * 8U + in_coeff] = sum;
			}
		}
		return matrix;
	}

	static uint8_t natural_to_physical_coeff(const uint8_t natural_coeff, const bool zigzag_columns) {
		if (!zigzag_columns) {
			return natural_coeff;
		}
		static constexpr std::array<uint8_t, 64> kNaturalToPhysical {
		    0,  1,  5,  6,  14, 15, 27, 28,
		    2,  4,  7,  13, 16, 26, 29, 42,
		    3,  8,  12, 17, 25, 30, 41, 43,
		    9,  11, 18, 24, 31, 40, 44, 53,
		    10, 19, 23, 32, 39, 45, 52, 54,
		    20, 22, 33, 38, 46, 51, 55, 60,
		    21, 34, 37, 47, 50, 56, 59, 61,
		    35, 36, 48, 49, 57, 58, 62, 63,
		};
		if (natural_coeff >= kNaturalToPhysical.size()) {
			throw std::runtime_error("invalid natural DCT coefficient index");
		}
		return kNaturalToPhysical[natural_coeff];
	}

	static JpegDctRowRef locate_row_in_shard(const ShardState& shard,
	                                         const uint32_t    local_image_index,
	                                         const uint32_t    semantic_slot_id,
	                                         const uint32_t    block_x,
	                                         const uint32_t    block_y) {
		if (local_image_index >= shard.metadata.images.size()) {
			throw std::runtime_error("JPEG DCT shard metadata does not contain the requested local image");
		}

		JpegDctRowRef ref;
		ref.shard_id          = shard.entry.shard_id;
		ref.local_image_index = local_image_index;
		ref.semantic_slot_id  = semantic_slot_id;
		ref.block_x           = block_x;
		ref.block_y           = block_y;

		const auto has_target =
		    image_has_block(shard.metadata.images[local_image_index], semantic_slot_id, block_x, block_y);
		const auto* group = find_group_or_null(shard, semantic_slot_id, block_x, block_y);
		if (group == nullptr) {
			if (!has_target) {
				return ref;
			}
			throw std::runtime_error("JPEG DCT block group was not found in shard metadata");
		}
		ref.fls_rowgroup_index    = group->fls_rowgroup_index;
		ref.row_start_in_rowgroup = group->row_start_in_rowgroup;

		uint32_t rank = 0;
		for (uint32_t image_idx = 0; image_idx < local_image_index; ++image_idx) {
			if (image_has_block(shard.metadata.images[image_idx], semantic_slot_id, block_x, block_y)) {
				++rank;
			}
		}
		ref.row_offset_in_block_group = rank;
		ref.present                   = has_target;
		ref.physical_row_index = group->row_start + ref.row_offset_in_block_group;
		return ref;
	}

	static uint32_t plan_ragged_rank(const ShardState&             shard,
	                                 const JpegDctBlockGroupIndex& group,
	                                 const uint32_t                local_image_index,
	                                 std::unordered_map<RankCursorKey, RankCursor, RankCursorKeyHash>& rank_cursors) {
		const auto key    = RankCursorKey {shard.entry.shard_id, group_index_in_shard(shard, group)};
		auto&      cursor = rank_cursors[key];
		if (local_image_index < cursor.next_local_image) {
			uint32_t rank = 0;
			for (uint32_t image_idx = 0; image_idx < local_image_index; ++image_idx) {
				if (image_has_block(
				        shard.metadata.images[image_idx], group.semantic_slot_id, group.block_x, group.block_y)) {
					++rank;
				}
			}
			return rank;
		}
		while (cursor.next_local_image < local_image_index) {
			if (image_has_block(shard.metadata.images[cursor.next_local_image],
			                    group.semantic_slot_id,
			                    group.block_x,
			                    group.block_y)) {
				++cursor.present_before_next;
			}
			++cursor.next_local_image;
		}
		return cursor.present_before_next;
	}

	static JpegDctRowRef
	locate_row_in_shard_for_plan(const ShardState&                                                 shard,
	                             const uint32_t                                                    local_image_index,
	                             const uint32_t                                                    semantic_slot_id,
	                             const uint32_t                                                    block_x,
	                             const uint32_t                                                    block_y,
	                             std::unordered_map<RankCursorKey, RankCursor, RankCursorKeyHash>& rank_cursors) {
		if (local_image_index >= shard.metadata.images.size()) {
			throw std::runtime_error("JPEG DCT shard metadata does not contain the requested local image");
		}

		JpegDctRowRef ref;
		ref.shard_id          = shard.entry.shard_id;
		ref.local_image_index = local_image_index;
		ref.semantic_slot_id  = semantic_slot_id;
		ref.block_x           = block_x;
		ref.block_y           = block_y;

		const auto has_target =
		    image_has_block(shard.metadata.images[local_image_index], semantic_slot_id, block_x, block_y);
		const auto* group = find_group_or_null(shard, semantic_slot_id, block_x, block_y);
		if (group == nullptr) {
			if (!has_target) {
				return ref;
			}
			throw std::runtime_error("JPEG DCT block group was not found in shard metadata");
		}
		ref.fls_rowgroup_index    = group->fls_rowgroup_index;
		ref.row_start_in_rowgroup = group->row_start_in_rowgroup;

		ref.row_offset_in_block_group = plan_ragged_rank(shard, *group, local_image_index, rank_cursors);
		ref.present                   = has_target;
		ref.physical_row_index = group->row_start + ref.row_offset_in_block_group;
		return ref;
	}

	detail::JpegDctDeviceBatchPlan plan_device_batch(const std::vector<JpegDctImageCropRequest>& requests,
	                                                 const JpegDctDeviceBatchOptions&            options) const {
		if (options.layout != JpegDctDeviceLayout::kImageMajorComponentBlockCoeff &&
		    options.layout != JpegDctDeviceLayout::kYcbcrDctGrid &&
		    options.layout != JpegDctDeviceLayout::kTransformedDctGrid) {
			throw std::runtime_error("unsupported JPEG DCT device output layout");
		}
		const bool output_transformed_dct_grid = options.layout == JpegDctDeviceLayout::kTransformedDctGrid;
		if (output_transformed_dct_grid != options.grid_transform.has_value()) {
			throw std::runtime_error(
			    "transformed DCT grid layout and grid_transform specification must be provided together");
		}
		if (options.grid_transform.has_value()) {
			validate_grid_transform_spec(*options.grid_transform);
		}

		detail::JpegDctDeviceBatchPlan plan;
		const auto resize_cache_counters_before = dct_resize_cache_counters();
			plan.layout                      = options.layout;
			if (options.grid_transform.has_value()) {
				plan.grid_transform = *options.grid_transform;
			}
			plan.selected_coefficients       = detail::normalize_coefficient_selection(options.coefficient_selection);
			plan.coefficient_selection_shape = detail::classify_coefficient_selection(plan.selected_coefficients);
			plan.coefficients_per_block      = plan.selected_coefficients.size();
			if (output_transformed_dct_grid && options.grid_transform->require_all_coefficients &&
			    !detail::selects_all_coefficients(plan.selected_coefficients)) {
				throw std::runtime_error(
				    "transformed DCT grid requires dct_coeffs=all because DCT resize uses all 64 source coefficients");
			}
			const bool output_ycbcr_dct_grid =
			    options.layout == JpegDctDeviceLayout::kYcbcrDctGrid ||
			    options.layout == JpegDctDeviceLayout::kTransformedDctGrid;
		const auto* grid_transform = options.grid_transform.has_value() ? &*options.grid_transform : nullptr;
		const bool materialize_projection_items =
		    output_ycbcr_dct_grid ||
		    plan.coefficient_selection_shape.kind != detail::JpegDctCoefficientSelectionKind::kAll;
		plan.decode_batch_rowgroups =
		    options.decode_batch_rowgroups == 0 ? kDefaultJpegDctDecodeBatchRowgroups : options.decode_batch_rowgroups;
		plan.rowgroup_prefetch.enabled = options.enable_rowgroup_prefetch;
		plan.rowgroup_prefetch.depth = options.rowgroup_prefetch_depth == 0 ? kDefaultJpegDctDeviceRowgroupPrefetchDepth
		                                                                    : options.rowgroup_prefetch_depth;
		plan.rowgroup_prefetch.workers            = options.rowgroup_prefetch_workers == 0
		                                                ? kDefaultJpegDctDeviceRowgroupPrefetchWorkers
		                                                : options.rowgroup_prefetch_workers;
		plan.rowgroup_prefetch.min_decode_batches = options.rowgroup_prefetch_min_decode_batches == 0
		                                                ? kDefaultJpegDctDeviceRowgroupPrefetchMinDecodeBatches
		                                                : options.rowgroup_prefetch_min_decode_batches;
		plan.image_layouts.reserve(requests.size());
		std::unordered_map<RankCursorKey, RankCursor, RankCursorKeyHash> rank_cursors;
		std::unordered_map<uint32_t, size_t>                             shard_plan_indices;
		std::vector<std::unordered_map<uint32_t, size_t>>                rowgroup_plan_indices;
		uint32_t                                                         grid_y_width         = 0;
		uint32_t                                                         grid_y_height        = 0;
		uint32_t                                                         grid_cbcr_width      = 0;
		uint32_t                                                         grid_cbcr_height     = 0;
		bool                                                             grid_y_shape_set     = false;
		bool                                                             grid_cbcr_shape_set  = false;

		const auto ensure_grid_component_shape = [&](const uint32_t semantic_slot_id,
		                                             const uint32_t width,
		                                             const uint32_t height) {
			if (!output_ycbcr_dct_grid) {
				return;
			}
			if (width == 0 || height == 0) {
				throw std::runtime_error("YCbCr DCT grid layout requires non-empty component crops");
			}
			auto ensure_shape = [&](bool& shape_set, uint32_t& expected_width, uint32_t& expected_height) {
				if (!shape_set) {
					expected_width  = width;
					expected_height = height;
					shape_set      = true;
					return;
				}
				if (expected_width != width || expected_height != height) {
					throw std::runtime_error(
					    "YCbCr DCT grid layout requires every image in the batch to use the same crop-derived "
					    "DCT grid shape");
				}
			};
			if (semantic_slot_id == 0) {
				ensure_shape(grid_y_shape_set, grid_y_width, grid_y_height);
			} else if (semantic_slot_id == 1 || semantic_slot_id == 2) {
				ensure_shape(grid_cbcr_shape_set, grid_cbcr_width, grid_cbcr_height);
			} else {
				throw std::runtime_error("YCbCr DCT grid layout supports only semantic slots 0, 1, and 2");
			}
		};

		const auto shard_plan_index_for = [&](const ShardState& shard) -> size_t {
			const auto found = shard_plan_indices.find(shard.entry.shard_id);
			if (found != shard_plan_indices.end()) {
				return found->second;
			}
			const size_t                   shard_plan_index = plan.shards.size();
			detail::JpegDctDeviceShardPlan shard_plan;
			shard_plan.shard_id = shard.entry.shard_id;
			shard_plan.fls_path = shard.fls_path;
			plan.shards.push_back(std::move(shard_plan));
			rowgroup_plan_indices.emplace_back();
			shard_plan_indices.emplace(shard.entry.shard_id, shard_plan_index);
			return shard_plan_index;
		};

		const auto rowgroup_plan_index_for = [&](const size_t   shard_plan_index,
		                                         const uint32_t rowgroup_index) -> size_t {
			auto&      rowgroup_indices = rowgroup_plan_indices[shard_plan_index];
			const auto found            = rowgroup_indices.find(rowgroup_index);
			if (found != rowgroup_indices.end()) {
				return found->second;
			}
			auto&                             shard_plan          = plan.shards[shard_plan_index];
			const size_t                      rowgroup_plan_index = shard_plan.rowgroups.size();
			detail::JpegDctDeviceRowgroupPlan rowgroup_plan;
			rowgroup_plan.rowgroup_index = rowgroup_index;
			shard_plan.rowgroups.push_back(std::move(rowgroup_plan));
			rowgroup_indices.emplace(rowgroup_index, rowgroup_plan_index);
			return rowgroup_plan_index;
		};

		std::unordered_map<uint64_t, uint32_t> fixed_axis_matrix_indices;
		const auto fixed_axis_matrix_index_for = [&](const uint16_t up_factor,
		                                              const uint16_t down_factor,
		                                              const uint16_t source_block,
		                                              const uint16_t output_block) {
			const auto key = (static_cast<uint64_t>(up_factor) << 48U) |
			                 (static_cast<uint64_t>(down_factor) << 32U) |
			                 (static_cast<uint64_t>(source_block) << 16U) |
			                 static_cast<uint64_t>(output_block);
			const auto found = fixed_axis_matrix_indices.find(key);
			if (found != fixed_axis_matrix_indices.end()) {
				return found->second;
			}
			const auto matrix = transformed_dct_grid_axis_matrix(up_factor, down_factor, source_block, output_block);
			const auto index = static_cast<uint32_t>(plan.fixed_resize_weight_matrices.size() / 64U);
			plan.fixed_resize_weight_matrices.insert(
			    plan.fixed_resize_weight_matrices.end(), matrix.begin(), matrix.end());
			fixed_axis_matrix_indices.emplace(key, index);
			return index;
		};

		for (size_t request_idx = 0; request_idx < requests.size(); ++request_idx) {
			const auto& request = requests[request_idx];
			const auto& shard   = shard_for_global_image(request.global_image_index);
			const auto  local_image_index =
			    static_cast<uint32_t>(request.global_image_index - shard.entry.first_global_image_index);
			if (local_image_index >= shard.metadata.images.size()) {
				throw std::runtime_error("JPEG DCT shard metadata does not contain the requested local image");
			}

			const auto& image = shard.metadata.images[local_image_index];
			const auto  crop  = effective_crop_box(image, request.source_crop);
			const JpegComponentMetadata* y_component        = nullptr;
			const JpegComponentMetadata* fixed_cb_component = nullptr;
			const JpegComponentMetadata* fixed_cr_component = nullptr;
			constexpr uint32_t           kInvalidFixedSlot  = std::numeric_limits<uint32_t>::max();
			uint32_t                     fixed_y_slot       = kInvalidFixedSlot;
			uint32_t                     fixed_cb_slot      = kInvalidFixedSlot;
			uint32_t                     fixed_cr_slot      = kInvalidFixedSlot;
			for (const auto& component : image.components) {
				if (component.semantic_slot_id == 0 && component.present) {
					y_component = &component;
					fixed_y_slot = component.semantic_slot_id;
				} else if (component.semantic_slot_id == 1 && component.present) {
					fixed_cb_slot      = component.semantic_slot_id;
					fixed_cb_component = &component;
				} else if (component.semantic_slot_id == 2 && component.present) {
					fixed_cr_slot      = component.semantic_slot_id;
					fixed_cr_component = &component;
				}
			}
			if (output_transformed_dct_grid && y_component == nullptr) {
				std::array<const JpegComponentMetadata*, 3> fallback_components {};
				for (const auto& component : image.components) {
					if (!component.present || component.width_in_blocks == 0 || component.height_in_blocks == 0) {
						continue;
					}
					if (component.local_component_index >= fallback_components.size()) {
						continue;
					}
					fallback_components[component.local_component_index] = &component;
				}
				if (fallback_components[0] != nullptr) {
					y_component  = fallback_components[0];
					fixed_y_slot = fallback_components[0]->semantic_slot_id;
				}
				if (fallback_components[1] != nullptr) {
					fixed_cb_slot      = fallback_components[1]->semantic_slot_id;
					fixed_cb_component = fallback_components[1];
				}
				if (fallback_components[2] != nullptr) {
					fixed_cr_slot      = fallback_components[2]->semantic_slot_id;
					fixed_cr_component = fallback_components[2];
				}
			}
			int32_t  fixed_y_x0 = 0;
			int32_t  fixed_y_y0 = 0;
			uint32_t fixed_y_w  = 0;
			uint32_t fixed_y_h  = 0;
			int32_t  fixed_chroma_x0 = 0;
			int32_t  fixed_chroma_y0 = 0;
			uint32_t fixed_chroma_w  = 0;
			uint32_t fixed_chroma_h  = 0;
			if (output_transformed_dct_grid) {
				if (y_component == nullptr || y_component->width_in_blocks == 0 || y_component->height_in_blocks == 0) {
					throw std::runtime_error("transformed DCT grid requires a present Y component");
				}
				if ((fixed_cb_component == nullptr) != (fixed_cr_component == nullptr)) {
					throw std::runtime_error(
					    "transformed DCT grid requires both Cb and Cr components when chroma is present");
				}
				if (fixed_cb_component != nullptr && fixed_cr_component != nullptr) {
					const auto sampling_matches = [&](const JpegComponentMetadata* chroma) {
						return chroma->h_samp_factor == fixed_cb_component->h_samp_factor &&
						       chroma->v_samp_factor == fixed_cb_component->v_samp_factor;
					};
					if (!sampling_matches(fixed_cr_component)) {
						throw std::runtime_error("transformed DCT grid requires matching Cb/Cr sampling factors");
					}
					if (!sampling_ratio_allowed(*y_component, *fixed_cb_component, *grid_transform)) {
						throw std::runtime_error(
						    "transformed DCT grid profile does not allow this chroma sampling ratio");
					}
				} else if (!grid_transform->allow_grayscale) {
					throw std::runtime_error("transformed DCT grid profile does not allow grayscale input");
				}
				fixed_y_w = closest_aligned_crop_extent(y_component->width_in_blocks,
				                                               grid_transform->y_output_width_blocks,
				                                               grid_transform->crop_reference_width_blocks,
				                                               grid_transform->preferred_small_crop_width_blocks);
				fixed_y_h = closest_aligned_crop_extent(y_component->height_in_blocks,
				                                               grid_transform->y_output_height_blocks,
				                                               grid_transform->crop_reference_height_blocks,
				                                               grid_transform->preferred_small_crop_height_blocks);
				fixed_y_x0 = floor_div_i32(
				    static_cast<int32_t>(y_component->width_in_blocks) - static_cast<int32_t>(fixed_y_w), 2);
				fixed_y_y0 = floor_div_i32(
				    static_cast<int32_t>(y_component->height_in_blocks) - static_cast<int32_t>(fixed_y_h), 2);
				const auto alignment = static_cast<int32_t>(grid_transform->crop_origin_alignment_blocks);
				fixed_y_x0 = floor_div_i32(fixed_y_x0, alignment) * alignment;
				fixed_y_y0 = floor_div_i32(fixed_y_y0, alignment) * alignment;
				fixed_chroma_x0 = floor_div_i32(
				    fixed_y_x0, static_cast<int32_t>(grid_transform->chroma_crop_scale_x));
				fixed_chroma_y0 = floor_div_i32(
				    fixed_y_y0, static_cast<int32_t>(grid_transform->chroma_crop_scale_y));
				fixed_chroma_w = std::max<uint32_t>(1U, fixed_y_w / grid_transform->chroma_crop_scale_x);
				fixed_chroma_h = std::max<uint32_t>(1U, fixed_y_h / grid_transform->chroma_crop_scale_y);
			}

			JpegDctDeviceImageLayout image_layout;
			image_layout.global_image_index = request.global_image_index;
			image_layout.block_offset       = static_cast<uint64_t>(plan.block_metadata.size());
			bool grid_seen_y                = false;
			bool grid_seen_cb               = false;
			bool grid_seen_cr               = false;

			for (const auto& component : image.components) {
				if (!component.present || component.width_in_blocks == 0 || component.height_in_blocks == 0) {
					continue;
				}
				const bool fixed_component_y =
				    output_transformed_dct_grid && component.semantic_slot_id == fixed_y_slot;
				const bool fixed_component_cb =
				    output_transformed_dct_grid && component.semantic_slot_id == fixed_cb_slot;
				const bool fixed_component_cr =
				    output_transformed_dct_grid && component.semantic_slot_id == fixed_cr_slot;
				if (output_transformed_dct_grid && !fixed_component_y && !fixed_component_cb && !fixed_component_cr) {
					continue;
				}
				uint32_t fixed_quant_table_index = 0;
				if (output_transformed_dct_grid) {
					const auto quant_table = std::find_if(
					    image.quant_tables.begin(), image.quant_tables.end(), [&](const JpegQuantTableMetadata& candidate) {
						    return component.quant_tbl_no >= 0 &&
						           candidate.table_id == static_cast<uint8_t>(component.quant_tbl_no);
					    });
					if (quant_table == image.quant_tables.end()) {
						throw std::runtime_error(
						    "transformed DCT grid requires the JPEG quantization table for every component");
					}
					fixed_quant_table_index = static_cast<uint32_t>(plan.fixed_quant_tables.size() / 64U);
					plan.fixed_quant_tables.insert(
					    plan.fixed_quant_tables.end(), quant_table->values.begin(), quant_table->values.end());
				}

				uint32_t x0 = std::min(component.width_in_blocks,
				                       floor_mul_div_u32(crop.x, component.width_in_blocks, image.image_width));
				uint32_t y0 = std::min(component.height_in_blocks,
				                       floor_mul_div_u32(crop.y, component.height_in_blocks, image.image_height));
				uint32_t x1 =
				    std::min(component.width_in_blocks,
				             ceil_mul_div_u32(crop.x + crop.width, component.width_in_blocks, image.image_width));
				uint32_t y1 =
				    std::min(component.height_in_blocks,
				             ceil_mul_div_u32(crop.y + crop.height, component.height_in_blocks, image.image_height));
				uint32_t out_w = x1 - x0;
				uint32_t out_h = y1 - y0;
				if (output_transformed_dct_grid) {
					if (fixed_component_y) {
						out_w = grid_transform->y_output_width_blocks;
						out_h = grid_transform->y_output_height_blocks;
					} else if (fixed_component_cb || fixed_component_cr) {
						out_w = grid_transform->cbcr_output_width_blocks;
						out_h = grid_transform->cbcr_output_height_blocks;
					}
				}
				if (!output_transformed_dct_grid) {
					ensure_grid_component_shape(component.semantic_slot_id, out_w, out_h);
				}
				if (output_transformed_dct_grid) {
					if (fixed_component_y) {
						grid_seen_y = true;
					} else if (fixed_component_cb) {
						grid_seen_cb = true;
					} else if (fixed_component_cr) {
						grid_seen_cr = true;
					}
				} else if (output_ycbcr_dct_grid) {
					if (component.semantic_slot_id == 0) {
						grid_seen_y = true;
					} else if (component.semantic_slot_id == 1) {
						grid_seen_cb = true;
					} else if (component.semantic_slot_id == 2) {
						grid_seen_cr = true;
					}
				}
				if (output_transformed_dct_grid) {
					int32_t  crop_x0 = static_cast<int32_t>(x0);
					int32_t  crop_y0 = static_cast<int32_t>(y0);
					uint32_t crop_w  = x1 - x0;
					uint32_t crop_h  = y1 - y0;
					if (fixed_component_y) {
						crop_x0 = fixed_y_x0;
						crop_y0 = fixed_y_y0;
						crop_w  = std::max<uint32_t>(1U, fixed_y_w);
						crop_h  = std::max<uint32_t>(1U, fixed_y_h);
					} else if (fixed_component_cb || fixed_component_cr) {
						crop_x0 = fixed_chroma_x0;
						crop_y0 = fixed_chroma_y0;
						crop_w  = fixed_chroma_w;
						crop_h  = fixed_chroma_h;
					}
						const auto fixed_relation = [](const uint32_t source, const uint32_t output) {
							struct Relation {
								uint16_t up_factor   = 0;
								uint16_t down_factor = 0;
							};
							if (source == 0U || output == 0U) {
								return Relation {};
							}
							const auto gcd         = std::gcd(source, output);
							const auto up_factor   = output / gcd;
							const auto down_factor = source / gcd;
							if (up_factor > std::numeric_limits<uint16_t>::max() ||
							    down_factor > std::numeric_limits<uint16_t>::max()) {
								return Relation {};
							}
							return Relation {static_cast<uint16_t>(up_factor), static_cast<uint16_t>(down_factor)};
						};
						const auto x_relation = fixed_relation(crop_w, out_w);
						const auto y_relation = fixed_relation(crop_h, out_h);
						const bool use_specialized_fixed_transform =
						    x_relation.up_factor != 0U && x_relation.down_factor != 0U &&
						    y_relation.up_factor != 0U && y_relation.down_factor != 0U;
						if (!use_specialized_fixed_transform) {
							throw std::runtime_error(
							    "transformed DCT grid source-to-output geometry exceeds supported factor range");
						}
					std::unordered_map<uint64_t, JpegDctRowRef> source_refs;
					if (!use_specialized_fixed_transform) {
						source_refs.reserve(static_cast<size_t>(crop_w) * static_cast<size_t>(crop_h));
					}
						for (uint32_t local_y = 0; local_y < crop_h; ++local_y) {
							const int32_t source_y_i = crop_y0 + static_cast<int32_t>(local_y);
							if (source_y_i < 0 || source_y_i >= static_cast<int32_t>(component.height_in_blocks)) {
								continue;
							}
							for (uint32_t local_x = 0; local_x < crop_w; ++local_x) {
								const int32_t source_x_i = crop_x0 + static_cast<int32_t>(local_x);
								if (source_x_i < 0 || source_x_i >= static_cast<int32_t>(component.width_in_blocks)) {
									continue;
								}
							const auto source_x = static_cast<uint32_t>(source_x_i);
							const auto source_y = static_cast<uint32_t>(source_y_i);
							auto ref = locate_row_in_shard_for_plan(
							    shard, local_image_index, component.semantic_slot_id, source_x, source_y, rank_cursors);
							if (!ref.present) {
								throw std::runtime_error(
								    "transformed DCT grid source block is missing from shard metadata");
							}
							if (!use_specialized_fixed_transform) {
								const auto source_key =
								    (static_cast<uint64_t>(local_y) << 32U) | static_cast<uint64_t>(local_x);
								source_refs.emplace(source_key, ref);
							}
							const auto metadata_block_index = static_cast<uint64_t>(plan.block_metadata.size());
							plan.block_metadata.push_back(JpegDctDeviceBlockMetadata {static_cast<uint32_t>(request_idx),
							                                                          request.global_image_index,
							                                                          component.semantic_slot_id,
							                                                          source_x,
							                                                          source_y});
							const auto shard_plan_index = shard_plan_index_for(shard);
							const auto rowgroup_plan_index =
							    rowgroup_plan_index_for(shard_plan_index, ref.fls_rowgroup_index);
							auto& rowgroup_plan = plan.shards[shard_plan_index].rowgroups[rowgroup_plan_index];
							rowgroup_plan.items.push_back(
							    detail::JpegDctDeviceGatherItem {ref.fls_rowgroup_index,
							                                     ref.row_start_in_rowgroup + ref.row_offset_in_block_group,
							                                     metadata_block_index});
							if (use_specialized_fixed_transform) {
								const auto output_x0 =
								    (static_cast<uint64_t>(local_x) * x_relation.up_factor) / x_relation.down_factor;
								const auto output_x1 =
								    ((static_cast<uint64_t>(local_x + 1U) * x_relation.up_factor) - 1U) /
								    x_relation.down_factor;
								const auto output_y0 =
								    (static_cast<uint64_t>(local_y) * y_relation.up_factor) / y_relation.down_factor;
								const auto output_y1 =
								    ((static_cast<uint64_t>(local_y + 1U) * y_relation.up_factor) - 1U) /
								    y_relation.down_factor;
								for (uint32_t output_y = static_cast<uint32_t>(output_y0);
								     output_y <= std::min<uint32_t>(static_cast<uint32_t>(output_y1), out_h - 1U);
								     ++output_y) {
								for (uint32_t output_x = static_cast<uint32_t>(output_x0);
									     output_x <= std::min<uint32_t>(static_cast<uint32_t>(output_x1), out_w - 1U);
									     ++output_x) {
									uint64_t output_block_index = 0;
									if (fixed_component_y) {
										output_block_index =
										    (static_cast<uint64_t>(request_idx) * grid_transform->y_output_height_blocks +
										     output_y) *
										        grid_transform->y_output_width_blocks +
										    output_x;
									} else {
										const auto channel = fixed_component_cb ? 0ULL : 1ULL;
										output_block_index =
										    ((static_cast<uint64_t>(request_idx) * 2U + channel) *
										         grid_transform->cbcr_output_height_blocks +
										     output_y) *
										        grid_transform->cbcr_output_width_blocks +
										    output_x;
									}
									const auto x_weight_matrix_index = fixed_axis_matrix_index_for(
									    x_relation.up_factor,
									    x_relation.down_factor,
									    static_cast<uint16_t>(local_x),
									    static_cast<uint16_t>(output_x));
									const auto y_weight_matrix_index = fixed_axis_matrix_index_for(
									    y_relation.up_factor,
									    y_relation.down_factor,
									    static_cast<uint16_t>(local_y),
									    static_cast<uint16_t>(output_y));
									rowgroup_plan.fixed_transform_items.push_back(
										    detail::JpegDctDeviceFixedTransformItem {
										        ref.fls_rowgroup_index,
										        ref.row_start_in_rowgroup + ref.row_offset_in_block_group,
										        output_block_index,
										        static_cast<uint32_t>(request_idx),
										        static_cast<uint16_t>(local_x),
										        static_cast<uint16_t>(local_y),
										        static_cast<uint16_t>(output_x),
										        static_cast<uint16_t>(output_y),
										        static_cast<uint8_t>(fixed_component_y ? 0U : (fixed_component_cb ? 1U : 2U)),
										        shard.metadata.zigzag_columns,
										        x_relation.down_factor,
										        y_relation.down_factor,
										        0U,
										        0U,
										        false,
										        false,
										        x_relation.up_factor,
										        y_relation.up_factor,
										        x_relation.down_factor,
										        y_relation.down_factor,
										        fixed_quant_table_index,
										        x_weight_matrix_index,
										        y_weight_matrix_index});
									}
								}
							}
						}
					}
					if (!use_specialized_fixed_transform) {
						const auto y_weights = dct_resize_axis_weights(crop_h, out_h);
						const auto x_weights = dct_resize_axis_weights(crop_w, out_w);
						for (const auto& wy : y_weights) {
							for (const auto& wx : x_weights) {
								const auto source_key =
								    (static_cast<uint64_t>(wy.in_block) << 32U) | static_cast<uint64_t>(wx.in_block);
								const auto found_ref = source_refs.find(source_key);
								if (found_ref == source_refs.end()) {
									continue;
								}
								const auto& ref = found_ref->second;
								uint64_t projection_block_index = 0;
								uint8_t  output_grid_tensor     = 0;
								if (fixed_component_y) {
									projection_block_index =
									    (static_cast<uint64_t>(request_idx) * grid_transform->y_output_height_blocks +
									     wy.out_block) *
									        grid_transform->y_output_width_blocks +
									    wx.out_block;
									output_grid_tensor = kJpegDctYcbcrDctGridTensorY;
								} else {
									const auto channel = fixed_component_cb ? 0ULL : 1ULL;
									projection_block_index =
									    ((static_cast<uint64_t>(request_idx) * 2U + channel) *
									         grid_transform->cbcr_output_height_blocks +
									     wy.out_block) *
									        grid_transform->cbcr_output_width_blocks +
									    wx.out_block;
									output_grid_tensor = kJpegDctYcbcrDctGridTensorCbCr;
								}
								const auto source_natural_coeff =
								    static_cast<uint8_t>(wy.in_coeff * 8U + wx.in_coeff);
								const auto output_natural_coeff =
								    static_cast<uint8_t>(wy.out_coeff * 8U + wx.out_coeff);
								const auto source_physical_coeff =
								    natural_to_physical_coeff(source_natural_coeff, shard.metadata.zigzag_columns);
								const auto shard_plan_index = shard_plan_index_for(shard);
								const auto rowgroup_plan_index =
								    rowgroup_plan_index_for(shard_plan_index, ref.fls_rowgroup_index);
								auto& rowgroup_plan = plan.shards[shard_plan_index].rowgroups[rowgroup_plan_index];
								rowgroup_plan.projection_items.push_back(detail::JpegDctDeviceProjectionItem {
								    ref.fls_rowgroup_index,
								    ref.row_start_in_rowgroup + ref.row_offset_in_block_group,
								    projection_block_index,
								    0U,
								    source_physical_coeff,
								    source_physical_coeff,
								    output_natural_coeff,
								    output_grid_tensor,
								    wy.weight * wx.weight});
							}
						}
					}
					continue;
				}
				for (uint32_t out_block_y = 0; out_block_y < out_h; ++out_block_y) {
					for (uint32_t out_block_x = 0; out_block_x < out_w; ++out_block_x) {
						const uint32_t source_w = std::max<uint32_t>(1U, x1 - x0);
						const uint32_t source_h = std::max<uint32_t>(1U, y1 - y0);
						const uint32_t block_x =
						    output_transformed_dct_grid
						        ? std::min<uint32_t>(component.width_in_blocks - 1U,
						                             x0 + floor_mul_div_u32(out_block_x, source_w, out_w))
						        : x0 + out_block_x;
						const uint32_t block_y =
						    output_transformed_dct_grid
						        ? std::min<uint32_t>(component.height_in_blocks - 1U,
						                             y0 + floor_mul_div_u32(out_block_y, source_h, out_h))
						        : y0 + out_block_y;
						auto ref = locate_row_in_shard_for_plan(
						    shard, local_image_index, component.semantic_slot_id, block_x, block_y, rank_cursors);
						if (!ref.present) {
							continue;
						}

						const auto metadata_block_index = static_cast<uint64_t>(plan.block_metadata.size());
						auto       projection_block_index = metadata_block_index;
						uint8_t    output_grid_tensor     = 0;
						if (output_ycbcr_dct_grid) {
							const auto local_block_y = static_cast<uint64_t>(out_block_y);
							const auto local_block_x = static_cast<uint64_t>(out_block_x);
							if (component.semantic_slot_id == 0) {
								projection_block_index = (static_cast<uint64_t>(request_idx) * grid_y_height +
								                          local_block_y) *
								                             grid_y_width +
								                         local_block_x;
								output_grid_tensor = kJpegDctYcbcrDctGridTensorY;
							} else {
								const auto channel = static_cast<uint64_t>(component.semantic_slot_id - 1U);
								projection_block_index =
								    ((static_cast<uint64_t>(request_idx) * 2U + channel) * grid_cbcr_height +
								     local_block_y) *
								        grid_cbcr_width +
								    local_block_x;
								output_grid_tensor = kJpegDctYcbcrDctGridTensorCbCr;
							}
						}
						plan.block_metadata.push_back(JpegDctDeviceBlockMetadata {static_cast<uint32_t>(request_idx),
						                                                          request.global_image_index,
						                                                          component.semantic_slot_id,
						                                                          block_x,
						                                                          block_y});
						const auto shard_plan_index = shard_plan_index_for(shard);
						const auto rowgroup_plan_index =
						    rowgroup_plan_index_for(shard_plan_index, ref.fls_rowgroup_index);
						auto& rowgroup_plan = plan.shards[shard_plan_index].rowgroups[rowgroup_plan_index];
						rowgroup_plan.items.push_back(
						    detail::JpegDctDeviceGatherItem {ref.fls_rowgroup_index,
						                                     ref.row_start_in_rowgroup + ref.row_offset_in_block_group,
						                                     metadata_block_index});
						if (materialize_projection_items) {
							const auto row_in_rowgroup = ref.row_start_in_rowgroup + ref.row_offset_in_block_group;
							for (size_t coeff_slot = 0; coeff_slot < plan.selected_coefficients.size(); ++coeff_slot) {
								const auto logical_coeff = plan.selected_coefficients[coeff_slot];
								const auto output_coeff =
								    output_ycbcr_dct_grid && shard.metadata.zigzag_columns
								        ? detail::kZigzagColumnToNaturalIndex[logical_coeff]
								        : logical_coeff;
								rowgroup_plan.projection_items.push_back(detail::JpegDctDeviceProjectionItem {
								    ref.fls_rowgroup_index,
								    row_in_rowgroup,
								    projection_block_index,
								    static_cast<uint16_t>(coeff_slot),
								    logical_coeff,
								    logical_coeff,
								    output_coeff,
								    output_grid_tensor});
							}
						}
					}
				}
			}
			if (output_ycbcr_dct_grid && !grid_seen_y) {
				throw std::runtime_error("YCbCr DCT grid layout requires a Y component per image");
			}
			if (output_ycbcr_dct_grid && !output_transformed_dct_grid && (!grid_seen_cb || !grid_seen_cr)) {
				throw std::runtime_error("YCbCr DCT grid layout requires Y, Cb, and Cr components per image");
			}

			image_layout.block_count =
			    static_cast<uint32_t>(static_cast<uint64_t>(plan.block_metadata.size()) - image_layout.block_offset);
			plan.image_layouts.push_back(image_layout);
			}
			if (output_ycbcr_dct_grid) {
				if (output_transformed_dct_grid) {
					plan.ycbcr_dct_grid_shape.y = {requests.size(),
					                                    1U,
					                                    grid_transform->y_output_height_blocks,
					                                    grid_transform->y_output_width_blocks,
					                                    8U,
					                                    8U};
					plan.ycbcr_dct_grid_shape.cbcr = {requests.size(),
					                                       2U,
					                                       grid_transform->cbcr_output_height_blocks,
					                                       grid_transform->cbcr_output_width_blocks,
					                                       8U,
					                                       8U};
				} else {
					if (!requests.empty() && (!grid_y_shape_set || !grid_cbcr_shape_set)) {
						throw std::runtime_error("YCbCr DCT grid layout could not determine Y/CbCr grid shapes");
					}
					plan.ycbcr_dct_grid_shape.y = {requests.size(), 1U, grid_y_height, grid_y_width, 8U, 8U};
					plan.ycbcr_dct_grid_shape.cbcr =
					    {requests.size(), 2U, grid_cbcr_height, grid_cbcr_width, 8U, 8U};
			}
		}

		std::unordered_set<uint64_t> fixed_transform_component_keys;
		std::unordered_set<uint64_t> fixed_transform_output_block_keys;
		if (output_transformed_dct_grid) {
			fixed_transform_component_keys.reserve(requests.size() * 3U);
			fixed_transform_output_block_keys.reserve(
			    (plan.ycbcr_dct_grid_shape.y_count() + plan.ycbcr_dct_grid_shape.cbcr_count()) / 64U);
		}
		for (auto& shard_plan : plan.shards) {
			std::sort(shard_plan.rowgroups.begin(), shard_plan.rowgroups.end(), [](const auto& lhs, const auto& rhs) {
				return lhs.rowgroup_index < rhs.rowgroup_index;
			});
			const auto& shard = shard_by_id(shard_plan.shard_id);
			for (auto& rowgroup_plan : shard_plan.rowgroups) {
				plan.rowgroups.push_back(
				    JpegDctDeviceRowgroupMetadata {shard_plan.shard_id, rowgroup_plan.rowgroup_index});
				if (rowgroup_plan.rowgroup_index >= shard.rowgroup_n_tuples.size()) {
					throw std::runtime_error("JPEG DCT planned rowgroup exceeds shard rowgroup metadata");
				}
				if (output_transformed_dct_grid) {
					std::unordered_set<uint32_t> source_rows;
					source_rows.reserve(std::min(
					    rowgroup_plan.fixed_transform_items.size(),
					    shard.rowgroup_n_tuples[rowgroup_plan.rowgroup_index]));
					for (const auto& item : rowgroup_plan.fixed_transform_items) {
						fixed_transform_component_keys.insert(
						    (static_cast<uint64_t>(item.image_index) << 8U) | item.component);
						source_rows.insert(item.row_in_rowgroup);
						fixed_transform_output_block_keys.insert(
						    (static_cast<uint64_t>(item.image_index) << 40U) |
						    (static_cast<uint64_t>(item.component) << 32U) |
						    (static_cast<uint64_t>(item.output_block_y) << 16U) |
						    item.output_block_x);
					}
					plan.fixed_transform_source_block_count += source_rows.size();
				}
				const auto full_vector_count =
				    row_count_to_vector_count(shard.rowgroup_n_tuples[rowgroup_plan.rowgroup_index]);
				auto selected_vectors = detail::selected_decode_vectors(
				    rowgroup_plan.items, full_vector_count, detail::kJpegDctDeviceUnpackNVectors);
				const auto planned_selected_vector_count = detail::selected_decode_vector_count(
				    selected_vectors, full_vector_count, detail::kJpegDctDeviceUnpackNVectors);
				const auto selected_chunks_fit = detail::selected_decode_chunks_fit(
				    selected_vectors, full_vector_count, detail::kJpegDctDeviceUnpackNVectors);
				const auto runtime_policy = detail::choose_jpeg_dct_runtime_policy(
				    planned_selected_vector_count, full_vector_count, selected_chunks_fit);
				rowgroup_plan.selected_vector_count = planned_selected_vector_count;
				rowgroup_plan.full_vector_count     = full_vector_count;
				rowgroup_plan.selected_chunks_fit   = selected_chunks_fit;
				rowgroup_plan.runtime_policy        = runtime_policy;
				rowgroup_plan.has_vector_plan       = true;
				if (runtime_policy.decision == detail::JpegDctRuntimePolicyDecision::kSelectedVectors) {
					rowgroup_plan.selected_gather_items = detail::remap_items_to_selected_vectors(
					    rowgroup_plan.items, selected_vectors, detail::kJpegDctDeviceUnpackNVectors);
					if (materialize_projection_items) {
						rowgroup_plan.selected_projection_items = detail::remap_projection_items_to_selected_vectors(
						    rowgroup_plan.projection_items, selected_vectors, detail::kJpegDctDeviceUnpackNVectors);
					}
					if (output_transformed_dct_grid) {
						rowgroup_plan.selected_fixed_transform_items =
						    detail::remap_fixed_transform_items_to_selected_vectors(
						        rowgroup_plan.fixed_transform_items, selected_vectors, detail::kJpegDctDeviceUnpackNVectors);
					}
				}
				rowgroup_plan.selected_vectors = std::move(selected_vectors);
				plan.planned_selected_vector_count += planned_selected_vector_count;
				plan.estimated_selected_vector_count +=
				    runtime_policy.decision == detail::JpegDctRuntimePolicyDecision::kFullRowgroup
				        ? full_vector_count
				        : planned_selected_vector_count;
				plan.full_vector_count += full_vector_count;
			}
		}
		plan.fixed_transform_component_count = fixed_transform_component_keys.size();
		plan.fixed_transform_output_block_count = fixed_transform_output_block_keys.size();
		plan.planned_saved_vector_count   = plan.full_vector_count - plan.planned_selected_vector_count;
		plan.estimated_saved_vector_count = plan.full_vector_count - plan.estimated_selected_vector_count;
		plan.planned_selected_vector_ratio =
		    plan.full_vector_count == 0
		        ? 0.0
		        : static_cast<double>(plan.planned_selected_vector_count) / static_cast<double>(plan.full_vector_count);
		plan.estimated_selected_vector_ratio = plan.full_vector_count == 0
		                                           ? 0.0
		                                           : static_cast<double>(plan.estimated_selected_vector_count) /
		                                                 static_cast<double>(plan.full_vector_count);
		const auto resize_cache_delta = dct_resize_cache_counter_delta(
		    resize_cache_counters_before, dct_resize_cache_counters());
		plan.resize_weight_build_ms = resize_cache_delta.resize_weight_build_ms;
		plan.dct_resize_weight_cache_hits = resize_cache_delta.resize_weight_hits;
		plan.dct_resize_weight_cache_misses = resize_cache_delta.resize_weight_misses;
		plan.dct_conversion_matrix_cache_hits = resize_cache_delta.conversion_hits;
		plan.dct_conversion_matrix_cache_misses = resize_cache_delta.conversion_misses;
		return plan;
	}

	static std::string device_plan_cache_key(const std::vector<JpegDctImageCropRequest>& requests,
	                                         const JpegDctDeviceBatchOptions&            options) {
		std::ostringstream key;
		key << static_cast<int>(options.layout) << ':'
		    << options.decode_batch_rowgroups << ':' << options.enable_rowgroup_prefetch << ':'
		    << options.rowgroup_prefetch_depth << ':' << options.rowgroup_prefetch_workers << ':'
		    << options.rowgroup_prefetch_min_decode_batches << ':';
		if (options.grid_transform.has_value()) {
			const auto& spec = *options.grid_transform;
			key << spec.y_output_width_blocks << ',' << spec.y_output_height_blocks << ','
			    << spec.cbcr_output_width_blocks << ',' << spec.cbcr_output_height_blocks << ','
			    << spec.crop_reference_width_blocks << ',' << spec.crop_reference_height_blocks << ','
			    << spec.crop_origin_alignment_blocks << ',' << spec.chroma_crop_scale_x << ','
			    << spec.chroma_crop_scale_y << ',' << spec.clamp_min << ',' << spec.clamp_max << ','
			    << spec.dequantize << ',' << spec.require_all_coefficients << ',' << spec.allow_grayscale << ':';
			for (const auto value : spec.preferred_small_crop_width_blocks) {
				key << value << ',';
			}
			key << ':';
			for (const auto value : spec.preferred_small_crop_height_blocks) {
				key << value << ',';
			}
			key << ':';
			for (const auto& ratio : spec.allowed_chroma_sampling_ratios) {
				key << ratio.horizontal_numerator << '/' << ratio.horizontal_denominator << ','
				    << ratio.vertical_numerator << '/' << ratio.vertical_denominator << ';';
			}
		} else {
			key << "none";
		}
		key << ':';
		const auto coefficients = detail::normalize_coefficient_selection(options.coefficient_selection);
		for (const auto coefficient : coefficients) {
			key << static_cast<unsigned>(coefficient) << ',';
		}
		key << ';';
		for (const auto& request : requests) {
			key << request.global_image_index << ',' << request.source_crop.x << ',' << request.source_crop.y << ','
			    << request.source_crop.width << ',' << request.source_crop.height << ';';
		}
		return key.str();
	}

	detail::JpegDctDeviceBatchPlan cached_device_batch_plan(
	    const std::vector<JpegDctImageCropRequest>& requests,
	    const JpegDctDeviceBatchOptions&            options) const {
		if (options.layout != JpegDctDeviceLayout::kTransformedDctGrid) {
			return plan_device_batch(requests, options);
		}
		const auto key = device_plan_cache_key(requests, options);
		{
			std::lock_guard<std::mutex> guard(device_plan_cache_mutex);
			const auto found = device_plan_cache.find(key);
			if (found != device_plan_cache.end()) {
				auto plan                                  = found->second;
				plan.resize_weight_build_ms                = 0.0;
				plan.dct_resize_weight_cache_hits          = 0;
				plan.dct_resize_weight_cache_misses        = 0;
				plan.dct_conversion_matrix_cache_hits      = 0;
				plan.dct_conversion_matrix_cache_misses    = 0;
				return plan;
			}
		}
		auto plan = plan_device_batch(requests, options);
		{
			std::lock_guard<std::mutex> guard(device_plan_cache_mutex);
			constexpr size_t kPlanCacheCapacity = 128;
			if (device_plan_cache.size() >= kPlanCacheCapacity) {
				device_plan_cache.erase(device_plan_cache.begin());
			}
			device_plan_cache.emplace(key, plan);
		}
		return plan;
	}

	JpegDctDeviceBatchPlanEstimate estimate_device_batch(const std::vector<JpegDctImageCropRequest>& requests,
	                                                     const JpegDctDeviceBatchOptions&            options) const {
		if (options.layout != JpegDctDeviceLayout::kImageMajorComponentBlockCoeff &&
		    options.layout != JpegDctDeviceLayout::kYcbcrDctGrid &&
		    options.layout != JpegDctDeviceLayout::kTransformedDctGrid) {
			throw std::runtime_error("unsupported JPEG DCT device output layout");
		}

		JpegDctDeviceBatchPlanEstimate estimate;
		estimate.layout = options.layout;
		std::unordered_map<uint64_t, size_t> seen_rowgroups;

		const auto add_rowgroup = [&](const ShardState& shard, const uint32_t rowgroup_index) {
			const auto key = (static_cast<uint64_t>(shard.entry.shard_id) << 32U) | rowgroup_index;
			if (!seen_rowgroups.emplace(key, estimate.rowgroups.size()).second) {
				return;
			}
			if (rowgroup_index >= shard.rowgroup_n_tuples.size()) {
				throw std::runtime_error("JPEG DCT estimated rowgroup exceeds shard rowgroup metadata");
			}
			estimate.rowgroups.push_back(JpegDctDeviceRowgroupMetadata {shard.entry.shard_id, rowgroup_index});
			estimate.full_vector_count += row_count_to_vector_count(shard.rowgroup_n_tuples[rowgroup_index]);
		};

		for (const auto& request : requests) {
			const auto& shard = shard_for_global_image(request.global_image_index);
			const auto  local_image_index =
			    static_cast<uint32_t>(request.global_image_index - shard.entry.first_global_image_index);
			if (local_image_index >= shard.metadata.images.size()) {
				throw std::runtime_error("JPEG DCT shard metadata does not contain the requested local image");
			}

			const auto& image = shard.metadata.images[local_image_index];
			const auto  crop  = effective_crop_box(image, request.source_crop);

			for (const auto& component : image.components) {
				if (!component.present || component.width_in_blocks == 0 || component.height_in_blocks == 0) {
					continue;
				}

				const uint32_t x0 = std::min(component.width_in_blocks,
				                             floor_mul_div_u32(crop.x, component.width_in_blocks, image.image_width));
				const uint32_t y0 = std::min(component.height_in_blocks,
				                             floor_mul_div_u32(crop.y, component.height_in_blocks, image.image_height));
				const uint32_t x1 =
				    std::min(component.width_in_blocks,
				             ceil_mul_div_u32(crop.x + crop.width, component.width_in_blocks, image.image_width));
				const uint32_t y1 =
				    std::min(component.height_in_blocks,
				             ceil_mul_div_u32(crop.y + crop.height, component.height_in_blocks, image.image_height));
				for (uint32_t block_y = y0; block_y < y1; ++block_y) {
					for (uint32_t block_x = x0; block_x < x1; ++block_x) {
						const auto* group = find_group_or_null(shard, component.semantic_slot_id, block_x, block_y);
						if (group == nullptr) {
							throw std::runtime_error("JPEG DCT block group was not found in shard metadata");
						}

						++estimate.block_count;
						add_rowgroup(shard, group->fls_rowgroup_index);
					}
				}
			}
		}

		std::sort(estimate.rowgroups.begin(), estimate.rowgroups.end(), [](const auto& lhs, const auto& rhs) {
			if (lhs.shard_id != rhs.shard_id) {
				return lhs.shard_id < rhs.shard_id;
			}
			return lhs.rowgroup_index < rhs.rowgroup_index;
		});
		return estimate;
	}

	void attach_device_runtime_resources(detail::JpegDctDeviceBatchPlan&  plan,
	                                     const JpegDctDeviceBatchOptions& options) const {
		if (!device_scratch) {
			device_scratch = detail::make_jpeg_dct_device_scratch();
		}
		plan.scratch = device_scratch.get();
		const bool cache_compatible =
		                              (options.layout == JpegDctDeviceLayout::kImageMajorComponentBlockCoeff ||
		                               options.layout == JpegDctDeviceLayout::kTransformedDctGrid) &&
		                              detail::selects_all_coefficients(plan.selected_coefficients);
		plan.cache_enabled = options.cache_capacity_bytes > 0 && cache_compatible;
		if (options.cache_capacity_bytes > 0 && cache_compatible) {
			if (device_cache == nullptr) {
				device_cache = std::make_unique<detail::JpegDctDeviceDecodedRowgroupCache>();
			}
			device_cache->set_capacity(options.cache_capacity_bytes);
			plan.cache = device_cache.get();
		} else if (device_cache != nullptr) {
			device_cache->set_capacity(0);
			plan.cache = nullptr;
		}
	}

	std::filesystem::path                                              root_dir;
	JpegDctShardManifest                                               manifest;
	std::vector<ShardState>                                            shards;
	std::shared_ptr<const uint8_t>                                     plan_owner_token = std::make_shared<uint8_t>(0);
	mutable std::unique_ptr<detail::JpegDctDeviceDecodedRowgroupCache> device_cache;
	mutable detail::JpegDctDeviceScratchPtr                            device_scratch;
	mutable std::mutex                                                 device_plan_cache_mutex;
	mutable std::unordered_map<std::string, detail::JpegDctDeviceBatchPlan> device_plan_cache;
};

JpegDctShardDatasetReader::JpegDctShardDatasetReader(const std::filesystem::path& manifest_path)
    : impl_(std::make_unique<Impl>(manifest_path)) {
}

JpegDctShardDatasetReader::~JpegDctShardDatasetReader() = default;

JpegDctShardDatasetReader::JpegDctShardDatasetReader(JpegDctShardDatasetReader&&) noexcept = default;

JpegDctShardDatasetReader& JpegDctShardDatasetReader::operator=(JpegDctShardDatasetReader&&) noexcept = default;

struct JpegDctDeviceBatchPreparedPlan::Impl {
	Impl(detail::JpegDctDeviceBatchPlan plan_in,
	     JpegDctDeviceBatchOptions      options_in,
	     std::shared_ptr<const uint8_t> owner_token_in)
	    : plan(std::move(plan_in))
	    , options(options_in)
	    , owner_token(std::move(owner_token_in)) {
	}

	detail::JpegDctDeviceBatchPlan plan;
	JpegDctDeviceBatchOptions      options;
	std::shared_ptr<const uint8_t> owner_token;
};

JpegDctDeviceBatchPreparedPlan::JpegDctDeviceBatchPreparedPlan() noexcept = default;

JpegDctDeviceBatchPreparedPlan::JpegDctDeviceBatchPreparedPlan(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
}

JpegDctDeviceBatchPreparedPlan::~JpegDctDeviceBatchPreparedPlan() = default;

JpegDctDeviceBatchPreparedPlan::JpegDctDeviceBatchPreparedPlan(JpegDctDeviceBatchPreparedPlan&&) noexcept = default;

JpegDctDeviceBatchPreparedPlan&
JpegDctDeviceBatchPreparedPlan::operator=(JpegDctDeviceBatchPreparedPlan&&) noexcept = default;

bool JpegDctDeviceBatchPreparedPlan::empty() const noexcept {
	return impl_ == nullptr;
}

JpegDctDeviceLayout JpegDctDeviceBatchPreparedPlan::layout() const noexcept {
	return impl_ ? impl_->plan.layout : JpegDctDeviceLayout::kImageMajorComponentBlockCoeff;
}

const std::vector<JpegDctDeviceImageLayout>& JpegDctDeviceBatchPreparedPlan::image_layouts() const noexcept {
	static const std::vector<JpegDctDeviceImageLayout> empty;
	return impl_ ? impl_->plan.image_layouts : empty;
}

const std::vector<JpegDctDeviceBlockMetadata>& JpegDctDeviceBatchPreparedPlan::block_metadata() const noexcept {
	static const std::vector<JpegDctDeviceBlockMetadata> empty;
	return impl_ ? impl_->plan.block_metadata : empty;
}

const std::vector<JpegDctDeviceRowgroupMetadata>& JpegDctDeviceBatchPreparedPlan::rowgroups() const noexcept {
	static const std::vector<JpegDctDeviceRowgroupMetadata> empty;
	return impl_ ? impl_->plan.rowgroups : empty;
}

size_t JpegDctDeviceBatchPreparedPlan::planned_selected_vector_count() const noexcept {
	return impl_ ? impl_->plan.planned_selected_vector_count : 0;
}

size_t JpegDctDeviceBatchPreparedPlan::estimated_selected_vector_count() const noexcept {
	return impl_ ? impl_->plan.estimated_selected_vector_count : 0;
}

size_t JpegDctDeviceBatchPreparedPlan::full_vector_count() const noexcept {
	return impl_ ? impl_->plan.full_vector_count : 0;
}

size_t JpegDctDeviceBatchPreparedPlan::planned_saved_vector_count() const noexcept {
	return impl_ ? impl_->plan.planned_saved_vector_count : 0;
}

size_t JpegDctDeviceBatchPreparedPlan::estimated_saved_vector_count() const noexcept {
	return impl_ ? impl_->plan.estimated_saved_vector_count : 0;
}

const std::vector<uint8_t>& JpegDctDeviceBatchPreparedPlan::selected_coefficients() const noexcept {
	static const std::vector<uint8_t> empty;
	return impl_ ? impl_->plan.selected_coefficients : empty;
}

size_t JpegDctDeviceBatchPreparedPlan::coefficients_per_block() const noexcept {
	return impl_ ? impl_->plan.coefficients_per_block : detail::kJpegDctCoefficientCount;
}

double JpegDctDeviceBatchPreparedPlan::planned_selected_vector_ratio() const noexcept {
	return impl_ ? impl_->plan.planned_selected_vector_ratio : 0.0;
}

double JpegDctDeviceBatchPreparedPlan::estimated_selected_vector_ratio() const noexcept {
	return impl_ ? impl_->plan.estimated_selected_vector_ratio : 0.0;
}

double JpegDctDeviceBatchPreparedPlan::planning_ms() const noexcept {
	return impl_ ? impl_->plan.planning_ms : 0.0;
}

uint64_t JpegDctShardDatasetReader::image_count() const noexcept {
	return impl_ == nullptr ? 0 : impl_->manifest.image_count;
}

JpegImageMetadata JpegDctShardDatasetReader::ImageMetadata(const uint32_t global_image_index) const {
	if (impl_ == nullptr) {
		throw std::runtime_error("JPEG DCT shard dataset reader is not initialized");
	}
	const auto& shard             = impl_->shard_for_global_image(global_image_index);
	const auto  local_image_index = static_cast<uint32_t>(global_image_index - shard.entry.first_global_image_index);
	if (local_image_index >= shard.metadata.images.size()) {
		throw std::runtime_error("JPEG DCT shard metadata does not contain the requested local image");
	}
	return shard.metadata.images[local_image_index];
}

JpegDctDeviceBatchPlanPreview
JpegDctShardDatasetReader::PlanDeviceDctBatch(const std::vector<JpegDctImageCropRequest>& requests,
                                              const JpegDctDeviceBatchOptions&            options) const {
	if (impl_ == nullptr) {
		throw std::runtime_error("JPEG DCT shard dataset reader is not initialized");
	}
	const auto plan_start = std::chrono::steady_clock::now();
	auto       plan       = impl_->plan_device_batch(requests, options);
	const auto plan_end   = std::chrono::steady_clock::now();

	JpegDctDeviceBatchPlanPreview preview;
	preview.layout                          = plan.layout;
	preview.image_layouts                   = std::move(plan.image_layouts);
	preview.block_metadata                  = std::move(plan.block_metadata);
	preview.rowgroups                       = std::move(plan.rowgroups);
	preview.planned_selected_vector_count   = plan.planned_selected_vector_count;
	preview.estimated_selected_vector_count = plan.estimated_selected_vector_count;
	preview.full_vector_count               = plan.full_vector_count;
	preview.planned_saved_vector_count      = plan.planned_saved_vector_count;
	preview.estimated_saved_vector_count    = plan.estimated_saved_vector_count;
	preview.selected_coefficients           = std::move(plan.selected_coefficients);
	preview.coefficients_per_block          = plan.coefficients_per_block;
	preview.planned_selected_vector_ratio   = plan.planned_selected_vector_ratio;
	preview.estimated_selected_vector_ratio = plan.estimated_selected_vector_ratio;
	preview.ycbcr_dct_grid_shape            = plan.ycbcr_dct_grid_shape;
	preview.planning_ms                     = std::chrono::duration<double, std::milli>(plan_end - plan_start).count();
	preview.resize_weight_build_ms          = plan.resize_weight_build_ms;
	preview.dct_resize_weight_cache_hits       = plan.dct_resize_weight_cache_hits;
	preview.dct_resize_weight_cache_misses     = plan.dct_resize_weight_cache_misses;
	preview.dct_conversion_matrix_cache_hits   = plan.dct_conversion_matrix_cache_hits;
	preview.dct_conversion_matrix_cache_misses = plan.dct_conversion_matrix_cache_misses;
	return preview;
}

JpegDctDeviceBatchPlanEstimate
JpegDctShardDatasetReader::EstimateDeviceDctBatch(const std::vector<JpegDctImageCropRequest>& requests,
                                                  const JpegDctDeviceBatchOptions&            options) const {
	if (impl_ == nullptr) {
		throw std::runtime_error("JPEG DCT shard dataset reader is not initialized");
	}
	const auto estimate_start = std::chrono::steady_clock::now();
	auto       estimate       = impl_->estimate_device_batch(requests, options);
	const auto estimate_end   = std::chrono::steady_clock::now();
	estimate.planning_ms      = std::chrono::duration<double, std::milli>(estimate_end - estimate_start).count();
	return estimate;
}

JpegDctDeviceBatchPreparedPlan
JpegDctShardDatasetReader::PrepareDeviceDctBatch(const std::vector<JpegDctImageCropRequest>& requests,
                                                 const JpegDctDeviceBatchOptions&            options) {
	if (impl_ == nullptr) {
		throw std::runtime_error("JPEG DCT shard dataset reader is not initialized");
	}
	const auto plan_start = std::chrono::steady_clock::now();
	auto       plan       = impl_->cached_device_batch_plan(requests, options);
	const auto plan_end   = std::chrono::steady_clock::now();
	plan.planning_ms      = std::chrono::duration<double, std::milli>(plan_end - plan_start).count();
	return JpegDctDeviceBatchPreparedPlan(
	    std::make_unique<JpegDctDeviceBatchPreparedPlan::Impl>(std::move(plan), options, impl_->plan_owner_token));
}

JpegDctDeviceBatch JpegDctShardDatasetReader::ReadPreparedDeviceDctBatch(JpegDctDeviceBatchPreparedPlan plan) {
	if (impl_ == nullptr) {
		throw std::runtime_error("JPEG DCT shard dataset reader is not initialized");
	}
	if (plan.impl_ == nullptr) {
		throw std::runtime_error("JPEG DCT prepared device batch plan is empty");
	}
	if (plan.impl_->owner_token != impl_->plan_owner_token) {
		throw std::runtime_error("JPEG DCT prepared device batch plan belongs to a different reader");
	}
	auto       detail_plan         = std::move(plan.impl_->plan);
	const auto requested_selection = detail::normalize_coefficient_selection(plan.impl_->options.coefficient_selection);
	if (requested_selection != detail_plan.selected_coefficients) {
		throw std::runtime_error("JPEG DCT prepared device batch coefficient selection was modified");
	}
	impl_->attach_device_runtime_resources(detail_plan, plan.impl_->options);
	plan.impl_.reset();
	return detail::execute_jpeg_dct_device_batch_plan(std::move(detail_plan));
}

JpegDctDeviceBatch JpegDctShardDatasetReader::ReadDeviceDctBatch(const std::vector<JpegDctImageCropRequest>& requests,
                                                                 const JpegDctDeviceBatchOptions&            options) {
	return ReadPreparedDeviceDctBatch(PrepareDeviceDctBatch(requests, options));
}

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
	ref.shard_id          = shard.entry.shard_id;
	ref.local_image_index = local_image_index;
	ref.semantic_slot_id  = semantic_slot_id;
	ref.block_x           = block_x;
	ref.block_y           = block_y;

	const auto has_target =
	    Impl::image_has_block(shard.metadata.images[local_image_index], semantic_slot_id, block_x, block_y);
	const auto* group = Impl::find_group_or_null(shard.metadata, semantic_slot_id, block_x, block_y);
	if (group == nullptr) {
		if (!has_target) {
			return ref;
		}
		throw std::runtime_error("JPEG DCT block group was not found in shard metadata");
	}
	ref.fls_rowgroup_index    = group->fls_rowgroup_index;
	ref.row_start_in_rowgroup = group->row_start_in_rowgroup;

	uint32_t rank = 0;
	for (uint32_t image_idx = 0; image_idx < local_image_index; ++image_idx) {
		if (Impl::image_has_block(shard.metadata.images[image_idx], semantic_slot_id, block_x, block_y)) {
			++rank;
		}
	}
	ref.row_offset_in_block_group = rank;
	ref.present                   = has_target;
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
	const auto&           shard = impl_->shard_for_global_image(global_image_index);
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
		const auto& rowgroup = materialized_rowgroup(ref.fls_rowgroup_index);
		const auto  materialized_row_idx =
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
