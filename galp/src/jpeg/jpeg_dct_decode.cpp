#include "jpeg/jpeg_dct_decode.hpp"
#include "galp/jpeg_dct_storage.hpp"
#include "jpeg/jpeg_dct_order.hpp"
#include <algorithm>
#include <csetjmp>
#include <cstdio>
#include <future>
#include <jpeglib.h>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace galp::jpeg::detail {
namespace {

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
	auto* error = reinterpret_cast<JpegErrorManager*>(cinfo->err);
	(*cinfo->err->format_message)(cinfo, error->message);
	longjmp(error->setjmp_buffer, 1);
}

extern "C" void jpeg_emit_message_bridge(j_common_ptr cinfo, int msg_level) {
	auto* error = reinterpret_cast<JpegErrorManager*>(cinfo->err);
	if (msg_level < 0) {
		++error->warning_count;
	}
}

FilePtr open_jpeg_file(const std::filesystem::path& path) {
	FILE* file = std::fopen(path.string().c_str(), "rb");
	if (file == nullptr) {
		throw std::runtime_error("failed to open JPEG file: " + path.string());
	}
	return FilePtr(file);
}

class JpegDecoder {
public:
	explicit JpegDecoder(std::filesystem::path input_path)
	    : path_(std::move(input_path))
	    , file_(open_jpeg_file(path_)) {
		cinfo.err               = jpeg_std_error(&error_.pub);
		error_.pub.error_exit   = jpeg_error_exit_bridge;
		error_.pub.emit_message = jpeg_emit_message_bridge;
		checked_void([&] { jpeg_create_decompress(&cinfo); });
		created_ = true;
	}

	JpegDecoder(const JpegDecoder&)            = delete;
	JpegDecoder& operator=(const JpegDecoder&) = delete;

	~JpegDecoder() {
		if (created_) {
			jpeg_destroy_decompress(&cinfo);
		}
	}

	void set_stdio_src() {
		checked_void([&] { jpeg_stdio_src(&cinfo, file_.get()); });
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

	JBLOCKARRAY access_virt_barray(jvirt_barray_ptr coefficient_array, JDIMENSION start_row) {
		return checked_value([&] {
			return (*cinfo.mem->access_virt_barray)(
			    reinterpret_cast<j_common_ptr>(&cinfo), coefficient_array, start_row, 1, FALSE);
		});
	}

	void finish_decompress() {
		checked_void([&] { jpeg_finish_decompress(&cinfo); });
	}

	[[nodiscard]] uint32_t warning_count() const noexcept {
		return error_.warning_count;
	}

	jpeg_decompress_struct cinfo {};

private:
	[[noreturn]] void throw_error() const {
		throw std::runtime_error("libjpeg failed while reading '" + path_.string() + "': " + error_.message);
	}

	template <typename Function>
	void checked_void(Function&& function) {
		if (setjmp(error_.setjmp_buffer) != 0) {
			throw_error();
		}
		function();
	}

	template <typename Function>
	auto checked_value(Function&& function) -> decltype(function()) {
		if (setjmp(error_.setjmp_buffer) != 0) {
			throw_error();
		}
		return function();
	}

	std::filesystem::path path_;
	FilePtr               file_;
	JpegErrorManager      error_ {};
	bool                  created_ = false;
};

std::vector<size_t> selected_component_indices(const int num_components, const JpegDctReaderOptions& options) {
	if (options.component_mode == JpegComponentMode::kAllComponents) {
		std::vector<size_t> indices(static_cast<size_t>(num_components));
		for (size_t index = 0; index < indices.size(); ++index) {
			indices[index] = index;
		}
		return indices;
	}
	if (options.selected_component_index < 0 || options.selected_component_index >= num_components) {
		std::ostringstream message;
		message << "selected JPEG component index " << options.selected_component_index << " is outside [0, "
		        << num_components << ")";
		throw std::runtime_error(message.str());
	}
	return {static_cast<size_t>(options.selected_component_index)};
}

DecodedDctRow block_to_row(const JCOEF* block, const bool use_zigzag_columns) {
	DecodedDctRow row {};
	for (size_t column = 0; column < row.size(); ++column) {
		const auto natural_index = use_zigzag_columns ? kZigzagColumnToNaturalIndex[column] : column;
		const auto value         = block[natural_index];
		if (value < std::numeric_limits<int16_t>::min() || value > std::numeric_limits<int16_t>::max()) {
			throw std::runtime_error("JPEG DCT coefficient is outside int16_t range");
		}
		row[column] = static_cast<int16_t>(value);
	}
	return row;
}

std::vector<size_t>
make_coord_to_index(const uint32_t width, const uint32_t height, const std::vector<MortonBlockCoord>& order) {
	std::vector<size_t> result(static_cast<size_t>(width) * height);
	for (size_t index = 0; index < order.size(); ++index) {
		result[static_cast<size_t>(order[index].y) * width + order[index].x] = index;
	}
	return result;
}

const JpegQuantTableMetadata* find_quant_table(const JpegImageMetadata& image, const int table_number) {
	if (table_number < 0) {
		return nullptr;
	}
	for (const auto& table : image.quant_tables) {
		if (table.table_id == static_cast<uint8_t>(table_number)) {
			return &table;
		}
	}
	return nullptr;
}

uint64_t quant_table_fingerprint(const JpegImageMetadata& image, const int table_number) {
	const auto* table = find_quant_table(image, table_number);
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

template <typename DecodeFunction>
std::vector<DecodedImage> decode_parallel(const std::vector<std::filesystem::path>& paths,
                                          const JpegDctReaderOptions&               options,
                                          const size_t                              requested_threads,
                                          DecodeFunction&&                          decode) {
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

	std::vector<DecodedImage>      images(paths.size());
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

void append_row(JpegDctTable& table, const DecodedDctRow& row) {
	for (size_t column = 0; column < row.size(); ++column) {
		table.columns[column].push_back(row[column]);
	}
	++table.row_count;
	++table.real_row_count;
}

size_t checked_add_rows(const size_t accumulated, const size_t rows) {
	if (rows > std::numeric_limits<size_t>::max() - accumulated) {
		throw std::runtime_error("JPEG DCT table row count overflow");
	}
	return accumulated + rows;
}

size_t decoded_image_row_count(const DecodedImage& image) {
	size_t row_count = 0;
	for (const auto& component : image.components) {
		row_count = checked_add_rows(row_count, component.blocks.size());
	}
	return row_count;
}

size_t decoded_row_count(const std::vector<DecodedImage>& images) {
	size_t row_count = 0;
	for (const auto& image : images) {
		row_count = checked_add_rows(row_count, decoded_image_row_count(image));
	}
	return row_count;
}

size_t layout_row_count(const DatasetLayout& layout) {
	size_t row_count = 0;
	for (const auto& components : layout.image_slot_components) {
		for (const auto* component : components) {
			if (component != nullptr) {
				row_count = checked_add_rows(row_count, component->blocks.size());
			}
		}
	}
	return row_count;
}

std::array<size_t, 64> reserve_table_rows(JpegDctTable& table, const size_t row_count) {
	std::array<size_t, 64> reserved_capacities {};
	for (size_t column = 0; column < table.columns.size(); ++column) {
		table.columns[column].reserve(row_count);
		reserved_capacities[column] = table.columns[column].capacity();
	}
	return reserved_capacities;
}

void update_peak_row_slots(const JpegDctTable& table, JpegDctTableBuildStats* stats) {
	if (stats == nullptr) {
		return;
	}
	stats->peak_coefficient_row_slots =
	    std::max(stats->peak_coefficient_row_slots, checked_add_rows(table.row_count, stats->remaining_decoded_rows));
}

void release_component_storage(DecodedComponent& component, JpegDctTableBuildStats* stats) {
	const auto decoded_rows = component.blocks.size();
	std::vector<DecodedDctRow>().swap(component.blocks);
	std::vector<size_t>().swap(component.coord_to_block_index);
	if (stats == nullptr) {
		return;
	}
	if (decoded_rows > stats->remaining_decoded_rows) {
		throw std::runtime_error("JPEG DCT decoded-row accounting underflow");
	}
	stats->remaining_decoded_rows -= decoded_rows;
	stats->released_decoded_rows = checked_add_rows(stats->released_decoded_rows, decoded_rows);
}

void release_image_storage(DecodedImage& image, JpegDctTableBuildStats* stats) {
	for (auto& component : image.components) {
		release_component_storage(component, stats);
	}
}

DecodedComponent* find_mutable_component_for_slot(DecodedImage& image, const ComponentSlot& slot) {
	for (auto& component : image.components) {
		if (component.metadata.component_id == slot.component_id) {
			return &component;
		}
	}
	return nullptr;
}

void finalize_build_stats(const JpegDctTable&           table,
                          const std::array<size_t, 64>& reserved_capacities,
                          const size_t                  expected_rows,
                          JpegDctTableBuildStats*       stats) {
	if (table.row_count != expected_rows) {
		throw std::runtime_error("JPEG DCT table row count changed after exact preallocation");
	}
	if (stats == nullptr) {
		return;
	}
	for (size_t column = 0; column < table.columns.size(); ++column) {
		if (table.columns[column].capacity() != reserved_capacities[column]) {
			++stats->column_capacity_growths;
		}
	}
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

bool same_semantic_slot(const JpegComponentMetadata& metadata, const ComponentSlot& slot) {
	return metadata.component_id == slot.component_id;
}

std::vector<std::vector<const DecodedComponent*>> make_image_slot_components(const std::vector<DecodedImage>&  images,
                                                                             const std::vector<ComponentSlot>& slots) {
	std::vector<std::vector<const DecodedComponent*>> result;
	result.reserve(images.size());
	for (const auto& image : images) {
		auto& row = result.emplace_back(slots.size(), nullptr);
		for (size_t slot_index = 0; slot_index < slots.size(); ++slot_index) {
			row[slot_index] = find_component_for_slot(image, slots[slot_index]);
		}
	}
	return result;
}

JpegImageMetadata normalized_image_metadata(const JpegImageMetadata&                    image,
                                            const std::vector<ComponentSlot>&           slots,
                                            const std::vector<const DecodedComponent*>& components,
                                            std::vector<JpegEncodingProfileMetadata>&   profiles) {
	JpegImageMetadata output;
	output.source_path      = image.source_path;
	output.image_width      = image.image_width;
	output.image_height     = image.image_height;
	output.data_precision   = image.data_precision;
	output.jpeg_color_space = image.jpeg_color_space;
	output.progressive      = image.progressive;
	output.warning_count    = image.warning_count;
	output.quant_tables     = image.quant_tables;
	output.markers          = image.markers;
	output.components.reserve(slots.size());
	for (size_t slot_index = 0; slot_index < slots.size(); ++slot_index) {
		const auto&           slot = slots[slot_index];
		JpegComponentMetadata metadata;
		metadata.semantic_slot_id        = slot.semantic_slot_id;
		metadata.component_index         = slot.component_index;
		metadata.component_id            = slot.component_id;
		metadata.padded_width_in_blocks  = slot.max_width_in_blocks;
		metadata.padded_height_in_blocks = slot.max_height_in_blocks;
		if (const auto* component = components[slot_index]) {
			metadata.local_component_index   = component->metadata.local_component_index;
			metadata.component_id            = component->metadata.component_id;
			metadata.width_in_blocks         = component->metadata.width_in_blocks;
			metadata.height_in_blocks        = component->metadata.height_in_blocks;
			metadata.h_samp_factor           = component->metadata.h_samp_factor;
			metadata.v_samp_factor           = component->metadata.v_samp_factor;
			metadata.quant_tbl_no            = component->metadata.quant_tbl_no;
			metadata.quant_table_fingerprint = component->metadata.quant_table_fingerprint;
			metadata.encoding_profile_id     = encoding_profile_id(profiles, image, component->metadata);
			metadata.present                 = true;
		} else {
			metadata.width_in_blocks  = 0;
			metadata.height_in_blocks = 0;
			metadata.present          = false;
		}
		output.components.push_back(metadata);
	}
	return output;
}

std::vector<JpegComponentMetadata> dataset_component_metadata(const std::vector<ComponentSlot>& slots) {
	std::vector<JpegComponentMetadata> result;
	result.reserve(slots.size());
	for (const auto& slot : slots) {
		JpegComponentMetadata metadata;
		metadata.semantic_slot_id        = slot.semantic_slot_id;
		metadata.component_index         = slot.component_index;
		metadata.component_id            = slot.component_id;
		metadata.width_in_blocks         = slot.max_width_in_blocks;
		metadata.height_in_blocks        = slot.max_height_in_blocks;
		metadata.padded_width_in_blocks  = slot.max_width_in_blocks;
		metadata.padded_height_in_blocks = slot.max_height_in_blocks;
		result.push_back(metadata);
	}
	return result;
}

const DecodedDctRow* find_component_block(const DecodedComponent& component, const uint32_t x, const uint32_t y) {
	if (x >= component.metadata.width_in_blocks || y >= component.metadata.height_in_blocks) {
		return nullptr;
	}
	const size_t flat_index = static_cast<size_t>(y) * component.metadata.width_in_blocks + x;
	if (flat_index >= component.coord_to_block_index.size()) {
		return nullptr;
	}
	const size_t block_index = component.coord_to_block_index[flat_index];
	return block_index < component.blocks.size() ? &component.blocks[block_index] : nullptr;
}

DatasetLayout make_dataset_layout(const std::vector<DecodedImage>&  images,
                                  const std::vector<ComponentSlot>* global_slots) {
	DatasetLayout layout;
	layout.slots                 = global_slots == nullptr ? normalize_component_slots(images) : *global_slots;
	layout.image_slot_components = make_image_slot_components(images, layout.slots);
	return layout;
}

} // namespace

DecodedImage decode_jpeg_coefficients(const std::filesystem::path& path, const JpegDctReaderOptions& options) {
	static_assert(sizeof(JCOEF) <= sizeof(int16_t), "JCOEF wider than int16_t requires an explicit conversion policy");
	JpegDecoder decoder(path);
	decoder.set_stdio_src();
	if (options.capture_metadata_markers) {
		decoder.save_metadata_markers();
	}
	decoder.read_header();
	auto&  cinfo              = decoder.cinfo;
	auto** coefficient_arrays = decoder.read_coefficients();

	DecodedImage image;
	image.metadata.source_path      = path;
	image.metadata.image_width      = cinfo.image_width;
	image.metadata.image_height     = cinfo.image_height;
	image.metadata.data_precision   = static_cast<uint8_t>(cinfo.data_precision);
	image.metadata.jpeg_color_space = static_cast<int>(cinfo.jpeg_color_space);
	image.metadata.progressive      = jpeg_has_multiple_scans(&cinfo) != 0;
	for (int table_id = 0; table_id < NUM_QUANT_TBLS; ++table_id) {
		const auto* table = cinfo.quant_tbl_ptrs[table_id];
		if (table == nullptr) {
			continue;
		}
		JpegQuantTableMetadata metadata;
		metadata.table_id = static_cast<uint8_t>(table_id);
		for (size_t index = 0; index < metadata.values.size(); ++index) {
			metadata.values[index] = table->quantval[index];
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
		const auto&      source = cinfo.comp_info[component_index];
		DecodedComponent component;
		component.metadata.semantic_slot_id        = static_cast<uint32_t>(component_index);
		component.metadata.component_index         = component_index;
		component.metadata.local_component_index   = component_index;
		component.metadata.component_id            = source.component_id;
		component.metadata.width_in_blocks         = source.width_in_blocks;
		component.metadata.height_in_blocks        = source.height_in_blocks;
		component.metadata.padded_width_in_blocks  = source.width_in_blocks;
		component.metadata.padded_height_in_blocks = source.height_in_blocks;
		component.metadata.h_samp_factor           = source.h_samp_factor;
		component.metadata.v_samp_factor           = source.v_samp_factor;
		component.metadata.quant_tbl_no            = source.quant_tbl_no;
		component.metadata.quant_table_fingerprint = quant_table_fingerprint(image.metadata, source.quant_tbl_no);
		const auto order =
		    make_block_order(source.width_in_blocks, source.height_in_blocks, options.use_z_curve_block_order);
		component.blocks.resize(order.size());
		component.coord_to_block_index = make_coord_to_index(source.width_in_blocks, source.height_in_blocks, order);
		for (JDIMENSION y = 0; y < source.height_in_blocks; ++y) {
			JBLOCKARRAY rows = decoder.access_virt_barray(coefficient_arrays[component_index], y);
			for (JDIMENSION x = 0; x < source.width_in_blocks; ++x) {
				const auto block_index =
				    component.coord_to_block_index[static_cast<size_t>(y) * source.width_in_blocks + x];
				component.blocks[block_index] = block_to_row(rows[0][x], options.use_zigzag_columns);
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
	auto&        cinfo = decoder.cinfo;
	DecodedImage image;
	image.metadata.source_path      = path;
	image.metadata.image_width      = cinfo.image_width;
	image.metadata.image_height     = cinfo.image_height;
	image.metadata.data_precision   = static_cast<uint8_t>(cinfo.data_precision);
	image.metadata.jpeg_color_space = static_cast<int>(cinfo.jpeg_color_space);
	image.metadata.progressive      = jpeg_has_multiple_scans(&cinfo) != 0;
	for (const auto component_index : selected_component_indices(cinfo.num_components, options)) {
		const auto&      source = cinfo.comp_info[component_index];
		DecodedComponent component;
		component.metadata.semantic_slot_id        = static_cast<uint32_t>(component_index);
		component.metadata.component_index         = component_index;
		component.metadata.local_component_index   = component_index;
		component.metadata.component_id            = source.component_id;
		component.metadata.width_in_blocks         = source.width_in_blocks;
		component.metadata.height_in_blocks        = source.height_in_blocks;
		component.metadata.padded_width_in_blocks  = source.width_in_blocks;
		component.metadata.padded_height_in_blocks = source.height_in_blocks;
		component.metadata.h_samp_factor           = source.h_samp_factor;
		component.metadata.v_samp_factor           = source.v_samp_factor;
		component.metadata.quant_tbl_no            = source.quant_tbl_no;
		image.metadata.components.push_back(component.metadata);
		image.components.push_back(std::move(component));
	}
	image.metadata.warning_count = decoder.warning_count();
	return image;
}

std::vector<DecodedImage> decode_jpeg_coefficients_parallel(const std::vector<std::filesystem::path>& paths,
                                                            const JpegDctReaderOptions&               options,
                                                            const size_t                              threads) {
	return decode_parallel(paths, options, threads, decode_jpeg_coefficients);
}

std::vector<DecodedImage> decode_jpeg_layouts_parallel(const std::vector<std::filesystem::path>& paths,
                                                       const JpegDctReaderOptions&               options,
                                                       const size_t                              threads) {
	return decode_parallel(paths, options, threads, decode_jpeg_layout);
}

void validate_supported_layout_options(const JpegDctReaderOptions& options) {
	if (options.compression_partition_policy != JpegCompressionPartitionPolicy::kBySemanticSlot) {
		throw std::runtime_error(
		    "JPEG DCT compression partition policy is not implemented yet; only kBySemanticSlot is supported");
	}
	if (options.coefficient_encoding != JpegDctCoefficientEncoding::kDense64FastLanes) {
		throw std::runtime_error(
		    "JPEG DCT coefficient encoding is not implemented yet; only kDense64FastLanes is supported");
	}
}

void validate_decoded_dataset(const std::vector<DecodedImage>& images) {
	if (images.empty()) {
		throw std::runtime_error("JPEG DCT dataset requires at least one image");
	}
}

const DecodedComponent* find_component_for_slot(const DecodedImage& image, const ComponentSlot& slot) {
	for (const auto& component : image.components) {
		if (same_semantic_slot(component.metadata, slot)) {
			return &component;
		}
	}
	return nullptr;
}

std::vector<ComponentSlot> normalize_component_slots(const std::vector<DecodedImage>& images) {
	std::vector<ComponentSlot> slots;
	for (const auto& image : images) {
		for (const auto& component : image.components) {
			auto found = std::find_if(slots.begin(), slots.end(), [&](const ComponentSlot& slot) {
				return same_semantic_slot(component.metadata, slot);
			});
			if (found == slots.end()) {
				slots.push_back(ComponentSlot {static_cast<uint32_t>(slots.size()),
				                               slots.size(),
				                               component.metadata.component_id,
				                               component.metadata.width_in_blocks,
				                               component.metadata.height_in_blocks});
			} else {
				found->max_width_in_blocks = std::max(found->max_width_in_blocks, component.metadata.width_in_blocks);
				found->max_height_in_blocks =
				    std::max(found->max_height_in_blocks, component.metadata.height_in_blocks);
			}
		}
	}
	return slots;
}

JpegDctTable make_single_image_table(DecodedImage image, const JpegDctReaderOptions& options) {
	validate_supported_layout_options(options);
	JpegDctTable table;
	const auto   expected_rows       = decoded_image_row_count(image);
	const auto   reserved_capacities = reserve_table_rows(table, expected_rows);
	for (size_t index = 0; index < image.metadata.components.size(); ++index) {
		auto& component                 = image.metadata.components[index];
		component.semantic_slot_id      = static_cast<uint32_t>(index);
		component.local_component_index = component.component_index;
		component.encoding_profile_id =
		    encoding_profile_id(table.metadata.encoding_profiles, image.metadata, component);
		if (index < image.components.size()) {
			image.components[index].metadata = component;
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
	for (auto& component : image.components) {
		const auto order = make_block_order(
		    component.metadata.width_in_blocks, component.metadata.height_in_blocks, options.use_z_curve_block_order);
		for (size_t block_index = 0; block_index < component.blocks.size(); ++block_index) {
			JpegDctBlockGroupIndex group;
			group.semantic_slot_id = component.metadata.semantic_slot_id;
			group.z_order_index    = static_cast<uint32_t>(block_index);
			group.block_x          = order[block_index].x;
			group.block_y          = order[block_index].y;
			group.row_start        = table.row_count;
			group.row_count        = 1;
			table.metadata.block_group_index.push_back(group);
			append_row(table, component.blocks[block_index]);
		}
		release_component_storage(component, nullptr);
	}
	table.block_group_count = table.metadata.block_group_index.size();
	finalize_build_stats(table, reserved_capacities, expected_rows, nullptr);
	return table;
}

JpegDctTable make_dataset_table(std::vector<DecodedImage>         images,
                                const JpegDctReaderOptions&       options,
                                const std::vector<ComponentSlot>* global_slots,
                                JpegDctPhysicalLayout             physical_layout,
                                JpegDctTableBuildStats*           build_stats) {
	validate_supported_layout_options(options);
	if (global_slots == nullptr) {
		validate_decoded_dataset(images);
	}
	const auto   layout = make_dataset_layout(images, global_slots);
	JpegDctTable table;
	const auto   expected_rows       = layout_row_count(layout);
	const auto   reserved_capacities = reserve_table_rows(table, expected_rows);
	if (build_stats != nullptr) {
		*build_stats                            = {};
		build_stats->expected_table_rows        = expected_rows;
		build_stats->initial_decoded_rows       = decoded_row_count(images);
		build_stats->remaining_decoded_rows     = build_stats->initial_decoded_rows;
		build_stats->peak_coefficient_row_slots = build_stats->initial_decoded_rows;
	}
	table.metadata.row_ordering                 = is_image_major_physical_layout(physical_layout)
	                                                  ? JpegDctRowOrdering::kDatasetImageMajorComponentBlockMajor
	                                                  : JpegDctRowOrdering::kDatasetComponentMajorBlockMajorImageMinor;
	table.metadata.compression_partition_policy = options.compression_partition_policy;
	table.metadata.coefficient_encoding         = options.coefficient_encoding;
	table.metadata.zigzag_columns               = options.use_zigzag_columns;
	table.metadata.image_major_spatial_order    = options.image_major_spatial_order;
	table.metadata.z_curve_block_order          = is_image_major_physical_layout(physical_layout)
	                                                  ? options.image_major_spatial_order == JpegDctSpatialOrder::kZOrder
	                                                  : options.use_z_curve_block_order;
	table.metadata.image_count                  = images.size();
	table.metadata.semantic_components          = dataset_component_metadata(layout.slots);
	table.metadata.images.reserve(images.size());
	for (size_t image_index = 0; image_index < images.size(); ++image_index) {
		table.metadata.images.push_back(normalized_image_metadata(images[image_index].metadata,
		                                                          layout.slots,
		                                                          layout.image_slot_components[image_index],
		                                                          table.metadata.encoding_profiles));
	}

	if (is_image_major_physical_layout(physical_layout)) {
		table.metadata.image_group_index.reserve(images.size());
		for (size_t image_index = 0; image_index < images.size(); ++image_index) {
			JpegDctImageGroupIndex group;
			group.local_image_index = static_cast<uint32_t>(image_index);
			group.row_start         = table.row_count;
			for (size_t slot_index = 0; slot_index < layout.slots.size(); ++slot_index) {
				const auto* component = layout.image_slot_components[image_index][slot_index];
				if (component == nullptr) {
					continue;
				}
				for (const auto& coordinate : make_block_order(component->metadata.width_in_blocks,
				                                               component->metadata.height_in_blocks,
				                                               options.image_major_spatial_order)) {
					const auto* row = find_component_block(*component, coordinate.x, coordinate.y);
					if (row == nullptr) {
						throw std::runtime_error("JPEG DCT image-major writer found a hole in a component grid");
					}
					append_row(table, *row);
					++group.row_count;
				}
			}
			table.metadata.image_group_index.push_back(group);
			update_peak_row_slots(table, build_stats);
			release_image_storage(images[image_index], build_stats);
		}
		finalize_build_stats(table, reserved_capacities, expected_rows, build_stats);
		return table;
	}

	for (size_t slot_index = 0; slot_index < layout.slots.size(); ++slot_index) {
		const auto& slot = layout.slots[slot_index];
		const auto  order =
		    make_block_order(slot.max_width_in_blocks, slot.max_height_in_blocks, options.use_z_curve_block_order);
		for (size_t order_index = 0; order_index < order.size(); ++order_index) {
			JpegDctBlockGroupIndex group;
			group.semantic_slot_id = slot.semantic_slot_id;
			group.z_order_index    = static_cast<uint32_t>(order_index);
			group.block_x          = order[order_index].x;
			group.block_y          = order[order_index].y;
			group.row_start        = table.row_count;
			for (const auto& components : layout.image_slot_components) {
				const auto* component = components[slot_index];
				const auto* row =
				    component == nullptr ? nullptr : find_component_block(*component, group.block_x, group.block_y);
				if (row != nullptr) {
					append_row(table, *row);
					++group.row_count;
				}
			}
			if (group.row_count != 0) {
				table.metadata.block_group_index.push_back(group);
			}
		}
		update_peak_row_slots(table, build_stats);
		for (auto& image : images) {
			if (auto* component = find_mutable_component_for_slot(image, slot)) {
				release_component_storage(*component, build_stats);
			}
		}
	}
	for (auto& image : images) {
		release_image_storage(image, build_stats);
	}
	table.block_group_count = table.metadata.block_group_index.size();
	finalize_build_stats(table, reserved_capacities, expected_rows, build_stats);
	return table;
}

JpegDctTable make_dataset_table(std::vector<DecodedImage> images, const JpegDctReaderOptions& options) {
	return make_dataset_table(
	    std::move(images), options, nullptr, JpegDctPhysicalLayout::kSpatialMajorImageMinor, nullptr);
}

} // namespace galp::jpeg::detail

namespace galp::jpeg {

JpegDctTable read_jpeg_dct_file(const std::filesystem::path& path, const JpegDctReaderOptions& options) {
	return detail::make_single_image_table(detail::decode_jpeg_coefficients(path, options), options);
}

JpegDctTable read_jpeg_dct_dataset(const std::vector<std::filesystem::path>& paths,
                                   const JpegDctReaderOptions&               options) {
	return detail::make_dataset_table(detail::decode_jpeg_coefficients_parallel(paths, options, 1), options);
}

} // namespace galp::jpeg
