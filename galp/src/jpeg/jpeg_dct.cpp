#include "galp/jpeg_dct.hpp"
#include "jpeg/jpeg_dct_order.hpp"
#include "fls/table/memory_table.hpp"
#include <algorithm>
#include <array>
#include <csetjmp>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <jpeglib.h>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace galp::jpeg {

namespace {

using DctRow = std::array<int16_t, 64>;

struct DecodedComponent {
	JpegComponentMetadata metadata;
	std::vector<DctRow>   blocks;
};

struct DecodedImage {
	JpegImageMetadata             metadata;
	std::vector<DecodedComponent> components;
};

struct ComponentSlot {
	size_t   component_index = 0;
	int      component_id    = 0;
	uint32_t max_width_in_blocks = 0;
	uint32_t max_height_in_blocks = 0;
	int      h_samp_factor = 0;
	int      v_samp_factor = 0;
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
};

extern "C" void jpeg_error_exit_bridge(j_common_ptr cinfo) {
	auto* err = reinterpret_cast<JpegErrorManager*>(cinfo->err);
	(*cinfo->err->format_message)(cinfo, err->message);
	longjmp(err->setjmp_buffer, 1);
}

FilePtr open_jpeg_file(const std::filesystem::path& path);

struct JpegDecoder {
	explicit JpegDecoder(std::filesystem::path input_path)
	    : path(std::move(input_path))
	    , file(open_jpeg_file(path)) {
		cinfo.err           = jpeg_std_error(&jerr.pub);
		jerr.pub.error_exit = jpeg_error_exit_bridge;
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

	void read_header() {
		checked_void([&] { jpeg_read_header(&cinfo, TRUE); });
	}

	jvirt_barray_ptr* read_coefficients() {
		return checked_value([&] { return jpeg_read_coefficients(&cinfo); });
	}

	JBLOCKARRAY access_virt_barray(const jvirt_barray_ptr coefficient_array, const JDIMENSION start_row) {
		return checked_value([&] {
			return (*cinfo.mem->access_virt_barray)(reinterpret_cast<j_common_ptr>(&cinfo),
			                                       coefficient_array,
			                                       start_row,
			                                       1,
			                                       FALSE);
		});
	}

	void finish_decompress() {
		checked_void([&] { jpeg_finish_decompress(&cinfo); });
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

DecodedImage decode_jpeg_coefficients(const std::filesystem::path& path, const JpegDctReaderOptions& options) {
	static_assert(sizeof(JCOEF) <= sizeof(int16_t), "JCOEF wider than int16_t requires explicit conversion policy");

	JpegDecoder decoder(path);
	decoder.set_stdio_src();
	decoder.read_header();
	auto& cinfo = decoder.cinfo;
	// libjpeg-turbo exposes the libjpeg-compatible coefficient API. This path reads the compressed-domain,
	// quantized DCT coefficient arrays directly; it does not hand-roll JPEG entropy decode, IDCT, or RGB decode.
	auto** coefficient_arrays = decoder.read_coefficients();

	DecodedImage image;
	image.metadata.source_path      = path;
	image.metadata.image_width      = cinfo.image_width;
	image.metadata.image_height     = cinfo.image_height;
	image.metadata.jpeg_color_space = static_cast<int>(cinfo.jpeg_color_space);
	image.metadata.progressive      = jpeg_has_multiple_scans(&cinfo) != 0;

	const auto indices = selected_component_indices(cinfo.num_components, options);
	image.components.reserve(indices.size());
	image.metadata.components.reserve(indices.size());

	for (const auto component_index : indices) {
		const auto& comp = cinfo.comp_info[component_index];

		DecodedComponent component;
		component.metadata.component_index  = component_index;
		component.metadata.component_id     = comp.component_id;
		component.metadata.width_in_blocks  = comp.width_in_blocks;
		component.metadata.height_in_blocks = comp.height_in_blocks;
		component.metadata.padded_width_in_blocks = comp.width_in_blocks;
		component.metadata.padded_height_in_blocks = comp.height_in_blocks;
		component.metadata.h_samp_factor    = comp.h_samp_factor;
		component.metadata.v_samp_factor    = comp.v_samp_factor;

		const auto order = detail::make_block_order(comp.width_in_blocks,
		                                            comp.height_in_blocks,
		                                            options.use_z_curve_block_order);
		component.blocks.reserve(order.size());
		for (const auto& coord : order) {
			JBLOCKARRAY block_rows = decoder.access_virt_barray(coefficient_arrays[component_index], coord.y);
			const JCOEF* block = block_rows[0][coord.x];
			component.blocks.push_back(block_to_row(block, options.use_zigzag_columns));
		}

		image.metadata.components.push_back(component.metadata);
		image.components.push_back(std::move(component));
	}

	decoder.finish_decompress();
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

JpegDctTable make_single_image_table(DecodedImage image, const JpegDctReaderOptions& options) {
	JpegDctTable table;
	table.metadata.images.push_back(std::move(image.metadata));
	table.metadata.row_ordering = JpegDctRowOrdering::kSingleImageComponentMajorBlockMajor;
	table.metadata.validation_mode = options.validation_mode;
	table.metadata.zigzag_columns = options.use_zigzag_columns;
	table.metadata.z_curve_block_order = options.use_z_curve_block_order;
	table.metadata.image_count = 1;

	for (const auto& component : image.components) {
		for (const auto& row : component.blocks) {
			append_row(table, row, false);
		}
	}
	table.block_group_count = table.row_count;
	return table;
}

const DecodedComponent* find_component_by_id(const DecodedImage& image, const int component_id) {
	for (const auto& component : image.components) {
		if (component.metadata.component_id == component_id) {
			return &component;
		}
	}
	return nullptr;
}

std::vector<ComponentSlot> normalize_component_slots(const std::vector<DecodedImage>& images) {
	std::vector<ComponentSlot> slots;
	if (images.empty()) {
		return slots;
	}

	auto find_slot = [&](const int component_id) -> ComponentSlot* {
		for (auto& slot : slots) {
			if (slot.component_id == component_id) {
				return &slot;
			}
		}
		return nullptr;
	};

	for (const auto& image : images) {
		for (const auto& component : image.components) {
			auto* slot = find_slot(component.metadata.component_id);
			if (slot == nullptr) {
				ComponentSlot new_slot;
				new_slot.component_index = slots.size();
				new_slot.component_id = component.metadata.component_id;
				new_slot.h_samp_factor = component.metadata.h_samp_factor;
				new_slot.v_samp_factor = component.metadata.v_samp_factor;
				new_slot.max_width_in_blocks = component.metadata.width_in_blocks;
				new_slot.max_height_in_blocks = component.metadata.height_in_blocks;
				slots.push_back(new_slot);
				continue;
			}

			if (slot->h_samp_factor != component.metadata.h_samp_factor ||
			    slot->v_samp_factor != component.metadata.v_samp_factor) {
				std::ostringstream msg;
				msg << "JPEG dataset component_id=" << component.metadata.component_id
				    << " has inconsistent sampling factors across images";
				throw std::runtime_error(msg.str());
			}

			slot->max_width_in_blocks = std::max(slot->max_width_in_blocks, component.metadata.width_in_blocks);
			slot->max_height_in_blocks = std::max(slot->max_height_in_blocks, component.metadata.height_in_blocks);
		}
	}
	return slots;
}

JpegImageMetadata normalized_image_metadata(const DecodedImage& image, const std::vector<ComponentSlot>& slots) {
	JpegImageMetadata out = image.metadata;
	out.components.clear();
	out.components.reserve(slots.size());
	for (const auto& slot : slots) {
		JpegComponentMetadata metadata;
		metadata.component_index = slot.component_index;
		metadata.component_id = slot.component_id;
		metadata.padded_width_in_blocks = slot.max_width_in_blocks;
		metadata.padded_height_in_blocks = slot.max_height_in_blocks;
		metadata.h_samp_factor = slot.h_samp_factor;
		metadata.v_samp_factor = slot.v_samp_factor;

		if (const auto* component = find_component_by_id(image, slot.component_id)) {
			metadata.width_in_blocks = component->metadata.width_in_blocks;
			metadata.height_in_blocks = component->metadata.height_in_blocks;
			metadata.present = true;
		} else {
			metadata.width_in_blocks = 0;
			metadata.height_in_blocks = 0;
			metadata.present = false;
		}
		out.components.push_back(metadata);
	}
	return out;
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
			if (actual.component_id != expected.component_id ||
			    actual.width_in_blocks != expected.width_in_blocks ||
			    actual.height_in_blocks != expected.height_in_blocks ||
			    actual.h_samp_factor != expected.h_samp_factor ||
			    actual.v_samp_factor != expected.v_samp_factor) {
				std::ostringstream msg;
				msg << "JPEG dataset image " << image_idx << " component " << component_idx
				    << " grid differs from image 0; expected component_id=" << expected.component_id
				    << " blocks=" << expected.width_in_blocks << "x" << expected.height_in_blocks
				    << " sampling=" << expected.h_samp_factor << "x" << expected.v_samp_factor
				    << ", got component_id=" << actual.component_id << " blocks=" << actual.width_in_blocks
				    << "x" << actual.height_in_blocks << " sampling=" << actual.h_samp_factor << "x"
				    << actual.v_samp_factor;
				throw std::runtime_error(msg.str());
			}
		}
	}
}

std::vector<size_t> make_coord_to_index(const DecodedComponent& component) {
	const auto order = detail::make_block_order(component.metadata.width_in_blocks,
	                                            component.metadata.height_in_blocks,
	                                            true);
	std::vector<size_t> coord_to_index(order.size());
	for (size_t block_idx = 0; block_idx < order.size(); ++block_idx) {
		const auto& coord = order[block_idx];
		coord_to_index[static_cast<size_t>(coord.y) * component.metadata.width_in_blocks + coord.x] = block_idx;
	}
	return coord_to_index;
}

std::unordered_map<const DecodedComponent*, std::vector<size_t>>
make_coord_indexes(const std::vector<DecodedImage>& images, const bool use_z_curve_block_order) {
	std::unordered_map<const DecodedComponent*, std::vector<size_t>> indexes;
	if (!use_z_curve_block_order) {
		return indexes;
	}
	for (const auto& image : images) {
		for (const auto& component : image.components) {
			indexes.emplace(&component, make_coord_to_index(component));
		}
	}
	return indexes;
}

const DctRow* find_component_block(const DecodedComponent&          component,
                                   const uint32_t                   x,
                                   const uint32_t                   y,
                                   const bool                       use_z_curve_block_order,
                                   const std::vector<size_t>* const coord_to_index) {
	if (x >= component.metadata.width_in_blocks || y >= component.metadata.height_in_blocks) {
		return nullptr;
	}
	const size_t flat_index = static_cast<size_t>(y) * component.metadata.width_in_blocks + x;
	if (!use_z_curve_block_order) {
		return flat_index < component.blocks.size() ? &component.blocks[flat_index] : nullptr;
	}
	if (coord_to_index == nullptr || flat_index >= coord_to_index->size()) {
		return nullptr;
	}
	const size_t block_index = (*coord_to_index)[flat_index];
	if (block_index >= component.blocks.size()) {
		return nullptr;
	}
	return &component.blocks[block_index];
}

JpegDctTable make_dataset_table(std::vector<DecodedImage> images, const JpegDctReaderOptions& options) {
	validate_dataset(images, options);
	const auto slots = normalize_component_slots(images);

	JpegDctTable table;
	table.metadata.row_ordering = JpegDctRowOrdering::kDatasetComponentMajorBlockMajorImageMinor;
	table.metadata.validation_mode = options.validation_mode;
	table.metadata.zigzag_columns = options.use_zigzag_columns;
	table.metadata.z_curve_block_order = options.use_z_curve_block_order;
	table.metadata.image_count = images.size();

	if (options.validation_mode == JpegDatasetValidationMode::kRequireSameComponentGrids) {
		table.metadata.images.push_back(normalized_image_metadata(images.front(), slots));
	} else {
		table.metadata.images.reserve(images.size());
		for (const auto& image : images) {
			table.metadata.images.push_back(normalized_image_metadata(image, slots));
		}
	}

	const DctRow padding_row {};
	const auto   coord_indexes = make_coord_indexes(images, options.use_z_curve_block_order);
	for (const auto& slot : slots) {
		const auto block_order =
		    detail::make_block_order(slot.max_width_in_blocks, slot.max_height_in_blocks, options.use_z_curve_block_order);
		table.block_group_count += block_order.size();
		for (const auto& block_coord : block_order) {
			for (const auto& image : images) {
				const auto* component = find_component_by_id(image, slot.component_id);
				const auto* coord_to_index =
				    component == nullptr || !options.use_z_curve_block_order ? nullptr : &coord_indexes.at(component);
				const auto* row = component == nullptr
				                      ? nullptr
				                      : find_component_block(*component,
				                                             block_coord.x,
				                                             block_coord.y,
				                                             options.use_z_curve_block_order,
				                                             coord_to_index);
				if (row != nullptr) {
					append_row(table, *row, false);
				} else if (options.validation_mode == JpegDatasetValidationMode::kPadToMaxComponentGrids) {
					append_row(table, padding_row, true);
				}
			}
		}
	}
	return table;
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

std::string dct_column_name(const size_t col) {
	std::ostringstream out;
	out << "dct_zz_" << std::setw(2) << std::setfill('0') << col;
	return out.str();
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

void write_jpeg_dct_metadata(const JpegDctDatasetMetadata& metadata, const std::filesystem::path& output_path) {
	std::ofstream out(output_path, std::ios::binary);
	if (!out) {
		throw std::runtime_error("failed to open JPEG DCT metadata output: " + output_path.string());
	}

	const auto* components = metadata.images.empty() ? nullptr : &metadata.images.front().components;
	const auto  component_count = components == nullptr ? 0 : components->size();

	const uint8_t magic[8] {'G', 'J', 'D', 'C', 'T', 'M', 'D', '1'};
	const auto write = [&](const auto& value) {
		out.write(reinterpret_cast<const char*>(&value), sizeof(value));
	};
	out.write(reinterpret_cast<const char*>(magic), sizeof(magic));
	const uint16_t version = 1;
	const uint16_t row_ordering = static_cast<uint16_t>(row_ordering_id(metadata.row_ordering));
	const uint16_t validation_mode = static_cast<uint16_t>(validation_mode_id(metadata.validation_mode));
	const uint16_t layout_flags =
	    static_cast<uint16_t>((metadata.zigzag_columns ? 1U : 0U) | (metadata.z_curve_block_order ? 2U : 0U));
	const uint64_t image_count = static_cast<uint64_t>(metadata.image_count);
	const uint32_t n_components = static_cast<uint32_t>(component_count);
	write(version);
	write(row_ordering);
	write(validation_mode);
	write(layout_flags);
	write(image_count);
	write(n_components);

	for (size_t component_idx = 0; component_idx < component_count; ++component_idx) {
		const auto& component = (*components)[component_idx];
		const uint32_t component_index = static_cast<uint32_t>(component.component_index);
		const int32_t  component_id = static_cast<int32_t>(component.component_id);
		const uint32_t width = component.width_in_blocks;
		const uint32_t height = component.height_in_blocks;
		const uint32_t padded_width = component.padded_width_in_blocks;
		const uint32_t padded_height = component.padded_height_in_blocks;
		const int32_t  h_samp = component.h_samp_factor;
		const int32_t  v_samp = component.v_samp_factor;
		write(component_index);
		write(component_id);
		write(width);
		write(height);
		write(padded_width);
		write(padded_height);
		write(h_samp);
		write(v_samp);
	}

	uint8_t has_image_records = 0;
	if (metadata.validation_mode != JpegDatasetValidationMode::kRequireSameComponentGrids &&
	    metadata.row_ordering == JpegDctRowOrdering::kDatasetComponentMajorBlockMajorImageMinor) {
		has_image_records = 1;
	}
	write(has_image_records);

	if (has_image_records != 0) {
		for (size_t image_idx = 0; image_idx < metadata.images.size(); ++image_idx) {
			const auto& image = metadata.images[image_idx];
			for (size_t component_idx = 0; component_idx < image.components.size(); ++component_idx) {
				const auto& component = image.components[component_idx];
				const uint8_t  present = component.present ? 1 : 0;
				const uint32_t width = component.width_in_blocks;
				const uint32_t height = component.height_in_blocks;
				write(present);
				write(width);
				write(height);
			}
		}
	}
}

void compress_jpeg_dct_to_fls(const JpegDctTable&          table,
                              const std::filesystem::path& fls_output_path,
                              const std::filesystem::path& metadata_output_path) {
	std::array<fastlanes::MemoryColumn, 64> columns;
	for (size_t col = 0; col < columns.size(); ++col) {
		columns[col].name = dct_column_name(col);
		columns[col].data = std::span<const int16_t>(table.columns[col].data(), table.columns[col].size());
	}

	const fastlanes::MemoryTable memory_table {std::span<const fastlanes::MemoryColumn>(columns.data(), columns.size())};
	fastlanes::write_memory_table_to_fls(memory_table, fls_output_path);
	write_jpeg_dct_metadata(table.metadata, metadata_output_path);
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
                                      const JpegDctReaderOptions&               options) {
	compress_jpeg_dct_to_fls(read_jpeg_dct_dataset(jpeg_paths, options), fls_output_path, metadata_output_path);
}

} // namespace galp::jpeg
