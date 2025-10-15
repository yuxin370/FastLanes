// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/connection.cpp
// ────────────────────────────────────────────────────────
#include "fls/connection.hpp" // for Connection
#include "fls/cfg/cfg.hpp"
#include "fls/common/alias.hpp"     // for make_unique<>, n_t, idx_t, fls_bool, FLS_TRUE
#include "fls/common/status.hpp"    // for Status
#include "fls/encoder/encoder.hpp"  // for Encoder
#include "fls/file/file_footer.hpp" // for FileFooter
#include "fls/file/file_header.hpp" // for FileHeader
#include "fls/flatbuffers/flatbuffers.hpp"
#include "fls/footer/operator_token_generated.h"
#include "fls/info.hpp"
#include "fls/json/fls_json.hpp"            // for JSON
#include "fls/reader/csv_reader.hpp"        // for CSVReader
#include "fls/reader/dctchannel_reader.hpp" // for DctChannelReader
#include "fls/reader/json_reader.hpp"       // for JSONReader
#include "fls/reader/table_reader.hpp"      // for TableReader
#include "fls/std/filesystem.hpp"           // for std::filesystem::directory_iterator, begin, path
#include "fls/std/string.hpp"               // for std::string
#include "fls/std/vector.hpp"               // for fastlanes::vector
#include "fls/table/rowgroup.hpp"           // for Rowgroup
#include "fls/table/table.hpp"              // for Table
#include "fls/wizard/wizard.hpp"            // for Wizard
#include <algorithm>                        // for std::ranges::none_of
#include <cctype>
#include <cstdint> // for uint64_t
#include <filesystem>
#include <memory> // for std::make_unique, unique_ptr
#include <set>
#include <stdexcept> // for std::runtime_error

namespace fastlanes {

Connection::Connection() {
	m_config = make_unique<Config>();
}

Connection::Connection(const Config& config) {
	m_config = make_unique<Config>(config);
}

Connection& Connection::read_csv(const path& dir_path) {
	m_table = CsvReader::Read(dir_path, *this);

	return *this;
}

Connection& Connection::read_dct(const ProcessedDCTChannel& channel, const int tag) {
	m_table = DctChannelReader::Read(channel, tag, *this);

	return *this;
}

Connection& Connection::read_jpeg(const std::string& path, const std::string& header_path) {
	auto image_header = JpegLoader::load_header(path);
	JpegLoader::dump_ImageHeader(image_header, header_path.c_str());
	m_table = DctChannelReader::Read(image_header.channel_dcts, *this);

	return *this;
}

// Helper: case-insensitive extension check
bool is_jpeg_file(const std::filesystem::path& p) {
	static const std::set<std::string> jpeg_exts = {".jpg", ".jpeg", ".JPG", ".JPEG"};
	auto                               ext       = p.extension().string();
	std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
	return jpeg_exts.count(ext) > 0;
}

ImageHeader deduplicate_headers(const std::vector<ImageHeader>& all_headers) {
	if (all_headers.empty()) {
		throw std::runtime_error("Input headers vector is empty.");
	}

	ImageHeader unified_header;
	unified_header.width   = all_headers[0].width;
	unified_header.height  = all_headers[0].height;
	unified_header.quality = all_headers[0].quality;

	std::unordered_map<ColorSpace, uint8_t> color_space_to_id;

	// Helper: check if two quant tables are identical
	auto quant_table_equal = [](const QuantTable& a, const QuantTable& b) -> bool {
		if (a.precision != b.precision)
			return false;
		for (int i = 0; i < 64; ++i) {
			if (a.data[i] != b.data[i])
				return false;
		}
		return true;
	};

	// Map a QuantTable to its unified index
	auto quant_table_to_id = [&](const QuantTable& qt) -> uint8_t {
		for (size_t i = 0; i < unified_header.quant_tables.size(); ++i) {
			if (quant_table_equal(unified_header.quant_tables[i], qt)) {
				return static_cast<uint8_t>(i);
			}
		}
		// Not found: add to unified_header
		QuantTable new_qt = qt;
		new_qt.id         = static_cast<uint8_t>(unified_header.quant_tables.size());
		unified_header.quant_tables.push_back(new_qt);
		return new_qt.id;
	};

	for (const auto& hdr : all_headers) {
		// 1. Size check
		if (unified_header.width != hdr.width || unified_header.height != hdr.height) {
			throw std::runtime_error("Different size images are not supported yet.");
		}

		// 2. Color space mapping
		if (hdr.color_spaces.empty()) {
			throw std::runtime_error("Image header has no color space info.");
		}
		const ColorSpace& img_color_space = hdr.color_spaces[0];
		uint8_t           color_space_id;
		auto              it = color_space_to_id.find(img_color_space);
		if (it != color_space_to_id.end()) {
			color_space_id = it->second;
		} else {
			color_space_id = static_cast<uint8_t>(unified_header.color_spaces.size());
			unified_header.color_spaces.push_back(img_color_space);
			color_space_to_id[img_color_space] = color_space_id;
		}

		// 3. Merge ChannelDCT
		for (const auto& src_channel : hdr.channel_dcts) {
			ChannelDCT new_channel     = src_channel; // copy
			new_channel.color_space_id = color_space_id;

			if (src_channel.qtable_id >= hdr.quant_tables.size()) {
				throw std::runtime_error("Invalid qtable_id in source channel");
			}
			const QuantTable& qt  = hdr.quant_tables[src_channel.qtable_id];
			new_channel.qtable_id = quant_table_to_id(qt);

			unified_header.channel_dcts.push_back(std::move(new_channel));
		}
	}

	return unified_header;
}

Connection& Connection::read_jpeg_dir(const std::string& dir_path, const std::string& header_path) {
	namespace fs = std::filesystem;

	if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
		throw std::runtime_error("Directory does not exist: " + dir_path);
	}

	std::vector<ImageHeader> all_headers;

	std::vector<fs::path> files;
	for (const auto& entry : fs::directory_iterator(dir_path)) {
		if (!entry.is_regular_file())
			continue;
		const auto& p = entry.path();
		if (is_jpeg_file(p)) {
			files.push_back(p);
		}
	}

	std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
		return a.filename().string() < b.filename().string();
	});

	for (size_t k = 0; k < files.size(); ++k) {
		printf("Reading header from: %s\n", files[k].string().c_str());
		auto header = JpegLoader::load_header(files[k].string());
		all_headers.push_back(std::move(header));
	}

	// Step 1: Iterate all JPEG files
	// for (const auto& entry : fs::directory_iterator(dir_path)) {
	//     if (is_jpeg_file(entry.path())) {
	//         try {
	//             auto header = JpegLoader::load_header(entry.path().string());
	//             all_headers.push_back(std::move(header));
	//         } catch (const std::exception& e) {
	//             // Optionally warn and skip
	//             continue;
	//         }
	//     }
	// }

	if (all_headers.empty()) {
		throw std::runtime_error("No valid JPEG files found in directory: " + dir_path);
	}

	// Step 2: Deduplicate headers
	ImageHeader unified_header = deduplicate_headers(all_headers);

	// printf("unified header is as follows:\n");
	// JpegLoader::print_image_header(unified_header);

	// Step 3: Dump unified header
	JpegLoader::dump_ImageHeader(unified_header, header_path.c_str());

	// Step 4: Flatten all DCTs into a single list expected by DctChannelReader::Read
	m_table = DctChannelReader::Read(unified_header.channel_dcts, *this);

	return *this;
}

Connection& Connection::read_json(const path& dir_path) {
	m_table = JsonReader::Read(dir_path, *this);

	return *this;
}

up<TableReader> Connection::read_fls(const path& file_path) {
	FileSystem::check_if_file_exists(file_path);

	// init
	return make_unique<TableReader>(file_path, *this);
}

void prepare_rowgroup(Rowgroup& rowgroup, const Config& config) {
	// init
	rowgroup.Init();

	// Only cast if schema wasn’t forced
	const bool shouldCast = !config.is_forced_schema && !config.is_forced_schema_pool;
	if (shouldCast) {
		rowgroup.Cast();
	}

	rowgroup.Finalize();
	rowgroup.GetStatistics();
}

void Connection::prepare_table() const {
	for (auto& rowgroup : m_table->m_rowgroups) {
		prepare_rowgroup(*rowgroup, *m_config);
	}
}

void Connection::write_footer(const path& file_path) const {
	// Write table descriptor

	const n_t        table_descriptor_size = FlatBuffers::Write(*this, file_path, *m_table_descriptor);
	const FileFooter file_footer {
	    m_table_descriptor->m_table_binary_size, table_descriptor_size, Info::get_magic_bytes()};

	FileFooter::Write(*this, file_path, file_footer);
}

up<Connection> connect() {
	return make_unique<Connection>();
}

Connection& Connection::spell() {
	if (m_table == nullptr) {
		/**/
		throw std::runtime_error("Data is not loaded.");
	}

	m_table_descriptor = Wizard::Spell(*this);

	return *this;
}

Connection& Connection::to_fls(const path& file_path) {
	if (exists(file_path)) {
		throw std::runtime_error("Fastlanes file already exists at: " + file_path.string());
	}

	// check if data is loaded into memory
	if (m_table == nullptr) {
		throw std::runtime_error("data is not loaded.");
	}

	prepare_table();

	//  make a rowgroup-get_descriptor if there is no rowgroup-get_descriptor .
	if (m_table_descriptor == nullptr) {
		spell();
	}

	FileHeader::Write(*this, file_path);

	// encode
	Encoder::encode(*this, file_path);

	if (m_config->enable_verbose) {
		fs::path json_file = file_path;
		json_file += ".json";
		JSON::write(*this, json_file, *m_table_descriptor);
	}

	// write the footer
	write_footer(file_path);

	return *this;
}

Status Connection::verify_fls(const path& file_path) {
	FileHeader file_header {};
	FileHeader::Load(file_header, file_path);

	if (file_header.magic_bytes != Info::get_magic_bytes()) {
		return Status::Error(Status::ErrorCode::ERR_5_INVALID_MAGIC_BYTES);
	}

	if (constexpr auto versions = Info::get_all_versions();
	    std::ranges::none_of(versions, [&](uint64_t v) { return file_header.version == v; })) {
		return Status::Error(Status::ErrorCode::ERR_6_INVALID_VERSION_BYTES);
	}

	FileFooter file_footer {};
	FileFooter::Load(file_footer, file_path);

	if (file_footer.magic_bytes != Info::get_magic_bytes()) {
		return Status::Error(Status::ErrorCode::ERR_5_INVALID_MAGIC_BYTES);
	}

	return Status::Ok();
}

Connection& Connection::reset() {
	m_table_descriptor.reset();
	m_table.reset();

	return *this;
}

Connection& Connection::project(const vector<idx_t>& idxs) {
	if (m_table == nullptr) {
		throw std::runtime_error("Data is not loaded.");
	}

	m_table = m_table->Project(idxs);

	return *this;
}

bool Connection::is_forced_schema_pool() const {
	return m_config->is_forced_schema_pool;
}

bool Connection::is_forced_schema() const {
	return m_config->is_forced_schema;
}

const vector<OperatorToken>& Connection::get_forced_schema_pool() const {
	//
	return m_config->forced_schema_pool;
}

Connection& Connection::force_schema_pool(const vector<OperatorToken>& operator_token) {
	m_config->is_forced_schema_pool = true;

	m_config->forced_schema_pool = operator_token;

	return *this;
}

Connection& Connection::force_schema(const vector<OperatorToken>& operator_token) {
	m_config->is_forced_schema = true;

	m_config->forced_schema = operator_token;

	return *this;
}

const vector<OperatorToken>& Connection::get_forced_schema() const {
	//
	return m_config->forced_schema;
}

Connection& Connection::set_n_vectors_per_rowgroup(n_t n_vector_per_rowgroup) {
	m_config->n_vector_per_rowgroup = n_vector_per_rowgroup;
	return *this;
}

Connection& Connection::set_sample_size(n_t n_vecs) {
	m_config->sample_size = n_vecs;
	return *this;
}

Connection& Connection::enable_verbose() {
	m_config->enable_verbose = true;

	return *this;
}

n_t Connection::get_sample_size() const {
	return m_config->sample_size;
}

Table& Connection::get_table() const {
	//
	return *m_table;
}

fls_bool Connection::is_footer_inlined() const {
	return m_config->inline_footer;
}

Connection& Connection::inline_footer() {
	m_config->inline_footer = FLS_TRUE;

	return *this;
}

string_view Connection::get_version() const {
	return Info::get_version();
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * Config
\*--------------------------------------------------------------------------------------------------------------------*/

Config::Config()
    : is_forced_schema_pool(false)
    , is_forced_schema(false)
    , sample_size(CFG::SAMPLER::SAMPLE_SIZE)
    , n_vector_per_rowgroup(CFG::RowGroup::N_VECTORS_PER_ROWGROUP)
    , inline_footer(CFG::Footer::IS_INLINED)
    , enable_verbose(CFG::Defaults::ENABLE_VERBOSE) {
}

} // namespace fastlanes
