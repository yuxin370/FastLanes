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
#include "fls/json/fls_json.hpp"       // for JSON
#include "fls/reader/csv_reader.hpp"   // for CSVReader
#include "fls/reader/json_reader.hpp"  // for JSONReader
#include "fls/reader/table_reader.hpp" // for TableReader
#include "fls/std/filesystem.hpp"      // for std::filesystem::directory_iterator, begin, path
#include "fls/std/string.hpp"          // for std::string
#include "fls/std/vector.hpp"          // for fastlanes::vector
#include "fls/table/memory_table.hpp"  // for MemoryTable
#include "fls/table/rowgroup.hpp"      // for Rowgroup
#include "fls/table/table.hpp"         // for Table
#include "fls/wizard/wizard.hpp"       // for Wizard
#include <algorithm>                   // for std::ranges::none_of
#include <atomic>
#include <chrono>
#include <cstdint> // for uint64_t
#include <filesystem>
#include <memory>    // for std::make_unique, unique_ptr
#include <stdexcept> // for std::runtime_error
#include <system_error>
#include <utility>

namespace fastlanes {

namespace {

constexpr auto TABLE_DESCRIPTOR_FILE_NAME = "table_descriptor.fbb";

path make_staging_directory(const path& final_path) {
	static std::atomic<uint64_t> counter {0};
	const auto parent = final_path.parent_path().empty() ? std::filesystem::current_path() : final_path.parent_path();
	const auto stamp  = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());

	for (uint64_t attempt = 0; attempt < 1024; ++attempt) {
		const auto id = stamp + counter.fetch_add(1, std::memory_order_relaxed) + attempt;
		const path candidate =
		    parent / (".fastlanes-stage-" + final_path.filename().string() + "-" + std::to_string(id));
		std::error_code ec;
		if (std::filesystem::create_directory(candidate, ec)) {
			return candidate;
		}
		if (ec && ec != std::errc::file_exists) {
			throw std::filesystem::filesystem_error("failed to create FastLanes staging directory", candidate, ec);
		}
	}
	throw std::runtime_error("failed to allocate a unique FastLanes staging directory");
}

class StagedOutput {
public:
	explicit StagedOutput(path final_path)
	    : m_final_path(std::move(final_path))
	    , m_directory(make_staging_directory(m_final_path))
	    , m_file_path(m_directory / m_final_path.filename()) {
	}

	~StagedOutput() {
		if (!m_published) {
			std::error_code ec;
			std::filesystem::remove_all(m_directory, ec);
		}
	}

	[[nodiscard]] const path& file_path() const {
		return m_file_path;
	}

	void publish(const bool has_external_footer, const bool has_json) {
		struct SidecarMove {
			path source;
			path target;
			path backup;
			bool target_was_backed_up = false;
			bool source_was_published = false;
		};

		vector<SidecarMove> sidecars;
		const auto          final_parent =
            m_final_path.parent_path().empty() ? std::filesystem::current_path() : m_final_path.parent_path();
		if (has_external_footer) {
			sidecars.push_back(
			    {m_directory / TABLE_DESCRIPTOR_FILE_NAME, final_parent / TABLE_DESCRIPTOR_FILE_NAME, {}});
		}
		if (has_json) {
			path source = m_file_path;
			source += ".json";
			path target = m_final_path;
			target += ".json";
			sidecars.push_back({std::move(source), std::move(target), {}});
		}

		try {
			for (std::size_t idx = 0; idx < sidecars.size(); ++idx) {
				auto& move = sidecars[idx];
				if (exists(move.target)) {
					move.backup = m_directory / ("sidecar-backup-" + std::to_string(idx));
					std::filesystem::rename(move.target, move.backup);
					move.target_was_backed_up = true;
				}
				std::filesystem::rename(move.source, move.target);
				move.source_was_published = true;
			}

			if (exists(m_final_path)) {
				throw std::runtime_error("Fastlanes file already exists at: " + m_final_path.string());
			}
			std::filesystem::rename(m_file_path, m_final_path);
		} catch (...) {
			for (auto move = sidecars.rbegin(); move != sidecars.rend(); ++move) {
				std::error_code ec;
				if (move->source_was_published) {
					std::filesystem::remove(move->target, ec);
				}
				if (move->target_was_backed_up) {
					ec.clear();
					std::filesystem::rename(move->backup, move->target, ec);
				}
			}
			throw;
		}

		m_published = true;
		std::error_code ec;
		std::filesystem::remove_all(m_directory, ec);
	}

private:
	path m_final_path;
	path m_directory;
	path m_file_path;
	bool m_published = false;
};

} // namespace

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

Connection& Connection::read_json(const path& dir_path) {
	m_table = JsonReader::Read(dir_path, *this);

	return *this;
}

Connection& Connection::read_memory(const MemoryTable& table, const MemoryTableOptions& options) {
	load_memory_table(*this, table, options);
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

	rowgroup.Finalize();
	rowgroup.GetStatistics();

	// Only cast if schema wasn’t forced
	const bool shouldCast = !config.is_forced_schema && !config.is_forced_schema_pool;
	if (shouldCast) {
		rowgroup.Cast();
	}

	// Populate bimap after Cast, so it reflects the final (possibly cast) column types
	rowgroup.PopulateBiMap();
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
	return to_fls(file_path, EncodingOptions {});
}

Connection& Connection::to_fls(const path& file_path, const EncodingOptions& options) {
	const auto total_started = std::chrono::steady_clock::now();
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
	const auto preparation_finished = std::chrono::steady_clock::now();

	m_last_encoding_stats = {};
	StagedOutput staged_output(file_path);
	FileHeader::Write(*this, staged_output.file_path());

	// encode
	const auto encoding_started  = std::chrono::steady_clock::now();
	m_last_encoding_stats        = Encoder::encode(*this, staged_output.file_path(), options);
	const auto encoding_finished = std::chrono::steady_clock::now();

	if (m_config->enable_verbose) {
		fs::path json_file = staged_output.file_path();
		json_file += ".json";
		JSON::write(*this, json_file, *m_table_descriptor);
	}

	// write the footer
	write_footer(staged_output.file_path());
	staged_output.publish(!static_cast<bool>(m_config->inline_footer), m_config->enable_verbose);
	const auto total_finished = std::chrono::steady_clock::now();
	m_last_encoding_stats.preparation_wall_seconds =
	    std::chrono::duration<double>(preparation_finished - total_started).count();
	m_last_encoding_stats.encoding_wall_seconds =
	    std::chrono::duration<double>(encoding_finished - encoding_started).count();
	m_last_encoding_stats.finalization_wall_seconds =
	    std::chrono::duration<double>(total_finished - encoding_finished).count();
	m_last_encoding_stats.total_wall_seconds = std::chrono::duration<double>(total_finished - total_started).count();

	return *this;
}

Status Connection::verify_fls(const path& file_path) {
	FileHeader file_header {};
	FileHeader::Load(file_header, file_path);

	if (file_header.magic_bytes != Info::get_magic_bytes()) {
		return Status::Error(Status::ErrorCode::ERR_5_INVALID_MAGIC_BYTES);
	}

	if (constexpr auto versions = Info::get_all_versions();
	    std::none_of(versions.begin(), versions.end(), [&](uint64_t v) { return file_header.version == v; })) {
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

void Connection::clear_forced_schema_state() {
	m_config->is_forced_schema = false;
	m_config->forced_schema.clear();
	m_config->is_forced_schema_pool = false;
	m_config->forced_schema_pool.clear();
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

const EncodingStats& Connection::get_last_encoding_stats() const {
	return m_last_encoding_stats;
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
