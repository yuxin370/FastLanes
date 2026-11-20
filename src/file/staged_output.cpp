// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/file/staged_output.cpp
// ────────────────────────────────────────────────────────
#include "staged_output.hpp"
#include "fls/std/filesystem.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <ranges>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace fastlanes::detail {

namespace {
constexpr auto STAGED_DESCRIPTOR_FILENAME = "table_descriptor.fbb";

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

} // namespace

StagedOutput::StagedOutput(path final_path)
    : m_final_path(std::move(final_path))
    , m_directory(make_staging_directory(m_final_path))
    , m_file_path(m_directory / "output" / m_final_path.filename()) {
	try {
		fs::create_directory(m_file_path.parent_path());
	} catch (...) {
		std::error_code ec;
		fs::remove(m_directory, ec);
		throw;
	}
}

StagedOutput::~StagedOutput() {
	if (!m_preserve_backups) {
		std::error_code ec;
		std::filesystem::remove_all(m_directory, ec);
	}
}

void StagedOutput::publish(const bool has_external_footer, const bool has_json) {
	struct SidecarMove {
		path source;
		path target;
		path backup;
		bool target_was_backed_up = false;
		bool source_was_published = false;
	};

	std::vector<SidecarMove> sidecars;
	const auto               final_parent =
        m_final_path.parent_path().empty() ? std::filesystem::current_path() : m_final_path.parent_path();
	if (has_external_footer) {
		sidecars.push_back(
		    {m_file_path.parent_path() / STAGED_DESCRIPTOR_FILENAME, final_parent / STAGED_DESCRIPTOR_FILENAME, {}});
	}
	if (has_json) {
		path source = m_file_path;
		source += ".json";
		path target = m_final_path;
		target += ".json";
		sidecars.push_back({std::move(source), std::move(target), {}});
	}

	try {
		for (auto& move : sidecars) {
			const auto existing = fs::symlink_status(move.target);
			if (fs::exists(existing)) {
				if (!fs::is_regular_file(existing)) {
					throw std::runtime_error("Cannot replace non-regular sidecar: " + move.target.string());
				}
				move.backup = m_directory / "backups" / move.target.filename();
				fs::create_directories(move.backup.parent_path());
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
		std::error_code    recovery_error;
		const SidecarMove* failed_move = nullptr;
		for (auto& sidecar : std::ranges::reverse_view(sidecars)) {
			std::error_code ec;
			if (sidecar.source_was_published) {
				std::filesystem::remove(sidecar.target, ec);
			}
			if (sidecar.target_was_backed_up && !ec) {
				std::filesystem::rename(sidecar.backup, sidecar.target, ec);
			}
			if (ec) {
				m_preserve_backups = true;
				if (!failed_move) {
					recovery_error = ec;
					failed_move    = &sidecar;
				}
			}
		}
		if (failed_move) {
			const auto& recovery_path = failed_move->target_was_backed_up ? failed_move->backup : m_directory;
			std::throw_with_nested(fs::filesystem_error("Failed to roll back FastLanes output; recovery files retained",
			                                            recovery_path,
			                                            failed_move->target,
			                                            recovery_error));
		}
		throw;
	}
}

} // namespace fastlanes::detail
