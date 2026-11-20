// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/file/staged_output.hpp
// ────────────────────────────────────────────────────────
#pragma once
#include "fls/api/api.hpp"
#include "fls/std/filesystem.hpp"

namespace fastlanes::detail {
class FLS_API StagedOutput {
public:
	explicit StagedOutput(path final_path);
	~StagedOutput();
	StagedOutput(const StagedOutput&)                        = delete;
	StagedOutput&             operator=(const StagedOutput&) = delete;
	[[nodiscard]] const path& file_path() const {
		return m_file_path;
	}
	void publish(bool has_external_footer, bool has_json);

private:
	path m_final_path;
	path m_directory;
	path m_file_path;
	bool m_preserve_backups = false;
};
} // namespace fastlanes::detail
