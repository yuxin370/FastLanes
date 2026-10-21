#ifndef FASTLENS_READER_HPP
#define FASTLENS_READER_HPP

#include "fastlens/options.hpp"
#include "fastlens/table.hpp"

#include <cstddef>
#include <filesystem>

namespace fastlens {

class Reader {
public:
	explicit Reader(std::filesystem::path fls_path);
	~Reader();

	Reader(Reader&&) noexcept;
	Reader& operator=(Reader&&) noexcept;

	Reader(const Reader&)            = delete;
	Reader& operator=(const Reader&) = delete;

	[[nodiscard]] size_t rowgroup_count() const;
	[[nodiscard]] Table  decompress(const DecompressOptions& options = {}) const;
	[[nodiscard]] const std::filesystem::path& path() const noexcept;

private:
	std::filesystem::path fls_path_;
};

} // namespace fastlens

#endif // FASTLENS_READER_HPP
