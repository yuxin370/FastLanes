#ifndef GALP_READER_HPP
#define GALP_READER_HPP

#include "galp/options.hpp"
#include "galp/table.hpp"

#include <cstddef>
#include <filesystem>

namespace galp {

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

} // namespace galp

#endif // GALP_READER_HPP
