#ifndef FASTLENS_FORMAT_RANGE_READER_HPP
#define FASTLENS_FORMAT_RANGE_READER_HPP

#include "fls/io/file.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace fastlens::format {

struct RangeReaderStats {
	uint64_t bytes_read  = 0U;
	uint64_t pread_count = 0U;
};

class RangeReader {
public:
	explicit RangeReader(const std::filesystem::path& path);
	void Read(uint64_t file_offset, size_t size, std::byte* destination, size_t destination_capacity);
	[[nodiscard]] const RangeReaderStats& stats() const noexcept {
		return stats_;
	}

private:
	std::shared_ptr<fastlanes::File> file_;
	RangeReaderStats                 stats_;
};

} // namespace fastlens::format

#endif // FASTLENS_FORMAT_RANGE_READER_HPP
