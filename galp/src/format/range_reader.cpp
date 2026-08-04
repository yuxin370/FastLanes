#include "format/range_reader.hpp"
#include <limits>
#include <stdexcept>

namespace galp::format {

RangeReader::RangeReader(const std::filesystem::path& path)
    : file_(std::make_shared<fastlanes::File>(path)) {
}

void RangeReader::Read(const uint64_t   file_offset,
                       const size_t     size,
                       std::byte* const destination,
                       const size_t     destination_capacity) {
	if (destination == nullptr || size > destination_capacity) {
		throw std::invalid_argument("GALP range reader destination is null or too small");
	}
	if (file_offset > file_->Size() || size > file_->Size() - file_offset) {
		throw std::out_of_range("GALP range reader request exceeds file bounds");
	}
	file_->ReadRangeUnchecked(destination, file_offset, size);
	if (size > std::numeric_limits<uint64_t>::max() - stats_.bytes_read) {
		throw std::overflow_error("GALP range reader byte counter overflow");
	}
	stats_.bytes_read += size;
	++stats_.pread_count;
}

} // namespace galp::format
