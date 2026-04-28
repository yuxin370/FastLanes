// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/file/file_footer.cpp
// ────────────────────────────────────────────────────────
#include "fls/file/file_footer.hpp"
#include "fls/common/alias.hpp"
#include "fls/common/status.hpp"
#include "fls/connection.hpp"
#include "fls/file/file_header.hpp"
#include "fls/info.hpp"
#include "fls/io/file.hpp"
#include "fls/io/io.hpp"
#include "fls/std/filesystem.hpp"

namespace fastlanes {

void FileFooter::Write(const Connection& connection, const path& file_path, const FileFooter& file_footer) {
	io file_io = make_unique<File>(file_path); // TODO[io]

	IO::append(file_io, reinterpret_cast<const char*>(&file_footer), sizeof(file_footer));
}

Status FileFooter::Load(FileFooter& file_footer, const path& file_path) {
	File file(file_path);
	return Load(file_footer, file);
}

Status FileFooter::Load(FileFooter& file_footer, File& file) {
	const auto file_size = file.Size();
	if (file_size < sizeof(FileHeader) + sizeof(FileFooter)) {
		return Status::Error(Status::ErrorCode::ERR_1_SMALL_FILE_SIZE);
	}
	file.ReadRange(&file_footer, file_size - sizeof(FileFooter), sizeof(FileFooter));

	if (file_footer.magic_bytes != Info::get_magic_bytes()) {
		return Status::Error(Status::ErrorCode::ERR_5_INVALID_MAGIC_BYTES);
	}

	return Status::Ok();
}
} // namespace fastlanes
