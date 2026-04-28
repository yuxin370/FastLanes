// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/file/file_header.cpp
// ────────────────────────────────────────────────────────
#include "fls/file/file_header.hpp"
#include "fls/common/alias.hpp"
#include "fls/common/status.hpp"
#include "fls/connection.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/info.hpp"
#include "fls/io/file.hpp"
#include "fls/io/io.hpp"
#include "fls/std/filesystem.hpp"

namespace fastlanes {

void FileHeader::Write(const Connection& connection, const path& file_path) {
	io         file_io = make_unique<File>(file_path); // TODO[io]
	FileHeader file_header {};

	file_header.magic_bytes            = Info::get_magic_bytes();
	file_header.version                = Info::get_version_bytes();
	file_header.settings.inline_footer = connection.is_footer_inlined();

	IO::append(file_io, reinterpret_cast<const char*>(&file_header), sizeof(file_header));
}

Status FileHeader::Load(FileHeader& file_header, const path& file_path) {
	File file(file_path);
	return Load(file_header, file);
}

Status FileHeader::Load(FileHeader& file_header, File& file) {
	if (const auto file_size = file.Size(); file_size < sizeof(FileHeader) + sizeof(FileFooter)) {
		return Status::Error(Status::ErrorCode::ERR_1_SMALL_FILE_SIZE);
	}
	file.ReadRange(&file_header, 0, sizeof(FileHeader));

	return Status::Ok();
}
} // namespace fastlanes
