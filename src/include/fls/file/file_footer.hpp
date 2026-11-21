// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/include/fls/file/file_footer.hpp
// ────────────────────────────────────────────────────────
#ifndef FLS_FILE_FILE_FOOTER_HPP
#define FLS_FILE_FILE_FOOTER_HPP

#include "fls/api/api.hpp"
#include "fls/common/alias.hpp"
#include "fls/common/status.hpp"
#include "fls/std/filesystem.hpp"

namespace fastlanes {
/*--------------------------------------------------------------------------------------------------------------------*/
class Connection;
class ReadSource;
class File;
/*--------------------------------------------------------------------------------------------------------------------*/

class FLS_API FileFooter {
public:
	static void   Write(const Connection& connection, const path& file_path, const FileFooter& file_footer);
	static Status Load(FileFooter& file_footer, const path& file_path);
	static Status Load(FileFooter& file_footer, File& file);
	static Status Load(FileFooter& file_footer, const ReadSource& source);

public:
	n_t table_descriptor_offset;
	n_t table_descriptor_size;
	n_t magic_bytes;
};

static_assert(sizeof(FileFooter) == 24);

} // namespace fastlanes

#endif // FLS_FILE_FILE_FOOTER_HPP
