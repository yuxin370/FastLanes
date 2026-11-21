// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/include/fls/io/file.hpp
// ────────────────────────────────────────────────────────
#ifndef FLS_IO_FILE_HPP
#define FLS_IO_FILE_HPP

#include "fls/api/api.hpp"
#include "fls/common/common.hpp"
#include "fls/io/read_source.hpp"
#include "fls/std/filesystem.hpp"
#include "fls/std/string.hpp"
#include <mutex>
#include <span>

namespace fastlanes {
/*--------------------------------------------------------------------------------------------------------------------*/
class Buf;
/*--------------------------------------------------------------------------------------------------------------------*/

using FileScatterReadTarget = ScatterReadTarget;
using FileRangeReadTarget   = RangeReadTarget;

class FLS_API File {
public:
	explicit File(const path& path);
	~File();
	File(const File&)            = delete;
	File& operator=(const File&) = delete;

public:
	// write to file_path
	void Write(const Buf& buf);
	// write to file_path
	void Read(Buf& buf);
	// write to file_path
	void Append(const Buf& buf);
	// Append
	void Append(const char* pointer, n_t size);
	//
	void ReadRange(Buf& buf, n_t offset, n_t size);
	//
	void ReadRange(void* dst, n_t offset, n_t size);
	//
	void ReadRangeUnchecked(void* dst, n_t offset, n_t size);
	// Read discontiguous file ranges into caller-provided storage.
	void ReadRangesUnchecked(std::span<const FileRangeReadTarget> targets);
	// Read one contiguous file range directly into multiple destination spans.
	// Returns the number of read operations performed.
	n_t ReadScatterUnchecked(std::span<const FileScatterReadTarget> targets, n_t offset);
	// get file size
	[[nodiscard]] n_t Size() const;

public:
	/// read from file_path and return string.
	static string read(const path& file_path);
	/// write to file_path
	static void write(const path& file_path, const string& dump);
	/// append to file_path
	static void append(const path& file_path, const string& dump);

private:
	void open_read_handle() const;
	void close_read_handle() const;
	void invalidate_size_cache();

private:
	path               m_path;
	up<std::ofstream>  m_of_stream;
	mutable std::mutex m_read_mutex;
#if defined(_WIN32)
	up<std::ifstream> m_if_stream;
#else
	mutable int m_fd = -1;
#endif
	mutable n_t  m_cached_size     = 0;
	mutable bool m_has_cached_size = false;
};
} // namespace fastlanes

#endif // FLS_IO_FILE_HPP
