// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/include/fls/io/file.hpp
// ────────────────────────────────────────────────────────
#ifndef FLS_IO_FILE_HPP
#define FLS_IO_FILE_HPP

#include "fls/common/common.hpp"
#include "fls/std/filesystem.hpp"
#include "fls/std/string.hpp"
#include <cstdint>
#include <mutex>
#include <span>

namespace fastlanes {
/*--------------------------------------------------------------------------------------------------------------------*/
class Buf;
/*--------------------------------------------------------------------------------------------------------------------*/

struct FileScatterReadTarget {
	void* data = nullptr;
	n_t   size = 0;
};

// One destination/file-offset pair for a discontiguous batched read. Unlike
// FileScatterReadTarget, adjacent entries do not imply contiguous source
// bytes.
struct FileRangeReadTarget {
	void* data   = nullptr;
	n_t   offset = 0;
	n_t   size   = 0;
};

struct FileRangeBatchReadResult {
	n_t    bytes                  = 0;
	n_t    read_request_count     = 0;
	n_t    completion_count       = 0;
	n_t    submit_syscall_count   = 0;
	n_t    wait_syscall_count     = 0;
	n_t    ring_mapped_bytes      = 0;
	n_t    newly_mapped_ring_bytes = 0;
};

struct FileIoUringState;

struct FileReadHandleStats {
	n_t current_open_handles = 0;
	n_t peak_open_handles    = 0;
	n_t open_count           = 0;
	n_t close_count          = 0;
};

class File {
public:
	explicit File(const path& path);
	~File();

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
	// Read one contiguous file range directly into multiple destination spans.
	// Returns the number of physical read syscalls issued.
	n_t ReadScatterUnchecked(std::span<const FileScatterReadTarget> targets, n_t offset);
	// Submit discontiguous file ranges in deterministic target order through a
	// persistent Linux io_uring. The call returns only after every range is
	// complete and never falls back to synchronous pread.
	FileRangeBatchReadResult ReadRangesIoUringUnchecked(
	    std::span<const FileRangeReadTarget> targets, uint32_t queue_depth);
	// get file size
	[[nodiscard]] n_t Size() const;

public:
	/// read from file_path and return string.
	static string read(const path& file_path);
	/// write to file_path
	static void write(const path& file_path, const string& dump);
	/// append to file_path
	static void append(const path& file_path, const string& dump);
	[[nodiscard]] static FileReadHandleStats read_handle_stats() noexcept;

private:
	void open_read_handle() const;
	void close_read_handle() const;
	void invalidate_size_cache();

private:
	path                      m_path;
	up<std::ofstream>         m_of_stream;
	mutable std::mutex        m_read_mutex;
	mutable up<FileIoUringState> m_io_uring;
#if !defined(_WIN32)
	mutable int               m_fd               = -1;
#endif
	mutable n_t               m_cached_size      = 0;
	mutable bool              m_has_cached_size  = false;
};
} // namespace fastlanes

#endif // FLS_IO_FILE_HPP
