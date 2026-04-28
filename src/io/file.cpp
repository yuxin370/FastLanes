// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/io/file.cpp
// ────────────────────────────────────────────────────────
#include "fls/io/file.hpp"
#include "fls/common/alias.hpp"
#include "fls/common/assert.hpp"
#include "fls/cor/lyt/buf.hpp"
#include "fls/std/filesystem.hpp"
#include "fls/std/string.hpp"
#if !defined(_WIN32)
#include <cerrno>
#endif
#include <cstddef>
#include <cstdint>    // for int64_t
#if !defined(_WIN32)
#include <cstring>
#include <fcntl.h>
#endif
#include <fstream>    // for std::ofstream
#include <ios>        // for std::ios, std::streamsize
#include <memory>     // for std::make_unique
#include <sstream>
#include <string>
#if !defined(_WIN32)
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <stdexcept> // for std::runtime_error

namespace fastlanes {

namespace {

#if !defined(_WIN32)
std::runtime_error make_io_error(const path& file_path, const std::string& action) {
	return std::runtime_error(action + ": " + file_path.string() + ": " + std::strerror(errno));
}

void pread_exact(const int fd, const path& file_path, void* dst, const n_t offset, const n_t size) {
	auto*  out       = reinterpret_cast<std::byte*>(dst);
	n_t    done      = 0;
	off_t  cur_off   = static_cast<off_t>(offset);

	while (done < size) {
		const size_t chunk = static_cast<size_t>(size - done);
		const auto   nread = ::pread(fd, out + done, chunk, cur_off);
		if (nread == 0) {
			throw std::runtime_error("unexpected EOF while reading: " + file_path.string());
		}
		if (nread < 0) {
			if (errno == EINTR) {
				continue;
			}
			throw make_io_error(file_path, "pread failed");
		}
		done += static_cast<n_t>(nread);
		cur_off += static_cast<off_t>(nread);
	}
}
#endif

bool range_exceeds_file(const n_t offset, const n_t size, const n_t file_size) {
	return offset > file_size || size > file_size - offset;
}

} // namespace

File::File(const path& path) // NOLINT
    : m_path(path) {
}

File::~File() {
	close_read_handle();

	if (m_of_stream != nullptr) {
		FileSystem::close(*m_of_stream);
	}
}

void File::Write(const Buf& buf) {
	if (m_of_stream == nullptr) {
		m_of_stream = make_unique<std::ofstream>(FileSystem::open_w(m_path));
	}
	//
	m_of_stream->write(reinterpret_cast<char*>(buf.data()), static_cast<int64_t>(buf.Size()));
	invalidate_size_cache();
}

void File::Read(Buf& buf) {
	const auto file_size = Size();
	if (file_size > buf.Capacity()) {
		buf.Resize(file_size);
	}
	buf.Reset();
	ReadRange(buf, 0, file_size);
}

void File::ReadRange(Buf& buf, const n_t offset, const n_t size) {
	const auto file_size = Size();
	if (range_exceeds_file(offset, size, file_size)) {
		throw std::runtime_error("read range exceeds file size: " + m_path.string());
	}
	if (size > buf.Capacity()) {
		buf.Resize(size);
	}
	buf.Reset();
	ReadRange(buf.mutable_data(), offset, size);
	buf.UnsafeAdvance(size);
}

void File::ReadRange(void* dst, const n_t offset, const n_t size) {
	const auto file_size = Size();
	if (range_exceeds_file(offset, size, file_size)) {
		throw std::runtime_error("read range exceeds file size: " + m_path.string());
	}
	ReadRangeUnchecked(dst, offset, size);
}

void File::ReadRangeUnchecked(void* dst, const n_t offset, const n_t size) {
#if defined(_WIN32)
	if (size == 0) {
		return;
	}
	auto stream = FileSystem::open_r_binary(m_path);
	stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
	if (!stream) {
		throw std::runtime_error("seek failed while reading: " + m_path.string());
	}
	stream.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(size));
	if (!stream) {
		throw std::runtime_error("read failed: " + m_path.string());
	}
#else
	open_read_handle();
	pread_exact(m_fd, m_path, dst, offset, size);
#endif
}

n_t File::Size() const {
#if defined(_WIN32)
	std::lock_guard<std::mutex> lock(m_read_mutex);
	if (!m_has_cached_size) {
		if (!exists(m_path)) {
			throw std::runtime_error("File does not exist");
		}
		m_cached_size     = static_cast<n_t>(fs::file_size(m_path));
		m_has_cached_size = true;
	}
	return m_cached_size;
#else
	open_read_handle();
	std::lock_guard<std::mutex> lock(m_read_mutex);
	return m_cached_size;
#endif
}

void File::Append(const Buf& buf) {
	if (m_of_stream == nullptr) {
		// Open file in append mode
		m_of_stream = std::make_unique<std::ofstream>(m_path.string(), std::ios::binary | std::ios::app);
	}
	m_of_stream->write(reinterpret_cast<char*>(buf.data()), static_cast<int64_t>(buf.Size()));
	invalidate_size_cache();
}

void File::Append(const char* pointer, n_t size) {
	if (m_of_stream == nullptr) {
		// Open file in append mode
		m_of_stream = std::make_unique<std::ofstream>(m_path.string(), std::ios::binary | std::ios::app);
	}
	m_of_stream->write(pointer, static_cast<int64_t>(size));
	invalidate_size_cache();
}

/*--------------------------------------------------------------------------------------------------------------------*\
 * STATIC
\*--------------------------------------------------------------------------------------------------------------------*/
string File::read(const path& file_path) {
	std::ifstream     json_stream = FileSystem::open_r(file_path);
	std::stringstream buffer;
	buffer << json_stream.rdbuf();
	return buffer.str();
}

void File::write(const path& dir_path, const string& dump) {
	auto file = FileSystem::open_w(dir_path);

	file << dump;

	FileSystem::close(file);
}

void File::append(const path& dir_path, const string& dump) {
	auto file = FileSystem::opend_app(dir_path);

	file << dump;

	FileSystem::close(file);
}

void File::open_read_handle() const {
#if defined(_WIN32)
	std::lock_guard<std::mutex> lock(m_read_mutex);
	if (!m_has_cached_size) {
		if (!exists(m_path)) {
			throw std::runtime_error("File does not exist");
		}
		m_cached_size     = static_cast<n_t>(fs::file_size(m_path));
		m_has_cached_size = true;
	}
#else
	std::lock_guard<std::mutex> lock(m_read_mutex);
	if (m_fd >= 0) {
		return;
	}

	m_fd = ::open(m_path.c_str(), O_RDONLY);
	if (m_fd < 0) {
		throw make_io_error(m_path, "open failed");
	}

	struct stat st {};
	if (::fstat(m_fd, &st) != 0) {
		const auto saved_errno = errno;
		::close(m_fd);
		m_fd = -1;
		errno = saved_errno;
		throw make_io_error(m_path, "fstat failed");
	}

	m_cached_size     = static_cast<n_t>(st.st_size);
	m_has_cached_size = true;

	// Hint the kernel that we'll be doing large sequential pread()s. The
	// stock 128 KiB read_ahead_kb is too conservative for rowgroup-sized
	// reads on NVMe Gen5; SEQUENTIAL doubles the readahead window and biases
	// page-cache eviction toward already-consumed regions. Failures are
	// non-fatal — the syscall is advisory.
#ifdef POSIX_FADV_SEQUENTIAL
	(void)::posix_fadvise(m_fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
#endif
}

void File::close_read_handle() const {
#if defined(_WIN32)
	return;
#else
	std::lock_guard<std::mutex> lock(m_read_mutex);
	if (m_fd >= 0) {
		::close(m_fd);
		m_fd = -1;
	}
#endif
}

void File::invalidate_size_cache() {
	std::lock_guard<std::mutex> lock(m_read_mutex);
	m_has_cached_size = false;
#if !defined(_WIN32)
	if (m_fd >= 0) {
		::close(m_fd);
		m_fd = -1;
	}
#endif
}
} // namespace fastlanes
