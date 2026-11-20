// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/io/file.cpp
// ────────────────────────────────────────────────────────
#include "fls/io/file.hpp"
#include "fls/common/alias.hpp"
#include "fls/cor/lyt/buf.hpp"
#include "fls/io/read_source.hpp"
#include "fls/std/filesystem.hpp"
#include "fls/std/string.hpp"
#include <algorithm>
#include <mutex>
#include <span>
#if !defined(_WIN32)
#include <cerrno>
#endif
#include <cstddef>
#include <cstdint> // for int64_t
#if !defined(_WIN32)
#include <cstring>
#include <fcntl.h>
#endif
#include <fstream> // for std::ofstream
#include <ios>     // for std::ios, std::streamsize
#include <memory>  // for std::make_unique
#include <sstream>
#include <string>
#if !defined(_WIN32)
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#endif
#include <stdexcept> // for std::runtime_error
#include <vector>

namespace fastlanes {

namespace {

#if !defined(_WIN32)
std::runtime_error make_io_error(const path& file_path, const std::string& action) {
	return std::runtime_error(action + ": " + file_path.string() + ": " + std::strerror(errno));
}

void pread_exact(const int fd, const path& file_path, void* dst, const n_t offset, const n_t size) {
	auto* out     = reinterpret_cast<std::byte*>(dst);
	n_t   done    = 0;
	auto  cur_off = static_cast<off_t>(offset);

	while (done < size) {
		const auto chunk = static_cast<size_t>(size - done);
		const auto nread = ::pread(fd, out + done, chunk, cur_off);
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
	if (!*m_of_stream) {
		throw std::runtime_error("write failed: " + m_path.string());
	}
	invalidate_size_cache();
}

void File::Read(Buf& buf) {
	ReadRange(buf, 0, Size());
}

void File::ReadRange(Buf& buf, const n_t offset, const n_t size) {
	ReadSource(*this).ReadRange(buf, offset, size);
}

void File::ReadRange(void* dst, const n_t offset, const n_t size) {
	ReadSource(*this).ReadRange(dst, offset, size);
}

void File::ReadRangeUnchecked(void* dst, const n_t offset, const n_t size) {
	if (size == 0) {
		return;
	}
	if (!dst) {
		throw std::invalid_argument("range read destination is null");
	}
#if defined(_WIN32)
	std::lock_guard lock(m_read_mutex);
	if (!m_if_stream) {
		m_if_stream = make_unique<std::ifstream>(FileSystem::open_r_binary(m_path));
	}
	auto& stream = *m_if_stream;
	stream.clear();
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

void File::ReadRangesUnchecked(const std::span<const FileRangeReadTarget> targets) {
	ReadSource(*this).ReadRanges(targets);
}

n_t File::ReadScatterUnchecked(const std::span<const FileScatterReadTarget> targets, const n_t offset) {
#if defined(_WIN32)
	n_t current_offset = offset;
	n_t read_count     = 0;
	for (const auto& target : targets) {
		if (target.size == 0) {
			continue;
		}
		if (target.data == nullptr) {
			throw std::invalid_argument("scatter read destination is null");
		}
		ReadRangeUnchecked(target.data, current_offset, target.size);
		current_offset += target.size;
		++read_count;
	}
	return read_count;
#else
	open_read_handle();
	const auto         configured_iov_max = ::sysconf(_SC_IOV_MAX);
	const size_t       iov_max            = configured_iov_max > 0 ? static_cast<size_t>(configured_iov_max) : 1024U;
	n_t                current_offset     = offset;
	n_t                read_count         = 0;
	std::vector<iovec> iovecs; // NOLINT(misc-include-cleaner): POSIX declares iovec in <sys/uio.h>.
	iovecs.reserve(std::min(iov_max, targets.size()));

	const auto flush = [&]() {
		if (iovecs.empty()) {
			return;
		}
		size_t first = 0;
		while (first < iovecs.size()) {
			const auto nread = ::preadv(m_fd,
			                            iovecs.data() + first,
			                            static_cast<int>(iovecs.size() - first),
			                            static_cast<off_t>(current_offset));
			if (nread == 0) {
				throw std::runtime_error("unexpected EOF while scatter-reading: " + m_path.string());
			}
			if (nread < 0) {
				if (errno == EINTR) {
					continue;
				}
				throw make_io_error(m_path, "preadv failed");
			}
			++read_count;
			current_offset += static_cast<n_t>(nread);
			auto consumed = static_cast<size_t>(nread);
			while (first < iovecs.size() && consumed >= iovecs[first].iov_len) {
				consumed -= iovecs[first].iov_len;
				++first;
			}
			if (consumed != 0U) {
				auto* const bytes      = static_cast<std::byte*>(iovecs[first].iov_base);
				iovecs[first].iov_base = bytes + consumed;
				iovecs[first].iov_len -= consumed;
			}
		}
		iovecs.clear();
	};

	for (const auto& target : targets) {
		if (target.size == 0) {
			continue;
		}
		if (target.data == nullptr) {
			throw std::invalid_argument("scatter read destination is null");
		}
		if (iovecs.size() == iov_max) {
			flush();
		}
		iovecs.push_back(iovec {target.data, static_cast<size_t>(target.size)});
	}
	flush();
	return read_count;
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
		m_of_stream = std::make_unique<std::ofstream>(m_path, std::ios::binary | std::ios::app);
	}
	m_of_stream->write(reinterpret_cast<char*>(buf.data()), static_cast<int64_t>(buf.Size()));
	if (!*m_of_stream) {
		throw std::runtime_error("append failed: " + m_path.string());
	}
	invalidate_size_cache();
}

void File::Append(const char* pointer, n_t size) {
	if (m_of_stream == nullptr) {
		// Open file in append mode
		m_of_stream = std::make_unique<std::ofstream>(m_path, std::ios::binary | std::ios::app);
	}
	m_of_stream->write(pointer, static_cast<int64_t>(size));
	if (!*m_of_stream) {
		throw std::runtime_error("append failed: " + m_path.string());
	}
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
		m_fd  = -1;
		errno = saved_errno;
		throw make_io_error(m_path, "fstat failed");
	}

	m_cached_size     = static_cast<n_t>(st.st_size);
	m_has_cached_size = true;

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
#if defined(_WIN32)
	m_if_stream.reset();
#else
	if (m_fd >= 0) {
		::close(m_fd);
		m_fd = -1;
	}
#endif
}
} // namespace fastlanes
