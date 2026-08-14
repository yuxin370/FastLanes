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
#include <algorithm>
#include <atomic>
#if !defined(_WIN32)
#include <cerrno>
#endif
#include <cstddef>
#include <cstdint>    // for int64_t
#include <limits>
#if defined(__linux__)
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#endif
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
#include <sys/uio.h>
#include <unistd.h>
#endif
#include <stdexcept> // for std::runtime_error
#include <vector>

namespace fastlanes {

namespace {

std::atomic<n_t> g_current_open_read_handles {0};
std::atomic<n_t> g_peak_open_read_handles {0};
std::atomic<n_t> g_read_handle_open_count {0};
std::atomic<n_t> g_read_handle_close_count {0};

void record_read_handle_open() noexcept {
	const auto current = g_current_open_read_handles.fetch_add(1, std::memory_order_relaxed) + 1;
	g_read_handle_open_count.fetch_add(1, std::memory_order_relaxed);
	auto peak = g_peak_open_read_handles.load(std::memory_order_relaxed);
	while (current > peak &&
	       !g_peak_open_read_handles.compare_exchange_weak(
	           peak, current, std::memory_order_relaxed, std::memory_order_relaxed)) {
	}
}

void record_read_handle_close() noexcept {
	g_current_open_read_handles.fetch_sub(1, std::memory_order_relaxed);
	g_read_handle_close_count.fetch_add(1, std::memory_order_relaxed);
}

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

#if defined(__linux__)
struct FileIoUringState {
	explicit FileIoUringState(const uint32_t requested_entries, const path& file_path)
	    : requested_entries(requested_entries) {
		if (requested_entries == 0U) {
			throw std::invalid_argument("io_uring queue depth must be positive");
		}
		io_uring_params params {};
		ring_fd = static_cast<int>(::syscall(SYS_io_uring_setup, requested_entries, &params));
		if (ring_fd < 0) {
			throw make_io_error(file_path, "io_uring_setup failed");
		}
		try {
			entries = params.sq_entries;
			if (entries == 0U || params.cq_entries == 0U) {
				throw std::runtime_error("io_uring_setup returned an empty queue: " + file_path.string());
			}
			sq_ring_bytes = params.sq_off.array + params.sq_entries * sizeof(uint32_t);
			cq_ring_bytes = params.cq_off.cqes + params.cq_entries * sizeof(io_uring_cqe);
			const bool single_mmap = (params.features & IORING_FEAT_SINGLE_MMAP) != 0U;
			if (single_mmap) {
				const size_t combined_bytes = std::max(sq_ring_bytes, cq_ring_bytes);
				sq_ring = map_ring(combined_bytes, IORING_OFF_SQ_RING, file_path);
				cq_ring = sq_ring;
				sq_ring_bytes = combined_bytes;
				cq_ring_bytes = 0U;
			} else {
				sq_ring = map_ring(sq_ring_bytes, IORING_OFF_SQ_RING, file_path);
				cq_ring = map_ring(cq_ring_bytes, IORING_OFF_CQ_RING, file_path);
			}
			sqes_bytes = params.sq_entries * sizeof(io_uring_sqe);
			sqes = static_cast<io_uring_sqe*>(map_ring(sqes_bytes, IORING_OFF_SQES, file_path));

			auto* const sq_base = static_cast<std::byte*>(sq_ring);
			auto* const cq_base = static_cast<std::byte*>(cq_ring);
			sq_head    = reinterpret_cast<uint32_t*>(sq_base + params.sq_off.head);
			sq_tail    = reinterpret_cast<uint32_t*>(sq_base + params.sq_off.tail);
			sq_mask    = reinterpret_cast<uint32_t*>(sq_base + params.sq_off.ring_mask);
			sq_entries = reinterpret_cast<uint32_t*>(sq_base + params.sq_off.ring_entries);
			sq_array   = reinterpret_cast<uint32_t*>(sq_base + params.sq_off.array);
			cq_head    = reinterpret_cast<uint32_t*>(cq_base + params.cq_off.head);
			cq_tail    = reinterpret_cast<uint32_t*>(cq_base + params.cq_off.tail);
			cq_mask    = reinterpret_cast<uint32_t*>(cq_base + params.cq_off.ring_mask);
			cq_entries = reinterpret_cast<uint32_t*>(cq_base + params.cq_off.ring_entries);
			cqes       = reinterpret_cast<io_uring_cqe*>(cq_base + params.cq_off.cqes);
			if (*sq_entries < requested_entries || *sq_entries != entries || *cq_entries < entries) {
				throw std::runtime_error("io_uring queue geometry is smaller than requested: " + file_path.string());
			}
		} catch (...) {
			cleanup();
			throw;
		}
	}

	~FileIoUringState() { cleanup(); }

	FileIoUringState(const FileIoUringState&) = delete;
	FileIoUringState& operator=(const FileIoUringState&) = delete;

	[[nodiscard]] size_t mapped_bytes() const noexcept {
		return sq_ring_bytes + cq_ring_bytes + sqes_bytes;
	}

	FileRangeBatchReadResult read(const int fd,
	                              const path& file_path,
	                              const std::span<const FileRangeReadTarget> targets) {
		struct Pending {
			std::byte* data      = nullptr;
			n_t        offset    = 0U;
			n_t        size      = 0U;
			n_t        completed = 0U;
		};
		std::vector<Pending> pending;
		pending.reserve(targets.size());
		FileRangeBatchReadResult stats {};
		stats.ring_mapped_bytes = mapped_bytes();
		for (const auto& target : targets) {
			if (target.size == 0U) {
				continue;
			}
			if (target.data == nullptr) {
				throw std::invalid_argument("io_uring read destination is null");
			}
			if (target.offset > static_cast<n_t>(std::numeric_limits<off_t>::max()) ||
			    target.size > static_cast<n_t>(std::numeric_limits<uint32_t>::max()) ||
			    target.size > std::numeric_limits<n_t>::max() - stats.bytes) {
				throw std::overflow_error("io_uring read range exceeds the supported offset/length domain");
			}
			stats.bytes += target.size;
			pending.push_back(Pending {static_cast<std::byte*>(target.data), target.offset, target.size, 0U});
		}

		for (size_t batch_begin = 0U; batch_begin < pending.size(); batch_begin += entries) {
			const size_t batch_end = std::min(pending.size(), batch_begin + entries);
			std::vector<size_t> active;
			active.reserve(batch_end - batch_begin);
			for (size_t index = batch_begin; index < batch_end; ++index) {
				active.push_back(index);
			}
			while (!active.empty()) {
				publish(active, fd, pending, file_path);
				submit(active.size(), file_path, stats);
				std::vector<int32_t> completions(pending.size(), std::numeric_limits<int32_t>::min());
				wait_and_collect(active.size(), completions, file_path, stats);
				std::vector<size_t> retry;
				retry.reserve(active.size());
				for (const size_t index : active) {
					const int32_t result = completions[index];
					if (result == std::numeric_limits<int32_t>::min()) {
						throw std::runtime_error("io_uring omitted a submitted completion: " + file_path.string());
					}
					if (result < 0) {
						errno = -result;
						throw make_io_error(file_path, "io_uring read failed");
					}
					if (result == 0) {
						throw std::runtime_error("unexpected EOF while io_uring-reading: " + file_path.string());
					}
					auto& item = pending[index];
					const n_t remaining = item.size - item.completed;
					if (static_cast<n_t>(result) > remaining) {
						throw std::runtime_error("io_uring completed more bytes than requested: " + file_path.string());
					}
					item.completed += static_cast<n_t>(result);
					if (item.completed != item.size) {
						retry.push_back(index);
					}
				}
				active = std::move(retry);
			}
		}
		return stats;
	}

	uint32_t requested_entries = 0U;
	uint32_t entries           = 0U;
	int      ring_fd           = -1;
	void*    sq_ring           = MAP_FAILED;
	void*    cq_ring           = MAP_FAILED;
	io_uring_sqe* sqes         = nullptr;
	size_t   sq_ring_bytes     = 0U;
	size_t   cq_ring_bytes     = 0U;
	size_t   sqes_bytes        = 0U;
	uint32_t* sq_head          = nullptr;
	uint32_t* sq_tail          = nullptr;
	uint32_t* sq_mask          = nullptr;
	uint32_t* sq_entries       = nullptr;
	uint32_t* sq_array         = nullptr;
	uint32_t* cq_head          = nullptr;
	uint32_t* cq_tail          = nullptr;
	uint32_t* cq_mask          = nullptr;
	uint32_t* cq_entries       = nullptr;
	io_uring_cqe* cqes         = nullptr;

private:
	void* map_ring(const size_t bytes, const off_t offset, const path& file_path) const {
		void* const mapped = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, ring_fd, offset);
		if (mapped == MAP_FAILED) {
			throw make_io_error(file_path, "io_uring mmap failed");
		}
		return mapped;
	}

	static uint32_t load_acquire(const uint32_t* value) noexcept {
		return __atomic_load_n(value, __ATOMIC_ACQUIRE);
	}

	static void store_release(uint32_t* value, const uint32_t updated) noexcept {
		__atomic_store_n(value, updated, __ATOMIC_RELEASE);
	}

	template <typename Pending>
	void publish(const std::vector<size_t>& active,
	             const int fd,
	             const std::vector<Pending>& pending,
	             const path& file_path) {
		const uint32_t head = load_acquire(sq_head);
		const uint32_t tail = load_acquire(sq_tail);
		if (active.size() > entries || tail - head > entries - active.size()) {
			throw std::runtime_error("io_uring submission queue has insufficient space: " + file_path.string());
		}
		for (size_t position = 0U; position < active.size(); ++position) {
			const size_t target_index = active[position];
			const auto&  item         = pending[target_index];
			const uint32_t sqe_index = (tail + static_cast<uint32_t>(position)) & *sq_mask;
			auto& sqe = sqes[sqe_index];
			std::memset(&sqe, 0, sizeof(sqe));
			sqe.opcode    = IORING_OP_READ;
			sqe.fd        = fd;
			sqe.off       = item.offset + item.completed;
			sqe.addr      = reinterpret_cast<uint64_t>(item.data + item.completed);
			sqe.len       = static_cast<uint32_t>(item.size - item.completed);
			sqe.user_data = target_index;
			sq_array[sqe_index] = sqe_index;
		}
		store_release(sq_tail, tail + static_cast<uint32_t>(active.size()));
	}

	void submit(const size_t count, const path& file_path, FileRangeBatchReadResult& stats) const {
		size_t remaining = count;
		while (remaining != 0U) {
			const int submitted = static_cast<int>(
			    ::syscall(SYS_io_uring_enter, ring_fd, remaining, 0U, 0U, nullptr, 0U));
			if (submitted < 0) {
				if (errno == EINTR) {
					continue;
				}
				throw make_io_error(file_path, "io_uring submission failed");
			}
			if (submitted == 0 || static_cast<size_t>(submitted) > remaining) {
				throw std::runtime_error("io_uring made invalid submission progress: " + file_path.string());
			}
			++stats.submit_syscall_count;
			remaining -= static_cast<size_t>(submitted);
		}
	}

	void wait_and_collect(const size_t expected,
	                      std::vector<int32_t>& completions,
	                      const path& file_path,
	                      FileRangeBatchReadResult& stats) const {
		size_t observed = 0U;
		while (observed < expected) {
			uint32_t head = load_acquire(cq_head);
			const uint32_t tail = load_acquire(cq_tail);
			while (head != tail && observed < expected) {
				const auto& cqe = cqes[head & *cq_mask];
				if (cqe.user_data >= completions.size() ||
				    completions[cqe.user_data] != std::numeric_limits<int32_t>::min()) {
					throw std::runtime_error("io_uring returned an invalid or duplicate completion: " +
					                         file_path.string());
				}
				completions[cqe.user_data] = cqe.res;
				++head;
				++observed;
				++stats.completion_count;
				++stats.read_request_count;
			}
			store_release(cq_head, head);
			if (observed == expected) {
				break;
			}
			const int ready = static_cast<int>(::syscall(
			    SYS_io_uring_enter, ring_fd, 0U, expected - observed, IORING_ENTER_GETEVENTS, nullptr, 0U));
			if (ready < 0) {
				if (errno == EINTR) {
					continue;
				}
				throw make_io_error(file_path, "io_uring completion wait failed");
			}
			++stats.wait_syscall_count;
		}
	}

	void cleanup() noexcept {
		if (sqes != nullptr && sqes != MAP_FAILED) {
			::munmap(sqes, sqes_bytes);
			sqes = nullptr;
		}
		if (cq_ring != MAP_FAILED && cq_ring != sq_ring) {
			::munmap(cq_ring, cq_ring_bytes);
			cq_ring = MAP_FAILED;
		}
		if (sq_ring != MAP_FAILED) {
			::munmap(sq_ring, sq_ring_bytes);
			sq_ring = MAP_FAILED;
		}
		if (ring_fd >= 0) {
			::close(ring_fd);
			ring_fd = -1;
		}
	}
};
#else
struct FileIoUringState {};
#endif

File::File(const path& path) // NOLINT
    : m_path(path) {
}

File::~File() {
	m_io_uring.reset();
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
	const long   configured_iov_max = ::sysconf(_SC_IOV_MAX);
	const size_t iov_max = configured_iov_max > 0 ? static_cast<size_t>(configured_iov_max) : 1024U;
	n_t          current_offset = offset;
	n_t          read_count     = 0;
	std::vector<iovec> iovecs;
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
			size_t consumed = static_cast<size_t>(nread);
			while (first < iovecs.size() && consumed >= iovecs[first].iov_len) {
				consumed -= iovecs[first].iov_len;
				++first;
			}
			if (consumed != 0U) {
				auto* const bytes = static_cast<std::byte*>(iovecs[first].iov_base);
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

FileRangeBatchReadResult File::ReadRangesIoUringUnchecked(
    const std::span<const FileRangeReadTarget> targets, const uint32_t queue_depth) {
#if defined(__linux__)
	if (queue_depth == 0U) {
		throw std::invalid_argument("io_uring queue depth must be positive");
	}
	open_read_handle();
	std::lock_guard<std::mutex> lock(m_read_mutex);
	const bool created = !m_io_uring;
	if (created) {
		m_io_uring = std::make_unique<FileIoUringState>(queue_depth, m_path);
	} else if (m_io_uring->requested_entries != queue_depth) {
		throw std::invalid_argument("io_uring queue depth changed for an open file");
	}
	try {
		auto result = m_io_uring->read(m_fd, m_path, targets);
		result.newly_mapped_ring_bytes = created ? result.ring_mapped_bytes : 0U;
		return result;
	} catch (...) {
		// Closing the ring cancels any request left in flight by an exceptional
		// submission/completion path. A formal io_uring mode never falls back.
		m_io_uring.reset();
		throw;
	}
#else
	(void)targets;
	(void)queue_depth;
	throw std::runtime_error("io_uring discontiguous reads require Linux");
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
	if (!*m_of_stream) {
		throw std::runtime_error("append failed: " + m_path.string());
	}
	invalidate_size_cache();
}

void File::Append(const char* pointer, n_t size) {
	if (m_of_stream == nullptr) {
		// Open file in append mode
		m_of_stream = std::make_unique<std::ofstream>(m_path.string(), std::ios::binary | std::ios::app);
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

FileReadHandleStats File::read_handle_stats() noexcept {
	return {g_current_open_read_handles.load(std::memory_order_relaxed),
	        g_peak_open_read_handles.load(std::memory_order_relaxed),
	        g_read_handle_open_count.load(std::memory_order_relaxed),
	        g_read_handle_close_count.load(std::memory_order_relaxed)};
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
	record_read_handle_open();

	struct stat st {};
	if (::fstat(m_fd, &st) != 0) {
		const auto saved_errno = errno;
		::close(m_fd);
		record_read_handle_close();
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
		record_read_handle_close();
		m_fd = -1;
	}
#endif
}

void File::invalidate_size_cache() {
	std::lock_guard<std::mutex> lock(m_read_mutex);
	m_has_cached_size = false;
	m_io_uring.reset();
#if !defined(_WIN32)
	if (m_fd >= 0) {
		::close(m_fd);
		record_read_handle_close();
		m_fd = -1;
	}
#endif
}
} // namespace fastlanes
