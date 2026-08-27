#include "jpeg/jpeg_dct_active_output_schedule.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <type_traits>
#include <utility>

namespace galp::jpeg::detail {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::array<char, 8> kMagic {'G', 'A', 'L', 'P', 'A', 'O', 'S', '1'};
constexpr uint32_t kFormatVersion = 1U;
constexpr size_t kHeaderBytes = 256U;
constexpr size_t kChecksumOffset = 32U;
constexpr size_t kChecksumBytes = sizeof(uint64_t);
constexpr size_t kIntervalBytes = 2U * sizeof(uint32_t);
constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

[[noreturn]] void fail(const std::string& message) {
	throw std::runtime_error("active-output schedule sidecar: " + message);
}

double elapsed_ms(const Clock::time_point begin, const Clock::time_point end) {
	return std::chrono::duration<double, std::milli>(end - begin).count();
}

uint64_t hash_bytes(uint64_t hash, const void* const data, const size_t size) noexcept {
	const auto* bytes = static_cast<const unsigned char*>(data);
	for (size_t index = 0U; index < size; ++index) {
		hash ^= bytes[index];
		hash *= kFnvPrime;
	}
	return hash;
}

template <typename T>
uint64_t hash_scalar(uint64_t hash, const T value) noexcept {
	std::array<unsigned char, sizeof(T)> encoded {};
	using Unsigned = std::make_unsigned_t<T>;
	const auto scalar = static_cast<Unsigned>(value);
	for (size_t index = 0U; index < sizeof(T); ++index) {
		encoded[index] = static_cast<unsigned char>((scalar >> (8U * index)) & 0xffU);
	}
	return hash_bytes(hash, encoded.data(), encoded.size());
}

uint64_t hash_float(uint64_t hash, const float value) noexcept {
	uint32_t bits = 0U;
	static_assert(sizeof(bits) == sizeof(value));
	std::memcpy(&bits, &value, sizeof(bits));
	return hash_scalar(hash, bits);
}

template <typename T>
void put_le(std::vector<std::byte>& bytes, const size_t offset, const T value) {
	if (offset > bytes.size() || sizeof(T) > bytes.size() - offset) {
		fail("encoder exceeded its output buffer");
	}
	using Unsigned = std::make_unsigned_t<T>;
	const auto scalar = static_cast<Unsigned>(value);
	for (size_t index = 0U; index < sizeof(T); ++index) {
		bytes[offset + index] = static_cast<std::byte>((scalar >> (8U * index)) & 0xffU);
	}
}

template <typename T>
T get_le(const std::byte* const bytes, const size_t size, const size_t offset, const std::string_view label) {
	if (offset > size || sizeof(T) > size - offset) {
		fail("truncated " + std::string(label));
	}
	using Unsigned = std::make_unsigned_t<T>;
	Unsigned value = 0U;
	for (size_t index = 0U; index < sizeof(T); ++index) {
		value |= static_cast<Unsigned>(std::to_integer<unsigned char>(bytes[offset + index])) << (8U * index);
	}
	return static_cast<T>(value);
}

uint64_t checksum_with_zeroed_field(const std::byte* const bytes, const size_t size) noexcept {
	if (size < kChecksumOffset + kChecksumBytes) {
		return 0U;
	}
	uint64_t hash = hash_bytes(kFnvOffsetBasis, bytes, kChecksumOffset);
	const std::array<std::byte, kChecksumBytes> zeros {};
	hash = hash_bytes(hash, zeros.data(), zeros.size());
	return hash_bytes(hash,
	                  bytes + kChecksumOffset + kChecksumBytes,
	                  size - kChecksumOffset - kChecksumBytes);
}

struct FileIdentity {
	dev_t device = 0;
	ino_t inode = 0;
	off_t size = 0;
	timespec mtime {};

	bool operator==(const FileIdentity& other) const noexcept {
		return device == other.device && inode == other.inode && size == other.size &&
		       mtime.tv_sec == other.mtime.tv_sec && mtime.tv_nsec == other.mtime.tv_nsec;
	}
};

struct Mapping {
	std::filesystem::path path;
	FileIdentity          identity;
	int                   fd = -1;
	const std::byte*      data = nullptr;
	size_t                size = 0U;

	~Mapping() {
		if (data != nullptr && size != 0U) {
			::munmap(const_cast<std::byte*>(data), size);
		}
		if (fd >= 0) {
			::close(fd);
		}
	}

	Mapping() = default;
	Mapping(const Mapping&) = delete;
	Mapping& operator=(const Mapping&) = delete;
};

struct MappingCacheEntry {
	std::filesystem::path path;
	std::shared_ptr<Mapping> mapping;
	uint64_t last_use = 0U;
};

struct MappingCache {
	std::mutex mutex;
	std::vector<MappingCacheEntry> entries;
	uint64_t clock = 0U;
};

MappingCache& mapping_cache() {
	static MappingCache cache;
	return cache;
}

FileIdentity file_identity(const int fd) {
	struct stat status {};
	if (::fstat(fd, &status) != 0) {
		fail("fstat failed: " + std::string(std::strerror(errno)));
	}
	return {status.st_dev, status.st_ino, status.st_size, status.st_mtim};
}

struct MappingLookup {
	std::shared_ptr<Mapping> mapping;
	uint64_t resident_bytes = 0U;
};

MappingLookup map_sidecar(const std::filesystem::path& path) {
	const int candidate_fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
	if (candidate_fd < 0) {
		if (errno == ENOENT) {
			return {};
		}
		fail("open failed for " + path.string() + ": " + std::strerror(errno));
	}
	const auto candidate_identity = file_identity(candidate_fd);
	if (candidate_identity.size <= 0 ||
	    static_cast<uint64_t>(candidate_identity.size) > std::numeric_limits<size_t>::max()) {
		::close(candidate_fd);
		fail("file size is outside the host address range");
	}

	auto& cache = mapping_cache();
	std::lock_guard lock(cache.mutex);
	++cache.clock;
	for (auto& entry : cache.entries) {
		if (entry.path == path && entry.mapping->identity == candidate_identity) {
			::close(candidate_fd);
			entry.last_use = cache.clock;
			uint64_t resident = 0U;
			for (const auto& current : cache.entries) {
				resident += current.mapping->size;
			}
			return {entry.mapping, resident};
		}
	}

	const auto size = static_cast<size_t>(candidate_identity.size);
	void* const address = ::mmap(nullptr, size, PROT_READ, MAP_SHARED, candidate_fd, 0);
	if (address == MAP_FAILED) {
		const auto error = errno;
		::close(candidate_fd);
		fail("mmap failed for " + path.string() + ": " + std::strerror(error));
	}
	auto mapping = std::make_shared<Mapping>();
	mapping->path = path;
	mapping->identity = candidate_identity;
	mapping->fd = candidate_fd;
	mapping->data = static_cast<const std::byte*>(address);
	mapping->size = size;

	cache.entries.erase(
	    std::remove_if(cache.entries.begin(), cache.entries.end(), [&](const auto& entry) {
		    return entry.path == path;
	    }),
	    cache.entries.end());
	cache.entries.push_back({path, mapping, cache.clock});
	auto resident_bytes = [&]() {
		uint64_t total = 0U;
		for (const auto& entry : cache.entries) {
			total += entry.mapping->size;
		}
		return total;
	};
	while (cache.entries.size() > kJpegDctActiveOutputScheduleMmapWindowCount ||
	       (resident_bytes() > kJpegDctActiveOutputScheduleMmapCapacityBytes && cache.entries.size() > 1U)) {
		const auto victim = std::min_element(cache.entries.begin(), cache.entries.end(), [](const auto& lhs, const auto& rhs) {
			return std::pair {lhs.last_use, lhs.path.string()} < std::pair {rhs.last_use, rhs.path.string()};
		});
		if (victim == cache.entries.end() || victim->mapping == mapping) {
			break;
		}
		cache.entries.erase(victim);
	}
	return {std::move(mapping), resident_bytes()};
}

struct Interval {
	uint32_t begin = 0U;
	uint32_t length = 0U;
};

struct EncodedSchedule {
	std::vector<std::byte> bytes;
	uint64_t interval_count = 0U;
};

EncodedSchedule encode_schedule(const JpegDctActiveOutputScheduleKey& key,
	                              const JpegDctDeviceBlockMajorActiveOutputSchedule& schedule) {
	if (key.workset_count == 0U || schedule.offsets.size() != static_cast<size_t>(key.workset_count) + 1U ||
	    schedule.offsets.front() != 0U || schedule.offsets.back() != schedule.active_output_blocks.size() ||
	    schedule.output_workset_ownership_count != schedule.active_output_blocks.size() ||
	    schedule.logical_output_block_count > std::numeric_limits<uint32_t>::max()) {
		fail("schedule geometry is inconsistent");
	}
	std::vector<uint64_t> interval_offsets(static_cast<size_t>(key.workset_count) + 1U, 0U);
	std::vector<Interval> intervals;
	for (uint32_t workset = 0U; workset < key.workset_count; ++workset) {
		const auto begin = schedule.offsets[workset];
		const auto end = schedule.offsets[workset + 1U];
		if (begin > end || end > schedule.active_output_blocks.size()) {
			fail("workset slice is outside the ownership vector");
		}
		uint64_t cursor = begin;
		uint32_t previous = 0U;
		bool have_previous = false;
		while (cursor < end) {
			const auto first = schedule.active_output_blocks[static_cast<size_t>(cursor++)];
			if (first >= schedule.logical_output_block_count || (have_previous && first <= previous)) {
				fail("workset slice is not sorted, unique, and bounded");
			}
			uint32_t length = 1U;
			previous = first;
			have_previous = true;
			while (cursor < end) {
				const auto next = schedule.active_output_blocks[static_cast<size_t>(cursor)];
				if (next <= previous) {
					fail("workset slice is not sorted and unique");
				}
				if (next != previous + 1U || length == std::numeric_limits<uint32_t>::max()) {
					break;
				}
				++cursor;
				++length;
				previous = next;
			}
			intervals.push_back({first, length});
		}
		interval_offsets[workset + 1U] = intervals.size();
	}

	const uint64_t offsets_bytes = interval_offsets.size() * sizeof(uint64_t);
	const uint64_t intervals_bytes = intervals.size() * kIntervalBytes;
	const uint64_t offsets_offset = kHeaderBytes;
	const uint64_t intervals_offset = offsets_offset + offsets_bytes;
	const uint64_t sidecar_bytes = intervals_offset + intervals_bytes;
	if (sidecar_bytes > std::numeric_limits<size_t>::max()) {
		fail("encoded sidecar exceeds the host address range");
	}
	EncodedSchedule encoded;
	encoded.bytes.resize(static_cast<size_t>(sidecar_bytes));
	encoded.interval_count = intervals.size();
	std::memcpy(encoded.bytes.data(), kMagic.data(), kMagic.size());
	put_le<uint32_t>(encoded.bytes, 8U, kFormatVersion);
	put_le<uint32_t>(encoded.bytes, 12U, kHeaderBytes);
	put_le<uint32_t>(encoded.bytes, 16U, key.planner_abi);
	put_le<uint32_t>(encoded.bytes, 20U, key.shard_id);
	put_le<uint64_t>(encoded.bytes, 24U, sidecar_bytes);
	put_le<uint64_t>(encoded.bytes, 32U, 0U);
	put_le<uint64_t>(encoded.bytes, 40U, jpeg_dct_active_output_schedule_key_digest(key));
	put_le<uint64_t>(encoded.bytes, 48U, key.canonical_plan_digest);
	put_le<uint64_t>(encoded.bytes, 56U, key.decision_digest);
	put_le<uint64_t>(encoded.bytes, 64U, key.transform_digest);
	put_le<uint64_t>(encoded.bytes, 72U, key.decode_workset_capacity_bytes);
	put_le<uint32_t>(encoded.bytes, 80U, key.decode_batch_rowgroups);
	put_le<uint32_t>(encoded.bytes, 84U, key.double_buffer_policy);
	put_le<uint32_t>(encoded.bytes, 88U, key.workset_count);
	put_le<uint32_t>(encoded.bytes, 92U, kIntervalBytes);
	put_le<uint64_t>(encoded.bytes, 96U, schedule.logical_output_block_count);
	put_le<uint64_t>(encoded.bytes, 104U, schedule.source_contribution_count);
	put_le<uint64_t>(encoded.bytes, 112U, schedule.source_contribution_visit_count);
	put_le<uint64_t>(encoded.bytes, 120U, schedule.output_workset_ownership_count);
	put_le<uint64_t>(encoded.bytes, 128U, intervals.size());
	put_le<uint64_t>(encoded.bytes, 136U, offsets_offset);
	put_le<uint64_t>(encoded.bytes, 144U, intervals_offset);
	put_le<uint64_t>(encoded.bytes, 152U, sidecar_bytes);
	for (size_t index = 0U; index < interval_offsets.size(); ++index) {
		put_le<uint64_t>(encoded.bytes, static_cast<size_t>(offsets_offset) + index * sizeof(uint64_t), interval_offsets[index]);
	}
	for (size_t index = 0U; index < intervals.size(); ++index) {
		const auto base = static_cast<size_t>(intervals_offset) + index * kIntervalBytes;
		put_le<uint32_t>(encoded.bytes, base, intervals[index].begin);
		put_le<uint32_t>(encoded.bytes, base + sizeof(uint32_t), intervals[index].length);
	}
	put_le<uint64_t>(encoded.bytes, kChecksumOffset,
	                 checksum_with_zeroed_field(encoded.bytes.data(), encoded.bytes.size()));
	return encoded;
}

class FileLock {
public:
	explicit FileLock(const std::filesystem::path& path) {
		fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
		if (fd_ < 0) {
			fail("cannot open lock " + path.string() + ": " + std::strerror(errno));
		}
		if (::flock(fd_, LOCK_EX) != 0) {
			const auto error = errno;
			::close(fd_);
			fd_ = -1;
			fail("cannot lock " + path.string() + ": " + std::strerror(error));
		}
	}

	~FileLock() {
		if (fd_ >= 0) {
			::flock(fd_, LOCK_UN);
			::close(fd_);
		}
	}

	FileLock(const FileLock&) = delete;
	FileLock& operator=(const FileLock&) = delete;

private:
	int fd_ = -1;
};

void write_all(const int fd, const std::byte* bytes, const size_t size) {
	size_t written = 0U;
	while (written < size) {
		const auto result = ::write(fd, bytes + written, size - written);
		if (result < 0 && errno == EINTR) {
			continue;
		}
		if (result <= 0) {
			fail("sidecar write made no progress: " + std::string(std::strerror(errno)));
		}
		written += static_cast<size_t>(result);
	}
}

} // namespace

uint64_t jpeg_dct_active_output_transform_digest(const JpegDctGridTransformSpec& transform) noexcept {
	uint64_t hash = kFnvOffsetBasis;
	hash = hash_scalar(hash, transform.y_output_width_blocks);
	hash = hash_scalar(hash, transform.y_output_height_blocks);
	hash = hash_scalar(hash, transform.cbcr_output_width_blocks);
	hash = hash_scalar(hash, transform.cbcr_output_height_blocks);
	hash = hash_scalar(hash, transform.crop_reference_width_blocks);
	hash = hash_scalar(hash, transform.crop_reference_height_blocks);
	hash = hash_scalar(hash, transform.crop_origin_alignment_blocks);
	hash = hash_scalar(hash, transform.chroma_crop_scale_x);
	hash = hash_scalar(hash, transform.chroma_crop_scale_y);
	hash = hash_scalar(hash, transform.clamp_min);
	hash = hash_scalar(hash, transform.clamp_max);
	hash = hash_scalar(hash, static_cast<uint32_t>(transform.output_data_type));
	hash = hash_float(hash, transform.output_add);
	hash = hash_float(hash, transform.output_scale);
	hash = hash_scalar(hash, static_cast<uint8_t>(transform.dequantize));
	hash = hash_scalar(hash, static_cast<uint8_t>(transform.require_all_coefficients));
	hash = hash_scalar(hash, static_cast<uint8_t>(transform.allow_grayscale));
	for (const auto value : transform.preferred_small_crop_width_blocks) {
		hash = hash_scalar(hash, value);
	}
	for (const auto value : transform.preferred_small_crop_height_blocks) {
		hash = hash_scalar(hash, value);
	}
	for (const auto& ratio : transform.allowed_chroma_sampling_ratios) {
		hash = hash_scalar(hash, ratio.horizontal_numerator);
		hash = hash_scalar(hash, ratio.horizontal_denominator);
		hash = hash_scalar(hash, ratio.vertical_numerator);
		hash = hash_scalar(hash, ratio.vertical_denominator);
	}
	return hash;
}

uint64_t jpeg_dct_active_output_decision_digest(
    const std::vector<JpegDctActiveOutputDecisionRecord>& decisions) noexcept {
	uint64_t hash = hash_scalar(kFnvOffsetBasis, static_cast<uint64_t>(decisions.size()));
	for (const auto& record : decisions) {
		hash = hash_scalar(hash, record.shard_id);
		hash = hash_scalar(hash, record.rowgroup_index);
		hash = hash_scalar(hash, record.workset_index);
	}
	return hash;
}

uint64_t jpeg_dct_active_output_schedule_key_digest(const JpegDctActiveOutputScheduleKey& key) noexcept {
	uint64_t hash = kFnvOffsetBasis;
	hash = hash_scalar(hash, key.shard_id);
	hash = hash_scalar(hash, key.canonical_plan_digest);
	hash = hash_scalar(hash, key.decision_digest);
	hash = hash_scalar(hash, key.transform_digest);
	hash = hash_scalar(hash, key.decode_workset_capacity_bytes);
	hash = hash_scalar(hash, key.decode_batch_rowgroups);
	hash = hash_scalar(hash, key.double_buffer_policy);
	hash = hash_scalar(hash, key.workset_count);
	hash = hash_scalar(hash, key.planner_abi);
	return hash;
}

std::filesystem::path jpeg_dct_active_output_schedule_path(const JpegDctActiveOutputScheduleKey& key) {
	std::array<char, 64> name {};
	const auto written = std::snprintf(name.data(), name.size(), "shard_%06u.active_output_schedule.bin", key.shard_id);
	if (written <= 0 || static_cast<size_t>(written) >= name.size()) {
		fail("failed to format the sidecar path");
	}
	return key.directory / name.data();
}

JpegDctActiveOutputScheduleIoResult load_jpeg_dct_active_output_schedule(
    const JpegDctActiveOutputScheduleKey& key) {
	JpegDctActiveOutputScheduleIoResult result;
	const auto load_start = Clock::now();
	try {
		const auto mapping_lookup = map_sidecar(jpeg_dct_active_output_schedule_path(key));
		if (!mapping_lookup.mapping) {
			return result;
		}
		const auto& mapping = *mapping_lookup.mapping;
		result.sidecar_bytes = mapping.size;
		result.mapped_bytes = mapping_lookup.resident_bytes;
		result.load_ms = elapsed_ms(load_start, Clock::now());
		const auto validation_start = Clock::now();
		const auto* bytes = mapping.data;
		const auto size = mapping.size;
		if (size < kHeaderBytes || std::memcmp(bytes, kMagic.data(), kMagic.size()) != 0 ||
		    get_le<uint32_t>(bytes, size, 8U, "format version") != kFormatVersion ||
		    get_le<uint32_t>(bytes, size, 12U, "header bytes") != kHeaderBytes ||
		    get_le<uint32_t>(bytes, size, 16U, "planner ABI") != key.planner_abi ||
		    get_le<uint32_t>(bytes, size, 20U, "shard id") != key.shard_id ||
		    get_le<uint64_t>(bytes, size, 24U, "sidecar bytes") != size ||
		    get_le<uint64_t>(bytes, size, 32U, "checksum") != checksum_with_zeroed_field(bytes, size) ||
		    get_le<uint64_t>(bytes, size, 40U, "key digest") != jpeg_dct_active_output_schedule_key_digest(key) ||
		    get_le<uint64_t>(bytes, size, 48U, "canonical plan digest") != key.canonical_plan_digest ||
		    get_le<uint64_t>(bytes, size, 56U, "decision digest") != key.decision_digest ||
		    get_le<uint64_t>(bytes, size, 64U, "transform digest") != key.transform_digest ||
		    get_le<uint64_t>(bytes, size, 72U, "workset capacity") != key.decode_workset_capacity_bytes ||
		    get_le<uint32_t>(bytes, size, 80U, "decode rowgroups") != key.decode_batch_rowgroups ||
		    get_le<uint32_t>(bytes, size, 84U, "double-buffer policy") != key.double_buffer_policy ||
		    get_le<uint32_t>(bytes, size, 88U, "workset count") != key.workset_count ||
		    get_le<uint32_t>(bytes, size, 92U, "interval bytes") != kIntervalBytes) {
			fail("header identity, ABI, or checksum mismatch");
		}
		const auto logical_count = get_le<uint64_t>(bytes, size, 96U, "logical output count");
		const auto source_count = get_le<uint64_t>(bytes, size, 104U, "source contribution count");
		const auto source_visits = get_le<uint64_t>(bytes, size, 112U, "source visit count");
		const auto ownership_count = get_le<uint64_t>(bytes, size, 120U, "ownership count");
		const auto interval_count = get_le<uint64_t>(bytes, size, 128U, "interval count");
		const auto offsets_offset = get_le<uint64_t>(bytes, size, 136U, "offsets offset");
		const auto intervals_offset = get_le<uint64_t>(bytes, size, 144U, "intervals offset");
		const auto payload_end = get_le<uint64_t>(bytes, size, 152U, "payload end");
		const uint64_t offsets_count = static_cast<uint64_t>(key.workset_count) + 1U;
		if (logical_count > std::numeric_limits<uint32_t>::max() || ownership_count > std::numeric_limits<size_t>::max() ||
		    interval_count > std::numeric_limits<size_t>::max() || offsets_offset != kHeaderBytes ||
		    offsets_count > (std::numeric_limits<uint64_t>::max() - offsets_offset) / sizeof(uint64_t) ||
		    intervals_offset != offsets_offset + offsets_count * sizeof(uint64_t) ||
		    interval_count > (std::numeric_limits<uint64_t>::max() - intervals_offset) / kIntervalBytes ||
		    payload_end != intervals_offset + interval_count * kIntervalBytes || payload_end != size) {
			fail("payload geometry is invalid");
		}
		std::vector<uint64_t> interval_offsets(static_cast<size_t>(offsets_count));
		for (size_t index = 0U; index < interval_offsets.size(); ++index) {
			interval_offsets[index] = get_le<uint64_t>(
			    bytes, size, static_cast<size_t>(offsets_offset) + index * sizeof(uint64_t), "interval offset");
		}
		if (interval_offsets.front() != 0U || interval_offsets.back() != interval_count ||
		    !std::is_sorted(interval_offsets.begin(), interval_offsets.end())) {
			fail("interval offsets are not monotonic and complete");
		}
		result.validation_ms = elapsed_ms(validation_start, Clock::now());

		const auto materialize_start = Clock::now();
		JpegDctDeviceBlockMajorActiveOutputSchedule schedule;
		schedule.offsets.assign(static_cast<size_t>(key.workset_count) + 1U, 0U);
		schedule.active_output_blocks.reserve(static_cast<size_t>(ownership_count));
		for (uint32_t workset = 0U; workset < key.workset_count; ++workset) {
			uint32_t previous_end = 0U;
			bool have_previous = false;
			for (uint64_t index = interval_offsets[workset]; index < interval_offsets[workset + 1U]; ++index) {
				const auto base = static_cast<size_t>(intervals_offset + index * kIntervalBytes);
				const auto begin = get_le<uint32_t>(bytes, size, base, "interval begin");
				const auto length = get_le<uint32_t>(bytes, size, base + sizeof(uint32_t), "interval length");
				if (length == 0U || begin >= logical_count || length > logical_count - begin ||
				    (have_previous && begin <= previous_end)) {
					fail("interval is empty, overlapping, or out of range");
				}
				if (length > ownership_count ||
				    schedule.active_output_blocks.size() > static_cast<size_t>(ownership_count - length)) {
					fail("interval expansion exceeds the ownership count");
				}
				for (uint32_t offset = 0U; offset < length; ++offset) {
					schedule.active_output_blocks.push_back(begin + offset);
				}
				previous_end = begin + length - 1U;
				have_previous = true;
			}
			schedule.offsets[workset + 1U] = schedule.active_output_blocks.size();
		}
		if (schedule.active_output_blocks.size() != ownership_count) {
			fail("interval expansion did not reproduce the ownership count");
		}
		schedule.logical_output_block_count = logical_count;
		schedule.source_contribution_count = source_count;
		schedule.source_contribution_visit_count = source_visits;
		schedule.output_workset_ownership_count = ownership_count;
		schedule.sidecar_mapping = std::static_pointer_cast<const void>(mapping_lookup.mapping);
		schedule.sidecar_bytes = mapping.size;
		schedule.sidecar_mapped_bytes = mapping_lookup.resident_bytes;
		schedule.sidecar_interval_count = interval_count;
		schedule.sidecar_hit = true;
		result.materialize_ms = elapsed_ms(materialize_start, Clock::now());
		schedule.sidecar_load_ms = result.load_ms;
		schedule.sidecar_validation_ms = result.validation_ms;
		schedule.sidecar_materialize_ms = result.materialize_ms;
		result.interval_count = interval_count;
		result.hit = true;
		result.schedule = std::move(schedule);
		return result;
	} catch (const std::exception& error) {
		result.load_ms = elapsed_ms(load_start, Clock::now());
		result.rejected = true;
		result.rejection_reason = error.what();
		return result;
	}
}

JpegDctActiveOutputScheduleIoResult persist_jpeg_dct_active_output_schedule(
    const JpegDctActiveOutputScheduleKey&                   key,
    const JpegDctDeviceBlockMajorActiveOutputSchedule& schedule) {
	JpegDctActiveOutputScheduleIoResult result;
	const auto started = Clock::now();
	try {
		if (key.directory.empty() || !std::filesystem::is_directory(key.directory)) {
			fail("output directory is missing: " + key.directory.string());
		}
		const auto path = jpeg_dct_active_output_schedule_path(key);
		FileLock lock(path.string() + ".lock");
		const auto existing = load_jpeg_dct_active_output_schedule(key);
		if (existing.hit) {
			return existing;
		}
		auto encoded = encode_schedule(key, schedule);
		const auto temporary = path.string() + ".tmp." + std::to_string(static_cast<unsigned long long>(::getpid()));
		int fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
		if (fd < 0) {
			fail("cannot create temporary sidecar " + temporary + ": " + std::strerror(errno));
		}
		try {
			write_all(fd, encoded.bytes.data(), encoded.bytes.size());
			if (::fsync(fd) != 0) {
				fail("fsync failed for temporary sidecar: " + std::string(std::strerror(errno)));
			}
			if (::close(fd) != 0) {
				fd = -1;
				fail("close failed for temporary sidecar: " + std::string(std::strerror(errno)));
			}
			fd = -1;
			if (::rename(temporary.c_str(), path.c_str()) != 0) {
				fail("atomic rename failed: " + std::string(std::strerror(errno)));
			}
			const int directory_fd = ::open(key.directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
			if (directory_fd < 0) {
				fail("cannot open sidecar directory for fsync: " + std::string(std::strerror(errno)));
			}
			const auto sync_status = ::fsync(directory_fd);
			const auto sync_error = errno;
			::close(directory_fd);
			if (sync_status != 0) {
				fail("directory fsync failed: " + std::string(std::strerror(sync_error)));
			}
		} catch (...) {
			if (fd >= 0) {
				::close(fd);
			}
			::unlink(temporary.c_str());
			throw;
		}
		result.persisted = true;
		result.sidecar_bytes = encoded.bytes.size();
		result.interval_count = encoded.interval_count;
		result.persist_ms = elapsed_ms(started, Clock::now());
		return result;
	} catch (const std::exception& error) {
		result.persist_ms = elapsed_ms(started, Clock::now());
		result.rejected = true;
		result.rejection_reason = error.what();
		return result;
	}
}

} // namespace galp::jpeg::detail
