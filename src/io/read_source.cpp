// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/io/read_source.cpp
// ────────────────────────────────────────────────────────
#include "fls/io/read_source.hpp"
#include "fls/common/alias.hpp"
#include "fls/cor/lyt/buf.hpp"
#include "fls/io/external_memory.hpp"
#include "fls/io/file.hpp"
#include <cstddef>
#include <cstring>
#include <span>
#include <stdexcept>
#include <variant>

namespace fastlanes {
namespace {
void check_range(n_t offset, n_t size, n_t available) {
	if (offset > available || size > available - offset) {
		throw std::runtime_error("read range exceeds source size");
	}
}
} // namespace
ReadSource::ReadSource(File& file)
    : m_source(&file) {
}
ReadSource::ReadSource(std::span<const std::byte> bytes)
    : m_source(bytes) {
}
ReadSource::ReadSource(const ExternalMemory& memory)
    : ReadSource(std::span<const std::byte>(memory.GetSpan())) {
}

n_t ReadSource::Size() const {
	if (auto file = std::get_if<File*>(&m_source)) {
		return (*file)->Size();
	}
	return std::get<std::span<const std::byte>>(m_source).size();
}

void ReadSource::read_unchecked(void* dst, n_t offset, n_t size) const {
	if (size == 0) {
		return;
	}
	if (!dst) {
		throw std::invalid_argument("range read destination is null");
	}
	if (auto file = std::get_if<File*>(&m_source)) {
		(*file)->ReadRangeUnchecked(dst, offset, size);
	} else {
		std::memmove(dst, std::get<std::span<const std::byte>>(m_source).data() + offset, size);
	}
}

void ReadSource::ReadRange(void* dst, n_t offset, n_t size) const {
	check_range(offset, size, Size());
	read_unchecked(dst, offset, size);
}

void ReadSource::ReadRange(Buf& dst, n_t offset, n_t size) const {
	check_range(offset, size, Size());
	if (size > dst.Capacity()) {
		dst.Resize(size);
	}
	dst.Reset();
	read_unchecked(dst.mutable_data(), offset, size);
	dst.UnsafeAdvance(size);
}

void ReadSource::ReadRanges(std::span<const RangeReadTarget> targets) const {
	const auto available = Size();
	for (const auto& target : targets) {
		check_range(target.offset, target.size, available);
		if (target.scatter.empty()) {
			read_unchecked(target.data, target.offset, target.size);
			continue;
		}
		n_t remaining = target.size;
		for (const auto& part : target.scatter) {
			if (!part.data || part.size == 0 || part.size > remaining) {
				throw std::invalid_argument("invalid scatter destination");
			}
			remaining -= part.size;
		}
		if (remaining != 0) {
			throw std::invalid_argument("scatter size mismatch");
		}
		if (auto file = std::get_if<File*>(&m_source)) {
			(*file)->ReadScatterUnchecked(target.scatter, target.offset);
		} else {
			auto offset = target.offset;
			for (const auto& part : target.scatter) {
				read_unchecked(part.data, offset, part.size);
				offset += part.size;
			}
		}
	}
}
} // namespace fastlanes
