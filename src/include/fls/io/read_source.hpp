// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/include/fls/io/read_source.hpp
// ────────────────────────────────────────────────────────
#pragma once
#include "fls/api/api.hpp"
#include "fls/common/alias.hpp"
#include <cstddef>
#include <span>
#include <variant>

namespace fastlanes {
class Buf;
class File;
class ExternalMemory;
struct ScatterReadTarget {
	void* data = nullptr;
	n_t   size = 0;
};

// One destination/file-offset pair for a discontiguous batched read. Unlike
// ScatterReadTarget, adjacent entries do not imply contiguous source
// bytes.
struct RangeReadTarget {
	void*                              data   = nullptr;
	n_t                                offset = 0;
	n_t                                size   = 0;
	std::span<const ScatterReadTarget> scatter {};
};

// A borrowed file handle or immutable byte span. Keep its backing storage alive
// while a TableReader uses it. All offsets and lengths are measured in bytes.
class FLS_API ReadSource {
public:
	ReadSource(File& file);
	ReadSource(std::span<const std::byte> bytes);
	explicit ReadSource(const ExternalMemory& memory);
	[[nodiscard]] n_t Size() const;
	void              ReadRange(void* dst, n_t offset, n_t size) const;
	void              ReadRange(Buf& dst, n_t offset, n_t size) const;
	void              ReadRanges(std::span<const RangeReadTarget> targets) const;

private:
	void                                            read_unchecked(void* dst, n_t offset, n_t size) const;
	std::variant<File*, std::span<const std::byte>> m_source;
};
} // namespace fastlanes
