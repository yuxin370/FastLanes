// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/io/io.cpp
// ────────────────────────────────────────────────────────
#include "fls/io/io.hpp"
#include "fls/common/alias.hpp"
#include "fls/common/common.hpp"
#include "fls/cor/lyt/buf.hpp"
#include "fls/io/external_memory.hpp"
#include "fls/io/file.hpp"
#include "fls/io/read_source.hpp"
#include "fls/std/span.hpp"
#include "fls/std/variant.hpp"
#include <stdexcept>
#include <variant>

namespace fastlanes {

void IO::flush(io& io, const Buf& buf) {
	// write the buffer
	visit(overloaded {
	          [&](up<File>& file) { file->Write(buf); },
	          [](auto&) { FLS_UNREACHABLE() },
	      },
	      io);
}

void IO::append(io& io, const Buf& buf) {
	// write the buffer
	visit(overloaded {
	          [&](up<File>& file) { file->Append(buf); },
	          [](auto&) { FLS_UNREACHABLE() },
	      },
	      io);
}

void IO::append(io& io, const char* pointer, n_t size) {
	// write the buffer
	visit(overloaded {
	          [&](up<File>& file) { file->Append(pointer, size); },
	          [](auto&) { FLS_UNREACHABLE() },
	      },
	      io);
}

namespace {
ReadSource read_source(const io& input) {
	if (const auto* file = std::get_if<up<File>>(&input); file && *file) {
		return {**file};
	}
	if (const auto* memory = std::get_if<up<ExternalMemory>>(&input); memory && *memory) {
		return ReadSource(**memory);
	}
	throw std::invalid_argument("read requires a file or external memory source");
}
} // namespace

void IO::read(const io& input, Buf& buf) {
	const auto source = read_source(input);
	source.ReadRange(buf, 0, source.Size());
}

void IO::range_read(const io& input, Buf& buf, n_t offset, n_t size) {
	read_source(input).ReadRange(buf, offset, size);
}

void IO::range_read(const io& input, void* dst, n_t offset, n_t size) {
	read_source(input).ReadRange(dst, offset, size);
}

n_t IO::get_size(const io& input) {
	return read_source(input).Size();
}

template <typename PT>
span<PT> IO::read(const io& io) {
	span<PT> span_result;
	visit(overloaded {
	          [&](const up<ExternalMemory>& memory) {
		          auto memory_span = memory->GetSpan();
		          span_result = span<PT> {reinterpret_cast<PT*>(memory_span.data()), memory_span.size() / sizeof(PT)};
	          },
	          [](auto&) { FLS_UNREACHABLE() },
	      },
	      io);
	return span_result;
}

template span<i64_pt>    IO::read(const io& io);
template span<dbl_pt>    IO::read(const io& io);
template span<std::byte> IO::read(const io& io);

} // namespace fastlanes
