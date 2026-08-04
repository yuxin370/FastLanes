// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/include/fls/expression/fsst_dictionary_input.hpp
// ────────────────────────────────────────────────────────
#ifndef FLS_EXPRESSION_FSST_DICTIONARY_INPUT_HPP
#define FLS_EXPRESSION_FSST_DICTIONARY_INPUT_HPP

#include "fls/common/alias.hpp"
#include "fls/common/string.hpp"
#include "fls/cor/lyt/buf.hpp"
#include <limits>
#include <stdexcept>

namespace fastlanes {

/**
 * Owns the three arrays consumed by the FSST dictionary encoders.
 *
 * Dictionary bytes are growable, so pointers into that storage must not be
 * published while strings are still being appended.  This builder enforces a
 * two-phase lifecycle: append stable value data first, then derive every
 * pointer from the final byte-buffer base in one pass.
 */
class FsstDictionaryInput {
public:
	FsstDictionaryInput() = default;

	explicit FsstDictionaryInput(const n_t initial_capacity)
	    : m_length_buf(initial_capacity)
	    , m_bytes_buf(initial_capacity)
	    , m_string_p_buf(initial_capacity) {
	}

	FsstDictionaryInput(const FsstDictionaryInput&)            = delete;
	FsstDictionaryInput& operator=(const FsstDictionaryInput&) = delete;

	void Append(const fls_string_t& value) {
		if (m_finalized) {
			throw std::logic_error("cannot append to a finalized FSST dictionary input");
		}
		if (m_count == std::numeric_limits<n_t>::max()) {
			throw std::overflow_error("FSST dictionary value count overflow");
		}
		m_bytes_buf.Append(value.p, value.length);
		m_length_buf.Append(&value.length, sizeof(len_t));
		m_count++;
	}

	void FinalizePointers() {
		if (m_finalized) {
			return;
		}
		if (m_count > std::numeric_limits<n_t>::max() / sizeof(uint8_t*)) {
			throw std::overflow_error("FSST dictionary pointer table size overflow");
		}

		m_string_p_buf.Reset();
		auto** pointers = m_string_p_buf.GetFixedSizeArray<uint8_t*>(m_count * sizeof(uint8_t*));
		auto*  lengths  = m_length_buf.mutable_data<len_t>();
		n_t    byte_offset {0};
		for (n_t index {0}; index < m_count; index++) {
			const auto length = static_cast<n_t>(lengths[index]);
			if (byte_offset > m_bytes_buf.Size() || length > m_bytes_buf.Size() - byte_offset) {
				throw std::logic_error("FSST dictionary lengths exceed materialized bytes");
			}
			pointers[index] = m_bytes_buf.data() + byte_offset;
			byte_offset += length;
		}
		if (byte_offset != m_bytes_buf.Size()) {
			throw std::logic_error("FSST dictionary lengths do not cover materialized bytes");
		}
		m_finalized = true;
	}

	[[nodiscard]] n_t Count() const {
		return m_count;
	}

	[[nodiscard]] n_t EncodedCapacityUpperBound() const {
		if (m_bytes_buf.Size() > (std::numeric_limits<n_t>::max() - 8) / 2) {
			throw std::overflow_error("FSST encoded output size overflow");
		}
		return m_bytes_buf.Size() * 2 + 8;
	}

	[[nodiscard]] n_t OffsetCapacityBytes() const {
		if (m_count == std::numeric_limits<n_t>::max() ||
		    m_count + 1 > std::numeric_limits<n_t>::max() / sizeof(ofs_t)) {
			throw std::overflow_error("FSST output offset table size overflow");
		}
		return (m_count + 1) * sizeof(ofs_t);
	}

	len_t* Lengths() {
		RequireFinalized();
		return m_length_buf.mutable_data<len_t>();
	}

	uint8_t** Strings() {
		RequireFinalized();
		return m_string_p_buf.mutable_data<uint8_t*>();
	}

	[[nodiscard]] const Buf& Bytes() const {
		return m_bytes_buf;
	}

private:
	void RequireFinalized() const {
		if (!m_finalized) {
			throw std::logic_error("FSST dictionary pointers were requested before finalization");
		}
	}

private:
	Buf  m_length_buf;
	Buf  m_bytes_buf;
	Buf  m_string_p_buf;
	n_t  m_count {0};
	bool m_finalized {false};
};

} // namespace fastlanes

#endif // FLS_EXPRESSION_FSST_DICTIONARY_INPUT_HPP
