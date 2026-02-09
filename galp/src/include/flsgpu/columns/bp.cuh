// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/columns/bp.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_COLUMNS_BP_CUH
#define FLSGPU_COLUMNS_BP_CUH

#include "flsgpu/columns/base.cuh"
#include "flsgpu/columns/parse_common.cuh"
#include "flsgpu/host-utils.cuh"
#include <cstddef>
#include <cstdint>

namespace flsgpu {
namespace device {

template <typename T>
struct BPColumn {
	using UINT_T = typename utils::same_width_uint<T>::type;
	size_t n_values;
	size_t n_vecs;

	UINT_T* packed_array;
	vbw_t*  bit_widths;
	size_t* vector_offsets;
};

} // namespace device

namespace host {

template <typename T>
struct BPColumn {
	using UINT_T        = typename utils::same_width_uint<T>::type;
	using DeviceColumnT = typename device::BPColumn<T>;

	size_t n_values;
	size_t n_packed_values;

	size_t get_n_values() const {
		return n_values;
	}
	size_t get_n_vecs() const {
		return utils::get_n_vecs_from_size(n_values);
	}

	UINT_T* packed_array;
	vbw_t*  bit_widths;
	size_t* vector_offsets;

	device::BPColumn<T> copy_to_device() const {
		const size_t branchless_extra_access_buffer = sizeof(T) * utils::get_n_lanes<T>() * 4;
		return device::BPColumn<T> {
		    n_values,
		    get_n_vecs(),
		    GPUArray<UINT_T>(n_packed_values, branchless_extra_access_buffer, packed_array).release(),
		    GPUArray<vbw_t>(get_n_vecs(), bit_widths).release(),
		    GPUArray<size_t>(get_n_vecs(), vector_offsets).release()};
	}
};

template <typename T>
void free_column(BPColumn<T> column) {
	delete[] column.packed_array;
	delete[] column.bit_widths;
	delete[] column.vector_offsets;
}

template <typename T>
void free_column(device::BPColumn<T> column) {
	free_device_pointer(column.packed_array);
	free_device_pointer(column.bit_widths);
	free_device_pointer(column.vector_offsets);
}

} // namespace host
} // namespace flsgpu

namespace reader::columns {
namespace detail {

template <typename T>
inline flsgpu::host::BPColumn<T> make_bp_from_raw(const T* raw, const size_t n_values) {
	using UINT_T                 = typename utils::same_width_uint<T>::type;
	const size_t n_vecs          = utils::get_n_vecs_from_size(n_values);
	const size_t n_packed_values = n_vecs * kVecSize;
	const vbw_t  bw              = static_cast<vbw_t>(sizeof(T) * 8);
	UINT_T*      packed_array    = new UINT_T[n_packed_values];
	vbw_t*       bit_widths      = new vbw_t[n_vecs];
	size_t*      vector_offsets  = new size_t[n_vecs];

	std::memset(packed_array, 0, n_packed_values * sizeof(UINT_T));
	std::memcpy(packed_array, raw, std::min(n_values, n_packed_values) * sizeof(UINT_T));

	for (size_t vi = 0; vi < n_vecs; ++vi) {
		bit_widths[vi]     = bw;
		vector_offsets[vi] = vi * kVecSize;
	}

	return flsgpu::host::BPColumn<T> {n_values, n_packed_values, packed_array, bit_widths, vector_offsets};
}

} // namespace detail

template <typename T>
inline ParseResultT<flsgpu::host::BPColumn<T>> parse_uncompressed(const ParseContext& ctx) {
	if (!ctx.operand_tokens || ctx.operand_tokens->size() < 1) {
		throw std::runtime_error("EXP_UNCOMPRESSED: missing operand tokens");
	}
	const auto seg_idx = static_cast<uint32_t>(ctx.operand_tokens->Get(ctx.operand_tokens->size() - 1));
	auto       seg     = ctx.column_view.GetSegment(seg_idx);
	auto*      raw     = detail::copy_segment_array<T>(seg);
	auto       host    = detail::make_bp_from_raw<T>(raw, ctx.n_values);
	delete[] raw;
	return ParseResultT<flsgpu::host::BPColumn<T>> {std::move(host)};
}

} // namespace reader::columns

#endif // FLSGPU_COLUMNS_BP_CUH
