// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/columns/ffor.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_COLUMNS_FFOR_CUH
#define FLSGPU_COLUMNS_FFOR_CUH

#include "flsgpu/columns/bp.cuh"
#include "flsgpu/columns/parse_common.cuh"

namespace flsgpu {
namespace device {

template <typename T>
struct FFORColumn {
	using UINT_T = typename utils::same_width_uint<T>::type;
	size_t      n_values;
	BPColumn<T> bp;
	UINT_T*     bases;
};

} // namespace device

namespace host {

template <typename T>
struct FFORColumn {
	using UINT_T        = typename utils::same_width_uint<T>::type;
	using DeviceColumnT = typename device::FFORColumn<T>;

	BPColumn<T> bp;
	UINT_T*     bases;

	size_t get_n_values() const {
		return bp.n_values;
	}
	size_t get_n_vecs() const {
		return bp.get_n_vecs();
	}

	device::FFORColumn<T> copy_to_device() const {
		return device::FFORColumn<T> {
		    get_n_values(), bp.copy_to_device(), GPUArray<UINT_T>(bp.get_n_vecs(), bases).release()};
	}

	device::FFORColumn<T> copy_to_device(cudaStream_t stream) const {
		if (stream == nullptr) {
			return copy_to_device();
		}
		return device::FFORColumn<T> {
		    get_n_values(), bp.copy_to_device(stream), GPUArray<UINT_T>(bp.get_n_vecs(), bases, stream).release()};
	}
};

template <typename T>
void free_column(FFORColumn<T> column) {
	free_column(column.bp);
	delete[] column.bases;
}

template <typename T>
void free_column(device::FFORColumn<T> column) {
	free_column(column.bp);
	free_device_pointer(column.bases);
}

} // namespace host
} // namespace flsgpu

namespace reader::columns {

template <typename T>
inline ParseResultT<flsgpu::host::FFORColumn<T>> parse_ffor(const ParseContext& ctx) {
	if (!ctx.operand_tokens || ctx.operand_tokens->size() < 3) {
		throw std::runtime_error("EXP_FFOR: missing operand tokens");
	}
	const size_t base_idx    = ctx.operand_tokens->size() - 1;
	const auto seg_bitpacked = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 2)));
	const auto seg_bw        = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 1)));
	const auto seg_base      = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 0)));

	auto bp_parts =
	    detail::parse_bp_segments<typename utils::same_width_uint<T>::type>(seg_bitpacked, seg_bw, ctx.n_vecs);
	auto* bases = detail::copy_segment_array<typename utils::same_width_uint<T>::type>(seg_base);

	flsgpu::host::BPColumn<T> bp {
	    ctx.n_values, bp_parts.n_packed, bp_parts.packed, bp_parts.bit_widths, bp_parts.vector_offsets};

	return ParseResultT<flsgpu::host::FFORColumn<T>> {flsgpu::host::FFORColumn<T> {bp, bases}};
}

} // namespace reader::columns

#endif // FLSGPU_COLUMNS_FFOR_CUH
