// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/columns/slpatch.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_COLUMNS_SLPATCH_CUH
#define FLSGPU_COLUMNS_SLPATCH_CUH

#include "flsgpu/columns/ffor.cuh"
#include "flsgpu/columns/parse_common.cuh"
#include "flsgpu/consts.cuh"

namespace flsgpu {
namespace device {

template <typename T>
struct SLPATCHColumn {
	using UINT_T = typename utils::same_width_uint<T>::type;
	size_t n_values;
	size_t n_vecs;

	FFORColumn<T> ffor; // base values

	size_t    n_exceptions;       // total number of exceptions
	size_t*   exceptions_offsets; // exception offsets in exception array
	T*        exceptions;         // exception values
	uint16_t* positions;          // exception positions in vectors
	uint16_t* counts;             // number of exceptions per vector
};

} // namespace device

namespace host {

template <typename T>
struct SLPATCHColumn {
	using UINT_T        = typename utils::same_width_uint<T>::type;
	using DeviceColumnT = typename device::SLPATCHColumn<T>;

	size_t n_values;
	size_t n_vecs;

	FFORColumn<T> ffor; // base values

	size_t    n_exceptions;       // total number of exceptions
	size_t*   exceptions_offsets; // exception values
	T*        exceptions;         // exception values
	uint16_t* positions;          // exception positions in vectors
	uint16_t* counts;             // number of exceptions per vector

	size_t get_n_values() const {
		return n_values;
	}

	size_t get_n_vecs() const {
		return n_vecs;
	}

	device::SLPATCHColumn<T> copy_to_device() const {
		size_t branchless_and_prefetch_buffer = consts::MAX_UNPACK_N_VECS;
		return device::SLPATCHColumn<T> {
		    n_values,
		    n_vecs,
		    ffor.copy_to_device(),
		    n_exceptions,
		    GPUArray<size_t>(n_vecs, exceptions_offsets).release(),
		    GPUArray<T>(n_exceptions, branchless_and_prefetch_buffer, exceptions).release(),
		    GPUArray<uint16_t>(n_exceptions, branchless_and_prefetch_buffer, positions).release(),
		    GPUArray<uint16_t>(n_vecs, counts).release(),
		};
	}
};

template <typename T>
void free_column(SLPATCHColumn<T> column) {
	free_column(column.ffor);
	delete[] column.exceptions_offsets;
	delete[] column.exceptions;
	delete[] column.positions;
	delete[] column.counts;
}

template <typename T>
void free_column(device::SLPATCHColumn<T> column) {
	free_column(column.ffor);
	free_device_pointer(column.exceptions_offsets);
	free_device_pointer(column.exceptions);
	free_device_pointer(column.positions);
	free_device_pointer(column.counts);
}

} // namespace host
} // namespace flsgpu

namespace reader::columns {

template <typename T>
inline ParseResultT<flsgpu::host::SLPATCHColumn<T>> parse_slpatch(const ParseContext& ctx) {
	if (!ctx.operand_tokens || ctx.operand_tokens->size() < 6) {
		throw std::runtime_error("EXP_FFOR_SLPATCH: missing operand tokens");
	}

	const size_t base_idx    = ctx.operand_tokens->size() - 1;
	const auto seg_bitpacked = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 2)));
	const auto seg_bw        = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 1)));
	const auto seg_base      = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 0)));

	const auto seg_exc = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 5)));
	const auto seg_pos = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 4)));
	const auto seg_cnt = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 3)));

	auto bp_parts =
	    detail::parse_bp_segments<typename utils::same_width_uint<T>::type>(seg_bitpacked, seg_bw, ctx.n_vecs);
	auto* bases = detail::copy_segment_array<typename utils::same_width_uint<T>::type>(seg_base);

	flsgpu::host::BPColumn<T> bp {
	    ctx.n_values, bp_parts.n_packed, bp_parts.packed, bp_parts.bit_widths, bp_parts.vector_offsets};
	flsgpu::host::FFORColumn<T> ffor {bp, bases};

	auto* counts     = detail::copy_segment_array<uint16_t>(seg_cnt);
	auto* positions  = detail::copy_segment_array<uint16_t>(seg_pos);
	auto* exceptions = detail::copy_segment_array<T>(seg_exc);

	auto exc = detail::build_exception_offsets(counts, ctx.n_vecs);

	return ParseResultT<flsgpu::host::SLPATCHColumn<T>> {flsgpu::host::SLPATCHColumn<T> {
	    ctx.n_values, ctx.n_vecs, ffor, exc.total, exc.offsets, exceptions, positions, counts}};
}

} // namespace reader::columns

#endif // FLSGPU_COLUMNS_SLPATCH_CUH
