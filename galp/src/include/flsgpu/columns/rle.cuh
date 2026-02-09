// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/columns/rle.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_COLUMNS_RLE_CUH
#define FLSGPU_COLUMNS_RLE_CUH

#include "flsgpu/columns/ffor.cuh"
#include "flsgpu/columns/parse_common.cuh"
#include "flsgpu/host-utils.cuh"
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace flsgpu {
namespace device {

template <typename T, typename IndexT>
struct RLEColumn {
	size_t             n_values;
	size_t             n_vecs;
	FFORColumn<IndexT> ffor;
	IndexT*            rsum_bases;   // n_vecs * n_lanes(IndexT)
	T*                 rle_values;   // concatenated per-vector values
	size_t*            rle_offsets;  // per-vector base offset into rle_values
	size_t             n_rle_values; // total values length
};

} // namespace device

namespace host {

template <typename T, typename IndexT>
struct RLEColumn {
	using DeviceColumnT = typename device::RLEColumn<T, IndexT>;

	size_t             n_values;
	size_t             n_vecs;
	FFORColumn<IndexT> ffor;
	IndexT*            rsum_bases;
	T*                 rle_values;
	size_t*            rle_offsets;
	size_t             n_rle_values;

	size_t get_n_values() const {
		return n_values;
	}

	device::RLEColumn<T, IndexT> copy_to_device() const {
		return device::RLEColumn<T, IndexT> {
		    n_values,
		    n_vecs,
		    ffor.copy_to_device(),
		    GPUArray<IndexT>(n_vecs * utils::get_n_lanes<IndexT>(), rsum_bases).release(),
		    GPUArray<T>(n_rle_values, rle_values).release(),
		    GPUArray<size_t>(n_vecs, rle_offsets).release(),
		    n_rle_values};
	}
};

template <typename T, typename IndexT>
void free_column(RLEColumn<T, IndexT> column) {
	free_column(column.ffor);
	delete[] column.rsum_bases;
	delete[] column.rle_values;
	delete[] column.rle_offsets;
}

template <typename T, typename IndexT>
void free_column(device::RLEColumn<T, IndexT> column) {
	free_column(column.ffor);
	free_device_pointer(column.rsum_bases);
	free_device_pointer(column.rle_values);
	free_device_pointer(column.rle_offsets);
}

} // namespace host
} // namespace flsgpu

namespace reader::columns {

template <typename T, typename IndexT>
inline ParseResultT<flsgpu::host::RLEColumn<T, IndexT>> parse_rle(const ParseContext& ctx) {
	if (!ctx.operand_tokens || ctx.operand_tokens->size() < 5) {
		throw std::runtime_error("EXP_RLE: missing operand tokens");
	}

	const size_t base_idx    = ctx.operand_tokens->size() - 1;
	const auto   seg_vals    = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 4)));
	const auto   seg_rsum    = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 3)));
	const auto seg_bitpacked = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 2)));
	const auto seg_bw        = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 1)));
	const auto seg_base      = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 0)));

	auto bp_parts =
	    detail::parse_bp_segments<typename utils::same_width_uint<IndexT>::type>(seg_bitpacked, seg_bw, ctx.n_vecs);

	auto* bases_ffor = detail::copy_segment_array<typename utils::same_width_uint<IndexT>::type>(seg_base);
	flsgpu::host::BPColumn<IndexT> bp {
	    ctx.n_values, bp_parts.n_packed, bp_parts.packed, bp_parts.bit_widths, bp_parts.vector_offsets};
	flsgpu::host::FFORColumn<IndexT> ffor {bp, bases_ffor};

	const size_t expected_bases = ctx.n_vecs * utils::get_n_lanes<IndexT>();
	auto*        rsum_bases     = detail::copy_segment_array<IndexT>(seg_rsum);
	if (seg_rsum.data_span.size() / sizeof(IndexT) != expected_bases) {
		throw std::runtime_error("EXP_RLE: rsum bases size mismatch");
	}

	auto entrypoints = detail::extract_entrypoints(seg_vals);
	if (entrypoints.size() != ctx.n_vecs) {
		throw std::runtime_error("EXP_RLE: values entrypoint count mismatch");
	}
	auto  offsets_vec = detail::build_vector_offsets<T>(entrypoints);
	auto* offsets     = new size_t[ctx.n_vecs];
	for (size_t i = 0; i < ctx.n_vecs; ++i) {
		offsets[i] = offsets_vec[i];
	}

	auto*        values = detail::copy_segment_array<T>(seg_vals);
	const size_t n_vals = seg_vals.data_span.size() / sizeof(T);

	flsgpu::host::RLEColumn<T, IndexT> host {ctx.n_values, ctx.n_vecs, ffor, rsum_bases, values, offsets, n_vals};
	return ParseResultT<flsgpu::host::RLEColumn<T, IndexT>> {std::move(host)};
}

} // namespace reader::columns

#endif // FLSGPU_COLUMNS_RLE_CUH
