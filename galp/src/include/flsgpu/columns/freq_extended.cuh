// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/columns/freq_extended.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_COLUMNS_FREQ_EXTENDED_CUH
#define FLSGPU_COLUMNS_FREQ_EXTENDED_CUH

#include "flsgpu/columns/base.cuh"
#include "flsgpu/columns/parse_common.cuh"
#include "flsgpu/consts.cuh"
#include "flsgpu/host-utils.cuh"
#include <limits>

namespace flsgpu {
namespace device {

template <typename T>
struct FREQExtendedColumn {
	using UINT_T = typename utils::same_width_uint<T>::type;
	size_t n_values;
	size_t n_vecs;

	T* frequent_value; // frequent values

	size_t    n_exceptions;       // total number of exceptions
	size_t*   exceptions_offsets; // expection offsets in exception array
	T*        exceptions;         // exception values
	uint16_t* positions;          // exception positions in vectors
	uint16_t* offsets_counts;     // offsets and counts per lane
};

} // namespace device

namespace host {

template <typename T>
struct FREQExtendedColumn {
	using UINT_T        = typename utils::same_width_uint<T>::type;
	using DeviceColumnT = typename device::FREQExtendedColumn<T>;

	size_t n_values;
	size_t n_vecs;

	T* frequent_value; // frequent values

	size_t    n_exceptions;       // total number of exceptions
	size_t*   exceptions_offsets; // expection offsets in exception array
	T*        exceptions;         // exception values
	uint16_t* positions;          // exception positions in vectors
	uint16_t* offsets_counts;     // offsets and counts per lane

	size_t get_n_values() const {
		return n_values;
	}
	size_t get_n_vecs() const {
		return n_vecs;
	}

	device::FREQExtendedColumn<T> copy_to_device() const {
		size_t branchless_and_prefetch_buffer = consts::MAX_UNPACK_N_VECS;
		return device::FREQExtendedColumn<T> {
		    n_values,
		    n_vecs,
		    GPUArray<T>(n_vecs, frequent_value).release(),
		    n_exceptions,
		    GPUArray<size_t>(n_vecs, exceptions_offsets).release(),
		    GPUArray<T>(n_exceptions, branchless_and_prefetch_buffer, exceptions).release(),
		    GPUArray<uint16_t>(n_exceptions, branchless_and_prefetch_buffer, positions).release(),
		    GPUArray<uint16_t>(n_vecs * utils::get_n_lanes<T>(), offsets_counts).release(),
		};
	}

	void copy_to_device(flsgpu::memory::DeviceArena& arena, device::FREQExtendedColumn<T>& out) const {
		const size_t buf = consts::MAX_UNPACK_N_VECS;
		auto i_fv      = arena.template add<T>(n_vecs, frequent_value);
		auto i_exc_off = arena.template add<size_t>(n_vecs, exceptions_offsets);
		auto i_exc     = arena.template add<T>(n_exceptions, exceptions, buf);
		auto i_pos     = arena.template add<uint16_t>(n_exceptions, positions, buf);
		auto i_oc      = arena.template add<uint16_t>(n_vecs * utils::get_n_lanes<T>(), offsets_counts);
		out.n_values     = n_values;
		out.n_vecs       = n_vecs;
		out.n_exceptions = n_exceptions;
		arena.resolve_to(reinterpret_cast<void**>(&out.frequent_value), i_fv);
		arena.resolve_to(reinterpret_cast<void**>(&out.exceptions_offsets), i_exc_off);
		arena.resolve_to(reinterpret_cast<void**>(&out.exceptions), i_exc);
		arena.resolve_to(reinterpret_cast<void**>(&out.positions), i_pos);
		arena.resolve_to(reinterpret_cast<void**>(&out.offsets_counts), i_oc);
	}
};

template <typename T>
void free_column(FREQExtendedColumn<T> column) {
	delete[] column.frequent_value;
	delete[] column.exceptions_offsets;
	delete[] column.exceptions;
	delete[] column.positions;
	delete[] column.offsets_counts;
}

template <typename T>
void free_column(device::FREQExtendedColumn<T> column) {
	free_device_pointer(column.frequent_value);
	free_device_pointer(column.exceptions_offsets);
	free_device_pointer(column.exceptions);
	free_device_pointer(column.positions);
	free_device_pointer(column.offsets_counts);
}

} // namespace host
} // namespace flsgpu

namespace reader::columns {

template <typename T>
inline ParseResultT<flsgpu::host::FREQExtendedColumn<T>> parse_frequency_extended(const ParseContext& ctx) {
	if (!ctx.operand_tokens || ctx.operand_tokens->size() < 4) {
		throw std::runtime_error("EXP_FREQUENCY_EXTENDED: missing operand tokens");
	}
	const size_t base_idx = ctx.operand_tokens->size() - 1;
	const auto   seg_fv   = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 3)));
	const auto   seg_exc  = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 2)));
	const auto   seg_pos  = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 1)));
	const auto   seg_cnt  = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 0)));

	if (seg_fv.data_span.size() != sizeof(T)) {
		throw std::runtime_error("EXP_FREQUENCY_EXTENDED: invalid frequent value size");
	}
	const auto fv     = *reinterpret_cast<const T*>(seg_fv.data_span.data());
	auto*      fv_arr = new T[ctx.n_vecs];
	for (size_t i = 0; i < ctx.n_vecs; ++i) {
		fv_arr[i] = fv;
	}

	auto* counts     = detail::copy_segment_array<uint16_t>(seg_cnt);
	auto* positions  = detail::copy_segment_array<uint16_t>(seg_pos);
	auto* exceptions = detail::copy_segment_array<T>(seg_exc);

	auto exc = detail::build_exception_offsets(counts, ctx.n_vecs);

	constexpr auto N_LANES         = utils::get_n_lanes<T>();
	constexpr auto VALUES_PER_LANE = utils::get_values_per_lane<T>();
	constexpr auto VEC_VALUES      = consts::VALUES_PER_VECTOR;

	auto* out_exceptions     = (exc.total ? new T[exc.total] : nullptr);
	auto* out_positions      = (exc.total ? new uint16_t[exc.total] : nullptr);
	auto* out_offsets_counts = new uint16_t[ctx.n_vecs * N_LANES];

	T        vec_exceptions[VEC_VALUES];
	uint16_t vec_exceptions_positions[VEC_VALUES];
	uint16_t lane_counts[N_LANES];
	static_assert(consts::VALUES_PER_VECTOR <= std::numeric_limits<uint16_t>::max(),
	              "FREQ position storage requires uint16_t-capable vector size");

	T*        c_exceptions         = exceptions;
	uint16_t* c_positions          = positions;
	T*        c_out_exceptions     = out_exceptions;
	uint16_t* c_out_positions      = out_positions;
	uint16_t* c_out_offsets_counts = out_offsets_counts;

	for (size_t vec_index = 0; vec_index < ctx.n_vecs; ++vec_index) {
		uint32_t vec_exception_count = counts[vec_index];

		for (size_t j = 0; j < N_LANES; ++j) {
			lane_counts[j] = 0;
		}

		for (size_t exception_index = 0; exception_index < vec_exception_count; ++exception_index) {
			T        exception = c_exceptions[exception_index];
			uint16_t position  = c_positions[exception_index];

			uint32_t lane                 = position % N_LANES;
			uint32_t lane_exception_count = lane_counts[lane];
			++lane_counts[lane];
			vec_exceptions[lane * VALUES_PER_LANE + lane_exception_count]           = exception;
			vec_exceptions_positions[lane * VALUES_PER_LANE + lane_exception_count] = position;
		}

		uint32_t vec_exceptions_counter = 0;
		for (size_t lane = 0; lane < N_LANES; ++lane) {
			uint32_t exc_in_lane_count = lane_counts[lane];
			for (size_t exc_in_lane = 0; exc_in_lane < exc_in_lane_count; ++exc_in_lane) {
				c_out_exceptions[vec_exceptions_counter] = vec_exceptions[lane * VALUES_PER_LANE + exc_in_lane];
				c_out_positions[vec_exceptions_counter] =
				    vec_exceptions_positions[lane * VALUES_PER_LANE + exc_in_lane];
				++vec_exceptions_counter;
			}

			c_out_offsets_counts[lane] = (exc_in_lane_count << 10) | (vec_exceptions_counter - exc_in_lane_count);
		}

		c_exceptions += vec_exception_count;
		c_positions += vec_exception_count;
		c_out_exceptions += vec_exception_count;
		c_out_positions += vec_exception_count;
		c_out_offsets_counts += N_LANES;
	}

	delete[] counts;
	delete[] positions;
	delete[] exceptions;

	return ParseResultT<flsgpu::host::FREQExtendedColumn<T>> {flsgpu::host::FREQExtendedColumn<T> {
	    ctx.n_values, ctx.n_vecs, fv_arr, exc.total, exc.offsets, out_exceptions, out_positions, out_offsets_counts}};
}

} // namespace reader::columns

#endif // FLSGPU_COLUMNS_FREQ_EXTENDED_CUH
