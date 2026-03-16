// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/columns/freq.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_COLUMNS_FREQ_CUH
#define FLSGPU_COLUMNS_FREQ_CUH

#include "flsgpu/columns/base.cuh"
#include "flsgpu/columns/freq_extended.cuh"
#include "flsgpu/columns/parse_common.cuh"
#include "flsgpu/consts.cuh"
#include "flsgpu/host-utils.cuh"
#include <limits>

namespace flsgpu {
namespace device {

template <typename T>
struct FREQColumn {
	using UINT_T = typename utils::same_width_uint<T>::type;
	size_t n_values;
	size_t n_vecs;

	T* frequent_value; // frequent values

	size_t    n_exceptions;       // total number of exceptions
	size_t*   exceptions_offsets; // expection offsets in exception array
	size_t*   positions_offsets;  // position offsets in position array
	T*        exceptions;         // exception values
	uint16_t* positions;          // exception positions in vectors
	uint16_t* counts;             // number of exceptions per vector
};

} // namespace device

namespace host {

template <typename T>
struct FREQColumn {
	using UINT_T        = typename utils::same_width_uint<T>::type;
	using DeviceColumnT = typename device::FREQColumn<T>;

	size_t n_values;
	size_t n_vecs;

	T* frequent_value; // frequent values

	size_t    n_exceptions;       // total number of exceptions
	size_t*   exceptions_offsets; // expection offsets in exception array
	size_t*   positions_offsets;  // position offsets
	T*        exceptions;         // exception values
	uint16_t* positions;          // exception positions in vectors
	uint16_t* counts;             // number of exceptions per vector

	size_t get_n_values() const {
		return n_values;
	}

	size_t get_n_vecs() const {
		return n_vecs;
	}

	device::FREQColumn<T> copy_to_device() const {
		size_t branchless_and_prefetch_buffer = consts::MAX_UNPACK_N_VECS;
		return device::FREQColumn<T> {
		    n_values,
		    n_vecs,
		    GPUArray<T>(n_vecs, frequent_value).release(),
		    n_exceptions,
		    GPUArray<size_t>(n_vecs, exceptions_offsets).release(),
		    GPUArray<size_t>(n_vecs, positions_offsets).release(),
		    GPUArray<T>(n_exceptions, branchless_and_prefetch_buffer, exceptions).release(),
		    GPUArray<uint16_t>(n_exceptions, branchless_and_prefetch_buffer, positions).release(),
		    GPUArray<uint16_t>(n_vecs, counts).release(),
		};
	}

	std::tuple<T*, uint16_t*, uint16_t*> convert_exceptions_to_lane_divided_format() const {
		constexpr auto N_LANES         = utils::get_n_lanes<T>();
		constexpr auto VALUES_PER_LANE = utils::get_values_per_lane<T>();

		// New exception allocations
		T*        out_exceptions     = reinterpret_cast<T*>(malloc(sizeof(T) * n_exceptions));
		uint16_t* out_positions      = reinterpret_cast<uint16_t*>(malloc(sizeof(uint16_t) * n_exceptions));
		uint16_t* out_offsets_counts = reinterpret_cast<uint16_t*>(malloc(sizeof(uint16_t) * get_n_vecs() * N_LANES));

		// Intermediate arrays for reordering positions and exceptions
		T        vec_exceptions[consts::VALUES_PER_VECTOR];
		uint16_t vec_exceptions_positions[consts::VALUES_PER_VECTOR];
		uint16_t lane_counts[N_LANES];
		static_assert(consts::VALUES_PER_VECTOR <= std::numeric_limits<uint16_t>::max(),
		              "FREQ position storage requires uint16_t-capable vector size");

		// Copies of pointers for pointer arithmetic
		T*        c_out_exceptions     = out_exceptions;
		uint16_t* c_out_positions      = out_positions;
		uint16_t* c_out_offsets_counts = out_offsets_counts;

		for (size_t vec_index {0}; vec_index < get_n_vecs(); ++vec_index) {
			uint32_t     vec_exception_count = counts[vec_index];
			const size_t exc_base            = exceptions_offsets[vec_index];
			const size_t pos_base            = positions_offsets[vec_index];

			// Reset counts
			for (size_t j {0}; j < N_LANES; ++j) {
				lane_counts[j] = 0;
			}

			// Split all exceptions into lanes
			for (size_t exception_index {0}; exception_index < vec_exception_count; ++exception_index) {
				T        exception = exceptions[exc_base + exception_index];
				uint16_t position  = positions[pos_base + exception_index];

				uint32_t lane                 = position % N_LANES;
				uint32_t lane_exception_count = lane_counts[lane];
				++lane_counts[lane];
				vec_exceptions[lane * VALUES_PER_LANE + lane_exception_count]           = exception;
				vec_exceptions_positions[lane * VALUES_PER_LANE + lane_exception_count] = position;
			}

			// Merge and concatenate all exceptions per lane into single contiguous array
			uint32_t vec_exceptions_counter = 0;
			for (size_t lane {0}; lane < N_LANES; ++lane) {
				uint32_t exc_in_lane_count = lane_counts[lane];
				for (size_t exc_in_lane {0}; exc_in_lane < exc_in_lane_count; ++exc_in_lane) {
					c_out_exceptions[vec_exceptions_counter] = vec_exceptions[lane * VALUES_PER_LANE + exc_in_lane];
					c_out_positions[vec_exceptions_counter] =
					    vec_exceptions_positions[lane * VALUES_PER_LANE + exc_in_lane];
					++vec_exceptions_counter;
				}

				c_out_offsets_counts[lane] = (exc_in_lane_count << 10) | (vec_exceptions_counter - exc_in_lane_count);
			}

			c_out_exceptions += vec_exception_count;
			c_out_positions += vec_exception_count;
			c_out_offsets_counts += utils::get_n_lanes<T>();
		}

		return std::make_tuple(out_exceptions, out_positions, out_offsets_counts);
	}

	FREQExtendedColumn<T> create_extended_column() const {
		auto [e_exceptions, e_positions, e_offsets_counts] = convert_exceptions_to_lane_divided_format();
		auto*  e_offsets                                   = new size_t[get_n_vecs()];
		size_t acc                                         = 0;
		for (size_t i = 0; i < get_n_vecs(); ++i) {
			e_offsets[i] = acc;
			acc += static_cast<size_t>(counts[i]);
		}
		return FREQExtendedColumn<T> {n_values,
		                              get_n_vecs(),
		                              utils::copy_array(frequent_value, get_n_vecs()),
		                              n_exceptions,
		                              e_offsets,
		                              e_exceptions,
		                              e_positions,
		                              e_offsets_counts};
	}
};

template <typename T>
void free_column(FREQColumn<T> column) {
	delete[] column.frequent_value;
	delete[] column.exceptions_offsets;
	delete[] column.positions_offsets;
	delete[] column.exceptions;
	delete[] column.positions;
	delete[] column.counts;
}

template <typename T>
void free_column(device::FREQColumn<T> column) {
	free_device_pointer(column.frequent_value);
	free_device_pointer(column.exceptions_offsets);
	free_device_pointer(column.positions_offsets);
	free_device_pointer(column.exceptions);
	free_device_pointer(column.positions);
	free_device_pointer(column.counts);
}

} // namespace host
} // namespace flsgpu

namespace reader::columns {

template <typename T>
inline ParseResultT<flsgpu::host::FREQColumn<T>> parse_frequency(const ParseContext& ctx) {
	if (!ctx.operand_tokens || ctx.operand_tokens->size() < 4) {
		throw std::runtime_error("EXP_FREQUENCY: missing operand tokens");
	}
	const size_t base_idx = ctx.operand_tokens->size() - 1;
	const auto   seg_fv   = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 3)));
	const auto   seg_exc  = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 2)));
	const auto   seg_pos  = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 1)));
	const auto   seg_cnt  = ctx.column_view.GetSegment(static_cast<uint32_t>(ctx.operand_tokens->Get(base_idx - 0)));

	if (seg_fv.data_span.size() != sizeof(T)) {
		throw std::runtime_error("EXP_FREQUENCY: invalid frequent value size");
	}
	const auto fv     = *reinterpret_cast<const T*>(seg_fv.data_span.data());
	auto*      fv_arr = new T[ctx.n_vecs];
	for (size_t i = 0; i < ctx.n_vecs; ++i) {
		fv_arr[i] = fv;
	}

	auto* counts = detail::copy_segment_array<uint16_t>(seg_cnt);
	auto  exc    = detail::build_exception_offsets_from_segment<T>(seg_exc, ctx.n_vecs);
	auto  pos    = detail::build_exception_offsets_from_segment<uint16_t>(seg_pos, ctx.n_vecs);

	auto* positions  = detail::copy_segment_array<uint16_t>(seg_pos);
	auto* exceptions = detail::copy_segment_array<T>(seg_exc);

	return ParseResultT<flsgpu::host::FREQColumn<T>> {flsgpu::host::FREQColumn<T> {
	    ctx.n_values, ctx.n_vecs, fv_arr, exc.total, exc.offsets, pos.offsets, exceptions, positions, counts}};
}

} // namespace reader::columns

#endif // FLSGPU_COLUMNS_FREQ_CUH
