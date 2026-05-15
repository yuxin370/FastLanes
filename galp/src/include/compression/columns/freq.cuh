// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/compression/columns/freq.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_COMPRESSION_COLUMNS_FREQ_CUH
#define GALP_COMPRESSION_COLUMNS_FREQ_CUH

#include "compression/columns/base.cuh"
#include "compression/columns/freq_extended.cuh"
#include "compression/consts.cuh"
#include "memory/device_arena.cuh"
#include "memory/gpu_array.cuh"
#include <limits>
#include <stdexcept>

namespace galp::codec {
namespace device {

template <typename T>
struct FREQColumn {
	using UINT_T = typename galp::codec::utils::same_width_uint<T>::type;
	size_t n_values;
	size_t n_vecs;

		T frequent_value; // per-column frequent value

		size_t    n_exceptions;       // total number of exceptions
		uint32_t* exceptions_offsets; // expection offsets in exception array
		T*        exceptions;         // exception values
		uint16_t* positions;          // exception positions in vectors
		uint16_t* counts;             // number of exceptions per vector
};

} // namespace device

namespace host {

template <typename T>
struct FREQColumn {
	using UINT_T        = typename galp::codec::utils::same_width_uint<T>::type;
	using DeviceColumnT = typename device::FREQColumn<T>;

	size_t n_values;
	size_t n_vecs;

		T frequent_value; // per-column frequent value

		size_t    n_exceptions;       // total number of exceptions
		HostArray<uint32_t> exceptions_offsets; // expection offsets in exception array
		HostArray<T>        exceptions;         // exception values
		HostArray<uint16_t> positions;          // exception positions in vectors
		HostArray<uint16_t> counts;             // number of exceptions per vector

	size_t get_n_values() const {
		return n_values;
	}

	size_t get_n_vecs() const {
		return n_vecs;
	}

	device::FREQColumn<T> copy_to_device() const {
		size_t branchless_and_prefetch_buffer = galp::codec::consts::MAX_UNPACK_N_VECS;
		return device::FREQColumn<T> {
		    n_values,
		    n_vecs,
		    frequent_value,
		    n_exceptions,
		    GPUArray<uint32_t>(n_vecs, exceptions_offsets).release(),
		    GPUArray<T>(n_exceptions, branchless_and_prefetch_buffer, exceptions).release(),
		    GPUArray<uint16_t>(n_exceptions, branchless_and_prefetch_buffer, positions).release(),
		    GPUArray<uint16_t>(n_vecs, counts).release(),
		};
	}

	void copy_to_device(galp::memory::DeviceArena& arena, device::FREQColumn<T>& out) const {
		const size_t buf       = galp::codec::consts::MAX_UNPACK_N_VECS;
		auto         i_exc_off = arena.template add<uint32_t>(n_vecs, exceptions_offsets);
		auto         i_exc     = arena.template add<T>(n_exceptions, exceptions, buf);
		auto         i_pos     = arena.template add<uint16_t>(n_exceptions, positions, buf);
		auto         i_cnt     = arena.template add<uint16_t>(n_vecs, counts);
		out.n_values           = n_values;
		out.n_vecs             = n_vecs;
		out.frequent_value     = frequent_value;
		out.n_exceptions       = n_exceptions;
		arena.resolve_to(reinterpret_cast<void**>(&out.exceptions_offsets), i_exc_off);
		arena.resolve_to(reinterpret_cast<void**>(&out.exceptions), i_exc);
		arena.resolve_to(reinterpret_cast<void**>(&out.positions), i_pos);
		arena.resolve_to(reinterpret_cast<void**>(&out.counts), i_cnt);
	}

	std::tuple<T*, uint16_t*, uint16_t*> convert_exceptions_to_lane_divided_format() const {
		constexpr auto N_LANES         = galp::codec::utils::get_n_lanes<T>();
		constexpr auto VALUES_PER_LANE = galp::codec::utils::get_values_per_lane<T>();

		// New exception allocations
		T*        out_exceptions     = new T[n_exceptions];
		uint16_t* out_positions      = new uint16_t[n_exceptions];
		uint16_t* out_offsets_counts = new uint16_t[get_n_vecs() * N_LANES];

		// Intermediate arrays for reordering positions and exceptions
		T        vec_exceptions[galp::codec::consts::VALUES_PER_VECTOR];
		uint16_t vec_exceptions_positions[galp::codec::consts::VALUES_PER_VECTOR];
		uint16_t lane_counts[N_LANES];
		static_assert(galp::codec::consts::VALUES_PER_VECTOR <= std::numeric_limits<uint16_t>::max(),
		              "FREQ position storage requires uint16_t-capable vector size");

		// Copies of pointers for pointer arithmetic
		T*        c_out_exceptions     = out_exceptions;
		uint16_t* c_out_positions      = out_positions;
		uint16_t* c_out_offsets_counts = out_offsets_counts;

		for (size_t vec_index {0}; vec_index < get_n_vecs(); ++vec_index) {
			uint32_t     vec_exception_count = counts[vec_index];
			const size_t exc_base            = exceptions_offsets[vec_index];

			// Reset counts
			for (size_t j {0}; j < N_LANES; ++j) {
				lane_counts[j] = 0;
			}

			// Split all exceptions into lanes
			for (size_t exception_index {0}; exception_index < vec_exception_count; ++exception_index) {
				T        exception = exceptions[exc_base + exception_index];
				uint16_t position  = positions[exc_base + exception_index];

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
			c_out_offsets_counts += galp::codec::utils::get_n_lanes<T>();
		}

		return std::make_tuple(out_exceptions, out_positions, out_offsets_counts);
	}

	FREQExtendedColumn<T> create_extended_column() const {
		auto [e_exceptions, e_positions, e_offsets_counts] = convert_exceptions_to_lane_divided_format();
		auto*  e_offsets                                   = new uint32_t[get_n_vecs()];
		size_t acc                                         = 0;
		for (size_t i = 0; i < get_n_vecs(); ++i) {
			if (acc > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
				throw std::overflow_error("FREQ exception offset exceeds uint32_t range");
			}
			e_offsets[i] = static_cast<uint32_t>(acc);
			acc += static_cast<size_t>(counts[i]);
		}
		return FREQExtendedColumn<T> {n_values,
		                              get_n_vecs(),
		                              frequent_value,
		                              n_exceptions,
		                              e_offsets,
		                              e_exceptions,
		                              e_positions,
		                              e_offsets_counts};
	}
};

template <typename T>
void free_column(FREQColumn<T>& column) {
	column.exceptions_offsets.reset();
	column.exceptions.reset();
	column.positions.reset();
	column.counts.reset();
}

template <typename T>
void free_column(device::FREQColumn<T> column) {
	free_device_pointer(column.exceptions_offsets);
	free_device_pointer(column.exceptions);
	free_device_pointer(column.positions);
	free_device_pointer(column.counts);
}

} // namespace host
} // namespace galp::codec

#endif // GALP_COMPRESSION_COLUMNS_FREQ_CUH
