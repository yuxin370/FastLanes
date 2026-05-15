// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/compression/columns/freq_extended.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_COMPRESSION_COLUMNS_FREQ_EXTENDED_CUH
#define GALP_COMPRESSION_COLUMNS_FREQ_EXTENDED_CUH

#include "compression/columns/base.cuh"
#include "compression/consts.cuh"
#include "memory/device_arena.cuh"
#include "memory/gpu_array.cuh"
#include <limits>

namespace galp::codec {
namespace device {

template <typename T>
struct FREQExtendedColumn {
	using UINT_T = typename galp::codec::utils::same_width_uint<T>::type;
	size_t n_values;
	size_t n_vecs;

		T frequent_value; // per-column frequent value

		size_t    n_exceptions;       // total number of exceptions
		uint32_t* exceptions_offsets; // expection offsets in exception array
		T*        exceptions;         // exception values
		uint16_t* positions;          // exception positions in vectors
		uint16_t* offsets_counts;     // offsets and counts per lane
};

} // namespace device

namespace host {

template <typename T>
struct FREQExtendedColumn {
	using UINT_T        = typename galp::codec::utils::same_width_uint<T>::type;
	using DeviceColumnT = typename device::FREQExtendedColumn<T>;

	size_t n_values;
	size_t n_vecs;

		T frequent_value; // per-column frequent value

		size_t    n_exceptions;       // total number of exceptions
		HostArray<uint32_t> exceptions_offsets; // expection offsets in exception array
		HostArray<T>        exceptions;         // exception values
		HostArray<uint16_t> positions;          // exception positions in vectors
		HostArray<uint16_t> offsets_counts;     // offsets and counts per lane

	size_t get_n_values() const {
		return n_values;
	}
	size_t get_n_vecs() const {
		return n_vecs;
	}

	device::FREQExtendedColumn<T> copy_to_device() const {
		size_t branchless_and_prefetch_buffer = galp::codec::consts::MAX_UNPACK_N_VECS;
		return device::FREQExtendedColumn<T> {
		    n_values,
		    n_vecs,
		    frequent_value,
		    n_exceptions,
		    GPUArray<uint32_t>(n_vecs, exceptions_offsets).release(),
		    GPUArray<T>(n_exceptions, branchless_and_prefetch_buffer, exceptions).release(),
		    GPUArray<uint16_t>(n_exceptions, branchless_and_prefetch_buffer, positions).release(),
		    GPUArray<uint16_t>(n_vecs * galp::codec::utils::get_n_lanes<T>(), offsets_counts).release(),
		};
	}

	void copy_to_device(galp::memory::DeviceArena& arena, device::FREQExtendedColumn<T>& out) const {
		const size_t buf       = galp::codec::consts::MAX_UNPACK_N_VECS;
		auto         i_exc_off = arena.template add<uint32_t>(n_vecs, exceptions_offsets);
		auto         i_exc     = arena.template add<T>(n_exceptions, exceptions, buf);
		auto         i_pos     = arena.template add<uint16_t>(n_exceptions, positions, buf);
		auto         i_oc      = arena.template add<uint16_t>(n_vecs * galp::codec::utils::get_n_lanes<T>(), offsets_counts);
		out.n_values           = n_values;
		out.n_vecs             = n_vecs;
		out.frequent_value     = frequent_value;
		out.n_exceptions       = n_exceptions;
		arena.resolve_to(reinterpret_cast<void**>(&out.exceptions_offsets), i_exc_off);
		arena.resolve_to(reinterpret_cast<void**>(&out.exceptions), i_exc);
		arena.resolve_to(reinterpret_cast<void**>(&out.positions), i_pos);
		arena.resolve_to(reinterpret_cast<void**>(&out.offsets_counts), i_oc);
	}
};

template <typename T>
void free_column(FREQExtendedColumn<T>& column) {
	column.exceptions_offsets.reset();
	column.exceptions.reset();
	column.positions.reset();
	column.offsets_counts.reset();
}

template <typename T>
void free_column(device::FREQExtendedColumn<T> column) {
	free_device_pointer(column.exceptions_offsets);
	free_device_pointer(column.exceptions);
	free_device_pointer(column.positions);
	free_device_pointer(column.offsets_counts);
}

} // namespace host
} // namespace galp::codec

#endif // GALP_COMPRESSION_COLUMNS_FREQ_EXTENDED_CUH
