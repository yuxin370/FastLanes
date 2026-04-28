// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/columns/freq_extended.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_COLUMNS_FREQ_EXTENDED_CUH
#define FLSGPU_COLUMNS_FREQ_EXTENDED_CUH

#include "flsgpu/columns/base.cuh"
#include "flsgpu/consts.cuh"
#include "flsgpu/memory/device_arena.cuh"
#include "flsgpu/memory/gpu_array.cuh"
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
		const size_t buf       = consts::MAX_UNPACK_N_VECS;
		auto         i_fv      = arena.template add<T>(n_vecs, frequent_value);
		auto         i_exc_off = arena.template add<size_t>(n_vecs, exceptions_offsets);
		auto         i_exc     = arena.template add<T>(n_exceptions, exceptions, buf);
		auto         i_pos     = arena.template add<uint16_t>(n_exceptions, positions, buf);
		auto         i_oc      = arena.template add<uint16_t>(n_vecs * utils::get_n_lanes<T>(), offsets_counts);
		out.n_values           = n_values;
		out.n_vecs             = n_vecs;
		out.n_exceptions       = n_exceptions;
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

#endif // FLSGPU_COLUMNS_FREQ_EXTENDED_CUH
