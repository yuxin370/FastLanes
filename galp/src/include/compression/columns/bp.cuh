// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/compression/columns/bp.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_COMPRESSION_COLUMNS_BP_CUH
#define GALP_COMPRESSION_COLUMNS_BP_CUH

#include "compression/columns/base.cuh"
#include "memory/device_arena.cuh"
#include "memory/gpu_array.cuh"
#include <cstddef>
#include <cstdint>

namespace galp::codec {
namespace device {

template <typename T>
struct BPColumn {
	using UINT_T = typename galp::codec::utils::same_width_uint<T>::type;
	size_t n_values;
	size_t n_vecs;

	UINT_T*   packed_array;
	vbw_t*    bit_widths;
	uint32_t* vector_offsets;
};

} // namespace device

namespace host {

template <typename T>
struct BPColumn {
	using UINT_T        = typename galp::codec::utils::same_width_uint<T>::type;
	using DeviceColumnT = typename device::BPColumn<T>;

	size_t n_values;
	size_t n_packed_values;

	size_t get_n_values() const {
		return n_values;
	}
	size_t get_n_vecs() const {
		return galp::codec::utils::get_n_vecs_from_size(n_values);
	}

	HostArray<UINT_T>   packed_array;
	HostArray<vbw_t>    bit_widths;
	HostArray<uint32_t> vector_offsets;

	device::BPColumn<T> copy_to_device() const {
		const size_t branchless_extra_access_buffer = sizeof(T) * galp::codec::utils::get_n_lanes<T>() * 4;
		return device::BPColumn<T> {
		    n_values,
		    get_n_vecs(),
		    GPUArray<UINT_T>(n_packed_values, branchless_extra_access_buffer, packed_array).release(),
		    GPUArray<vbw_t>(get_n_vecs(), bit_widths).release(),
		    GPUArray<uint32_t>(get_n_vecs(), vector_offsets).release()};
	}

	void copy_to_device(galp::memory::DeviceArena& arena, device::BPColumn<T>& out) const {
		const size_t buffer_elems = galp::codec::utils::get_n_lanes<T>() * 4;
		auto         i_packed     = arena.template add<UINT_T>(n_packed_values, packed_array, buffer_elems);
		auto         i_bw         = arena.template add<vbw_t>(get_n_vecs(), bit_widths);
		auto         i_offsets    = arena.template add<uint32_t>(get_n_vecs(), vector_offsets);
		out.n_values              = n_values;
		out.n_vecs                = get_n_vecs();
		arena.resolve_to(reinterpret_cast<void**>(&out.packed_array), i_packed);
		arena.resolve_to(reinterpret_cast<void**>(&out.bit_widths), i_bw);
		arena.resolve_to(reinterpret_cast<void**>(&out.vector_offsets), i_offsets);
	}
};

template <typename T>
void free_column(BPColumn<T>& column) {
	column.packed_array.reset();
	column.bit_widths.reset();
	column.vector_offsets.reset();
}

template <typename T>
void free_column(device::BPColumn<T> column) {
	free_device_pointer(column.packed_array);
	free_device_pointer(column.bit_widths);
	free_device_pointer(column.vector_offsets);
}

} // namespace host
} // namespace galp::codec

#endif // GALP_COMPRESSION_COLUMNS_BP_CUH
