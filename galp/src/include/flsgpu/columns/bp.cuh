// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/columns/bp.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_COLUMNS_BP_CUH
#define FLSGPU_COLUMNS_BP_CUH

#include "flsgpu/columns/base.cuh"
#include "flsgpu/memory/device_arena.cuh"
#include "flsgpu/memory/gpu_array.cuh"
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
	uint32_t* vector_offsets;
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
	uint32_t* vector_offsets;

	device::BPColumn<T> copy_to_device() const {
		const size_t branchless_extra_access_buffer = sizeof(T) * utils::get_n_lanes<T>() * 4;
		return device::BPColumn<T> {
		    n_values,
		    get_n_vecs(),
		    GPUArray<UINT_T>(n_packed_values, branchless_extra_access_buffer, packed_array).release(),
		    GPUArray<vbw_t>(get_n_vecs(), bit_widths).release(),
		    GPUArray<uint32_t>(get_n_vecs(), vector_offsets).release()};
	}

	void copy_to_device(flsgpu::memory::DeviceArena& arena, device::BPColumn<T>& out) const {
		const size_t buffer_elems = utils::get_n_lanes<T>() * 4;
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

#endif // FLSGPU_COLUMNS_BP_CUH
