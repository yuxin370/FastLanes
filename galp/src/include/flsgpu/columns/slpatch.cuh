// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/columns/slpatch.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_COLUMNS_SLPATCH_CUH
#define FLSGPU_COLUMNS_SLPATCH_CUH

#include "flsgpu/columns/ffor.cuh"
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
	size_t*   positions_offsets;  // position offsets in position array
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

	device::SLPATCHColumn<T> copy_to_device() const {
		size_t branchless_and_prefetch_buffer = consts::MAX_UNPACK_N_VECS;
		return device::SLPATCHColumn<T> {
		    n_values,
		    n_vecs,
		    ffor.copy_to_device(),
		    n_exceptions,
		    GPUArray<size_t>(n_vecs, exceptions_offsets).release(),
		    GPUArray<size_t>(n_vecs, positions_offsets).release(),
		    GPUArray<T>(n_exceptions, branchless_and_prefetch_buffer, exceptions).release(),
		    GPUArray<uint16_t>(n_exceptions, branchless_and_prefetch_buffer, positions).release(),
		    GPUArray<uint16_t>(n_vecs, counts).release(),
		};
	}

	void copy_to_device(flsgpu::memory::DeviceArena& arena, device::SLPATCHColumn<T>& out) const {
		const size_t buf             = consts::MAX_UNPACK_N_VECS;
		const size_t bp_buffer_elems = utils::get_n_lanes<T>() * 4;
		using UINT_T_BP              = typename utils::same_width_uint<T>::type;
		auto i_packed  = arena.template add<UINT_T_BP>(ffor.bp.n_packed_values, ffor.bp.packed_array, bp_buffer_elems);
		auto i_bw      = arena.template add<vbw_t>(ffor.bp.get_n_vecs(), ffor.bp.bit_widths);
		auto i_bp_off  = arena.template add<size_t>(ffor.bp.get_n_vecs(), ffor.bp.vector_offsets);
		auto i_bases   = arena.template add<UINT_T_BP>(ffor.bp.get_n_vecs(), ffor.bases);
		auto i_exc_off = arena.template add<size_t>(n_vecs, exceptions_offsets);
		auto i_pos_off = arena.template add<size_t>(n_vecs, positions_offsets);
		auto i_exc     = arena.template add<T>(n_exceptions, exceptions, buf);
		auto i_pos     = arena.template add<uint16_t>(n_exceptions, positions, buf);
		auto i_cnt     = arena.template add<uint16_t>(n_vecs, counts);
		out.n_values   = n_values;
		out.n_vecs     = n_vecs;
		out.n_exceptions     = n_exceptions;
		out.ffor.n_values    = ffor.get_n_values();
		out.ffor.bp.n_values = ffor.bp.n_values;
		out.ffor.bp.n_vecs   = ffor.bp.get_n_vecs();
		arena.resolve_to(reinterpret_cast<void**>(&out.ffor.bp.packed_array), i_packed);
		arena.resolve_to(reinterpret_cast<void**>(&out.ffor.bp.bit_widths), i_bw);
		arena.resolve_to(reinterpret_cast<void**>(&out.ffor.bp.vector_offsets), i_bp_off);
		arena.resolve_to(reinterpret_cast<void**>(&out.ffor.bases), i_bases);
		arena.resolve_to(reinterpret_cast<void**>(&out.exceptions_offsets), i_exc_off);
		arena.resolve_to(reinterpret_cast<void**>(&out.positions_offsets), i_pos_off);
		arena.resolve_to(reinterpret_cast<void**>(&out.exceptions), i_exc);
		arena.resolve_to(reinterpret_cast<void**>(&out.positions), i_pos);
		arena.resolve_to(reinterpret_cast<void**>(&out.counts), i_cnt);
	}
};

template <typename T>
void free_column(SLPATCHColumn<T> column) {
	free_column(column.ffor);
	delete[] column.exceptions_offsets;
	delete[] column.positions_offsets;
	delete[] column.exceptions;
	delete[] column.positions;
	delete[] column.counts;
}

template <typename T>
void free_column(device::SLPATCHColumn<T> column) {
	free_column(column.ffor);
	free_device_pointer(column.exceptions_offsets);
	free_device_pointer(column.positions_offsets);
	free_device_pointer(column.exceptions);
	free_device_pointer(column.positions);
	free_device_pointer(column.counts);
}

} // namespace host
} // namespace flsgpu

#endif // FLSGPU_COLUMNS_SLPATCH_CUH
