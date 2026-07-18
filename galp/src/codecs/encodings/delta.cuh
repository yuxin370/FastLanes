// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
#ifndef GALP_COMPRESSION_COLUMNS_DELTA_CUH
#define GALP_COMPRESSION_COLUMNS_DELTA_CUH

#include "codecs/encodings/ffor.cuh"

namespace galp::codec {
namespace device {

template <typename T>
struct DELTAColumn {
	using UINT_T = typename galp::codec::utils::same_width_uint<T>::type;
	size_t             n_values;
	FFORColumn<UINT_T> ffor;
	UINT_T*            rsum_bases;
};

} // namespace device

namespace host {

template <typename T>
struct DELTAColumn {
	using UINT_T        = typename galp::codec::utils::same_width_uint<T>::type;
	using DeviceColumnT = typename device::DELTAColumn<T>;

	FFORColumn<UINT_T> ffor;
	HostArray<UINT_T>  rsum_bases;

	size_t get_n_values() const {
		return ffor.get_n_values();
	}
	size_t get_n_vecs() const {
		return ffor.get_n_vecs();
	}

	device::DELTAColumn<T> copy_to_device() const {
		return device::DELTAColumn<T> {
		    get_n_values(),
		    ffor.copy_to_device(),
		    GPUArray<UINT_T>(get_n_vecs() * galp::codec::utils::get_n_lanes<T>(), rsum_bases).release()};
	}

	void copy_to_device(galp::memory::DeviceArena& arena, device::DELTAColumn<T>& out) const {
		ffor.copy_to_device(arena, out.ffor);
		const auto i_rsum =
		    arena.template add<UINT_T>(get_n_vecs() * galp::codec::utils::get_n_lanes<T>(), rsum_bases);
		out.n_values = get_n_values();
		arena.resolve_to(reinterpret_cast<void**>(&out.rsum_bases), i_rsum);
	}
};

template <typename T>
void free_column(DELTAColumn<T>& column) {
	free_column(column.ffor);
	column.rsum_bases.reset();
}

template <typename T>
void free_column(device::DELTAColumn<T> column) {
	free_column(column.ffor);
	free_device_pointer(column.rsum_bases);
}

} // namespace host
} // namespace galp::codec

#endif // GALP_COMPRESSION_COLUMNS_DELTA_CUH
