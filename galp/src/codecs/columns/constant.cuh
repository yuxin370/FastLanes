// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/codecs/columns/constant.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_COMPRESSION_COLUMNS_CONSTANT_CUH
#define GALP_COMPRESSION_COLUMNS_CONSTANT_CUH

#include "codecs/columns/base.cuh"
#include "cuda/memory/device_arena.cuh"
#include <cstddef>
#include <cstdint>

namespace galp::codec {
namespace device {

template <typename T>
struct CONSTANTColumn {
	size_t n_values;
	T      value;
};

} // namespace device

namespace host {

template <typename T>
struct CONSTANTColumn {
	using DeviceColumnT = typename device::CONSTANTColumn<T>;
	size_t n_values;
	T      value;

	size_t get_n_values() const {
		return n_values;
	}
	size_t get_n_vecs() const {
		return galp::codec::utils::get_n_vecs_from_size(n_values);
	}

	device::CONSTANTColumn<T> copy_to_device() const {
		return device::CONSTANTColumn<T> {n_values, value};
	}

	void copy_to_device(galp::memory::DeviceArena& /*arena*/, device::CONSTANTColumn<T>& out) const {
		out = copy_to_device();
	}
};

template <typename T>
void free_column(CONSTANTColumn<T>& column) {
	(void)column;
}

template <typename T>
void free_column(device::CONSTANTColumn<T> column) {
	(void)column;
}

} // namespace host
} // namespace galp::codec

#endif // GALP_COMPRESSION_COLUMNS_CONSTANT_CUH
