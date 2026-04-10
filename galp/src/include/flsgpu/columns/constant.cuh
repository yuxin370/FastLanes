// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/columns/constant.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_COLUMNS_CONSTANT_CUH
#define FLSGPU_COLUMNS_CONSTANT_CUH

#include "flsgpu/columns/base.cuh"
#include "flsgpu/columns/parse_common.cuh"
#include "flsgpu/host-utils.cuh"
#include <cstddef>
#include <cstdint>

namespace flsgpu {
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
		return utils::get_n_vecs_from_size(n_values);
	}

	device::CONSTANTColumn<T> copy_to_device() const {
		return device::CONSTANTColumn<T> {n_values, value};
	}

	device::CONSTANTColumn<T> copy_to_device(cudaStream_t stream) const {
		(void)stream;
		return copy_to_device();
	}

	void copy_to_device(flsgpu::memory::DeviceArena& /*arena*/, device::CONSTANTColumn<T>& out) const {
		out = copy_to_device();
	}
};

template <typename T>
void free_column(CONSTANTColumn<T> column) {
	(void)column;
}

template <typename T>
void free_column(device::CONSTANTColumn<T> column) {
	(void)column;
}

} // namespace host
} // namespace flsgpu

namespace reader::columns {

template <typename T>
inline ParseResultT<flsgpu::host::CONSTANTColumn<T>> parse_constant(const ParseContext& ctx) {
	if (!ctx.col_desc.max()) {
		throw std::runtime_error("EXP_CONSTANT: missing max value");
	}
	const auto* bin = ctx.col_desc.max()->binary_data();
	if (!bin || bin->size() != sizeof(T)) {
		throw std::runtime_error("EXP_CONSTANT: invalid constant size");
	}
	const auto value = *reinterpret_cast<const T*>(bin->data());
	return ParseResultT<flsgpu::host::CONSTANTColumn<T>> {flsgpu::host::CONSTANTColumn<T> {ctx.n_values, value}};
}

} // namespace reader::columns

#endif // FLSGPU_COLUMNS_CONSTANT_CUH
