// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/compression/columns/alp_extended.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_COMPRESSION_COLUMNS_ALP_EXTENDED_CUH
#define GALP_COMPRESSION_COLUMNS_ALP_EXTENDED_CUH

#include "alp.hpp"
#include "compression/columns/ffor.cuh"
#include "compression/consts.cuh"
#include "memory/gpu_array.cuh"
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace galp::codec {
namespace device {

template <typename T>
struct ALPExtendedColumn {
	using INT_T  = typename galp::codec::utils::same_width_int<T>::type;
	using UINT_T = typename galp::codec::utils::same_width_uint<T>::type;
	size_t             n_values;
	FFORColumn<UINT_T> ffor;

		INT_T*   factors;
		T*       fractions;
		uint8_t* factor_indices;
		uint8_t* fraction_indices;

		size_t              n_exceptions;
		uint32_t*           exceptions_offsets;
		T*                  exceptions;
		uint16_t*           positions;
		uint16_t*           offsets_counts;
};

} // namespace device

namespace host {

template <typename T>
struct ALPExtendedColumn {
	using INT_T         = typename galp::codec::utils::same_width_int<T>::type;
	using UINT_T        = typename galp::codec::utils::same_width_uint<T>::type;
	using DeviceColumnT = typename device::ALPExtendedColumn<T>;

	FFORColumn<UINT_T> ffor;

		HostArray<uint8_t> factor_indices;
		HostArray<uint8_t> fraction_indices;

		size_t    n_exceptions;
		HostArray<uint32_t> exceptions_offsets;
		HostArray<T>        exceptions;
		HostArray<uint16_t> positions;
		HostArray<uint16_t> offsets_counts;

	size_t compressed_size_bytes_alp_extended;

	size_t get_n_values() const {
		return ffor.bp.n_values;
	}
	size_t get_n_vecs() const {
		return ffor.bp.get_n_vecs();
	}

	double get_compression_ratio() const {
		return static_cast<double>(ffor.bp.n_values * sizeof(T)) /
		       static_cast<double>(compressed_size_bytes_alp_extended);
	}

	device::ALPExtendedColumn<T> copy_to_device() const {
		size_t branchless_and_prefetch_buffer = galp::codec::consts::MAX_UNPACK_N_VECS;
		return device::ALPExtendedColumn<T> {
		    get_n_values(),
		    ffor.copy_to_device(),
		    GPUArray<INT_T>(galp::codec::consts::as<T>::FACT_ARR_COUNT, alp::Constants<T>::FACT_ARR.data()).release(),
		    GPUArray<T>(galp::codec::consts::as<T>::FRAC_ARR_COUNT, alp::Constants<T>::FRAC_ARR.data()).release(),
		    GPUArray<uint8_t>(ffor.bp.get_n_vecs(), factor_indices).release(),
		    GPUArray<uint8_t>(ffor.bp.get_n_vecs(), fraction_indices).release(),
		    n_exceptions,
		    GPUArray<uint32_t>(ffor.bp.get_n_vecs(), exceptions_offsets).release(),
		    GPUArray<T>(n_exceptions, branchless_and_prefetch_buffer, exceptions).release(),
		    GPUArray<uint16_t>(n_exceptions, branchless_and_prefetch_buffer, positions).release(),
		    GPUArray<uint16_t>(ffor.bp.get_n_vecs() * galp::codec::utils::get_n_lanes<T>(), offsets_counts).release(),
		};
	}
};

template <typename T>
void free_column(ALPExtendedColumn<T>& column) {
	free_column(column.ffor);
	column.factor_indices.reset();
	column.fraction_indices.reset();
	column.exceptions_offsets.reset();
	column.exceptions.reset();
	column.positions.reset();
	column.offsets_counts.reset();
}

template <typename T>
void free_column(device::ALPExtendedColumn<T> column) {
	free_column(column.ffor);
	free_device_pointer(column.factors);
	free_device_pointer(column.fractions);
	free_device_pointer(column.factor_indices);
	free_device_pointer(column.fraction_indices);
	free_device_pointer(column.exceptions_offsets);
	free_device_pointer(column.exceptions);
	free_device_pointer(column.positions);
	free_device_pointer(column.offsets_counts);
}

} // namespace host
} // namespace galp::codec

#endif // GALP_COMPRESSION_COLUMNS_ALP_EXTENDED_CUH
