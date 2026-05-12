// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/flsgpu/columns/alp.cuh
// ────────────────────────────────────────────────────────
#ifndef FLSGPU_COLUMNS_ALP_CUH
#define FLSGPU_COLUMNS_ALP_CUH

#include "alp.hpp"
#include "flsgpu/columns/alp_extended.cuh"
#include "flsgpu/columns/ffor.cuh"
#include "flsgpu/consts.cuh"
#include "flsgpu/memory/gpu_array.cuh"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <tuple>

namespace galp::codec {
namespace device {

template <typename T>
struct ALPColumn {
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
		uint16_t*           counts;
};

} // namespace device

namespace host {

template <typename T>
struct ALPColumn {
	using INT_T         = typename galp::codec::utils::same_width_int<T>::type;
	using UINT_T        = typename galp::codec::utils::same_width_uint<T>::type;
	using DeviceColumnT = typename device::ALPColumn<T>;

	FFORColumn<UINT_T> ffor;

		HostArray<uint8_t> factor_indices;
		HostArray<uint8_t> fraction_indices;

		size_t    n_exceptions;
		HostArray<uint32_t> exceptions_offsets;
		HostArray<T>        exceptions;
		HostArray<uint16_t> positions;
		HostArray<uint16_t> counts;

	size_t compressed_size_bytes_alp;
	size_t compressed_size_bytes_alp_extended;

	size_t get_n_values() const {
		return ffor.bp.n_values;
	}
	size_t get_n_vecs() const {
		return ffor.bp.get_n_vecs();
	}

	double get_compression_ratio() const {
		return static_cast<double>(ffor.bp.n_values * sizeof(T)) / static_cast<double>(compressed_size_bytes_alp);
	}

	device::ALPColumn<T> copy_to_device() const {
		return device::ALPColumn<T> {
		    get_n_values(),
		    ffor.copy_to_device(),
		    GPUArray<INT_T>(galp::codec::consts::as<T>::FACT_ARR_COUNT, alp::Constants<T>::FACT_ARR.data()).release(),
		    GPUArray<T>(galp::codec::consts::as<T>::FRAC_ARR_COUNT, alp::Constants<T>::FRAC_ARR.data()).release(),
		    GPUArray<uint8_t>(ffor.bp.get_n_vecs(), factor_indices).release(),
		    GPUArray<uint8_t>(ffor.bp.get_n_vecs(), fraction_indices).release(),
		    n_exceptions,
		    GPUArray<uint32_t>(ffor.bp.get_n_vecs(), exceptions_offsets).release(),
		    GPUArray<T>(n_exceptions, exceptions).release(),
		    GPUArray<uint16_t>(n_exceptions, positions).release(),
		    GPUArray<uint16_t>(ffor.bp.get_n_vecs(), counts).release(),
		};
	}

	std::tuple<T*, uint16_t*, uint16_t*> convert_exceptions_to_lane_divided_format() const {
		constexpr auto N_LANES         = galp::codec::utils::get_n_lanes<T>();
		constexpr auto VALUES_PER_LANE = galp::codec::utils::get_values_per_lane<T>();

		// New exception allocations
		T*        out_exceptions = new T[n_exceptions];
		uint16_t* out_positions  = new uint16_t[n_exceptions];
		uint16_t* out_offsets_counts =
		    new uint16_t[ffor.get_n_vecs() * N_LANES];

		// Intermediate arrays for reordering positions and exceptions
		T        vec_exceptions[galp::codec::consts::VALUES_PER_VECTOR];
		T        vec_exceptions_positions[galp::codec::consts::VALUES_PER_VECTOR];
		uint16_t lane_counts[N_LANES];

		// Copies of pointers for pointer arithmetic
		T*        c_exceptions         = exceptions;
		uint16_t* c_positions          = positions;
		T*        c_out_exceptions     = out_exceptions;
		uint16_t* c_out_positions      = out_positions;
		uint16_t* c_out_offsets_counts = out_offsets_counts;

		for (size_t vec_index {0}; vec_index < ffor.get_n_vecs(); ++vec_index) {
			uint32_t vec_exception_count = counts[vec_index];

			// Reset counts
			for (size_t j {0}; j < N_LANES; ++j) {
				lane_counts[j] = 0;
			}

			// Split all exceptions into lanes
			for (size_t exception_index {0}; exception_index < vec_exception_count; ++exception_index) {
				T        exception = c_exceptions[exception_index];
				uint16_t position  = c_positions[exception_index];

				uint32_t lane                 = position % N_LANES;
				uint32_t lane_exception_count = lane_counts[lane];
				++lane_counts[lane];
				vec_exceptions[lane * VALUES_PER_LANE + lane_exception_count]           = exception;
				vec_exceptions_positions[lane * VALUES_PER_LANE + lane_exception_count] = position;
			}

			// Merge and concatenate all exceptions per lane into single contiguous
			// array
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

			c_exceptions += vec_exception_count;
			c_positions += vec_exception_count;
			c_out_exceptions += vec_exception_count;
			c_out_positions += vec_exception_count;
			c_out_offsets_counts += galp::codec::utils::get_n_lanes<T>();
		}

		return std::make_tuple(out_exceptions, out_positions, out_offsets_counts);
	}

	ALPExtendedColumn<T> create_extended_column() const {
		auto [e_exceptions, e_positions, e_offsets_counts] = convert_exceptions_to_lane_divided_format();
		return ALPExtendedColumn<T> {FFORColumn<UINT_T> {
		                                 BPColumn<UINT_T> {
		                                     ffor.bp.n_values,
		                                     ffor.bp.n_packed_values,
		                                     galp::codec::utils::copy_array(ffor.bp.packed_array.get(), ffor.bp.n_packed_values),
		                                     galp::codec::utils::copy_array(ffor.bp.bit_widths.get(), get_n_vecs()),
		                                     galp::codec::utils::copy_array(ffor.bp.vector_offsets.get(), get_n_vecs()),
		                                 },
		                                 galp::codec::utils::copy_array(ffor.bases.get(), get_n_vecs()),
		                             },
		                             galp::codec::utils::copy_array(factor_indices.get(), get_n_vecs()),
		                             galp::codec::utils::copy_array(fraction_indices.get(), get_n_vecs()),
		                             n_exceptions,
		                             galp::codec::utils::copy_array(exceptions_offsets.get(), get_n_vecs()),
		                             e_exceptions,
		                             e_positions,
		                             e_offsets_counts,
		                             compressed_size_bytes_alp_extended};
	}
};

template <typename T>
void free_column(ALPColumn<T>& column) {
	free_column(column.ffor);
	column.factor_indices.reset();
	column.fraction_indices.reset();
	column.exceptions_offsets.reset();
	column.exceptions.reset();
	column.positions.reset();
	column.counts.reset();
}

template <typename T>
void free_column(device::ALPColumn<T> column) {
	free_column(column.ffor);
	free_device_pointer(column.factors);
	free_device_pointer(column.fractions);
	free_device_pointer(column.factor_indices);
	free_device_pointer(column.fraction_indices);
	free_device_pointer(column.exceptions_offsets);
	free_device_pointer(column.exceptions);
	free_device_pointer(column.positions);
	free_device_pointer(column.counts);
}

} // namespace host
} // namespace galp::codec

#endif // FLSGPU_COLUMNS_ALP_CUH
