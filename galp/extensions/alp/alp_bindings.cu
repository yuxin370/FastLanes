// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/extensions/alp/alp_bindings.cu
// ────────────────────────────────────────────────────────
#include "alp.hpp"
#include "galp_extensions/alp/alp_bindings.cuh"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace galp::codec::alp {
constexpr int MAX_ATTEMPTS_TO_ENCODE = 10000;

inline uint32_t checked_u32_offset(const size_t value, const char* field) {
	if (value > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
		throw std::overflow_error(std::string(field) + " exceeds uint32_t range");
	}
	return static_cast<uint32_t>(value);
}

template <typename T>
size_t get_bytes_overhead_size_per_alp_vector() {
	return sizeof(uint8_t) + // bit_width
	       sizeof(uint8_t) + // factor-idx
	       sizeof(uint8_t) + // exponent-idx
	       sizeof(T);        // ffor base
	                         // + 32; // Overhead of vector offset in packed array
	                         // + 32; // Overhead of vector offset in exception array
};

template <typename T>
size_t get_bytes_overhead_size_per_alp_extended_vector() {
	return get_bytes_overhead_size_per_alp_vector<T>() +
	       sizeof(uint16_t) * galp::codec::utils::get_n_lanes<T>(); // pos + offset per lane
	                                                   // data parallel_format;
};

template <typename T>
size_t get_bytes_vector_compressed_size_without_overhead(const uint8_t bit_width, const uint16_t exceptions_count) {
	constexpr size_t line_size               = galp::codec::utils::get_n_lanes<T>() * sizeof(T);
	constexpr size_t exception_value_size    = sizeof(T);
	constexpr size_t exception_position_size = sizeof(uint16_t);

	return bit_width * line_size + exceptions_count * (exception_value_size + exception_position_size);
}

template <typename T>
bool is_compressable(const T* input_array, const size_t count) {
	std::vector<T> sample_array(count);
	::alp::state<T> alpstate;

	bool is_possible = false;
	for (int32_t attempts = 0; attempts < MAX_ATTEMPTS_TO_ENCODE; ++attempts) {
		::alp::encoder<T>::init(input_array, count, sample_array.data(), alpstate);

		if ((is_possible = alpstate.scheme == ::alp::Scheme::ALP)) {
			break;
		}
	}

	return is_possible;
}

template <typename T>
::alp::state<T> configure_alpstate(const T* input_array, const size_t n_values) {
	std::vector<T> sample_array(n_values);
	::alp::state<T> alpstate;

	bool successful_encoding = false;
	for (int32_t attempts = 0; attempts < MAX_ATTEMPTS_TO_ENCODE; ++attempts) {
		::alp::encoder<T>::init(input_array, n_values, sample_array.data(), alpstate);

		if ((successful_encoding = alpstate.scheme == ::alp::Scheme::ALP)) {
			break;
		}
	}
	if (!successful_encoding) {
		throw EncodingException();
	}
	return alpstate;
}

template <typename T>
galp::codec::host::ALPColumn<T> encode(const T* input_array, const size_t n_values, const bool print_compression_info) {
	using INT_T  = typename galp::codec::utils::same_width_int<T>::type;
	using UINT_T = typename galp::codec::utils::same_width_uint<T>::type;

	const size_t n_vecs   = galp::codec::utils::get_n_vecs_from_size(n_values);
	::alp::state<T>     alpstate = configure_alpstate(input_array, n_values);

	// Intermediate arrays
	std::vector<UINT_T>   packed_array_tmp(galp::codec::consts::VALUES_PER_VECTOR);
	std::vector<T>        exceptions_tmp(galp::codec::consts::VALUES_PER_VECTOR);
	std::vector<uint16_t> positions_tmp(galp::codec::consts::VALUES_PER_VECTOR);

	auto v_packed_array = std::vector<UINT_T>();
	auto v_exceptions   = std::vector<T>();
	auto v_positions    = std::vector<uint16_t>();

	// Final arrays
	auto bit_widths         = std::make_unique<vbw_t[]>(n_vecs);
	auto vector_offsets     = std::make_unique<uint32_t[]>(n_vecs);
	auto bases              = std::make_unique<UINT_T[]>(n_vecs);
	auto factor_indices     = std::make_unique<uint8_t[]>(n_vecs);
	auto fraction_indices   = std::make_unique<uint8_t[]>(n_vecs);
	auto exceptions_offsets = std::make_unique<uint32_t[]>(n_vecs);
	auto counts             = std::make_unique<uint16_t[]>(n_vecs);

	size_t compressed_vector_sizes = 0;
	std::vector<INT_T> encoded_array(galp::codec::consts::VALUES_PER_VECTOR);
	size_t exceptions_offset       = 0;
	size_t vector_offset           = 0;

	size_t bit_widths_sum = 0;
	for (size_t vi {0}; vi < n_vecs; vi++) {
		::alp::encoder<T>::encode(input_array, exceptions_tmp.data(), positions_tmp.data(), encoded_array.data(), alpstate, nullptr);
		::alp::encoder<T>::analyze_ffor(encoded_array.data(), bit_widths[vi], reinterpret_cast<INT_T*>(&bases[vi]));

		counts[vi] = alpstate.n_exceptions;
		bit_widths_sum += bit_widths[vi];
		fastlanes::generated::ffor::fallback::scalar::ffor(
		    reinterpret_cast<UINT_T*>(encoded_array.data()), packed_array_tmp.data(), bit_widths[vi], &bases[vi]);

		input_array += galp::codec::consts::VALUES_PER_VECTOR;
		size_t compressed_values_size = galp::codec::utils::get_compressed_vector_size<T>(bit_widths[vi]);

		v_exceptions.insert(v_exceptions.end(), exceptions_tmp.data(), exceptions_tmp.data() + counts[vi]);
		v_positions.insert(v_positions.end(), positions_tmp.data(), positions_tmp.data() + counts[vi]);

		v_packed_array.insert(v_packed_array.end(), packed_array_tmp.data(), packed_array_tmp.data() + compressed_values_size);

		factor_indices[vi]     = alpstate.fac;
		fraction_indices[vi]   = alpstate.exp;
		exceptions_offsets[vi] = checked_u32_offset(exceptions_offset, "ALP exception offset");
		exceptions_offset += counts[vi];
		vector_offsets[vi] = checked_u32_offset(vector_offset, "ALP vector offset");
		vector_offset += compressed_values_size;

		compressed_vector_sizes += get_bytes_vector_compressed_size_without_overhead<T>(bit_widths[vi], counts[vi]);
	}

	size_t compressed_alp_bytes_size = compressed_vector_sizes + n_vecs * get_bytes_overhead_size_per_alp_vector<T>();
	size_t compressed_alp_extended_bytes_size =
	    compressed_vector_sizes + n_vecs * get_bytes_overhead_size_per_alp_extended_vector<T>();

	auto packed_array = std::make_unique<UINT_T[]>(v_packed_array.size());
	auto exceptions   = std::make_unique<T[]>(v_exceptions.size());
	auto positions    = std::make_unique<uint16_t[]>(v_positions.size());

	std::copy(v_packed_array.begin(), v_packed_array.end(), packed_array.get());
	std::copy(v_exceptions.begin(), v_exceptions.end(), exceptions.get());
	std::copy(v_positions.begin(), v_positions.end(), positions.get());

	if (print_compression_info) {
		const size_t input_size = n_values * sizeof(T);
		printf("ALP_COMPRESSION_PARAMETERS,%zu,%f,%f,%f,%f\n",
		       n_vecs,
		       static_cast<double>(input_size) / static_cast<double>(compressed_alp_bytes_size),
		       static_cast<double>(input_size) / static_cast<double>(compressed_alp_extended_bytes_size),
		       static_cast<double>(bit_widths_sum) / static_cast<double>(n_vecs),
		       static_cast<double>(v_exceptions.size()) / static_cast<double>(n_vecs));
	}

	return galp::codec::host::ALPColumn<T> {
	    galp::codec::host::FFORColumn<UINT_T> {
	        galp::codec::host::BPColumn<UINT_T> {
	            n_values,
	            v_packed_array.size(),
	            packed_array.release(),
	            bit_widths.release(),
	            vector_offsets.release(),
	        },
	        bases.release(),
	    },
	    factor_indices.release(),
	    fraction_indices.release(),
	    v_exceptions.size(),
	    exceptions_offsets.release(),
	    exceptions.release(),
	    positions.release(),
	    counts.release(),
	    compressed_alp_bytes_size,
	    compressed_alp_extended_bytes_size,
	};
}

template <typename T>
T* decode(const galp::codec::host::ALPColumn<T>& column, T* output_array) {
	const size_t n_vecs = galp::codec::utils::get_n_vecs_from_size(column.ffor.bp.n_values);

	T* c_output_array = output_array;
	for (size_t vi {0}; vi < n_vecs; ++vi) {
		generated::falp::fallback::scalar::falp(column.ffor.bp.packed_array + column.ffor.bp.vector_offsets[vi],
		                                        c_output_array,
		                                        column.ffor.bp.bit_widths[vi],
		                                        &column.ffor.bases[vi],
		                                        column.factor_indices[vi],
		                                        column.fraction_indices[vi]);

		::alp::state<T> alpstate;
		alpstate.n_exceptions = column.counts[vi];

		::alp::decoder<T>::patch_exceptions(c_output_array,
		                                  column.exceptions + column.exceptions_offsets[vi],
		                                  column.positions + column.exceptions_offsets[vi],
		                                  alpstate);

		c_output_array += galp::codec::consts::VALUES_PER_VECTOR;
	}

	return output_array;
}

template <typename T>
T* decode(const galp::codec::host::ALPExtendedColumn<T>& column, T* output_array) {
	constexpr unsigned N_LANES = galp::codec::utils::get_n_lanes<T>();
	const size_t       n_vecs  = galp::codec::utils::get_n_vecs_from_size(column.ffor.bp.n_values);

	T* c_output_array = output_array;
	for (size_t vi {0}; vi < n_vecs; ++vi) {
		generated::falp::fallback::scalar::falp(column.ffor.bp.packed_array + column.ffor.bp.vector_offsets[vi],
		                                        c_output_array,
		                                        column.ffor.bp.bit_widths[vi],
		                                        &column.ffor.bases[vi],
		                                        column.factor_indices[vi],
		                                        column.fraction_indices[vi]);

		// Reconstruct total count
		uint16_t count = 0;
		for (size_t offset_count_i {0}; offset_count_i < N_LANES; ++offset_count_i) {
			count += column.offsets_counts[vi * N_LANES + offset_count_i] >> 10;
		}

		::alp::state<T> alpstate;
		alpstate.n_exceptions = count; // fix me

		::alp::decoder<T>::patch_exceptions(c_output_array,
		                                  column.exceptions + column.exceptions_offsets[vi],
		                                  column.positions + column.exceptions_offsets[vi],
		                                  alpstate);

		c_output_array += galp::codec::consts::VALUES_PER_VECTOR;
	}

	return output_array;
}

template bool is_compressable(const float* input_array, const size_t n_values);
template bool is_compressable(const double* input_array, const size_t n_values);

template galp::codec::host::ALPColumn<float>
encode(const float* input_array, const size_t n_values, const bool print_compression_info);
template galp::codec::host::ALPColumn<double>
encode(const double* input_array, const size_t n_values, const bool print_compression_info);

template float*  decode(const galp::codec::host::ALPColumn<float>& column, float* output_array);
template double* decode(const galp::codec::host::ALPColumn<double>& column, double* output_array);
template float*  decode(const galp::codec::host::ALPExtendedColumn<float>& column, float* output_array);
template double* decode(const galp::codec::host::ALPExtendedColumn<double>& column, double* output_array);

} // namespace galp::codec::alp
