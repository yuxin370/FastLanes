// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/compression/utils.cuh
// ────────────────────────────────────────────────────────
#ifndef FASTLANES_UTILS_H
#define FASTLANES_UTILS_H

#include "compression/consts.cuh"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace galp::codec::utils { // internal functions
	              //
template <typename T>
struct same_width_int {
	using type = typename std::conditional<
	    sizeof(T) == 8,
	    int64_t,
	    typename std::conditional<sizeof(T) == 4,
	                              int32_t,
	                              typename std::conditional<sizeof(T) == 2, int16_t, int8_t>::type>::type>::type;
};

template <typename T>
struct same_width_uint {
	using type = typename std::conditional<
	    sizeof(T) == 8,
	    uint64_t,
	    typename std::conditional<sizeof(T) == 4,
	                              uint32_t,
	                              typename std::conditional<sizeof(T) == 2, uint16_t, uint8_t>::type>::type>::type;
};

template <typename T, typename returnT = int32_t>
constexpr returnT sizeof_in_bits() {
	return sizeof(T) * 8;
}

template <typename T>
constexpr T min(const T& a, const T& b) {
	// C++-11 and older do not offer constexpr, that is why this is added here
	return a <= b ? a : b;
}

template <typename T_in, typename T_out>
T_out reinterpret_type(const T_in& in) {
	static_assert(std::is_trivially_copyable<T_in>::value, "input type must be trivially copyable");
	static_assert(std::is_trivially_copyable<T_out>::value, "output type must be trivially copyable");
	T_out out {};
	std::memcpy(&out, &in, min(sizeof(T_in), sizeof(T_out)));
	return out;
}

template <typename T>
constexpr T h_set_first_n_bits(const int32_t count) {
	return (count < sizeof_in_bits<T>() ? static_cast<T>((T {1} << int32_t {count}) - T {1}) : static_cast<T>(~T {0}));
}

template <typename T>
constexpr T set_first_n_bits(const int32_t count) {
	using UINT_T = typename same_width_uint<T>::type;
	static_assert(std::is_integral<T>::value, "T must be an integer type");

	if (count <= 0) {
		return T {0};
	}
	constexpr int32_t width = galp::codec::utils::sizeof_in_bits<UINT_T>();
	if (count >= width) {
		return static_cast<T>(std::numeric_limits<UINT_T>::max());
	}
	const UINT_T mask = (UINT_T {1} << count) - UINT_T {1};
	return static_cast<T>(mask);
}

template <typename T>
constexpr int32_t get_lane_bitwidth() {
	return sizeof_in_bits<T>();
}

template <typename T>
constexpr int32_t get_n_lanes() {
	return galp::codec::consts::REGISTER_WIDTH / get_lane_bitwidth<T>();
}

template <typename T>
constexpr int32_t get_values_per_lane() {
	return galp::codec::consts::VALUES_PER_VECTOR / get_n_lanes<T>();
}

template <typename T>
constexpr int32_t get_compressed_vector_size(int32_t value_bit_width) {
	return (galp::codec::consts::VALUES_PER_VECTOR * value_bit_width) / sizeof_in_bits<T>();
}

constexpr size_t get_n_vecs_from_size(const size_t size) {
	return (size + galp::codec::consts::VALUES_PER_VECTOR - 1) / galp::codec::consts::VALUES_PER_VECTOR;
}

template <typename T>
T* copy_array(const T* in, const size_t n_elements) {
	T* out = new T[n_elements];
	std::memcpy(out, in, sizeof(T) * n_elements);
	return out;
}

} // namespace galp::codec::utils

#endif // FASTLANES_UTILS_H
