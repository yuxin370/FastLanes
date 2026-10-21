// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// fastlens/extensions/alp/include/fastlens_extensions/alp/alp_bindings.cuh
// ────────────────────────────────────────────────────────
#ifndef ALP_BINDINGS_CUH
#define ALP_BINDINGS_CUH

#include "alp.hpp"
#include "codecs/decode/alp.cuh"
#include <cstddef>
#include <cstdint>
#include <exception>
#include <type_traits>

namespace fastlens::codec::alp {

class EncodingException : public std::exception {
public:
	using std::exception::what;
	const char* what() {
		return "Could not encode data with desired encoding.";
	}
};

// Test if data can be decoded in specified type
template <typename T>
bool is_compressable(const T* input_array, const size_t n_elements);

	template <typename T>
	fastlens::codec::host::ALPColumn<T>
encode(const T* input_array, const size_t n_elements, const bool print_compression_info = false);

template <typename T>
T* decode(const fastlens::codec::host::ALPColumn<T>& column, T* output_array);

template <typename T>
T* decode(const fastlens::codec::host::ALPExtendedColumn<T>& column, T* output_array);

} // namespace fastlens::codec::alp

#endif // ALP_BINDINGS_CUH
