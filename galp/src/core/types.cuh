// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/types.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_TYPES_CUH
#define ENGINE_TYPES_CUH

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace galp::format {

enum class DataType {
	I8,
	I16,
	U32,
	U64,
	F32,
	F64,
};

template <DataType>
struct CppType;

template <>
struct CppType<DataType::I8> {
	using type = int8_t;
};
template <>
struct CppType<DataType::I16> {
	using type = int16_t;
};
template <>
struct CppType<DataType::U32> {
	using type = uint32_t;
};
template <>
struct CppType<DataType::U64> {
	using type = uint64_t;
};
template <>
struct CppType<DataType::F32> {
	using type = float;
};
template <>
struct CppType<DataType::F64> {
	using type = double;
};

template <typename T>
struct ToDataType;

template <>
struct ToDataType<int8_t> : std::integral_constant<DataType, DataType::I8> {};
template <>
struct ToDataType<int16_t> : std::integral_constant<DataType, DataType::I16> {};
template <>
struct ToDataType<uint32_t> : std::integral_constant<DataType, DataType::U32> {};
template <>
struct ToDataType<uint64_t> : std::integral_constant<DataType, DataType::U64> {};
template <>
struct ToDataType<float> : std::integral_constant<DataType, DataType::F32> {};
template <>
struct ToDataType<double> : std::integral_constant<DataType, DataType::F64> {};

inline constexpr std::array<DataType, 2> kSupportedDataTypes = {DataType::I8, DataType::I16};

inline constexpr bool is_supported_data_type(DataType dt) {
	for (auto v : kSupportedDataTypes) {
		if (v == dt) {
			return true;
		}
	}
	return false;
}

inline constexpr std::string_view data_type_to_string(DataType dt) {
	switch (dt) {
	case DataType::I8:
		return "i8";
	case DataType::I16:
		return "i16";
	case DataType::U32:
		return "u32";
	case DataType::U64:
		return "u64";
	case DataType::F32:
		return "f32";
	case DataType::F64:
		return "f64";
	default:
		return "unknown";
	}
}

inline DataType string_to_data_type(std::string_view str) {
	if (str == "i8") {
		return DataType::I8;
	}
	if (str == "i16") {
		return DataType::I16;
	}
	if (str == "u32") {
		return DataType::U32;
	}
	if (str == "u64") {
		return DataType::U64;
	}
	if (str == "f32") {
		return DataType::F32;
	}
	if (str == "f64") {
		return DataType::F64;
	}
	throw std::invalid_argument("Unknown data type");
}

} // namespace galp::format

#endif // ENGINE_TYPES_CUH
