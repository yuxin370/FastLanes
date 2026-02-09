// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/types.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_TYPES_CUH
#define ENGINE_TYPES_CUH

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <variant>

namespace types {

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

} // namespace types

namespace dispatch {

template <typename... Ts>
struct TypeList {};

using SupportedTypes = TypeList<int8_t, int16_t>;

template <typename T, typename List>
struct IsIn;

template <typename T, typename... Ts>
struct IsIn<T, TypeList<Ts...>> : std::bool_constant<(std::is_same_v<T, Ts> || ...)> {};

template <typename T>
inline constexpr bool is_supported_type_v = IsIn<T, SupportedTypes>::value;

inline constexpr bool is_supported_data_type(types::DataType dt) {
	return types::is_supported_data_type(dt);
}

template <typename List>
struct DecompressVariantBuilder;

template <typename... Ts>
struct DecompressVariantBuilder<TypeList<Ts...>> {
	using type = std::variant<std::unique_ptr<Ts[]>...>;
};

using DecompressResult = typename DecompressVariantBuilder<SupportedTypes>::type;

template <typename T>
inline DecompressResult make_result(T* ptr) {
	return DecompressResult {std::unique_ptr<T[]>(ptr)};
}

template <typename... Ts, typename F>
inline void for_each_type(TypeList<Ts...>, F&& f) {
	(f(std::type_identity<Ts> {}), ...);
}

} // namespace dispatch

#endif // ENGINE_TYPES_CUH
