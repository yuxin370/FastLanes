// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/data/value_store.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_DATA_VALUE_STORE_CUH
#define ENGINE_DATA_VALUE_STORE_CUH

#include <cstdint>
#include <memory>
#include <type_traits>
#include <variant>

namespace galp::execution {

template <typename... Ts>
struct TypeList {};

using SupportedTypes = TypeList<int8_t, int16_t>;

template <typename T, typename List>
struct IsIn;

template <typename T, typename... Ts>
struct IsIn<T, TypeList<Ts...>> : std::bool_constant<(std::is_same_v<T, Ts> || ...)> {};

template <typename T>
inline constexpr bool is_supported_type_v = IsIn<T, SupportedTypes>::value;

template <typename List>
struct ValueStoreBuilder;

template <typename... Ts>
struct ValueStoreBuilder<TypeList<Ts...>> {
	using type = std::variant<std::shared_ptr<Ts[]>...>;
};

using ValueStore = typename ValueStoreBuilder<SupportedTypes>::type;

template <typename T>
inline ValueStore make_value_store(T* ptr) {
	return ValueStore {std::shared_ptr<T[]>(ptr, std::default_delete<T[]>())};
}

template <typename... Ts, typename F>
inline void for_each_type(TypeList<Ts...>, F&& f) {
	(f(std::type_identity<Ts> {}), ...);
}

} // namespace galp::execution

#endif // ENGINE_DATA_VALUE_STORE_CUH
