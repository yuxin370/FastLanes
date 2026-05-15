// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/execution/internal/unpack_dispatch.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_INTERNAL_UNPACK_DISPATCH_CUH
#define ENGINE_EXECUTION_INTERNAL_UNPACK_DISPATCH_CUH

#include "engine/execution/config.cuh"
#include "compression/consts.cuh"
#include <stdexcept>
#include <type_traits>

namespace galp::runtime {

using galp::execution::ExecutionConfig;

inline void validate_unpack_config(const ExecutionConfig& cfg) {
	if (cfg.unpack_n_values != 1) {
		throw std::invalid_argument("unsupported unpack config: only unpack_n_values=1 is currently supported");
	}

	switch (cfg.unpack_n_vectors) {
	case 1:
	case galp::codec::consts::MAX_UNPACK_N_VECS:
		return;
	default:
		throw std::invalid_argument("unsupported unpack config: supported tuples are (1,1) and (4,1)");
	}
}

template <typename Fn>
decltype(auto) with_unpack_config(const ExecutionConfig& cfg, Fn&& fn) {
	validate_unpack_config(cfg);

	switch (cfg.unpack_n_vectors) {
	case 1:
		return fn(std::integral_constant<unsigned, 1> {}, std::integral_constant<unsigned, 1> {});
	case galp::codec::consts::MAX_UNPACK_N_VECS:
		return fn(std::integral_constant<unsigned, galp::codec::consts::MAX_UNPACK_N_VECS> {},
		          std::integral_constant<unsigned, 1> {});
	default:
		throw std::invalid_argument("unsupported unpack config: supported tuples are (1,1) and (4,1)");
	}
}

} // namespace galp::runtime

#endif // ENGINE_EXECUTION_INTERNAL_UNPACK_DISPATCH_CUH
