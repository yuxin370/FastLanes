// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/unpack_dispatch.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_ENGINE_UNPACK_DISPATCH_CUH
#define GALP_ENGINE_UNPACK_DISPATCH_CUH

#include "codecs/consts.cuh"
#include "engine/config.cuh"
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
	case 2:
	case galp::codec::consts::MAX_UNPACK_N_VECS:
		return;
	default:
		throw std::invalid_argument("unsupported unpack config: supported tuples are (1,1), (2,1), and (4,1)");
	}
}

template <typename Fn>
decltype(auto) with_unpack_config(const ExecutionConfig& cfg, Fn&& fn) {
	validate_unpack_config(cfg);

	switch (cfg.unpack_n_vectors) {
	case 1:
		return fn(std::integral_constant<unsigned, 1> {}, std::integral_constant<unsigned, 1> {});
	case 2:
		return fn(std::integral_constant<unsigned, 2> {}, std::integral_constant<unsigned, 1> {});
	case galp::codec::consts::MAX_UNPACK_N_VECS:
		return fn(std::integral_constant<unsigned, galp::codec::consts::MAX_UNPACK_N_VECS> {},
		          std::integral_constant<unsigned, 1> {});
	default:
		throw std::invalid_argument("unsupported unpack config: supported tuples are (1,1), (2,1), and (4,1)");
	}
}

template <typename Fn>
decltype(auto) with_delta_decoder(const ExecutionConfig& cfg, Fn&& fn) {
	switch (galp::execution::resolve_delta_decoder(cfg.delta_decoder)) {
	case galp::execution::DeltaDecoder::Stateful:
		return fn(std::integral_constant<galp::execution::DeltaDecoder, galp::execution::DeltaDecoder::Stateful> {});
	case galp::execution::DeltaDecoder::Register:
		return fn(std::integral_constant<galp::execution::DeltaDecoder, galp::execution::DeltaDecoder::Register> {});
	case galp::execution::DeltaDecoder::Auto:
		break;
	default:
		break;
	}
	throw std::invalid_argument("unsupported DELTA decoder");
}

} // namespace galp::runtime

#endif // GALP_ENGINE_UNPACK_DISPATCH_CUH
