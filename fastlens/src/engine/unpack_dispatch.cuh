// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// fastlens/src/engine/unpack_dispatch.cuh
// ────────────────────────────────────────────────────────
#ifndef FASTLENS_ENGINE_UNPACK_DISPATCH_CUH
#define FASTLENS_ENGINE_UNPACK_DISPATCH_CUH

#include "codecs/consts.cuh"
#include "engine/config.cuh"
#include <stdexcept>
#include <type_traits>

namespace fastlens::runtime {

using fastlens::execution::ExecutionConfig;

inline void validate_unpack_config(const ExecutionConfig& cfg) {
	if (cfg.unpack_n_values != 1) {
		throw std::invalid_argument("unsupported unpack config: only unpack_n_values=1 is currently supported");
	}

	switch (cfg.unpack_n_vectors) {
	case 1:
	case 2:
	case fastlens::codec::consts::MAX_UNPACK_N_VECS:
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
	case fastlens::codec::consts::MAX_UNPACK_N_VECS:
		return fn(std::integral_constant<unsigned, fastlens::codec::consts::MAX_UNPACK_N_VECS> {},
		          std::integral_constant<unsigned, 1> {});
	default:
		throw std::invalid_argument("unsupported unpack config: supported tuples are (1,1), (2,1), and (4,1)");
	}
}

template <typename Fn>
decltype(auto) with_delta_decoder(const ExecutionConfig& cfg, Fn&& fn) {
	switch (fastlens::execution::resolve_delta_decoder(cfg.delta_decoder)) {
	case fastlens::execution::DeltaDecoder::Stateful:
		return fn(std::integral_constant<fastlens::execution::DeltaDecoder, fastlens::execution::DeltaDecoder::Stateful> {});
	case fastlens::execution::DeltaDecoder::Register:
		return fn(std::integral_constant<fastlens::execution::DeltaDecoder, fastlens::execution::DeltaDecoder::Register> {});
	case fastlens::execution::DeltaDecoder::Auto:
		break;
	default:
		break;
	}
	throw std::invalid_argument("unsupported DELTA decoder");
}

} // namespace fastlens::runtime

#endif // FASTLENS_ENGINE_UNPACK_DISPATCH_CUH
