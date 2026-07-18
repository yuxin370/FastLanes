// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/operators/dict_ref_resolver.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_DICT_REF_RESOLVER_CUH
#define ENGINE_EXECUTION_DICT_REF_RESOLVER_CUH

#include "core/expression.cuh"
#include "fls/expression/rpn.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace galp::execution {

namespace detail {

template <typename>
struct DictRefTraits {
	static constexpr bool is_dict_ref = false;
};

template <typename T, typename IndexT>
struct DictRefTraits<galp::codec::host::DICTREFColumn<T, IndexT>> {
	static constexpr bool is_dict_ref = true;
	using value_type                  = T;
	using index_type                  = IndexT;
};

template <typename>
struct DictFforTraits {
	static constexpr bool is_dict_ffor = false;
};

template <typename T, typename IndexT>
struct DictFforTraits<galp::codec::host::DICTFFORColumn<T, IndexT>> {
	static constexpr bool is_dict_ffor = true;
	using index_type                   = IndexT;
};

template <typename>
struct DictSlpatchTraits {
	static constexpr bool is_dict_slpatch = false;
};

template <typename T, typename IndexT>
struct DictSlpatchTraits<galp::codec::host::DICTSLPATCHColumn<T, IndexT>> {
	static constexpr bool is_dict_slpatch = true;
	using index_type                      = IndexT;
};

template <typename HostColumnT>
inline constexpr bool is_dict_ref_column_v = DictRefTraits<std::decay_t<HostColumnT>>::is_dict_ref;

template <typename HostColumnT>
inline constexpr bool is_dict_ffor_column_v = DictFforTraits<std::decay_t<HostColumnT>>::is_dict_ffor;

template <typename HostColumnT>
inline constexpr bool is_dict_slpatch_column_v = DictSlpatchTraits<std::decay_t<HostColumnT>>::is_dict_slpatch;

inline bool is_dict_ref_payload(const galp::execution::EncodedPayload& payload) {
	return std::visit([](const auto& column) { return is_dict_ref_column_v<decltype(column)>; }, payload);
}

struct DictRefResolveResult {
	galp::execution::EncodedPayload payload;
	fastlanes::OperatorToken        token;
};

template <typename ValueT, typename IndexT>
constexpr fastlanes::OperatorToken dict_ffor_token() {
	if constexpr (std::is_same_v<ValueT, int8_t> && std::is_same_v<IndexT, uint8_t>) {
		return fastlanes::OperatorToken::EXP_DICT_I08_FFOR_U08;
	} else if constexpr (std::is_same_v<ValueT, int16_t> && std::is_same_v<IndexT, uint8_t>) {
		return fastlanes::OperatorToken::EXP_DICT_I16_FFOR_U08;
	} else {
		static_assert(std::is_same_v<ValueT, int16_t> && std::is_same_v<IndexT, uint16_t>);
		return fastlanes::OperatorToken::EXP_DICT_I16_FFOR_U16;
	}
}

template <typename ValueT, typename IndexT>
constexpr fastlanes::OperatorToken dict_slpatch_token() {
	if constexpr (std::is_same_v<ValueT, int8_t> && std::is_same_v<IndexT, uint8_t>) {
		return fastlanes::OperatorToken::EXP_DICT_I08_FFOR_SLPATCH_U08;
	} else if constexpr (std::is_same_v<ValueT, int16_t> && std::is_same_v<IndexT, uint8_t>) {
		return fastlanes::OperatorToken::EXP_DICT_I16_FFOR_SLPATCH_U08;
	} else {
		static_assert(std::is_same_v<ValueT, int16_t> && std::is_same_v<IndexT, uint16_t>);
		return fastlanes::OperatorToken::EXP_DICT_I16_FFOR_SLPATCH_U16;
	}
}

template <typename ValueT, typename IndexT>
DictRefResolveResult make_dict_ffor_result(galp::codec::host::DICTREFColumn<ValueT, IndexT>& dict_ref,
	                                       galp::codec::host::FFORColumn<IndexT>              index) {
	auto keys = std::move(dict_ref.keys);
	return DictRefResolveResult {
	    galp::codec::host::DICTFFORColumn<ValueT, IndexT> {std::move(index), std::move(keys), dict_ref.key_count},
	    dict_ffor_token<ValueT, IndexT>()};
}

template <typename ValueT, typename IndexT>
DictRefResolveResult make_dict_slpatch_result(
	galp::codec::host::DICTREFColumn<ValueT, IndexT>& dict_ref,
	galp::codec::host::SLPATCHColumn<IndexT>           index) {
	auto keys = std::move(dict_ref.keys);
	return DictRefResolveResult {
	    galp::codec::host::DICTSLPATCHColumn<ValueT, IndexT> {std::move(index), std::move(keys), dict_ref.key_count},
	    dict_slpatch_token<ValueT, IndexT>()};
}

template <typename ValueT, typename IndexT>
std::optional<DictRefResolveResult>
try_resolve_dict_ref(galp::codec::host::DICTREFColumn<ValueT, IndexT>& dict_ref,
	                 const galp::execution::EncodedPayload&            index_payload) {
	using SignedIndexT = std::make_signed_t<IndexT>;
	std::optional<DictRefResolveResult> result;

	std::visit(
	    [&](const auto& index_col) {
		    using IndexHostT = std::decay_t<decltype(index_col)>;
		    if constexpr (std::is_same_v<IndexHostT, galp::codec::host::FFORColumn<SignedIndexT>>) {
			    result.emplace(make_dict_ffor_result(
			        dict_ref, galp::codec::host::make_index_ffor_from_ffor<IndexT>(index_col)));
		    } else if constexpr (std::is_same_v<IndexHostT, galp::codec::host::BPColumn<SignedIndexT>>) {
			    result.emplace(
			        make_dict_ffor_result(dict_ref, galp::codec::host::make_index_ffor_from_bp<IndexT>(index_col)));
		    } else if constexpr (std::is_same_v<IndexHostT, galp::codec::host::SLPATCHColumn<SignedIndexT>>) {
			    result.emplace(make_dict_slpatch_result(
			        dict_ref, galp::codec::host::make_index_slpatch_from_slpatch<IndexT>(index_col)));
		    } else if constexpr (is_dict_ffor_column_v<IndexHostT>) {
			    using SourceIndexT = typename DictFforTraits<IndexHostT>::index_type;
			    if constexpr (std::is_same_v<SourceIndexT, IndexT>) {
				    result.emplace(make_dict_ffor_result(
				        dict_ref, galp::codec::host::make_index_ffor_from_ffor<IndexT>(index_col.ffor)));
			    }
		    } else if constexpr (is_dict_slpatch_column_v<IndexHostT>) {
			    using SourceIndexT = typename DictSlpatchTraits<IndexHostT>::index_type;
			    if constexpr (std::is_same_v<SourceIndexT, IndexT>) {
				    result.emplace(make_dict_slpatch_result(
				        dict_ref, galp::codec::host::make_index_slpatch_from_slpatch<IndexT>(index_col.index)));
			    }
		    }
	    },
	    index_payload);
	return result;
}

inline std::string dependency_cycle_message(const std::vector<size_t>& stack, const size_t repeated) {
	std::ostringstream message;
	message << "DICTREF dependency cycle detected: ";
	const auto begin = std::find(stack.begin(), stack.end(), repeated);
	for (auto it = begin; it != stack.end(); ++it) {
		if (it != begin) {
			message << " -> ";
		}
		message << *it;
	}
	if (begin != stack.end()) {
		message << " -> " << repeated;
	} else {
		message << repeated;
	}
	return message.str();
}

inline size_t resolve_alias_target(const std::vector<galp::expression::Expression>& expressions,
                                   const size_t                                     dict_ref_column,
                                   size_t                                           source_index) {
	for (size_t depth = 0; depth <= expressions.size(); ++depth) {
		if (source_index >= expressions.size()) {
			std::ostringstream message;
			message << "DICTREF column " << dict_ref_column << " references source column " << source_index
			        << ", but the rowgroup has " << expressions.size() << " columns";
			throw std::out_of_range(message.str());
		}
		const auto* source = expressions[source_index].column;
		if (source == nullptr) {
			std::ostringstream message;
			message << "DICTREF column " << dict_ref_column << " source column " << source_index << " is missing";
			throw std::runtime_error(message.str());
		}
		if (!source->alias_of.has_value()) {
			return source_index;
		}
		source_index = *source->alias_of;
	}
	throw std::runtime_error("DICTREF dependency alias cycle detected");
}

template <typename ValueT, typename IndexT>
DictRefResolveResult resolve_typed_dict_ref(galp::codec::host::DICTREFColumn<ValueT, IndexT>& dict_ref,
	                                       const galp::execution::Column&                    source,
	                                       const size_t                                      dict_ref_column,
	                                       const size_t                                      source_column) {
	static_assert(std::is_unsigned_v<IndexT>);
	if (dict_ref.key_count > static_cast<size_t>(std::numeric_limits<IndexT>::max()) + 1U) {
		std::ostringstream message;
		message << "DICTREF column " << dict_ref_column << " has " << dict_ref.key_count
		        << " keys, which cannot be addressed by a " << (sizeof(IndexT) * 8U) << "-bit index";
		throw std::out_of_range(message.str());
	}
	if (dict_ref.key_count != 0 && dict_ref.keys.get() == nullptr) {
		std::ostringstream message;
		message << "DICTREF column " << dict_ref_column << " has a missing dictionary key segment";
		throw std::runtime_error(message.str());
	}
	const size_t source_n_values = std::visit([](const auto& column) { return column.get_n_values(); }, source.host);
	if (source_n_values != dict_ref.n_values) {
		std::ostringstream message;
		message << "DICTREF column " << dict_ref_column << " has " << dict_ref.n_values
		        << " values, but source column " << source_column << " has " << source_n_values;
		throw std::runtime_error(message.str());
	}
	if (auto resolved = try_resolve_dict_ref(dict_ref, source.host)) {
		return std::move(*resolved);
	}

	std::ostringstream message;
	message << "DICTREF column " << dict_ref_column << " expects a " << (sizeof(IndexT) * 8U)
	        << "-bit BP/FFOR/SLPATCH index payload (or a local dictionary with the same index width), but source column "
	        << source_column << " has token " << fastlanes::token_to_string(source.token);
	throw std::runtime_error(message.str());
}

inline DictRefResolveResult resolve_typed_payload(galp::execution::EncodedPayload& dict_ref_payload,
	                                              const galp::execution::Column&   source,
	                                              const size_t                     dict_ref_column,
	                                              const size_t                     source_column) {
	std::optional<DictRefResolveResult> result;
	std::visit(
	    [&](auto& dict_ref) {
		    using DictRefT = std::decay_t<decltype(dict_ref)>;
		    if constexpr (is_dict_ref_column_v<DictRefT>) {
			    result.emplace(resolve_typed_dict_ref(dict_ref, source, dict_ref_column, source_column));
		    }
	    },
	    dict_ref_payload);
	if (!result.has_value()) {
		throw std::runtime_error("internal DICTREF resolver type mismatch");
	}
	return std::move(*result);
}

} // namespace detail

inline bool has_unresolved_dict_ref(const galp::execution::EncodedPayload& payload) {
	return detail::is_dict_ref_payload(payload);
}

inline void resolve_dict_refs(std::vector<galp::expression::Expression>& expressions) {
	bool has_dict_ref = false;
	for (const auto& expression : expressions) {
		if (expression.column != nullptr && detail::is_dict_ref_payload(expression.column->host)) {
			has_dict_ref = true;
			break;
		}
	}
	if (!has_dict_ref) {
		return;
	}

	enum class VisitState : uint8_t {
		Unvisited,
		Visiting,
		Done,
	};
	std::vector<VisitState> state(expressions.size(), VisitState::Unvisited);
	std::vector<size_t>     stack;

	auto visit = [&](auto&& self, const size_t idx, const size_t root_dict_ref) -> void {
		if (idx >= expressions.size()) {
			std::ostringstream message;
			message << "DICTREF column " << root_dict_ref << " references source column " << idx
			        << ", but the rowgroup has " << expressions.size() << " columns";
			throw std::out_of_range(message.str());
		}
		if (state[idx] == VisitState::Done) {
			return;
		}
		if (state[idx] == VisitState::Visiting) {
			throw std::runtime_error(detail::dependency_cycle_message(stack, idx));
		}

		state[idx] = VisitState::Visiting;
		stack.push_back(idx);
		auto* column = expressions[idx].column;
		if (column == nullptr) {
			std::ostringstream message;
			message << "DICTREF column " << root_dict_ref << " source column " << idx << " is missing";
			throw std::runtime_error(message.str());
		}

		std::optional<size_t> dependency;
		if (column->alias_of.has_value()) {
			dependency = *column->alias_of;
		} else {
			std::visit(
			    [&](const auto& host_column) {
				    using HostColumnT = std::decay_t<decltype(host_column)>;
				    if constexpr (detail::is_dict_ref_column_v<HostColumnT>) {
					    dependency = host_column.index_column_index;
				    }
			    },
			    column->host);
		}
		if (dependency.has_value()) {
			self(self, *dependency, root_dict_ref);
		}

		column = expressions[idx].column;
		if (column != nullptr && detail::is_dict_ref_payload(column->host)) {
			size_t source_index = 0;
			std::visit(
			    [&](const auto& host_column) {
				    using HostColumnT = std::decay_t<decltype(host_column)>;
				    if constexpr (detail::is_dict_ref_column_v<HostColumnT>) {
					    source_index = host_column.index_column_index;
				    }
			    },
			    column->host);
			source_index = detail::resolve_alias_target(expressions, idx, source_index);
			const auto* source = expressions[source_index].column;
			if (source == nullptr) {
				std::ostringstream message;
				message << "DICTREF column " << idx << " source column " << source_index << " is missing";
				throw std::runtime_error(message.str());
			}
			auto resolved = detail::resolve_typed_payload(column->host, *source, idx, source_index);
			column->host  = std::move(resolved.payload);
			column->token = resolved.token;
			// Converted index buffers are owned by the resolved local dictionary.
			// Keys remain borrowed from Rowgroup::backing_storage until upload is complete.
			column->host_owned_by_backing = false;
			column->backing_base          = nullptr;
			column->backing_bytes         = 0;
			column->backing_is_pinned     = false;
		}

		stack.pop_back();
		state[idx] = VisitState::Done;
	};

	for (size_t idx = 0; idx < expressions.size(); ++idx) {
		if (expressions[idx].column != nullptr && detail::is_dict_ref_payload(expressions[idx].column->host)) {
			visit(visit, idx, idx);
		}
	}
}

} // namespace galp::execution

#endif // ENGINE_EXECUTION_DICT_REF_RESOLVER_CUH
