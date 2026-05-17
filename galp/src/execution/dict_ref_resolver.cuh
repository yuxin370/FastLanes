// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/execution/dict_ref_resolver.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_DICT_REF_RESOLVER_CUH
#define ENGINE_EXECUTION_DICT_REF_RESOLVER_CUH

#include "core/expression.cuh"
#include <stdexcept>
#include <vector>

namespace galp::execution {

namespace detail {

inline size_t resolve_alias_index(const std::vector<galp::expression::Expression>& expressions, const size_t idx) {
	if (idx >= expressions.size()) {
		throw std::out_of_range("alias index out of range");
	}
	size_t cur = idx;
	for (size_t depth = 0; depth <= expressions.size(); ++depth) {
		if (cur >= expressions.size()) {
			throw std::out_of_range("alias target out of range");
		}
		const auto* col = expressions[cur].column;
		if (!col || !col->alias_of.has_value()) {
			return cur;
		}
		cur = *col->alias_of;
	}
	throw std::runtime_error("alias cycle detected");
}

struct DictRefResolveResult {
	galp::execution::EncodedPayload payload;
	fastlanes::OperatorToken token;
};

inline DictRefResolveResult
resolve_dictref_i8_u8_from_index(const galp::codec::host::DICTREFColumn<int8_t, uint8_t>& dict_ref,
                                 const galp::execution::EncodedPayload&                     index_payload) {
	DictRefResolveResult out {};
	bool                 handled = false;

	std::visit(
	    [&](auto&& index_col) {
		    using IndexHostT = std::decay_t<decltype(index_col)>;
		    if constexpr (std::is_same_v<IndexHostT, galp::codec::host::FFORColumn<int8_t>>) {
			    auto  idx_ffor = galp::codec::host::make_ffor_u8_from_ffor_i8(index_col);
				    auto* keys     = galp::codec::utils::copy_array(dict_ref.keys.get(), dict_ref.key_count);
			    out.payload =
			        galp::codec::host::DICTFFORColumn<int8_t, uint8_t> {std::move(idx_ffor), keys, dict_ref.key_count};
			    out.token = fastlanes::OperatorToken::EXP_DICT_I08_FFOR_U08;
			    handled   = true;
		    } else if constexpr (std::is_same_v<IndexHostT, galp::codec::host::BPColumn<int8_t>>) {
			    auto  idx_ffor = galp::codec::host::make_ffor_u8_from_bp_i8(index_col);
				    auto* keys     = galp::codec::utils::copy_array(dict_ref.keys.get(), dict_ref.key_count);
			    out.payload =
			        galp::codec::host::DICTFFORColumn<int8_t, uint8_t> {std::move(idx_ffor), keys, dict_ref.key_count};
			    out.token = fastlanes::OperatorToken::EXP_DICT_I08_FFOR_U08;
			    handled   = true;
		    } else if constexpr (std::is_same_v<IndexHostT, galp::codec::host::SLPATCHColumn<int8_t>>) {
			    auto  idx_slpatch = galp::codec::host::make_slpatch_u8_from_slpatch_i8(index_col);
				    auto* keys        = galp::codec::utils::copy_array(dict_ref.keys.get(), dict_ref.key_count);
			    out.payload =
			        galp::codec::host::DICTSLPATCHColumn<int8_t, uint8_t> {std::move(idx_slpatch), keys, dict_ref.key_count};
			    out.token = fastlanes::OperatorToken::EXP_DICT_I08_FFOR_SLPATCH_U08;
			    handled   = true;
		    }
	    },
	    index_payload);

	if (!handled) {
		throw std::runtime_error("DICTREF: index column must be BP/FFOR/SLPATCH int8");
	}
	return out;
}

} // namespace detail

inline void resolve_dict_refs(std::vector<galp::expression::Expression>& expressions) {
	bool has_dict_ref = false;
	for (const auto& expression : expressions) {
		const auto* col = expression.column;
		if (col != nullptr && std::holds_alternative<galp::codec::host::DICTREFColumn<int8_t, uint8_t>>(col->host)) {
			has_dict_ref = true;
			break;
		}
	}
	if (!has_dict_ref) {
		return;
	}

	enum class VisitState : uint8_t {
		Unvisited = 0,
		Visiting  = 1,
		Done      = 2,
	};

	std::vector<VisitState> state(expressions.size(), VisitState::Unvisited);
	auto                    resolve_one = [&](auto&& self, const size_t idx) -> void {
        if (idx >= expressions.size()) {
            throw std::out_of_range("expression index out of range");
        }
        if (state[idx] == VisitState::Done) {
            return;
        }
        if (state[idx] == VisitState::Visiting) {
            throw std::runtime_error("cycle detected while resolving dict refs");
        }

        state[idx] = VisitState::Visiting;
        auto* col  = expressions[idx].column;
        if (col != nullptr && std::holds_alternative<galp::codec::host::DICTREFColumn<int8_t, uint8_t>>(col->host)) {
            const auto& dict_ref = std::get<galp::codec::host::DICTREFColumn<int8_t, uint8_t>>(col->host);
            const auto src_idx = detail::resolve_alias_index(expressions, dict_ref.index_column_index);
            self(self, src_idx);
            auto* src_col = expressions[src_idx].column;
            if (!src_col) {
                throw std::runtime_error("DICTREF: referenced index column is null");
            }
            auto resolved = detail::resolve_dictref_i8_u8_from_index(dict_ref, src_col->host);
            col->host  = std::move(resolved.payload);
            col->token = resolved.token;
            // The resolved payload owns its buffers. Clear the borrowed-backing
            // state so free_rowgroup releases it.
            col->host_owned_by_backing = false;
            col->backing_base          = nullptr;
            col->backing_bytes         = 0;
        }
        state[idx] = VisitState::Done;
	};

	for (size_t i = 0; i < expressions.size(); ++i) {
		resolve_one(resolve_one, i);
	}
}

} // namespace galp::execution

#endif // ENGINE_EXECUTION_DICT_REF_RESOLVER_CUH
