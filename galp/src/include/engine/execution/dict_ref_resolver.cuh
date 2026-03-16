// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/execution/dict_ref_resolver.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_DICT_REF_RESOLVER_CUH
#define ENGINE_EXECUTION_DICT_REF_RESOLVER_CUH

#include "engine/expression.cuh"
#include <stdexcept>
#include <vector>

namespace dispatch {

namespace detail {

inline size_t resolve_alias_index(const std::vector<expr::Expression>& expressions, const size_t idx) {
	if (idx >= expressions.size()) {
		throw std::out_of_range("alias index out of range");
	}
	std::vector<uint8_t> visited(expressions.size(), 0);
	size_t               cur = idx;
	for (;;) {
		if (cur >= expressions.size()) {
			throw std::out_of_range("alias target out of range");
		}
		if (visited[cur]) {
			throw std::runtime_error("alias cycle detected");
		}
		visited[cur]    = 1;
		const auto* col = expressions[cur].column;
		if (!col || !col->alias_of.has_value()) {
			return cur;
		}
		cur = *col->alias_of;
	}
}

struct DictRefResolveResult {
	dispatch::EncodedPayload payload;
	fastlanes::OperatorToken token;
};

inline DictRefResolveResult
resolve_dictref_i8_u8_from_index(const flsgpu::host::DICTREFColumn<int8_t, uint8_t>& dict_ref,
                                 const dispatch::EncodedPayload&                     index_payload) {
	DictRefResolveResult out {};
	bool                 handled = false;

	std::visit(
	    [&](auto&& index_col) {
		    using IndexHostT = std::decay_t<decltype(index_col)>;
		    if constexpr (std::is_same_v<IndexHostT, flsgpu::host::FFORColumn<int8_t>>) {
			    auto  idx_ffor = reader::columns::make_ffor_u8_from_ffor_i8(index_col);
			    auto* keys     = utils::copy_array(dict_ref.keys, dict_ref.key_count);
			    out.payload =
			        flsgpu::host::DICTFFORColumn<int8_t, uint8_t> {std::move(idx_ffor), keys, dict_ref.key_count};
			    out.token = fastlanes::OperatorToken::EXP_DICT_I08_FFOR_U08;
			    handled   = true;
		    } else if constexpr (std::is_same_v<IndexHostT, flsgpu::host::BPColumn<int8_t>>) {
			    auto  idx_ffor = reader::columns::make_ffor_u8_from_bp_i8(index_col);
			    auto* keys     = utils::copy_array(dict_ref.keys, dict_ref.key_count);
			    out.payload =
			        flsgpu::host::DICTFFORColumn<int8_t, uint8_t> {std::move(idx_ffor), keys, dict_ref.key_count};
			    out.token = fastlanes::OperatorToken::EXP_DICT_I08_FFOR_U08;
			    handled   = true;
		    } else if constexpr (std::is_same_v<IndexHostT, flsgpu::host::SLPATCHColumn<int8_t>>) {
			    auto  idx_slpatch = reader::columns::make_slpatch_u8_from_slpatch_i8(index_col);
			    auto* keys        = utils::copy_array(dict_ref.keys, dict_ref.key_count);
			    out.payload =
			        flsgpu::host::DICTSLPATCHColumn<int8_t, uint8_t> {std::move(idx_slpatch), keys, dict_ref.key_count};
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

inline void sync_expression_ops_after_resolve(std::vector<expr::Expression>& expressions) {
	for (auto& expression : expressions) {
		if (!expression.column) {
			expression.ops.clear();
			continue;
		}
		expression.ops = expr::ops_for_token(expression.column->token);
		if (expression.ops.empty()) {
			throw std::runtime_error("unsupported operator token after dict-ref resolve");
		}
	}
}

inline void resolve_dict_refs(std::vector<expr::Expression>& expressions) {
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
        if (col != nullptr && std::holds_alternative<flsgpu::host::DICTREFColumn<int8_t, uint8_t>>(col->host)) {
            const auto dict_ref = std::get<flsgpu::host::DICTREFColumn<int8_t, uint8_t>>(col->host);
            const auto src_idx = detail::resolve_alias_index(expressions, dict_ref.index_column_index);
            self(self, src_idx);
            auto* src_col = expressions[src_idx].column;
            if (!src_col) {
                throw std::runtime_error("DICTREF: referenced index column is null");
            }
            auto resolved = detail::resolve_dictref_i8_u8_from_index(dict_ref, src_col->host);
            flsgpu::host::free_column(dict_ref);
            col->host  = std::move(resolved.payload);
            col->token = resolved.token;
        }
        state[idx] = VisitState::Done;
	};

	for (size_t i = 0; i < expressions.size(); ++i) {
		resolve_one(resolve_one, i);
	}
}

} // namespace dispatch

#endif // ENGINE_EXECUTION_DICT_REF_RESOLVER_CUH
