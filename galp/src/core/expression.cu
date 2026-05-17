// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/core/expression.cu
// ────────────────────────────────────────────────────────
#include "core/expression.cuh"
#include <stdexcept>

namespace galp::expression {

std::vector<Expression> assemble(galp::execution::Rowgroup& rowgroup) {
	std::vector<Expression> out;
	out.reserve(rowgroup.columns.size());
	for (auto& col : rowgroup.columns) {
		if (!is_supported_token(col.token)) {
			throw std::runtime_error("unsupported operator token in expression assembly");
		}
		out.push_back(Expression {&col});
	}
	return out;
}

} // namespace galp::expression
