// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/expression.cu
// ────────────────────────────────────────────────────────
#include "engine/expression.cuh"
#include <stdexcept>

namespace expr {

std::vector<Expression> assemble(dispatch::Rowgroup& rowgroup) {
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

} // namespace expr
