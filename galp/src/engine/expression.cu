// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/expression.cu
// ────────────────────────────────────────────────────────
#include "engine/expression.cuh"

namespace expr {

std::vector<Expression> assemble(dispatch::Rowgroup& rowgroup) {
	std::vector<Expression> out;
	out.reserve(rowgroup.columns.size());
	for (auto& col : rowgroup.columns) {
		auto ops = ops_for_token(col.token);
		if (ops.empty()) {
			throw std::runtime_error("unsupported operator token in expression assembly");
		}
		out.push_back(Expression {&col, std::move(ops)});
	}
	return out;
}

} // namespace expr
