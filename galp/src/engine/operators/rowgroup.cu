// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/operators/rowgroup.cu
// ────────────────────────────────────────────────────────
#include "engine/materialization/metadata.cuh"
#include "engine/unpack_dispatch.cuh"
#include "cuda/launch/launch.cuh"
#include "engine/operators/rowgroup.cuh"
#include "engine/workset/append.cuh"
#include "engine/workset/upload.cuh"

namespace galp::execution {
namespace {

RowgroupData decompress_rowgroup_impl(std::vector<galp::expression::Expression>& expressions, const ExecutionConfig& cfg) {
	runtime::validate_unpack_config(cfg);
	runtime::ExecutionWorkset      workset {};
	runtime::ExecutionWorksetGuard guard(workset);
	runtime::append_expressions(workset, expressions, cfg);
	runtime::upload_workset(workset, cfg);
	runtime::run_workset(workset, 1, cfg);
	return runtime::materialize_workset(workset, expressions, cfg);
}
} // namespace

RowgroupData decompress_rowgroup(std::vector<galp::expression::Expression>& expressions, const ExecutionConfig& cfg) {
	return decompress_rowgroup_impl(expressions, cfg);
}

RowgroupData decompress_rowgroup(const std::vector<galp::expression::Expression>& expressions, const ExecutionConfig& cfg) {
	auto mutable_expressions = expressions;
	return decompress_rowgroup_impl(mutable_expressions, cfg);
}
} // namespace galp::execution
