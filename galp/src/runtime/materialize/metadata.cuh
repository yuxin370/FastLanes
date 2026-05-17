// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/runtime/materialize/metadata.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_RUNTIME_MATERIALIZE_METADATA_CUH
#define ENGINE_RUNTIME_MATERIALIZE_METADATA_CUH

#include "runtime/materialize/pinned_d2h.cuh"

namespace galp::runtime {

size_t       column_n_values(const galp::expression::Expression& expression);
size_t       resolve_alias(const std::vector<galp::expression::Expression>& expressions, size_t idx);
void         apply_aliases(RowgroupData&                                    data,
                           const std::vector<galp::expression::Expression>& expressions,
                           const ExecutionConfig&                           cfg);
void         populate_materialized_metadata(RowgroupData&                                    data,
                                            const std::vector<galp::expression::Expression>& expressions,
                                            const ExecutionConfig&                           cfg);
RowgroupData materialize_workset(ExecutionWorkset&                                workset,
                                 const std::vector<galp::expression::Expression>& expressions,
                                 const ExecutionConfig&                           cfg);
void         release_workset(ExecutionWorkset& workset, bool preserve_resources = false);

struct ExecutionWorksetGuard {
	ExecutionWorkset* workset = nullptr;
	bool              active  = true;

	explicit ExecutionWorksetGuard(ExecutionWorkset& ws);
	~ExecutionWorksetGuard() noexcept;

	void dismiss();
};

} // namespace galp::runtime

#endif // ENGINE_RUNTIME_MATERIALIZE_METADATA_CUH
