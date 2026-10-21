// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// fastlens/src/engine/materialization/metadata.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_RUNTIME_MATERIALIZE_METADATA_CUH
#define ENGINE_RUNTIME_MATERIALIZE_METADATA_CUH

#include "engine/materialization/pinned_d2h.cuh"

namespace fastlens::runtime {

size_t       column_n_values(const fastlens::expression::Expression& expression);
size_t       resolve_alias(const std::vector<fastlens::expression::Expression>& expressions, size_t idx);
void         apply_aliases(RowgroupData&                                    data,
                           const std::vector<fastlens::expression::Expression>& expressions,
                           const ExecutionConfig&                           cfg);
void         populate_materialized_metadata(RowgroupData&                                    data,
                                            const std::vector<fastlens::expression::Expression>& expressions,
                                            const ExecutionConfig&                           cfg);
RowgroupData materialize_workset(ExecutionWorkset&                                workset,
                                 const std::vector<fastlens::expression::Expression>& expressions,
                                 const ExecutionConfig&                           cfg);
void release_workset(ExecutionWorkset& workset,
                     bool              preserve_resources        = false,
                     bool              h2d_already_complete      = false,
                     double*           timing_event_destroy_ms   = nullptr);

struct ExecutionWorksetGuard {
	ExecutionWorkset* workset = nullptr;
	bool              active  = true;

	explicit ExecutionWorksetGuard(ExecutionWorkset& ws);
	~ExecutionWorksetGuard() noexcept;

	void dismiss();
};

} // namespace fastlens::runtime

#endif // ENGINE_RUNTIME_MATERIALIZE_METADATA_CUH
