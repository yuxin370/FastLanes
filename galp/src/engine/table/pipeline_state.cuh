// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/table/pipeline_state.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_ENGINE_TABLE_PIPELINE_STATE_CUH
#define GALP_ENGINE_TABLE_PIPELINE_STATE_CUH

#include "engine/pipeline/rowgroup_prefetch_types.cuh"
#include "engine/table/chunk_state.cuh"
#include <optional>
#include <vector>

namespace galp::runtime::detail {

size_t data_type_size(fastlanes::DataType dt);
size_t rowgroup_logical_bytes(const fastlanes::RowgroupDescriptor* rg);
size_t count_active_columns(const std::vector<galp::expression::Expression>& expressions);
size_t max_rowgroup_storage_bytes(galp::format::FlsReader& rdr, size_t start, size_t end);
size_t max_rowgroup_storage_bytes(galp::format::FlsReader& rdr, const std::vector<size_t>& rowgroups);
void   check_rowgroup_index(size_t n_rowgroups, const std::optional<size_t>& rowgroup);
void   check_rowgroup_schedule(size_t n_rowgroups, const std::vector<size_t>& rowgroups);
void   validate_table_request(const TableExecutionRequest& request);
bool   use_whole_table_pipeline(const TableExecutionRequest& request);
size_t choose_prefetch_workers(size_t requested, size_t rowgroup_count, size_t max_rowgroups_per_chunk);
size_t choose_compute_inflight_chunks(const TableDecompressionConfig& config, size_t max_rowgroups_per_chunk);
size_t choose_rowgroup_prefetch_depth(const TableDecompressionConfig& config,
                                      size_t                          max_rowgroups_per_chunk,
                                      size_t                          compute_inflight_chunks);
std::optional<size_t> env_size_value(const char* name);
bool                  env_mode_is(const char* raw, const char* a, const char* b, const char* c = "");
size_t                choose_pinned_prewarm_slots(size_t pooled_slots,
                                                  size_t total_rowgroups,
                                                  size_t max_rowgroups_per_chunk,
                                                  size_t compute_inflight_chunks,
                                                  size_t prefetch_depth_slots,
                                                  size_t active_io_workers);
size_t choose_direct_pinned_prewarm_slots(size_t pooled_slots, size_t total_rowgroups, size_t max_rowgroups_per_chunk);
RowgroupReadResult read_rowgroup(galp::format::FlsReader&                         rdr,
                                 size_t                                           rowgroup_index,
                                 const std::shared_ptr<PinnedRowgroupBufferPool>& pinned_pool = {});

} // namespace galp::runtime::detail

#endif // GALP_ENGINE_TABLE_PIPELINE_STATE_CUH
