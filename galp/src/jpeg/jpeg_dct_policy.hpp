#ifndef GALP_JPEG_DCT_POLICY_HPP
#define GALP_JPEG_DCT_POLICY_HPP

#include "jpeg/jpeg_dct_plan_types.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace galp::jpeg::detail {

inline constexpr double   kMaxSelectedVectorRatioForPushdown = 0.75;
inline constexpr size_t   kMinSavedVectorsForPushdown        = 4;
inline constexpr unsigned kJpegDctDeviceUnpackNVectors       = 1;

// The generic sparse reader reconstructs a logical rowgroup after issuing its
// physical reads.  Automatic storage selection therefore needs to price both
// I/O fragmentation and that extra memory traffic; vector ratio alone is not a
// useful proxy for either cost.
struct JpegDctSparseStorageCost {
	size_t full_storage_bytes             = 0U;
	size_t sparse_storage_bytes           = 0U;
	size_t sparse_pread_count             = 0U;
	bool   requires_logical_materialization = true;
};

enum class JpegDctSparseStoragePolicyReason {
	kPredictedFaster,
	kNoByteSavings,
	kNoPhysicalReads,
	kFragmentationDominates,
};

struct JpegDctSparseStoragePolicyResult {
	bool                             use_sparse_read       = false;
	JpegDctSparseStoragePolicyReason reason                = JpegDctSparseStoragePolicyReason::kNoByteSavings;
	double                           full_estimated_ns     = 0.0;
	double                           sparse_estimated_ns   = 0.0;
};

// Unified automatic comparison for the three observable read strategies. The
// model prices planning work, pread fragmentation, physical bytes, decoded
// bytes and resident workset bytes. A strategy that exceeds the workset budget
// is never selected while a budget-fitting candidate exists.
struct JpegDctAdaptiveReadCost {
	size_t full_storage_bytes              = 0U;
	size_t run_interval_storage_bytes      = 0U;
	size_t run_interval_pread_count        = 0U;
	size_t selected_vector_count           = 0U;
	size_t full_vector_count               = 0U;
	size_t decoded_bytes_per_vector        = 0U;
	size_t decode_workset_capacity_bytes   = 0U;
	bool   selected_decode_supported       = false;
	bool   run_interval_supported          = false;
	bool   run_requires_full_materialization = true;
};

struct JpegDctAdaptiveReadPolicyResult {
	JpegDctReadStrategy strategy = JpegDctReadStrategy::kFullRowgroup;
	double run_interval_estimated_ns = 0.0;
	double bitmap_estimated_ns       = 0.0;
	double full_rowgroup_estimated_ns = 0.0;
	size_t selected_resident_bytes = 0U;
	size_t full_resident_bytes     = 0U;
	bool   selected_fits_memory    = false;
	bool   full_fits_memory        = false;
};

JpegDctRuntimePolicyResult
choose_jpeg_dct_runtime_policy(size_t selected_vector_count, size_t full_vector_count, bool selected_chunks_fit);

JpegDctSparseStoragePolicyResult choose_jpeg_dct_sparse_storage_policy(const JpegDctSparseStorageCost& cost);
JpegDctAdaptiveReadPolicyResult choose_jpeg_dct_adaptive_read_policy(const JpegDctAdaptiveReadCost& cost);

std::vector<uint8_t>             normalize_coefficient_selection(const JpegDctCoefficientSelection& selection);
bool                             selects_all_coefficients(const std::vector<uint8_t>& selected_coefficients);
JpegDctCoefficientSelectionShape classify_coefficient_selection(const std::vector<uint8_t>& selected_coefficients);

std::vector<uint32_t>                selected_decode_vectors(const std::vector<JpegDctDeviceGatherItem>& items,
                                                             size_t                                      rowgroup_n_vecs,
                                                             unsigned                                    unpack_n_vectors_cfg);
std::vector<JpegDctDeviceGatherItem> remap_items_to_selected_vectors(const std::vector<JpegDctDeviceGatherItem>& items,
                                                                     const std::vector<uint32_t>& selected_vectors,
                                                                     unsigned                     unpack_n_vectors_cfg);
std::vector<JpegDctDeviceProjectionItem>
remap_projection_items_to_selected_vectors(const std::vector<JpegDctDeviceProjectionItem>& items,
                                           const std::vector<uint32_t>&                    selected_vectors,
                                           unsigned                                        unpack_n_vectors_cfg);
std::vector<JpegDctDeviceFixedTransformItem>
         remap_fixed_transform_items_to_selected_vectors(const std::vector<JpegDctDeviceFixedTransformItem>& items,
                                                         const std::vector<uint32_t>&                        selected_vectors,
                                                         unsigned unpack_n_vectors_cfg);
bool     selected_decode_chunks_fit(const std::vector<uint32_t>& selected_vectors,
                                    size_t                       rowgroup_n_vecs,
                                    unsigned                     unpack_n_vectors_cfg);
size_t   selected_decode_vector_count(const std::vector<uint32_t>& selected_vectors,
                                      size_t                       rowgroup_n_vecs,
                                      unsigned                     unpack_n_vectors_cfg);
std::vector<uint32_t> expand_selected_decode_chunks(const std::vector<uint32_t>& selected_vectors,
                                                    size_t                       rowgroup_n_vecs,
                                                    unsigned                     unpack_n_vectors_cfg);
std::vector<uint32_t> build_logical_to_compact_vector_remap(const std::vector<uint32_t>& selected_vectors,
                                                            size_t                       logical_rowgroup_n_vecs,
                                                            unsigned                     unpack_n_vectors_cfg);
unsigned constrain_jpeg_dct_batch_unpack_n_vectors(unsigned                         preferred_unpack_n_vectors,
                                                   const JpegDctDeviceRowgroupPlan& rowgroup_plan);

struct JpegDctDeviceRowgroupPrefetchPlan {
	bool                enabled                             = false;
	bool                disabled_by_config                  = false;
	bool                disabled_by_all_hits                = false;
	bool                disabled_by_small_batch_count       = false;
	bool                disabled_by_selected_vector_miss    = false;
	size_t              initial_cache_hit_rowgroup_count    = 0;
	size_t              candidate_rowgroup_count            = 0;
	size_t              selected_vector_miss_rowgroup_count = 0;
	size_t              skipped_repeated_rowgroup_count     = 0;
	std::vector<size_t> rowgroup_indices;
	std::vector<std::vector<uint32_t>> selected_vectors;
	std::vector<bool>   use_prefetch_for_position;
	std::vector<bool>   initial_cache_hit_for_position;
	std::vector<bool>   skipped_repeated_for_position;
};

JpegDctDeviceRowgroupPrefetchPlan
plan_jpeg_dct_rowgroup_prefetch_from_hits(const JpegDctDeviceShardPlan&              shard,
                                          const std::vector<bool>&                   cache_hit_by_position,
                                          const JpegDctDeviceRowgroupPrefetchConfig& config,
                                          size_t                                     effective_decode_batch_rowgroups,
                                          size_t                                     decoded_cache_capacity_bytes = 0);

} // namespace galp::jpeg::detail

#endif // GALP_JPEG_DCT_POLICY_HPP
