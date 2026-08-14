// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/reader.cuh
// ────────────────────────────────────────────────────────
#ifndef FLS_READER_CUH
#define FLS_READER_CUH

#include "format/rowgroup_io.cuh"
#include "format/schema_plan.cuh"
#include "fls/footer/table_descriptor.hpp"
#include "fls/io/file.hpp"
#include "galp/sparse_read_recipe.hpp"
#include "galp/sparse_vector_bundle.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace galp::format {

struct SparseReaderInitializationStats {
	double descriptor_open_ms          = 0.0;
	double zero_copy_schema_plan_build_ms = 0.0;
	double source_validation_ms        = 0.0;
	double sparse_access_index_build_ms = 0.0;
	double sparse_recipe_load_ms       = 0.0;
	double sparse_recipe_validation_ms = 0.0;
	double sparse_recipe_rehydrate_ms  = 0.0;
	double sparse_recipe_rehydrate_service_ms = 0.0;
	size_t sparse_recipe_rehydrate_workers = 0U;
	size_t sparse_recipe_source_metadata_bytes = 0U;
	size_t sparse_recipe_source_metadata_pread_count = 0U;
	size_t sparse_recipe_sidecar_bytes = 0U;
	size_t sparse_recipe_record_count  = 0U;
	bool   sparse_recipe_loaded        = false;
};

// DeviceArena places independently registered backing regions on 256-byte
// boundaries. Compact batch reads preserve the same boundary between
// rowgroups so coalescing adjacent host regions cannot turn a later rowgroup
// into a misaligned device subspan.
inline constexpr size_t kCompactBatchRowgroupAlignment = 256U;

class SparseVectorReadPlan {
public:
	enum class SubmissionBackend {
		kSynchronousPread,
		kIoUring,
	};

	enum class Backend {
		kFullRowgroup,
		kSourceRanges,
		kBoundedSourceRanges,
		kBundleRuns,
		kBundleEnvelope,
		kBundlePacked,
	};

	struct Impl;
	SparseVectorReadPlan() noexcept = default;
	[[nodiscard]] bool empty() const noexcept { return !impl_; }
	[[nodiscard]] size_t rowgroup_index() const noexcept;
	[[nodiscard]] size_t selected_vector_count() const noexcept;
	[[nodiscard]] size_t storage_bytes() const noexcept;
	[[nodiscard]] size_t full_storage_bytes() const noexcept;
	[[nodiscard]] size_t merged_gap_bytes() const noexcept;
	[[nodiscard]] size_t estimated_pread_count() const noexcept;
	[[nodiscard]] bool recipe_hit() const noexcept;
	[[nodiscard]] double recipe_lookup_ms() const noexcept;
	[[nodiscard]] double recipe_rehydrate_ms() const noexcept;
	[[nodiscard]] size_t recipe_source_metadata_bytes() const noexcept;
	[[nodiscard]] size_t recipe_source_metadata_pread_count() const noexcept;
	[[nodiscard]] double endpoint_resolution_ms() const noexcept;
	[[nodiscard]] double range_gather_ms() const noexcept;
	[[nodiscard]] double range_sort_coalesce_ms() const noexcept;
	[[nodiscard]] std::vector<SparseReadRange> exact_source_ranges() const;
	[[nodiscard]] std::vector<SparseReadRange> physical_source_ranges() const;
	[[nodiscard]] std::vector<SparseReadRange> merged_hole_ranges() const;
	[[nodiscard]] SparseVectorReadPlan with_bounded_coalescing(
	    const SparseReadBoundedRowgroupResult& result,
	    SubmissionBackend submission_backend = SubmissionBackend::kSynchronousPread,
	    uint32_t io_uring_queue_depth = 0U) const;
	[[nodiscard]] SubmissionBackend submission_backend() const noexcept;
	[[nodiscard]] uint32_t io_uring_queue_depth() const noexcept;
	[[nodiscard]] Backend backend() const noexcept;
	[[nodiscard]] bool uses_sparse_read() const noexcept;
	[[nodiscard]] bool uses_packed_device_scatter() const noexcept;

private:
	friend class FlsReader;
	explicit SparseVectorReadPlan(std::shared_ptr<const Impl> impl) noexcept : impl_(std::move(impl)) {}
	std::shared_ptr<const Impl> impl_;
};

namespace detail {

struct SparseDatasetAccessIndex;
struct SparseReadRecipeIndex;

fastlanes::TableDescriptorHandle load_table_descriptor(fastlanes::File&              file,
	                                                    const std::filesystem::path& file_path);
fastlanes::TableDescriptorHandle load_table_descriptor(const std::filesystem::path& file_path);
bool validate_sparse_column_operators(const fastlanes::ColumnDescriptor& column, std::string* reason = nullptr);
struct SparseVectorBundleIndex;

} // namespace detail

class CompactDescriptorV3;

struct FlsReaderOptions {
	bool load_column_names                 = true;
	bool enable_sparse_vector_reads         = true;
	bool build_shared_zero_copy_schema_plan = true;
	// A recipe sidecar is read-only at runtime. Missing, invalid, or non-matching
	// records fall back to the synchronous source-index path; readers never grow
	// or rewrite this file while serving reads.
	std::filesystem::path sparse_read_recipe_path;
	uint64_t              sparse_read_recipe_source_fingerprint = 0U;
	// Zero preserves the rowgroup-lazy synchronous fallback. A positive value
	// prehydrates immutable recipe access records with a bounded worker pool.
	size_t                sparse_read_recipe_rehydrate_workers = 0U;
};

// Immutable descriptor/schema/index state shared by short-lived payload
// readers.  Keeping this state separate lets the training reader bound open
// payload readers without repeatedly mmap'ing and validating compact-v3
// descriptors when random sampling revisits an evicted shard.
class FlsReaderStaticMetadata {
public:
	FlsReaderStaticMetadata(const FlsReaderStaticMetadata&)            = delete;
	FlsReaderStaticMetadata& operator=(const FlsReaderStaticMetadata&) = delete;
	~FlsReaderStaticMetadata();

	[[nodiscard]] size_t retained_bytes() const noexcept;

private:
	struct Impl;
	explicit FlsReaderStaticMetadata(std::shared_ptr<const Impl> impl) noexcept;

	std::shared_ptr<const Impl> impl_;
	friend class FlsReader;
};

class FlsReader {
public:
	explicit FlsReader(const std::filesystem::path& file_path,
	                   bool                         load_column_names          = true,
	                   bool                         enable_sparse_vector_reads = true);
	explicit FlsReader(const std::filesystem::path& file_path, const FlsReaderOptions& options);
	FlsReader(const std::filesystem::path&                 file_path,
	          const FlsReaderOptions&                      options,
	          std::shared_ptr<const FlsReaderStaticMetadata> static_metadata);

	const fastlanes::TableDescriptor* table_descriptor() const;
	size_t                            rowgroup_count() const;
	size_t                            rowgroup_storage_bytes(size_t rowgroup_idx) const;
	bool                              has_sparse_vector_bundle() const noexcept;
	bool                              is_compact_v3() const noexcept;
	[[nodiscard]] const SparseReaderInitializationStats& sparse_initialization_stats() const noexcept;
	[[nodiscard]] std::shared_ptr<const FlsReaderStaticMetadata> share_static_metadata() const noexcept;
	[[nodiscard]] size_t static_metadata_bytes() const noexcept;
	size_t                            compact_batch_backing_bytes(
	                               const std::vector<size_t>& rowgroup_indices) const;
	// Return the largest whole-image Compact-v3 backing requirements. Each
	// value includes rowgroup alignment and is padded at the image boundary, so
	// sums remain a safe upper bound when several images share one shard arena.
	// Descriptor pages faulted by the full scan are released before returning;
	// the validated mmap and pointer identities remain intact.
	[[nodiscard]] std::vector<size_t> compact_largest_image_backing_bytes(size_t limit) const;
	bool                              sparse_vector_read_supported(size_t rowgroup_idx,
	                                                              std::string* reason = nullptr) const;
	SparseVectorReadPlan compile_sparse_vector_read_plan(size_t                       rowgroup_idx,
	                                                     const std::vector<uint32_t>& selected_vectors,
	                                                     bool packed_device_scatter = false) const;

	void read_rowgroup_bytes_into(size_t              rowgroup_idx,
	                              std::byte*          backing_data,
	                              size_t              backing_capacity,
	                              ZeroCopyReadTiming* timing = nullptr);
	void read_rowgroup_bytes_selected_vectors_into(size_t                       rowgroup_idx,
	                                               const std::vector<uint32_t>& selected_vectors,
	                                               std::byte*                   backing_data,
	                                               size_t                       backing_capacity,
	                                               ZeroCopyReadTiming*          timing = nullptr);
	void read_rowgroup_bytes_selected_columns_into(size_t                      rowgroup_idx,
	                                               const std::vector<uint8_t>& selected_columns,
	                                               std::byte*                  backing_data,
	                                               size_t                      backing_capacity,
	                                               ZeroCopyReadTiming*         timing = nullptr);

	ZeroCopyRowgroup make_zero_copy_rowgroup_from_backing(size_t                rowgroup_idx,
	                                                      std::shared_ptr<void> backing_owner,
	                                                      std::byte*            backing_data,
	                                                      size_t                backing_capacity,
	                                                      bool                  backing_is_pinned = false,
	                                                      ZeroCopyReadTiming*   timing            = nullptr,
	                                                      bool                  prefer_compact_direct_geometry = false);

	ZeroCopyRowgroup read_rowgroup_zero_copy_into(size_t                rowgroup_idx,
	                                              std::shared_ptr<void> backing_owner,
	                                              std::byte*            backing_data,
	                                              size_t                backing_capacity,
	                                              bool                  backing_is_pinned = false,
	                                              ZeroCopyReadTiming*   timing            = nullptr);
	ZeroCopyRowgroup read_rowgroup_zero_copy_selected_vectors_into(
	    size_t                       rowgroup_idx,
	    const std::vector<uint32_t>& selected_vectors,
	    std::shared_ptr<void>        backing_owner,
	    std::byte*                   backing_data,
	    size_t                       backing_capacity,
	    bool                         backing_is_pinned = false,
	    ZeroCopyReadTiming*          timing            = nullptr);
	ZeroCopyRowgroup read_rowgroup_zero_copy_selected_columns_into(
	    size_t                      rowgroup_idx,
	    const std::vector<uint8_t>& selected_columns,
	    std::shared_ptr<void>       backing_owner,
	    std::byte*                  backing_data,
	    size_t                      backing_capacity,
	    bool                        backing_is_pinned = false,
	    ZeroCopyReadTiming*         timing            = nullptr);

	ZeroCopyRowgroup read_rowgroup_zero_copy(size_t rowgroup_idx = 0, ZeroCopyReadTiming* timing = nullptr);
	ZeroCopyRowgroup read_rowgroup_zero_copy_selected_vectors(size_t                       rowgroup_idx,
	                                                          const std::vector<uint32_t>& selected_vectors,
	                                                          ZeroCopyReadTiming*          timing = nullptr);
	ZeroCopyRowgroup read_rowgroup_zero_copy_selected_columns(size_t                      rowgroup_idx,
	                                                          const std::vector<uint8_t>& selected_columns,
	                                                          ZeroCopyReadTiming*         timing = nullptr);
	// Compact-v3 payloads are physically ordered by rowgroup. Sort the requested
	// rowgroups by payload offset, join adjacent payloads into preadv runs, and
	// preserve the caller's rowgroup order in the returned views.
	std::vector<ZeroCopyRowgroup>
	read_compact_rowgroups_zero_copy_scatter(const std::vector<size_t>&       rowgroup_indices,
	                                         std::vector<ZeroCopyReadTiming>* timings = nullptr,
	                                         size_t                           view_workers = 1U);
	std::vector<ZeroCopyRowgroup> read_compact_rowgroups_zero_copy_scatter_into(
	    const std::vector<size_t>&       rowgroup_indices,
	    std::shared_ptr<void>            backing_owner,
	    std::byte*                       backing_data,
	    size_t                           backing_capacity,
	    bool                             backing_is_pinned = false,
	    std::vector<ZeroCopyReadTiming>* timings           = nullptr,
	    size_t                           view_workers      = 1U);
	std::vector<ZeroCopyRowgroup>
	read_compact_rowgroups_zero_copy_selected_columns(const std::vector<size_t>&       rowgroup_indices,
	                                                  const std::vector<uint8_t>&      selected_columns,
	                                                  std::vector<ZeroCopyReadTiming>* timings = nullptr,
	                                                  size_t                           view_workers = 1U);
	std::vector<ZeroCopyRowgroup> read_compact_rowgroups_zero_copy_selected_columns_into(
	    const std::vector<size_t>&       rowgroup_indices,
	    const std::vector<uint8_t>&      selected_columns,
	    std::shared_ptr<void>            backing_owner,
	    std::byte*                       backing_data,
	    size_t                           backing_capacity,
	    bool                             backing_is_pinned = false,
	    std::vector<ZeroCopyReadTiming>* timings           = nullptr,
	    size_t                           view_workers      = 1U);
	ZeroCopyRowgroup      read_rowgroup_zero_copy_selected_vectors_packed(size_t                       rowgroup_idx,
	                                                                      const std::vector<uint32_t>& selected_vectors,
	                                                                      ZeroCopyReadTiming*          timing = nullptr);
	ZeroCopyRowgroup      read_rowgroup_zero_copy_compiled(const SparseVectorReadPlan& plan,
	                                                       ZeroCopyReadTiming*         timing = nullptr);
	Rowgroup              materialize_zero_copy_rowgroup(ZeroCopyRowgroup zero_copy) const;
	Rowgroup              read_rowgroup_zero_copy_materialized(size_t rowgroup_idx = 0);
	Rowgroup              read_rowgroup(size_t rowgroup_idx = 0);
	std::vector<Rowgroup> read_table();

private:
	ZeroCopySchemaPlan build_shared_zero_copy_schema_plan() const;

	std::shared_ptr<fastlanes::File>                        m_file;
	std::shared_ptr<const fastlanes::TableDescriptorHandle> m_table_descriptor;
	std::shared_ptr<CompactDescriptorV3>                    m_compact_descriptor;
	std::shared_ptr<const detail::SparseVectorBundleIndex>  m_sparse_vector_bundle;
	std::shared_ptr<const detail::SparseDatasetAccessIndex> m_sparse_access_index;
	bool                                                    m_load_column_names = true;
	std::shared_ptr<const ZeroCopySchemaPlan>               m_zero_copy_schema_plan;
	std::shared_ptr<const uint8_t>                          m_sparse_plan_owner = std::make_shared<const uint8_t>(0U);
	std::shared_ptr<const detail::SparseReadRecipeIndex>     m_sparse_read_recipe;
	SparseReaderInitializationStats                         m_sparse_initialization_stats;
	std::shared_ptr<const FlsReaderStaticMetadata>          m_static_metadata;
};

} // namespace galp::format

#endif // FLS_READER_CUH
