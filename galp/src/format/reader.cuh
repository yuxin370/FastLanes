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
#include "galp/sparse_vector_bundle.hpp"
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace galp::format {

// DeviceArena places independently registered backing regions on 256-byte
// boundaries. Compact batch reads preserve the same boundary between
// rowgroups so coalescing adjacent host regions cannot turn a later rowgroup
// into a misaligned device subspan.
inline constexpr size_t kCompactBatchRowgroupAlignment = 256U;

class SparseVectorReadPlan {
public:
	enum class Backend {
		kFullRowgroup,
		kSourceRanges,
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
	[[nodiscard]] size_t estimated_pread_count() const noexcept;
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
};

class FlsReader {
public:
	explicit FlsReader(const std::filesystem::path& file_path,
	                   bool                         load_column_names          = true,
	                   bool                         enable_sparse_vector_reads = true);
	explicit FlsReader(const std::filesystem::path& file_path, const FlsReaderOptions& options);

	const fastlanes::TableDescriptor* table_descriptor() const;
	size_t                            rowgroup_count() const;
	size_t                            rowgroup_storage_bytes(size_t rowgroup_idx) const;
	bool                              has_sparse_vector_bundle() const noexcept;
	bool                              is_compact_v3() const noexcept;
	size_t                            compact_batch_backing_bytes(
	                               const std::vector<size_t>& rowgroup_indices) const;
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
};

} // namespace galp::format

#endif // FLS_READER_CUH
