// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/reader.cuh
// ────────────────────────────────────────────────────────
#ifndef FLS_READER_CUH
#define FLS_READER_CUH

#include "engine/format/rowgroup_io.cuh"
#include "engine/format/schema_plan.cuh"
#include "fls/footer/table_descriptor.hpp"
#include "fls/io/file.hpp"
#include <cstddef>
#include <filesystem>
#include <memory>
#include <vector>

namespace galp::format {

namespace detail {

fastlanes::TableDescriptorHandle load_table_descriptor(fastlanes::File& file, const std::filesystem::path& file_path);
fastlanes::TableDescriptorHandle load_table_descriptor(const std::filesystem::path& file_path);

} // namespace detail

class FlsReader {
public:
	explicit FlsReader(const std::filesystem::path& file_path, bool load_column_names = true);

	const fastlanes::TableDescriptor* table_descriptor() const;
	size_t                            rowgroup_count() const;
	size_t                            rowgroup_storage_bytes(size_t rowgroup_idx) const;

	void read_rowgroup_bytes_into(size_t              rowgroup_idx,
	                              std::byte*          backing_data,
	                              size_t              backing_capacity,
	                              ZeroCopyReadTiming* timing = nullptr);

	ZeroCopyRowgroup make_zero_copy_rowgroup_from_backing(size_t                rowgroup_idx,
	                                                      std::shared_ptr<void> backing_owner,
	                                                      std::byte*            backing_data,
	                                                      size_t                backing_capacity,
	                                                      bool                  backing_is_pinned = false,
	                                                      ZeroCopyReadTiming*   timing            = nullptr);

	ZeroCopyRowgroup read_rowgroup_zero_copy_into(size_t                rowgroup_idx,
	                                              std::shared_ptr<void> backing_owner,
	                                              std::byte*            backing_data,
	                                              size_t                backing_capacity,
	                                              bool                  backing_is_pinned = false,
	                                              ZeroCopyReadTiming*   timing            = nullptr);

	ZeroCopyRowgroup      read_rowgroup_zero_copy(size_t rowgroup_idx = 0, ZeroCopyReadTiming* timing = nullptr);
	Rowgroup              materialize_zero_copy_rowgroup(ZeroCopyRowgroup zero_copy) const;
	Rowgroup              read_rowgroup_zero_copy_materialized(size_t rowgroup_idx = 0);
	Rowgroup              read_rowgroup(size_t rowgroup_idx = 0);
	std::vector<Rowgroup> read_table();

private:
	ZeroCopySchemaPlan build_shared_zero_copy_schema_plan() const;

	std::shared_ptr<fastlanes::File>                        m_file;
	std::shared_ptr<const fastlanes::TableDescriptorHandle> m_table_descriptor;
	bool                                                    m_load_column_names = true;
	std::shared_ptr<const ZeroCopySchemaPlan>               m_zero_copy_schema_plan;
};

} // namespace galp::format

#endif // FLS_READER_CUH
