#ifndef GALP_OPTIONS_HPP
#define GALP_OPTIONS_HPP

#include <cstddef>

namespace galp {

enum class DataType {
	I8,
	I16,
};

enum class TableDecompressionScope {
	PerRowgroup,
	WholeTable,
};

struct AdvancedOptions {
	bool   enable_rowgroup_prefetch    = true;
	size_t prefetch_depth              = 4;
	size_t prefetch_workers            = 0;
	size_t max_prefetch_storage_bytes  = 0;
	size_t streaming_target_work_items = 1u << 18;
	size_t streaming_target_rowgroups  = 1;
};

struct DecompressOptions {
	bool                    write_output = true;
	TableDecompressionScope scope        = TableDecompressionScope::PerRowgroup;
	AdvancedOptions         advanced     = {};
	// Compatibility fields. Prefer `advanced` for new code; these fields are
	// still honored so existing CLI/scripts keep their behavior.
	bool   enable_rowgroup_prefetch    = true;
	size_t prefetch_depth              = 4;
	size_t prefetch_workers            = 0;
	size_t max_prefetch_storage_bytes  = 0;
	size_t streaming_target_work_items = 1u << 18;
	size_t streaming_target_rowgroups  = 1;
};

} // namespace galp

#endif // GALP_OPTIONS_HPP
