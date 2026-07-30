#pragma once

#include "fls/footer/table_descriptor_generated.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace galp::metadata {

struct InternedObjectStats {
	uint64_t references = 0;
	uint64_t emitted    = 0;
};

struct InternedDescriptorStats {
	InternedObjectStats strings;
	InternedObjectStats rpns;
	InternedObjectStats expression_results;
	InternedObjectStats expression_vectors;
	InternedObjectStats segment_descriptors;
	InternedObjectStats segment_vectors;
	InternedObjectStats binary_values;
	InternedObjectStats decimals;
	InternedObjectStats child_vectors;
	uint64_t            columns   = 0;
	uint64_t            rowgroups = 0;
};

struct InternedDescriptorResult {
	std::vector<uint8_t>    bytes;
	InternedDescriptorStats stats;
};

// Re-packs the current TableDescriptor schema as a FlatBuffer DAG. Exact repeated
// leaf tables and vectors are shared; all scalar values and null-vs-empty states
// remain unchanged. No FLS payload bytes are interpreted or rewritten.
InternedDescriptorResult pack_interned_table_descriptor(const fastlanes::TableDescriptor& table,
                                                        std::size_t                       original_descriptor_size);

} // namespace galp::metadata
