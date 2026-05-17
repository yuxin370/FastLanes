// galp/benchmarks/include/galp_bench/nvcomp/types.cuh
#ifndef GALP_BENCH_NVCOMP_TYPES_CUH
#define GALP_BENCH_NVCOMP_TYPES_CUH

#include <string>

namespace galp::bench::nvcomp {

enum ComparisonType {
	DECOMPRESSION,
	DECOMPRESSION_QUERY,
};

ComparisonType string_to_comparison_type(const std::string& str);
std::string    comparison_type_to_string(const ComparisonType type);

enum CompressionType {
	THRUST,
	ALP,
	GALP,
	BITCOMP,
	BITCOMP_SPARSE,
	LZ4,
	ZSTD,
	DEFLATE,
	GDEFLATE,
	SNAPPY,
};

CompressionType string_to_compression_type(const std::string& str);
std::string     compression_type_to_string(const CompressionType type);

} // namespace galp::bench::nvcomp

#endif // GALP_BENCH_NVCOMP_TYPES_CUH
