// galp/benchmark/nvcomp/types.cu
#include "galp_bench/nvcomp/types.cuh"

#include <stdexcept>
#include <unordered_map>

namespace galp::bench::nvcomp {

ComparisonType string_to_comparison_type(const std::string& str) {
	static const std::unordered_map<std::string, ComparisonType> mapping = {
	    {"decompression", ComparisonType::DECOMPRESSION},
	    {"decompression_query", ComparisonType::DECOMPRESSION_QUERY},
	};

	auto it = mapping.find(str);
	if (it != mapping.end()) {
		return it->second;
	}

	throw std::invalid_argument("Unknown comparison type: " + str);
}

std::string comparison_type_to_string(const ComparisonType type) {
	switch (type) {
	case ComparisonType::DECOMPRESSION:
		return "decompression";
	case ComparisonType::DECOMPRESSION_QUERY:
		return "decompression_query";
	default:
		throw std::invalid_argument("Could not parse comparison type");
	}
}

CompressionType string_to_compression_type(const std::string& str) {
	static const std::unordered_map<std::string, CompressionType> mapping = {
	    {"Thrust", CompressionType::THRUST},
	    {"ALP", CompressionType::ALP},
	    {"GALP", CompressionType::GALP},
	    {"Bitcomp", CompressionType::BITCOMP},
	    {"BitcompSparse", CompressionType::BITCOMP_SPARSE},
	    {"LZ4", CompressionType::LZ4},
	    {"zstd", CompressionType::ZSTD},
	    {"Deflate", CompressionType::DEFLATE},
	    {"GDeflate", CompressionType::GDEFLATE},
	    {"Snappy", CompressionType::SNAPPY},
	};

	auto it = mapping.find(str);
	if (it != mapping.end()) {
		return it->second;
	}

	throw std::invalid_argument("Unknown compression type: " + str);
}

std::string compression_type_to_string(const CompressionType type) {
	switch (type) {
	case CompressionType::THRUST:
		return "Thrust";
	case CompressionType::ALP:
		return "ALP";
	case CompressionType::GALP:
		return "GALP";
	case CompressionType::BITCOMP:
		return "Bitcomp";
	case CompressionType::BITCOMP_SPARSE:
		return "BitcompSparse";
	case CompressionType::LZ4:
		return "LZ4";
	case CompressionType::ZSTD:
		return "zstd";
	case CompressionType::DEFLATE:
		return "Deflate";
	case CompressionType::GDEFLATE:
		return "GDeflate";
	case CompressionType::SNAPPY:
		return "Snappy";
	default:
		throw std::invalid_argument("Could not parse decompresor");
	}
}

} // namespace galp::bench::nvcomp
