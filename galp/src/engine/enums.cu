// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/enums.cu
// ────────────────────────────────────────────────────────
#include "engine/enums.cuh"
#include <unordered_map>

namespace enums {

Kernel string_to_kernel(const std::string& str) {
	static const std::unordered_map<std::string, Kernel> mapping = {
	    {"decompress", Kernel::Decompress},
	    {"query", Kernel::Query},
	    {"query-multi-column", Kernel::QueryMultiColumn},
	};

	auto it = mapping.find(str);
	if (it != mapping.end()) {
		return it->second;
	}

	throw std::invalid_argument("Unknown kernel type: " + str);
}

Unpacker string_to_unpacker(const std::string& str) {
	static const std::unordered_map<std::string, Unpacker> mapping = {
	    {"none", Unpacker::None},
	    {"dummy", Unpacker::Dummy},
	    {"old-fls", Unpacker::OldFls},
	    {"switch-case", Unpacker::SwitchCase},
	    {"stateless", Unpacker::Stateless},
	    {"stateless-branchless", Unpacker::StatelessBranchless},
	    {"stateful-cache", Unpacker::StatefulCache},
	    {"stateful-local-1", Unpacker::StatefulLocal1},
	    {"stateful-local-2", Unpacker::StatefulLocal2},
	    {"stateful-local-4", Unpacker::StatefulLocal4},
	    {"stateful-shared-1", Unpacker::StatefulShared1},
	    {"stateful-shared-2", Unpacker::StatefulShared2},
	    {"stateful-shared-4", Unpacker::StatefulShared4},
	    {"stateful-register-1", Unpacker::StatefulRegister1},
	    {"stateful-register-2", Unpacker::StatefulRegister2},
	    {"stateful-register-4", Unpacker::StatefulRegister4},
	    {"stateful-register-branchless-1", Unpacker::StatefulRegisterBranchless1},
	    {"stateful-register-branchless-2", Unpacker::StatefulRegisterBranchless2},
	    {"stateful-register-branchless-4", Unpacker::StatefulRegisterBranchless4},
	    {"stateful-branchless", Unpacker::StatefulBranchless},
	};

	auto it = mapping.find(str);
	if (it != mapping.end()) {
		return it->second;
	}

	throw std::invalid_argument("Unknown unpacker type: " + str);
}

Expander string_to_expander(const std::string& str) {
	static const std::unordered_map<std::string, Expander> mapping = {
	    {"none", Expander::None},
	    {"dummy", Expander::Dummy},
	    {"stateful", Expander::Stateful},
	    {"stateful-cache", Expander::StatefulCache},
	    {"stateful-shuffle", Expander::StatefulShuffle},
	    {"prefetch-stateful", Expander::PrefetchStateful},
	    {"stateful-advance", Expander::StatefulAdvance},
	    {"stateful-extended", Expander::StatefulExtended},
	    {"branchless", Expander::Branchless},
	    {"prefetch-branchless", Expander::PrefetchBranchless}};

	auto it = mapping.find(str);
	if (it != mapping.end()) {
		return it->second;
	}

	throw std::invalid_argument("Unknown expander type: " + str);
}

Patcher string_to_patcher(const std::string& str) {
	static const std::unordered_map<std::string, Patcher> mapping = {
	    {"none", Patcher::None},
	    {"dummy", Patcher::Dummy},
	    {"stateless", Patcher::Stateless},
	    {"stateful", Patcher::Stateful},
	    {"naive", Patcher::Naive},
	    {"naive-branchless", Patcher::NaiveBranchless},
	    {"prefetch-position", Patcher::PrefetchPosition},
	    {"prefetch-all", Patcher::PrefetchAll},
	    {"prefetch-all-branchless", Patcher::PrefetchAllBranchless},
	};

	auto it = mapping.find(str);
	if (it != mapping.end()) {
		return it->second;
	}

	throw std::invalid_argument("Unknown patcher type: " + str);
}

Encoding string_to_encoding(const std::string& str) {
	static const std::unordered_map<std::string, Encoding> mapping = {
	    {"alp", Encoding::ALP},
	    {"bit-packing", Encoding::BIT_PACKING},
	    {"ffor", Encoding::FFOR},
	    {"frequency", Encoding::FREQUENCY},
	    {"cross-rle", Encoding::CROSS_RLE},
	    {"dictionary", Encoding::DICTIONARY},
	    {"slpatch", Encoding::SLPATCH},
	    {"constant", Encoding::CONSTANT},
	    {"rle", Encoding::RLE},
	    {"dict-slpatch", Encoding::DICT_SLPATCH},
	};

	auto it = mapping.find(str);
	if (it != mapping.end()) {
		return it->second;
	}

	throw std::invalid_argument("Unknown encoding type: " + str);
}

std::string encoding_to_string(const Encoding type) {
	switch (type) {
	case Encoding::ALP:
		return "alp";
	case Encoding::BIT_PACKING:
		return "bit-packing";
	case Encoding::FFOR:
		return "ffor";
	case Encoding::FREQUENCY:
		return "frequency";
	case Encoding::CROSS_RLE:
		return "cross-rle";
	case Encoding::DICTIONARY:
		return "dictionary";
	case Encoding::SLPATCH:
		return "slpatch";
	case Encoding::CONSTANT:
		return "constant";
	case Encoding::RLE:
		return "rle";
	case Encoding::DICT_SLPATCH:
		return "dict-slpatch";
	default:
		throw std::invalid_argument("Could not parse encoding");
	}
}

} // namespace enums

namespace enums_nvcomp {
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
} // namespace enums_nvcomp
