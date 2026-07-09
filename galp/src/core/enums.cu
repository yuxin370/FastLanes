// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/core/enums.cu
// ────────────────────────────────────────────────────────
#include "core/enums.cuh"
#include <unordered_map>

namespace galp::format {

Kernel string_to_kernel(const std::string& str) {
	static const std::unordered_map<std::string, Kernel> mapping = {
	    {"decompress", Kernel::Decompress},
	    {"query", Kernel::Query},
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

} // namespace galp::format
