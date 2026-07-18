// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/enums.cuh
// ────────────────────────────────────────────────────────
#ifndef ENUMS_CUH
#define ENUMS_CUH

#include <stdexcept>
#include <string>

namespace galp::format {

enum class Kernel {
	Decompress,
	Query,
};

enum class Unpacker {
	None,
	Dummy,
	OldFls,
	SwitchCase,
	Stateless,
	StatelessBranchless,
	StatefulCache,
	StatefulLocal1,
	StatefulLocal2,
	StatefulLocal4,
	StatefulShared1,
	StatefulShared2,
	StatefulShared4,
	StatefulRegister1,
	StatefulRegister2,
	StatefulRegister4,
	StatefulRegisterBranchless1,
	StatefulRegisterBranchless2,
	StatefulRegisterBranchless4,
	StatefulBranchless,
};

enum class Patcher {
	None,
	Dummy,
	Stateless,
	Stateful,
	Naive,
	NaiveBranchless,
	PrefetchPosition,
	PrefetchAll,
	PrefetchAllBranchless,
};

enum class Expander {
	None,
	Dummy,
	Stateful,
	StatefulCache,
	PrefetchStateful,
	StatefulShuffle,
	StatefulAdvance,
	StatefulExtended,
	Branchless,
	PrefetchBranchless
};

enum class Print {
	PrintNothing,
	PrintDebug,
	PrintDebugExit0,
};

enum class Encoding {
	ALP,
	BIT_PACKING,
	FFOR,
	DELTA,
	FREQUENCY,
	CROSS_RLE,
	DICTIONARY,
	SLPATCH,
	CONSTANT,
	RLE,
	DICT_SLPATCH,
};

Kernel      string_to_kernel(const std::string& str);
Unpacker    string_to_unpacker(const std::string& str);
Patcher     string_to_patcher(const std::string& str);
Expander    string_to_expander(const std::string& str);
Encoding    string_to_encoding(const std::string& str);
std::string encoding_to_string(const Encoding type);

} // namespace galp::format

#endif // ENUMS_CUH
