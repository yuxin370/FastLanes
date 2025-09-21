// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/include/fls/reader/dctchannel_reader.hpp
// ────────────────────────────────────────────────────────
#ifndef FLS_READER_DCT_CHANEEL_READER_HPP
#define FLS_READER_DCT_CHANEEL_READER_HPP

#include "fls/common/alias.hpp" // for up, idx_t
#include "fls/std/filesystem.hpp"
#include "fls/jpeg/jpeg_loader.hpp"

namespace fastlanes {
/*--------------------------------------------------------------------------------------------------------------------*/
class Table;
struct RowgroupDescriptorT;
class Connection;
/*--------------------------------------------------------------------------------------------------------------------*/


/*--------------------------------------------------------------------------------------------------------------------*\
 * DctChannelReader
\*--------------------------------------------------------------------------------------------------------------------*/
class DctChannelReader {
public:
	// struct ProcessedDCTCTable {
	// 	// std::vector<ZigzagBlock> raw_blocks;                      // 原始 zigzag 编码的块
	// 	up<Table> raw_index_tb;                   // 原始编码块的index
	// 	up<Table> raw_blocks_tb;                  // 原始编码块，由于含0过少而单独存
	// 	up<Table> nonzero_values_tb;              // 仅包含非零元素
	// 	up<Table> mixed_run_encoding_pattern_tb;  // zero/nonzero 编码序列
	// };

	static up<Table> Read(const ProcessedDCTChannel& channel, const int tag,const Connection& connection); 

};

} // namespace fastlanes

#endif // FLS_READER_DCT_CHANEEL_READER_HPP
