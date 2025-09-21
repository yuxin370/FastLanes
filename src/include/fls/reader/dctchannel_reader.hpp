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
	static up<Table> Read(const ProcessedDCTChannel& channel, const int tag,const Connection& connection); 

};

} // namespace fastlanes

#endif // FLS_READER_DCT_CHANEEL_READER_HPP
