// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/include/fls/reader/dctchannel_reader.hpp
// ────────────────────────────────────────────────────────
#ifndef FLS_READER_DCT_CHANEEL_READER_HPP
#define FLS_READER_DCT_CHANEEL_READER_HPP

#include "fls/common/alias.hpp" // for up, idx_t
#include "fls/jpeg/jpeg_loader.hpp"
#include "fls/std/filesystem.hpp"
#include "fls/std/vector.hpp"
#include "fls/table/rowgroup.hpp"
#include <string>

namespace fastlanes {
/*--------------------------------------------------------------------------------------------------------------------*/
class Table;
struct RowgroupDescriptorT;
class Connection;
struct ColumnDescriptorT;
/*--------------------------------------------------------------------------------------------------------------------*/

// Forward declarations for struct creation functions
up<ColumnDescriptorT> make_dct_channel_struct_column_descriptor(const string& name);
void                  ingest_dct_channel_into_struct(col_pt& dct_struct_col, const ProcessedDCTChannel& channel);

/*--------------------------------------------------------------------------------------------------------------------*\
 * DctChannelReader
\*--------------------------------------------------------------------------------------------------------------------*/
class DctChannelReader {
public:
	// Original function for single channel with tag-based storage
	static up<Table> Read(const ProcessedDCTChannel& channel, const int tag, const Connection& connection);
	static up<Table> Read(const std::vector<ChannelDCT>& channel, const Connection& connection);
	// New functions for struct-based storage
	// static up<Table> ReadMultipleChannels(const std::vector<ProcessedDCTChannel>& channels, const Connection&
	// connection); static void AddChannelToTable(Table& table, const ProcessedDCTChannel& channel, const Connection&
	// connection);
};

} // namespace fastlanes

#endif // FLS_READER_DCT_CHANEEL_READER_HPP
