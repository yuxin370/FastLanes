// Experiment-only CPU bridge. Binary stdin/out is an IPC buffer, not a dataset format.
#include "fls/connection.hpp"
#include "fls/json/nlohmann/json.hpp"
#include "fls/table/memory_table.hpp"
#include "format/compact_descriptor_v3.hpp"
#include "galp/jpeg_dct_storage.hpp"
#include "jpeg/jpeg_dct_decode.hpp"
#include "jpeg/jpeg_dct_metadata.hpp"
#include "jpeg/jpeg_dct_order.hpp"
#include "jpeg/jpeg_dct_shard_reader.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sys/resource.h>
#include <unistd.h>
using namespace galp::jpeg;
namespace d = galp::jpeg::detail;
template <class T>
void get(T* p, size_t n) {
	if (!std::cin.read(reinterpret_cast<char*>(p), n * sizeof(T)))
		throw std::runtime_error("truncated IPC input");
}
template <class T>
void put(const T* p, size_t n) {
	std::cout.write(reinterpret_cast<const char*>(p), n * sizeof(T));
}
// Same spatial-major table builder and block-group-aligned 128-vector
// rowgroup contract as the native B6 writer; target component shapes are supplied by the input profile.
void encode_block_major(std::vector<d::DecodedImage> images, const std::string& base, size_t threads) {
	for (const auto& image : images)
		for (const auto& qt : image.metadata.quant_tables)
			if (!std::all_of(qt.values.begin(), qt.values.end(), [](auto value) { return value == 1; }))
				throw std::runtime_error("DCTNet Q100 block-major profile requires unit quantization tables");
	auto table = d::make_dataset_table(std::move(images), {}, nullptr, JpegDctPhysicalLayout::kSpatialMajorImageMinor);
	std::vector<uint64_t> groups;
	uint64_t              rows = 0;
	for (auto& group : table.metadata.block_group_index) {
		if (rows && rows + group.row_count > 128 * 1024) {
			groups.push_back(rows);
			rows = 0;
		}
		group.fls_rowgroup_index    = static_cast<uint32_t>(groups.size());
		group.row_start_in_rowgroup = static_cast<uint32_t>(rows);
		rows += group.row_count;
	}
	if (rows)
		groups.push_back(rows);
	std::array<fastlanes::MemoryColumn, 64> columns;
	for (size_t c = 0; c < 64; ++c) {
		columns[c].name = "dct_zz_" + (c < 10 ? std::string("0") : std::string()) + std::to_string(c);
		columns[c].data = std::span<const int16_t>(table.columns[c]);
	}
	fastlanes::MemoryTable        mt {std::span<const fastlanes::MemoryColumn>(columns)};
	fastlanes::MemoryTableOptions mo;
	mo.n_vectors_per_rowgroup = 128;
	mo.rowgroup_n_tuples      = std::span<const fastlanes::n_t>(groups);
	fastlanes::Connection conn;
	fastlanes::load_memory_table(conn, mt, mo);
	conn.inline_footer();
	fastlanes::EncodingOptions options;
	options.worker_count = threads;
	auto start           = std::chrono::steady_clock::now();
	conn.to_fls(base + ".fls", options);
	write_jpeg_dct_metadata(table.metadata, base + ".meta.bin", JpegDctMetadataWriterOptions {});
	const auto&    stats  = conn.get_last_encoding_stats();
	nlohmann::json result = {
	    {"images", table.metadata.images.size()},
	    {"rowgroups", groups.size()},
	    {"physical_layout", "dct-major/spatial-major-image-minor"},
	    {"rowgroup_vectors", 128},
	    {"fls_bytes", std::filesystem::file_size(base + ".fls")},
	    {"encoding_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()},
	    {"effective_encoding_threads", stats.effective_worker_count}};
	std::cout << result.dump() << '\n';
}
void read_image(const std::filesystem::path&  fls,
                const JpegDctDatasetMetadata& md,
                size_t                        index,
                d::JpegDctShardCpuReader&     reader) {
	size_t                                     idx  = index;
	auto&                                      im   = md.images.at(idx);
	auto                                       desc = galp::format::CompactDescriptorV3::Open(fls);
	auto                                       rec  = desc.image(idx);
	std::vector<d::JpegDctMaterializeBlockRef> refs;
	for (uint32_t c = 0; c < rec.component_count; c++) {
		auto cmp = desc.component(rec.first_component + c);
		for (uint32_t y = 0; y < cmp.height_in_blocks; y++)
			for (uint32_t x = 0; x < cmp.width_in_blocks; x++) {
				uint64_t rank =
				    cmp.row_offset + d::block_order_rank(cmp.width_in_blocks,
				                                         cmp.height_in_blocks,
				                                         x,
				                                         y,
				                                         static_cast<JpegDctSpatialOrder>(rec.spatial_order));
				d::JpegDctMaterializeBlockRef ref;
				ref.semantic_slot_id          = cmp.semantic_slot_id;
				ref.block_x                   = x;
				ref.block_y                   = y;
				ref.row.present               = true;
				ref.row.fls_rowgroup_index    = rec.first_rowgroup + static_cast<uint32_t>(rank / 1024);
				ref.row.row_start_in_rowgroup = rank % 1024;
				refs.push_back(ref);
			}
	}
	auto     start   = std::chrono::steady_clock::now();
	auto     decoded = reader.MaterializeImage(fls, static_cast<uint32_t>(idx), refs);
	double   seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	uint64_t payload = 0;
	for (uint32_t r = 0; r < rec.rowgroup_count; r++)
		payload += desc.rowgroup(rec.first_rowgroup + r).payload_size;
	std::cerr << "{\"read_decode_seconds\":" << seconds << ",\"payload_read_bytes\":" << payload << "}\n";
	uint32_t count = static_cast<uint32_t>(im.components.size());
	put(&count, 1);
	size_t off = 0;
	for (auto& c : im.components) {
		uint32_t shape[3] = {static_cast<uint32_t>(c.local_component_index), c.height_in_blocks, c.width_in_blocks};
		put(shape, 3);
		auto qt = std::find_if(
		    im.quant_tables.begin(), im.quant_tables.end(), [&](auto& q) { return q.table_id == c.quant_tbl_no; });
		if (qt == im.quant_tables.end())
			throw std::runtime_error("missing quantization table");
		put(qt->values.data(), 64);
		size_t blocks = size_t(shape[1]) * shape[2];
		for (size_t b = 0; b < blocks; b++) {
			auto                    row = decoded.blocks.at(off++).coefficients;
			std::array<int16_t, 64> natural;
			for (size_t k = 0; k < 64; k++)
				natural[md.zigzag_columns ? d::kZigzagColumnToNaturalIndex[k] : k] = row[k];
			put(natural.data(), 64);
		}
	}
}
int main(int argc, char** argv) {
	try {
		if (argc < 2)
			throw std::runtime_error("encode OUTPUT_BASE COUNT THREADS | read FLS META LOCAL_INDEX");
		if (std::string(argv[1]) == "encode" || std::string(argv[1]) == "encode-block") {
			if (argc != 5 && argc != 6)
				throw std::runtime_error("encode OUTPUT_BASE COUNT THREADS [GRID_BLOCKS]");
			const uint32_t grid = argc == 6 ? static_cast<uint32_t>(std::stoul(argv[5])) : 56;
			if (grid == 0)
				throw std::runtime_error("GRID_BLOCKS must be positive");
			size_t                       n = std::stoul(argv[3]);
			std::vector<d::DecodedImage> images(n);
			for (auto& im : images) {
				uint32_t len;
				get(&len, 1);
				std::string path(len, '\0');
				get(path.data(), len);
				im.metadata.source_path      = path;
				im.metadata.image_width      = grid * 8;
				im.metadata.image_height     = grid * 8;
				im.metadata.jpeg_color_space = 3;
				for (uint32_t c = 0; c < 3; c++) {
					JpegQuantTableMetadata q;
					q.table_id = static_cast<uint8_t>(c);
					get(q.values.data(), 64);
					im.metadata.quant_tables.push_back(q);
					d::DecodedComponent dc;
					auto&               m   = dc.metadata;
					m.component_index       = c;
					m.local_component_index = c;
					m.component_id          = c + 1;
					m.semantic_slot_id      = c;
					m.width_in_blocks = m.height_in_blocks = m.padded_width_in_blocks = m.padded_height_in_blocks =
					    grid;
					m.h_samp_factor = m.v_samp_factor = 1;
					m.quant_tbl_no                    = c;
					// Match GALP's existing encoding-profile quantization-table key.
					m.quant_table_fingerprint = 1469598103934665603ULL;
					for (auto value : q.values) {
						m.quant_table_fingerprint ^= static_cast<uint64_t>(value & 0xffU);
						m.quant_table_fingerprint *= 1099511628211ULL;
						m.quant_table_fingerprint ^= static_cast<uint64_t>((value >> 8U) & 0xffU);
						m.quant_table_fingerprint *= 1099511628211ULL;
					}
					dc.blocks.resize(grid * grid);
					dc.coord_to_block_index.resize(grid * grid);
					std::iota(dc.coord_to_block_index.begin(), dc.coord_to_block_index.end(), 0);
					get(dc.blocks[0].data(),
					    grid * grid * 64); // Natural frequency order; convert explicitly to GALP zigzag columns.
					for (auto& block : dc.blocks) {
						auto natural = block;
						for (size_t k = 0; k < 64; k++)
							block[k] = natural[d::kZigzagColumnToNaturalIndex[k]];
					}
					im.metadata.components.push_back(m);
					im.components.push_back(std::move(dc));
				}
			}
			if (std::string(argv[1]) == "encode-block") {
				encode_block_major(std::move(images), argv[2], std::stoul(argv[4]));
				return 0;
			}
			JpegDctReaderOptions opts;
			auto                 table = d::make_dataset_table(
                std::move(images), opts, nullptr, JpegDctPhysicalLayout::kImageMajorVectorRowgroups);
			galp::format::CompactV3BuildOptions compact;
			compact.spatial_order = static_cast<uint32_t>(opts.image_major_spatial_order);
			std::vector<uint64_t> rg;
			for (size_t i = 0; i < n; i++) {
				auto& g                 = table.metadata.image_group_index[i];
				g.fls_rowgroup_index    = static_cast<uint32_t>(rg.size());
				g.row_start_in_rowgroup = 0;
				galp::format::CompactV3ImageInput ci;
				ci.first_rowgroup     = static_cast<uint32_t>(rg.size());
				ci.real_row_count     = g.row_count;
				ci.first_physical_row = g.row_start;
				for (uint32_t left = g.row_count; left;) {
					uint32_t count = std::min(1024u, left);
					rg.push_back(count);
					left -= count;
				}
				ci.rowgroup_count = static_cast<uint32_t>(rg.size()) - ci.first_rowgroup;
				for (uint32_t c = 0; c < 3; c++)
					ci.components.push_back({c, grid, grid, grid, grid, c * grid * grid, c});
				compact.images.push_back(ci);
			}
			std::array<fastlanes::MemoryColumn, 64> cols;
			for (size_t c = 0; c < 64; c++) {
				cols[c].name = "dct_zz_" + (c < 10 ? std::string("0") : std::string()) + std::to_string(c);
				cols[c].data = std::span<const int16_t>(table.columns[c]);
			}
			fastlanes::MemoryTable        mt {std::span<const fastlanes::MemoryColumn>(cols)};
			fastlanes::MemoryTableOptions mo;
			mo.n_vectors_per_rowgroup = 1;
			mo.rowgroup_n_tuples      = std::span<const fastlanes::n_t>(rg);
			fastlanes::Connection conn;
			fastlanes::load_memory_table(conn, mt, mo);
			conn.inline_footer();
			fastlanes::EncodingOptions eo;
			eo.worker_count    = std::stoul(argv[4]);
			std::string base   = argv[2];
			const char* tmpdir = std::getenv("TMPDIR");
			const char* home   = std::getenv("HOME");
			if (!tmpdir && !home)
				throw std::runtime_error("TMPDIR or HOME is required for encoder staging");
			const auto temporary_root = tmpdir ? std::filesystem::path(tmpdir) : std::filesystem::path(home) / "tmp";
			std::filesystem::create_directories(temporary_root);
			const auto standard     = temporary_root / (std::filesystem::path(base).filename().string() + "." +
                                                    std::to_string(getpid()) + ".standard.tmp");
			auto       encode_start = std::chrono::steady_clock::now();
			conn.to_fls(standard, eo);
			const auto temporary_bytes = std::filesystem::file_size(standard);
			double     encode_seconds =
			    std::chrono::duration<double>(std::chrono::steady_clock::now() - encode_start).count();
			const auto& encoding_stats = conn.get_last_encoding_stats();
			auto        report         = galp::format::compact_standard_fls_to_v3(standard, base + ".fls", compact);
			write_jpeg_dct_metadata(table.metadata, base + ".meta.bin", JpegDctMetadataWriterOptions {});
			std::filesystem::remove(standard);
			struct rusage usage {};
			getrusage(RUSAGE_SELF, &usage);
			std::cout << "{\"encoding_seconds\":" << encode_seconds << ",\"peak_rss_kib\":" << usage.ru_maxrss
			          << ",\"temporary_bytes\":" << temporary_bytes
			          << ",\"preparation_seconds\":" << encoding_stats.preparation_wall_seconds
			          << ",\"payload_encoding_seconds\":" << encoding_stats.encoding_wall_seconds
			          << ",\"finalization_seconds\":" << encoding_stats.finalization_wall_seconds
			          << ",\"effective_encoding_threads\":" << encoding_stats.effective_worker_count
			          << ",\"images\":" << n << ",\"rowgroups\":" << rg.size()
			          << ",\"payload_bytes\":" << report.payload_bytes << ",\"fls_bytes\":" << report.output_file_bytes
			          << ",\"descriptor_bytes\":" << report.compact_descriptor_bytes << "}\n";
		} else if (std::string(argv[1]) == "read") {
			if (argc != 5)
				throw std::runtime_error("read FLS META LOCAL_INDEX");
			d::JpegDctShardCpuReader reader;
			auto                     md = d::read_jpeg_dct_metadata_file(argv[3]);
			read_image(argv[2], md, std::stoul(argv[4]), reader);
		} else if (std::string(argv[1]) == "read-manifest") {
			if (argc != 3)
				throw std::runtime_error("read-manifest MANIFEST (uint32 global IDs on stdin)");
			auto                     manifest = read_jpeg_dct_shard_manifest(argv[2]);
			auto                     root     = std::filesystem::path(argv[2]).parent_path();
			uint32_t                 id;
			d::JpegDctShardCpuReader reader;
			JpegDctDatasetMetadata   md;
			std::string              current_meta;
			while (std::cin.read(reinterpret_cast<char*>(&id), 4)) {
				auto it = std::find_if(manifest.shards.begin(), manifest.shards.end(), [&](auto& e) {
					return id >= e.first_global_image_index && id < e.first_global_image_index + e.image_count;
				});
				if (it == manifest.shards.end())
					throw std::runtime_error("image ordinal outside manifest");
				if (current_meta != it->metadata_file_name) {
					md           = d::read_jpeg_dct_metadata_file(root / it->metadata_file_name);
					current_meta = it->metadata_file_name;
				}
				read_image(root / it->fls_file_name, md, id - it->first_global_image_index, reader);
				std::cout.flush();
			}
		} else if (std::string(argv[1]) == "info-manifest") {
			if (argc != 3)
				throw std::runtime_error("info-manifest MANIFEST");
			const auto     manifest = read_jpeg_dct_shard_manifest(argv[2]);
			const auto     root     = std::filesystem::path(argv[2]).parent_path();
			nlohmann::json info;
			info["image_count"]         = manifest.image_count;
			info["version"]             = manifest.version;
			info["physical_layout"]     = manifest.physical_layout;
			info["shard_count"]         = manifest.shards.size();
			info["rowgroup_vectors"]    = manifest.rowgroup_vectors;
			info["rowgroups_per_shard"] = manifest.rowgroups_per_shard;
			info["source_paths_stored"] = false; // Existing GALP metadata does not serialize source_path.
			info["shards"]              = nlohmann::json::array();
			for (const auto& entry : manifest.shards) {
				const auto     metadata = d::read_jpeg_dct_metadata_file(root / entry.metadata_file_name);
				nlohmann::json shard;
				shard["shard_id"]                 = entry.shard_id;
				shard["first_global_image_index"] = entry.first_global_image_index;
				shard["image_count"]              = entry.image_count;
				shard["metadata_image_count"]     = metadata.images.size();
				shard["fls_bytes"]                = std::filesystem::file_size(root / entry.fls_file_name);
				shard["metadata_bytes"]           = std::filesystem::file_size(root / entry.metadata_file_name);
				shard["payload_size"]             = entry.payload_size;
				shard["source_paths"]             = nlohmann::json::array();
				for (const auto& image : metadata.images)
					shard["source_paths"].push_back(image.source_path.string());
				info["shards"].push_back(std::move(shard));
			}
			std::cout << info.dump() << '\n';
		} else if (std::string(argv[1]) == "merge-block") {
			std::filesystem::path root = argv[2];
			JpegDctShardManifest  m;
			m.physical_layout = "dct-major/spatial-major-image-minor";
			for (int i = 3; i < argc; ++i) {
				JpegDctShardManifestEntry e;
				e.shard_id                 = i - 3;
				e.first_global_image_index = m.image_count;
				e.fls_file_name            = std::string(argv[i]) + ".fls";
				e.metadata_file_name       = std::string(argv[i]) + ".meta.bin";
				auto md                    = d::read_jpeg_dct_metadata_file(root / e.metadata_file_name);
				e.image_count              = static_cast<uint32_t>(md.images.size());
				e.block_group_count        = static_cast<uint32_t>(md.block_group_index.size());
				for (const auto& g : md.block_group_index)
					e.real_row_count += g.row_count;
				e.physical_row_count = e.real_row_count;
				e.rowgroup_count     = md.block_group_index.back().fls_rowgroup_index + 1;
				e.fls_file_size      = std::filesystem::file_size(root / e.fls_file_name);
				e.metadata_file_size = std::filesystem::file_size(root / e.metadata_file_name);
				m.image_count += e.image_count;
				m.rowgroups_per_shard = std::max(m.rowgroups_per_shard, e.rowgroup_count);
				m.shards.push_back(e);
			}
			write_jpeg_dct_shard_manifest(m, root / "manifest.bin.tmp");
			std::filesystem::rename(root / "manifest.bin.tmp", root / "manifest.bin");
		} else if (std::string(argv[1]) == "merge") {
			if (argc < 4)
				throw std::runtime_error("merge OUTPUT_DIR RELATIVE_SHARD_BASE ... (ordered)");
			std::filesystem::path root = argv[2];
			JpegDctShardManifest  m;
			m.version             = 3;
			m.rowgroup_vectors    = 1;
			m.rowgroups_per_shard = 8192;
			m.physical_layout     = "image-major-vector-rowgroups";
			m.descriptor_kind     = "galp-compact-v1";
			m.vector_size         = 1024;
			m.spatial_order_name  = "tiled-z32";
			m.spatial_order       = JpegDctSpatialOrder::kTiledZ32;
			for (int i = 3; i < argc; i++) {
				JpegDctShardManifestEntry e;
				e.shard_id                 = i - 3;
				e.first_global_image_index = m.image_count;
				e.fls_file_name            = std::string(argv[i]) + ".fls";
				e.metadata_file_name       = std::string(argv[i]) + ".meta.bin";
				auto desc                  = galp::format::CompactDescriptorV3::Open(root / e.fls_file_name);
				e.image_count              = static_cast<uint32_t>(desc.image_count());
				e.rowgroup_count           = static_cast<uint32_t>(desc.rowgroup_count());
				for (size_t j = 0; j < desc.image_count(); j++)
					e.real_row_count += desc.image(j).real_row_count;
				e.physical_row_count      = e.real_row_count;
				e.fls_file_size           = std::filesystem::file_size(root / e.fls_file_name);
				e.metadata_file_size      = std::filesystem::file_size(root / e.metadata_file_name);
				e.payload_size            = desc.payload_bytes();
				e.payload_crc64           = desc.payload_crc64();
				e.compact_descriptor_size = desc.descriptor_bytes();
				m.image_count += e.image_count;
				m.rowgroups_per_shard = std::max(m.rowgroups_per_shard, e.rowgroup_count);
				m.shards.push_back(e);
			}
			write_jpeg_dct_shard_manifest(m, root / "manifest.bin.tmp");
			std::filesystem::rename(root / "manifest.bin.tmp", root / "manifest.bin");
		} else
			throw std::runtime_error("unknown command");
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "dctnet_storage: " << e.what() << '\n';
		return 1;
	}
}
