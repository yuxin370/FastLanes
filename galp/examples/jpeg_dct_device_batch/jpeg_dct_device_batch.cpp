#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <filesystem>
#include <galp/jpeg_dct.hpp>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

uint32_t parse_u32(const char* value, const char* label) {
	try {
		const auto parsed = std::stoull(value);
		if (parsed > UINT32_MAX) {
			throw std::out_of_range(label);
		}
		return static_cast<uint32_t>(parsed);
	} catch (const std::exception&) { throw std::runtime_error(std::string("invalid ") + label + ": " + value); }
}

size_t parse_size(const char* value, const char* label) {
	try {
		const auto parsed = std::stoull(value);
		if (parsed > std::numeric_limits<size_t>::max()) {
			throw std::out_of_range(label);
		}
		return static_cast<size_t>(parsed);
	} catch (const std::exception&) { throw std::runtime_error(std::string("invalid ") + label + ": " + value); }
}

void print_usage(const char* prog) {
	std::cerr << "usage: " << prog
	          << " <manifest.bin> [--crop x y width height] [--window-images n] [--cache-capacity-mib n]"
	             " [--materialize-all] [--verbose] [image_id ...]\n"
	          << "       omit image_id to process every image in the manifest using windowed execution\n";
}

const char* layout_name(const galp::jpeg::JpegDctDeviceLayout layout) {
	switch (layout) {
	case galp::jpeg::JpegDctDeviceLayout::kImageMajorComponentBlockCoeff:
		return "image-major-component-block-coeff";
	}
	return "unknown";
}

struct RunStats {
	size_t windows                  = 0;
	size_t requests                 = 0;
	size_t blocks                   = 0;
	size_t coefficients             = 0;
	size_t bytes                    = 0;
	size_t rowgroups                = 0;
	size_t repeated_rowgroups       = 0;
	size_t cache_hits               = 0;
	size_t cache_misses             = 0;
	size_t cache_inserts            = 0;
	size_t cache_evictions          = 0;
	size_t cache_resident_bytes     = 0;
	size_t cache_resident_rowgroups = 0;
	size_t peak_window_bytes        = 0;
	size_t peak_window_blocks       = 0;
	size_t peak_window_images       = 0;
	double read_batch_ms            = 0.0;
};

RunStats run_window(galp::jpeg::JpegDctShardDatasetReader&   reader,
                    const std::vector<uint32_t>&             image_ids,
                    const size_t                             begin,
                    const size_t                             end,
                    const galp::jpeg::JpegDctCropBox&        crop,
                    const size_t                             cache_capacity_bytes,
                    const bool                               verbose,
                    std::set<std::pair<uint32_t, uint32_t>>& seen_rowgroups) {
	std::vector<galp::jpeg::JpegDctImageCropRequest> requests;
	requests.reserve(end - begin);
	for (size_t i = begin; i < end; ++i) {
		requests.push_back(galp::jpeg::JpegDctImageCropRequest {image_ids[i], crop});
	}

	galp::jpeg::JpegDctDeviceBatchOptions options;
	options.cache_capacity_bytes = cache_capacity_bytes;
	const auto read_start        = std::chrono::steady_clock::now();
	auto       batch             = reader.ReadDeviceDctBatch(requests, options);
	const auto read_end          = std::chrono::steady_clock::now();

	RunStats stats;
	stats.windows                  = 1;
	stats.requests                 = requests.size();
	stats.blocks                   = batch.block_count();
	stats.coefficients             = batch.coefficient_count();
	stats.bytes                    = batch.coefficient_bytes();
	stats.rowgroups                = batch.rowgroup_count();
	const auto cache_stats         = batch.cache_stats();
	stats.cache_hits               = cache_stats.hits;
	stats.cache_misses             = cache_stats.misses;
	stats.cache_inserts            = cache_stats.inserts;
	stats.cache_evictions          = cache_stats.evictions;
	stats.cache_resident_bytes     = cache_stats.resident_bytes;
	stats.cache_resident_rowgroups = cache_stats.resident_rowgroups;
	stats.peak_window_bytes        = batch.coefficient_bytes();
	stats.peak_window_blocks       = batch.block_count();
	stats.peak_window_images       = batch.image_count();
	stats.read_batch_ms            = std::chrono::duration<double, std::milli>(read_end - read_start).count();
	for (const auto& rowgroup : batch.rowgroups()) {
		if (!seen_rowgroups.insert({rowgroup.shard_id, rowgroup.rowgroup_index}).second) {
			++stats.repeated_rowgroups;
		}
	}

	std::cout << "window[" << begin << "," << end << ")"
	          << " images=" << batch.image_count() << " blocks=" << batch.block_count()
	          << " rowgroups=" << batch.rowgroup_count() << " repeated_rowgroups=" << stats.repeated_rowgroups
	          << " cache_hits=" << stats.cache_hits << " cache_misses=" << stats.cache_misses
	          << " cache_inserts=" << stats.cache_inserts << " cache_evictions=" << stats.cache_evictions
	          << " cache_resident_rowgroups=" << stats.cache_resident_rowgroups
	          << " cache_resident_bytes=" << stats.cache_resident_bytes << " coefficients=" << batch.coefficient_count()
	          << " bytes=" << batch.coefficient_bytes() << " read_batch_ms=" << stats.read_batch_ms
	          << " device_coefficients=" << static_cast<const void*>(batch.device_coefficients()) << "\n";

	if (verbose) {
		const auto& image_layouts = batch.image_layouts();
		for (size_t i = 0; i < image_layouts.size(); ++i) {
			const auto& layout = image_layouts[i];
			std::cout << "  image[" << i << "] global_id=" << layout.global_image_index
			          << " block_offset=" << layout.block_offset << " blocks=" << layout.block_count << "\n";
		}
		const auto& block_metadata = batch.block_metadata();
		const auto  preview_count  = std::min<size_t>(block_metadata.size(), 8);
		for (size_t i = 0; i < preview_count; ++i) {
			const auto& block = block_metadata[i];
			std::cout << "  block[" << i << "] image=" << block.global_image_index
			          << " component=" << block.semantic_slot_id << " x=" << block.block_x << " y=" << block.block_y
			          << "\n";
		}
	}

	return stats;
}

} // namespace

int main(const int argc, char** argv) {
	try {
		if (argc < 2) {
			print_usage(argv[0]);
			return 2;
		}

		const std::filesystem::path manifest_path {argv[1]};
		galp::jpeg::JpegDctCropBox  crop {};
		bool                        crop_specified       = false;
		bool                        materialize_all      = false;
		bool                        verbose              = false;
		size_t                      window_images        = 256;
		size_t                      cache_capacity_bytes = 0;
		std::vector<uint32_t>       image_ids;
		for (int i = 2; i < argc; ++i) {
			const std::string arg = argv[i];
			if (arg == "--crop") {
				if (i + 4 >= argc) {
					print_usage(argv[0]);
					return 2;
				}
				crop.x         = parse_u32(argv[++i], "crop x");
				crop.y         = parse_u32(argv[++i], "crop y");
				crop.width     = parse_u32(argv[++i], "crop width");
				crop.height    = parse_u32(argv[++i], "crop height");
				crop_specified = true;
				continue;
			}
			if (arg == "--window-images") {
				if (i + 1 >= argc) {
					print_usage(argv[0]);
					return 2;
				}
				window_images = parse_size(argv[++i], "window-images");
				if (window_images == 0) {
					throw std::runtime_error("window-images must be greater than zero");
				}
				continue;
			}
			if (arg == "--cache-capacity-mib") {
				if (i + 1 >= argc) {
					print_usage(argv[0]);
					return 2;
				}
				const auto mib = parse_size(argv[++i], "cache-capacity-mib");
				if (mib > std::numeric_limits<size_t>::max() / (1024U * 1024U)) {
					throw std::runtime_error("cache-capacity-mib is too large");
				}
				cache_capacity_bytes = mib * 1024U * 1024U;
				continue;
			}
			if (arg == "--materialize-all") {
				materialize_all = true;
				continue;
			}
			if (arg == "--verbose") {
				verbose = true;
				continue;
			}
			image_ids.push_back(parse_u32(argv[i], "image_id"));
		}
		int        device_count = 0;
		const auto status       = cudaGetDeviceCount(&device_count);
		if (status != cudaSuccess || device_count == 0) {
			std::cerr << "no CUDA device is available\n";
			return 3;
		}

		galp::jpeg::JpegDctShardDatasetReader reader(manifest_path);
		const bool                            full_dataset = image_ids.empty();
		if (full_dataset) {
			const auto count = reader.image_count();
			if (count > std::numeric_limits<uint32_t>::max()) {
				throw std::runtime_error("manifest image count exceeds uint32_t image ids");
			}
			image_ids.reserve(static_cast<size_t>(count));
			for (uint64_t image_id = 0; image_id < count; ++image_id) {
				image_ids.push_back(static_cast<uint32_t>(image_id));
			}
		}

		std::cout << "cuda_devices=" << device_count << "\n";
		std::cout << "dataset_images=" << reader.image_count() << " requests=" << image_ids.size();
		if (full_dataset) {
			std::cout << " selection=full-dataset";
		} else {
			std::cout << " selection=explicit";
		}
		if (crop_specified) {
			std::cout << " crop=" << crop.x << "," << crop.y << "," << crop.width << "," << crop.height;
		} else {
			std::cout << " crop=full-image";
		}
		if (materialize_all) {
			std::cout << " execution=materialize-all";
		} else {
			std::cout << " execution=windowed window_images=" << window_images;
		}
		std::cout << " cache_capacity_bytes=" << cache_capacity_bytes;
		std::cout << "\n";
		std::cout << "layout=" << layout_name(galp::jpeg::JpegDctDeviceLayout::kImageMajorComponentBlockCoeff) << "\n";

		RunStats                                total;
		std::set<std::pair<uint32_t, uint32_t>> seen_rowgroups;
		const auto                              effective_window = materialize_all ? image_ids.size() : window_images;
		for (size_t begin = 0; begin < image_ids.size(); begin += effective_window) {
			const size_t end = std::min(image_ids.size(), begin + effective_window);
			const auto   window =
			    run_window(reader, image_ids, begin, end, crop, cache_capacity_bytes, verbose, seen_rowgroups);
			total.windows += window.windows;
			total.requests += window.requests;
			total.blocks += window.blocks;
			total.coefficients += window.coefficients;
			total.bytes += window.bytes;
			total.rowgroups += window.rowgroups;
			total.repeated_rowgroups += window.repeated_rowgroups;
			total.cache_hits += window.cache_hits;
			total.cache_misses += window.cache_misses;
			total.cache_inserts += window.cache_inserts;
			total.cache_evictions += window.cache_evictions;
			total.cache_resident_bytes     = window.cache_resident_bytes;
			total.cache_resident_rowgroups = window.cache_resident_rowgroups;
			total.peak_window_bytes        = std::max(total.peak_window_bytes, window.peak_window_bytes);
			total.peak_window_blocks       = std::max(total.peak_window_blocks, window.peak_window_blocks);
			total.peak_window_images       = std::max(total.peak_window_images, window.peak_window_images);
			total.read_batch_ms += window.read_batch_ms;
		}

		std::cout << "summary windows=" << total.windows << " images=" << total.requests << " blocks=" << total.blocks
		          << " rowgroup_visits=" << total.rowgroups << " unique_rowgroups=" << seen_rowgroups.size()
		          << " repeated_rowgroups=" << total.repeated_rowgroups << " cache_hits=" << total.cache_hits
		          << " cache_misses=" << total.cache_misses << " cache_inserts=" << total.cache_inserts
		          << " cache_evictions=" << total.cache_evictions
		          << " cache_resident_rowgroups=" << total.cache_resident_rowgroups
		          << " cache_resident_bytes=" << total.cache_resident_bytes << " coefficients=" << total.coefficients
		          << " logical_bytes=" << total.bytes << " peak_window_images=" << total.peak_window_images
		          << " peak_window_blocks=" << total.peak_window_blocks
		          << " peak_window_bytes=" << total.peak_window_bytes << " total_read_batch_ms=" << total.read_batch_ms
		          << " avg_window_read_batch_ms="
		          << (total.windows == 0 ? 0.0 : total.read_batch_ms / static_cast<double>(total.windows)) << "\n";

		return 0;
	} catch (const std::exception& e) {
		std::cerr << "galp_jpeg_dct_device_batch: " << e.what() << "\n";
		return 1;
	}
}
