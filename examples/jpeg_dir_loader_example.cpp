// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// examples/jpeg_dir_loader_example.cpp
// ────────────────────────────────────────────────────────
#include "fastlanes.hpp"
#include "fls/connection.hpp"
#include "fls/jpeg/jpeg_loader.hpp"
#include "fls/printer/az_printer.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <tuple>
#include <vector>

using namespace fastlanes; // NOLINT
namespace fs = std::filesystem;

// ★ 工具：判断扩展名
static inline bool is_jpeg_ext(const fs::path& p) {
	if (!p.has_extension())
		return false;
	auto ext = p.extension().string();
	std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
	return (ext == ".jpg" || ext == ".jpeg");
}

// ★ 工具：按位追加 metadata（每块 1 bit）
static void append_metadata_bits(std::vector<uint8_t>&       dst_bytes,
                                 size_t&                     dst_nbits,
                                 const std::vector<uint8_t>& src_bytes,
                                 size_t                      src_nbits) {
	for (size_t i = 0; i < src_nbits; ++i) {
		bool bit = (src_bytes[i / 8] >> (i % 8)) & 0x1;
		if ((dst_nbits % 8) == 0)
			dst_bytes.push_back(0);
		if (bit)
			dst_bytes.back() |= (1u << (dst_nbits % 8));
		dst_nbits++;
	}
}

// ★ 工具：把单图结果累加进总结果
static void append_processed(ProcessedDCTChannel& acc, const ProcessedDCTChannel& cur) {
	// 1) metadata（按位）
	size_t dst_bits = acc.total_blocks; // 已有的 bit 数 = 已有的 block 数
	append_metadata_bits(acc.metadata, dst_bits, cur.metadata, cur.total_blocks);
	acc.total_blocks = dst_bits; // 更新总 block 数

	// 2) 其它顺序拼接
	acc.DC_values.insert(acc.DC_values.end(), cur.DC_values.begin(), cur.DC_values.end());
	acc.AC_values.insert(acc.AC_values.end(), cur.AC_values.begin(), cur.AC_values.end());
	acc.mix_run_nonzero_values.insert(
	    acc.mix_run_nonzero_values.end(), cur.mix_run_nonzero_values.begin(), cur.mix_run_nonzero_values.end());
	acc.mix_run_pattern.insert(acc.mix_run_pattern.end(), cur.mix_run_pattern.begin(), cur.mix_run_pattern.end());
}

int main(int argc, char** argv) {
	if (argc < 3) {
		std::cerr << "用法: ./jpeg_loader_demo <image_dir> <fls_output_dir>\n";
		return 1;
	}

	fs::path image_dir(argv[1]);
	fs::path out_dir(argv[2]);

	if (!fs::exists(image_dir) || !fs::is_directory(image_dir)) {
		std::cerr << "错误: 输入路径不是目录: " << image_dir << "\n";
		return 1;
	}
	if (!fs::exists(out_dir)) {
		fs::create_directories(out_dir);
	}

	// 读取目录下所有 JPG/JPEG
	std::vector<fs::path> images;
	for (const auto& entry : fs::directory_iterator(image_dir)) {
		if (entry.is_regular_file() && is_jpeg_ext(entry.path())) {
			images.push_back(entry.path());
		}
	}
	std::sort(images.begin(), images.end());
	if (images.empty()) {
		std::cerr << "目录中没有 *.jpg / *.jpeg 文件\n";
		return 1;
	}

	// ★ 第一步：在第一张图上估计（left, mid, right），全目录复用
	auto   first_header = JpegLoader::load_header(images.front().string());
	size_t first_blocks = 0;
	for (const auto& ch : first_header.channel_dcts)
		first_blocks += ch.blocks.size();
	auto [left, mid, right] = compute_adaptive_split(first_header, first_blocks, 1000);

	std::cout << "[Split] left=" << left << " mid=" << mid << " right=" << right << "\n";

	// ★ 第二步：累加所有图片
	ProcessedDCTChannel acc; // 注意：acc.metadata 初始为空，total_blocks=0
	for (const auto& img : images) {
		try {
			auto hdr  = JpegLoader::load_header(img.string());
			auto proc = JpegLoader::process_channel(hdr, left, mid, right); // 统一切分
			append_processed(acc, proc);
			std::cout << "Processed: " << img.filename().string() << "\n";
		} catch (const std::exception& ex) { std::cerr << "跳过文件（出错） " << img << " : " << ex.what() << "\n"; }
	}

	// ★ 第三步：一次把累加器写入四个 FLS
	try {
		auto con_dc  = connect();
		auto con_ac  = connect();
		auto con_mix = connect();
		auto con_run = connect();

		con_dc->set_n_vectors_per_rowgroup(64).read_dct(acc, 1);  // DC
		con_ac->set_n_vectors_per_rowgroup(64).read_dct(acc, 2);  // AC
		con_mix->set_n_vectors_per_rowgroup(64).read_dct(acc, 3); // mix_run_nonzero_values
		con_run->set_n_vectors_per_rowgroup(64).read_dct(acc, 4); // mix_run_pattern

		con_dc->to_fls(out_dir / "DC.fls");
		con_ac->to_fls(out_dir / "AC.fls");
		con_mix->to_fls(out_dir / "mix_nonzero.fls"); // 可改名
		con_run->to_fls(out_dir / "mix_run.fls");     // 可改名

		std::cout << "✅ 写入完成: " << out_dir << "\n";
		return EXIT_SUCCESS;
	} catch (std::exception& ex) {
		az_printer::bold_red_cout << "-- Error: " << ex.what() << std::endl;
		return EXIT_FAILURE;
	}
}
