// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/test/reader_test.cu
// ────────────────────────────────────────────────────────
#include "engine/execution/rowgroup.cuh"
#include "engine/execution/table.cuh"
#include "engine/reader.cuh"
#include "fls/connection.hpp"
#include "fls/expression/data_type.hpp"
#include "fls/expression/rpn.hpp"
#include "fls/reader/table_reader.hpp"
#include "fls/table/rowgroup.hpp"
#include "flsgpu/structs.cuh"
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <unordered_set>

namespace {

std::filesystem::path pick_fls_file() {
	const char* env_path = std::getenv("FLS_READER_TEST_FILE");
	if (env_path && std::filesystem::exists(env_path)) {
		return std::filesystem::path(env_path);
	}

	const std::filesystem::path candidate1 = "/home/tangyuxin/cleanFastlanes/FastLanes/data/fls/galp-test/data.fls";
	if (std::filesystem::exists(candidate1)) {
		return candidate1;
	}

	const std::filesystem::path candidate2 = "/home/tangyuxin/cleanFastlanes/FastLanes/python/tests/result/data.fls";
	if (std::filesystem::exists(candidate2)) {
		return candidate2;
	}

	return {};
}

size_t get_n_values(const reader::HostColumnVariant& host) {
	return std::visit([](auto&& col) { return col.get_n_values(); }, host);
}

const char* op_kind_to_string(const expr::OperatorKind kind) {
	switch (kind) {
	case expr::OperatorKind::UNCOMPRESSED:
		return "UNCOMPRESSED";
	case expr::OperatorKind::UNFFOR:
		return "UNFFOR";
	case expr::OperatorKind::SLPATCH:
		return "SLPATCH";
	case expr::OperatorKind::FREQUENCY:
		return "FREQUENCY";
	case expr::OperatorKind::CROSS_RLE:
		return "CROSS_RLE";
	case expr::OperatorKind::DICT:
		return "DICT";
	case expr::OperatorKind::CONSTANT:
		return "CONSTANT";
	}
	return "UNKNOWN";
}

std::string ops_to_string(const expr::Expression& expression) {
	std::ostringstream os;
	for (size_t i = 0; i < expression.ops.size(); ++i) {
		if (i > 0) {
			os << "->";
		}
		os << op_kind_to_string(expression.ops[i]);
	}
	return os.str();
}

template <typename T>
std::string format_value(T value) {
	if constexpr (std::is_same_v<T, int8_t>) {
		return std::to_string(static_cast<int>(value));
	} else if constexpr (std::is_same_v<T, uint8_t>) {
		return std::to_string(static_cast<unsigned>(value));
	} else {
		return std::to_string(static_cast<long long>(value));
	}
}

template <typename OutT, typename ExpT>
std::string
format_window(const OutT* out, const std::vector<ExpT>& expected, size_t start, size_t end, bool cast_out_to_u8) {
	std::ostringstream os;
	for (size_t row = start; row < end; ++row) {
		if (row > start) {
			os << ", ";
		}
		if (cast_out_to_u8) {
			os << format_value(static_cast<uint8_t>(out[row]));
		} else {
			os << format_value(out[row]);
		}
		os << "/" << format_value(expected[row]);
	}
	return os.str();
}

std::unordered_set<fastlanes::OperatorToken> supported_tokens() {
	return {
	    fastlanes::OperatorToken::EXP_FFOR_SLPATCH_I08,
	    fastlanes::OperatorToken::EXP_FREQUENCY_I08,
	    fastlanes::OperatorToken::EXP_CROSS_RLE_I08,
	    fastlanes::OperatorToken::EXP_FFOR_SLPATCH_I16,
	    fastlanes::OperatorToken::EXP_CONSTANT_I08,
	    fastlanes::OperatorToken::EXP_DICT_I08_FFOR_SLPATCH_U08,
	    fastlanes::OperatorToken::EXP_UNCOMPRESSED_I08,
	    fastlanes::OperatorToken::EXP_FFOR_I08,
	    fastlanes::OperatorToken::EXP_FFOR_I16,
	    fastlanes::OperatorToken::EXP_DICT_I08_U08,
	};
}

bool rowgroup_supported(const fastlanes::RowgroupDescriptor*                rg,
                        const std::unordered_set<fastlanes::OperatorToken>& supported,
                        std::vector<fastlanes::OperatorToken>&              unsupported) {
	unsupported.clear();
	if (!rg || !rg->m_column_descriptors()) {
		return false;
	}
	for (uint32_t i = 0; i < rg->m_column_descriptors()->size(); ++i) {
		const auto* col = rg->m_column_descriptors()->Get(i);
		if (!col) {
			unsupported.push_back(fastlanes::OperatorToken::EXP_EQUAL);
			continue;
		}
		const auto* rpn = col->encoding_rpn();
		if (!rpn || !rpn->operator_tokens() || rpn->operator_tokens()->size() != 1) {
			unsupported.push_back(fastlanes::OperatorToken::EXP_EQUAL);
			continue;
		}
		const auto token = rpn->operator_tokens()->Get(0);
		if (!supported.count(token)) {
			unsupported.push_back(token);
		}
	}
	return unsupported.empty();
}

void compare_rowgroup_outputs(const reader::Rowgroup&              rowgroup,
                              const fastlanes::Rowgroup&           expected_rowgroup,
                              const fastlanes::RowgroupDescriptor* rg,
                              const std::vector<expr::Expression>& expressions,
                              bool                                 verbose,
                              size_t*                              compared_columns_out,
                              const dispatch::RowgroupData*        precomputed  = nullptr,
                              bool                                 free_columns = true) {
	ASSERT_NE(rg, nullptr);
	ASSERT_NE(rg->m_column_descriptors(), nullptr);
	ASSERT_EQ(rowgroup.columns.size(), rg->m_column_descriptors()->size());
	ASSERT_EQ(rowgroup.n_vecs, static_cast<size_t>(rg->m_n_vec()));

	const size_t expected_rows = static_cast<size_t>(expected_rowgroup.RowCount());
	ASSERT_GE(rowgroup.n_values, expected_rows);

	dispatch::RowgroupData        local_result;
	const dispatch::RowgroupData* rowgroup_result_ptr = precomputed;
	if (rowgroup_result_ptr == nullptr) {
		local_result        = dispatch::decompress_rowgroup(expressions);
		rowgroup_result_ptr = &local_result;
	}
	ASSERT_EQ(rowgroup_result_ptr->columns.size(), rowgroup.columns.size());

	size_t compared_columns = 0;

	for (size_t i = 0; i < rowgroup.columns.size(); ++i) {
		const auto& col_desc = *rg->m_column_descriptors()->Get(static_cast<uint32_t>(i));
		const auto* rpn      = col_desc.encoding_rpn();
		ASSERT_NE(rpn, nullptr);
		ASSERT_EQ(rpn->operator_tokens()->size(), 1U);

		const auto expected_token = rpn->operator_tokens()->Get(0);
		EXPECT_EQ(rowgroup.columns[i].token, expected_token);
		EXPECT_EQ(expressions[i].column->host.index(), rowgroup.columns[i].host.index());
		EXPECT_EQ(get_n_values(rowgroup.columns[i].host), rowgroup.n_values);

		const auto* name_ptr   = col_desc.name();
		const auto  col_name   = name_ptr ? name_ptr->str() : std::string("<unnamed>");
		const auto  dtype_name = fastlanes::ToStr(col_desc.data_type());
		const auto  token_str  = fastlanes::token_to_string(expected_token);
		const auto  ops_str    = ops_to_string(expressions[i]);
		const auto  n_values   = get_n_values(rowgroup.columns[i].host);

		std::ostringstream trace;
		trace << "col=" << i << " name=" << col_name << " dtype=" << dtype_name << " token=" << token_str
		      << " ops=" << ops_str << " n_values=" << n_values << " n_vecs=" << rowgroup.n_vecs
		      << " expected_rows=" << expected_rows;
		if (rowgroup.columns[i].skip_decompress) {
			trace << " skip_decompress=1";
		}
		SCOPED_TRACE(trace.str());
		if (verbose) {
			std::cerr << "[ReaderTest] " << trace.str() << "\n";
		}

		if (rowgroup.columns[i].skip_decompress) {
			if (free_columns) {
				std::visit([](auto& host_col) { flsgpu::host::free_column(host_col); }, rowgroup.columns[i].host);
			}
			continue;
		}

		ASSERT_TRUE(rowgroup_result_ptr->columns[i].has_value());
		auto& column_data = *rowgroup_result_ptr->columns[i];
		std::visit([&](auto& ptr) { ASSERT_NE(ptr, nullptr); }, column_data.values);

		bool compared_this = false;
		std::visit(
		    [&](auto& ptr) {
			    using OutT      = std::remove_pointer_t<decltype(ptr.get())>;
			    const OutT* out = ptr.get();

			    if (auto* col =
			            std::get_if<fastlanes::up<fastlanes::col_i08>>(&expected_rowgroup.internal_rowgroup[i])) {
				    if constexpr (std::is_same_v<OutT, int8_t>) {
					    compared_this    = true;
					    const auto& data = (*col)->data;
					    ASSERT_GE(data.size(), expected_rows);
					    bool   mismatch = false;
					    size_t row_idx  = 0;
					    for (size_t row = 0; row < expected_rows; ++row) {
						    if (out[row] != data[row]) {
							    mismatch = true;
							    row_idx  = row;
							    break;
						    }
					    }
					    if (mismatch) {
						    const size_t       start = row_idx > 4 ? row_idx - 4 : 0;
						    const size_t       end   = std::min(row_idx + 5, expected_rows);
						    std::ostringstream os;
						    os << "Mismatch at column " << i << " row " << row_idx
						       << " out=" << format_value(out[row_idx]) << " expected=" << format_value(data[row_idx])
						       << "\n";
						    os << "Expected data size=" << data.size() << " rowgroup_n_values=" << rowgroup.n_values
						       << "\n";
						    os << "Window out/expected [" << start << "," << end
						       << "): " << format_window(out, data, start, end, false);
						    ADD_FAILURE() << os.str();
					    }
				    } else {
					    ADD_FAILURE() << "column " << i << " expected int8 output, got int16";
					    compared_this = true;
				    }
			    } else if (auto* col = std::get_if<fastlanes::up<fastlanes::col_i16>>(
			                   &expected_rowgroup.internal_rowgroup[i])) {
				    if constexpr (std::is_same_v<OutT, int16_t>) {
					    compared_this    = true;
					    const auto& data = (*col)->data;
					    ASSERT_GE(data.size(), expected_rows);
					    bool   mismatch = false;
					    size_t row_idx  = 0;
					    for (size_t row = 0; row < expected_rows; ++row) {
						    if (out[row] != data[row]) {
							    mismatch = true;
							    row_idx  = row;
							    break;
						    }
					    }
					    if (mismatch) {
						    const size_t       start = row_idx > 4 ? row_idx - 4 : 0;
						    const size_t       end   = std::min(row_idx + 5, expected_rows);
						    std::ostringstream os;
						    os << "Mismatch at column " << i << " row " << row_idx
						       << " out=" << format_value(out[row_idx]) << " expected=" << format_value(data[row_idx])
						       << "\n";
						    os << "Expected data size=" << data.size() << " rowgroup_n_values=" << rowgroup.n_values
						       << "\n";
						    os << "Window out/expected [" << start << "," << end
						       << "): " << format_window(out, data, start, end, false);
						    ADD_FAILURE() << os.str();
					    }
				    } else {
					    ADD_FAILURE() << "column " << i << " expected int16 output, got int8";
					    compared_this = true;
				    }
			    } else if (auto* col = std::get_if<fastlanes::up<fastlanes::u08_col_t>>(
			                   &expected_rowgroup.internal_rowgroup[i])) {
				    if constexpr (std::is_same_v<OutT, int8_t>) {
					    compared_this    = true;
					    const auto& data = (*col)->data;
					    ASSERT_GE(data.size(), expected_rows);
					    bool   mismatch = false;
					    size_t row_idx  = 0;
					    for (size_t row = 0; row < expected_rows; ++row) {
						    if (static_cast<uint8_t>(out[row]) != data[row]) {
							    mismatch = true;
							    row_idx  = row;
							    break;
						    }
					    }
					    if (mismatch) {
						    const size_t       start = row_idx > 4 ? row_idx - 4 : 0;
						    const size_t       end   = std::min(row_idx + 5, expected_rows);
						    std::ostringstream os;
						    os << "Mismatch at column " << i << " row " << row_idx
						       << " out=" << format_value(static_cast<uint8_t>(out[row_idx]))
						       << " expected=" << format_value(data[row_idx]) << "\n";
						    os << "Expected data size=" << data.size() << " rowgroup_n_values=" << rowgroup.n_values
						       << "\n";
						    os << "Window out/expected [" << start << "," << end
						       << "): " << format_window(out, data, start, end, true);
						    ADD_FAILURE() << os.str();
					    }
				    } else {
					    ADD_FAILURE() << "column " << i << " expected uint8 output, got int16";
					    compared_this = true;
				    }
			    }
		    },
		    column_data.values);

		if (compared_this) {
			++compared_columns;
		}

		if (free_columns) {
			std::visit([](auto& host_col) { flsgpu::host::free_column(host_col); }, rowgroup.columns[i].host);
		}
	}

	if (compared_columns_out) {
		*compared_columns_out = compared_columns;
	}
}

} // namespace

TEST(Reader, ParseFlsRowgroup0) {
	const auto fls_path = pick_fls_file();
	if (fls_path.empty()) {
		GTEST_SKIP() << "No .fls file found for reader test (set FLS_READER_TEST_FILE).";
	}

	int         device_count = 0;
	const auto  cuda_status  = cudaGetDeviceCount(&device_count);
	const char* cuda_error   = cudaGetErrorString(cuda_status);
	if (cuda_status != cudaSuccess || device_count <= 0) {
		GTEST_SKIP() << "CUDA device not available for reader test: " << cuda_error;
	}

	const bool verbose = std::getenv("FLS_READER_TEST_VERBOSE") != nullptr;

	auto conn = fastlanes::connect();
	ASSERT_TRUE(conn != nullptr);
	auto table_reader = conn->read_fls(fls_path);
	ASSERT_TRUE(table_reader != nullptr);
	auto rowgroup_reader = table_reader->get_rowgroup_reader(0);
	ASSERT_TRUE(rowgroup_reader != nullptr);
	auto expected_rowgroup = rowgroup_reader->materialize();
	ASSERT_NE(expected_rowgroup.get(), nullptr);
	const size_t expected_rows = static_cast<size_t>(expected_rowgroup->RowCount());

	const auto  td_handle = reader::detail::load_table_descriptor(fls_path);
	const auto* td        = td_handle.Get();
	ASSERT_NE(td, nullptr);
	ASSERT_GT(td->m_rowgroup_descriptors()->size(), 0U);

	const auto* rg = td->m_rowgroup_descriptors()->Get(0);
	ASSERT_NE(rg, nullptr);

	const auto                            supported = supported_tokens();
	std::vector<fastlanes::OperatorToken> unsupported;
	const bool                            all_supported = rowgroup_supported(rg, supported, unsupported);

	if (!all_supported) {
		std::stringstream ss;
		ss << "Reader test skipped: unsupported operator tokens in file: ";
		for (auto t : unsupported) {
			ss << fastlanes::token_to_string(t) << " ";
		}
		GTEST_SKIP() << ss.str();
	}

	reader::reader rdr(fls_path);
	auto           rowgroup    = rdr.read_rowgroup(0);
	auto           expressions = expr::assemble(rowgroup);

	size_t compared_columns = 0;
	compare_rowgroup_outputs(rowgroup, *expected_rowgroup, rg, expressions, verbose, &compared_columns);

	if (compared_columns == 0) {
		GTEST_SKIP() << "No comparable int8/int16 columns in rowgroup for reader validation.";
	}
}

TEST(Reader, DecompressTable) {
	const auto fls_path = pick_fls_file();
	if (fls_path.empty()) {
		GTEST_SKIP() << "No .fls file found for reader test (set FLS_READER_TEST_FILE).";
	}

	int        device_count = 0;
	const auto cuda_status  = cudaGetDeviceCount(&device_count);
	if (cuda_status != cudaSuccess || device_count <= 0) {
		GTEST_SKIP() << "CUDA device not available for reader test.";
	}

	reader::reader rdr(fls_path);
	const size_t   expected_rowgroups = rdr.rowgroup_count();
	ASSERT_GT(expected_rowgroups, 0U);

	const auto supported = supported_tokens();
	auto       conn      = fastlanes::connect();
	ASSERT_TRUE(conn != nullptr);
	auto table_reader = conn->read_fls(fls_path);
	ASSERT_TRUE(table_reader != nullptr);

	const auto  td_handle = reader::detail::load_table_descriptor(fls_path);
	const auto* td        = td_handle.Get();
	ASSERT_NE(td, nullptr);
	ASSERT_GT(td->m_rowgroup_descriptors()->size(), 0U);

	auto should_decompress = [&](size_t rg_idx) {
		const auto* rg = td->m_rowgroup_descriptors()->Get(static_cast<uint32_t>(rg_idx));
		if (rg == nullptr) {
			ADD_FAILURE() << "Rowgroup descriptor missing at index " << rg_idx;
			return false;
		}
		std::vector<fastlanes::OperatorToken> unsupported;
		if (!rowgroup_supported(rg, supported, unsupported)) {
			if (std::getenv("FLS_READER_TEST_VERBOSE") != nullptr) {
				std::stringstream ss;
				ss << "[ReaderTest] Rowgroup " << rg_idx << " skipped: unsupported tokens: ";
				for (auto t : unsupported) {
					ss << fastlanes::token_to_string(t) << " ";
				}
				std::cerr << ss.str() << "\n";
			}
			return false;
		}
		return true;
	};

	size_t expected_total_columns = 0;
	for (uint32_t rg_idx = 0; rg_idx < td->m_rowgroup_descriptors()->size(); ++rg_idx) {
		if (!should_decompress(rg_idx)) {
			continue;
		}
		const auto* rg = td->m_rowgroup_descriptors()->Get(rg_idx);
		ASSERT_NE(rg, nullptr);
		expected_total_columns += rg->m_column_descriptors()->size();
	}

	bool compared_any = false;
	for (const auto scope :
	     {dispatch::TableDecompressionScope::PerRowgroup, dispatch::TableDecompressionScope::WholeTable}) {
		SCOPED_TRACE(scope == dispatch::TableDecompressionScope::PerRowgroup ? "PerRowgroup" : "WholeTable");
		size_t                             total_compared = 0;
		dispatch::TableDecompressionConfig cfg {};
		cfg.scope = scope;

		const auto table_result = dispatch::decompress_table(
		    fls_path,
		    cfg,
		    should_decompress,
		    [&](size_t                               rg_idx,
		        reader::Rowgroup&                    rowgroup,
		        const std::vector<expr::Expression>& expressions,
		        const dispatch::RowgroupData&        result) {
			    const auto* rg = td->m_rowgroup_descriptors()->Get(static_cast<uint32_t>(rg_idx));
			    ASSERT_NE(rg, nullptr);

			    auto rowgroup_reader = table_reader->get_rowgroup_reader(static_cast<fastlanes::n_t>(rg_idx));
			    ASSERT_TRUE(rowgroup_reader != nullptr);
			    auto expected_rowgroup = rowgroup_reader->materialize();
			    ASSERT_NE(expected_rowgroup.get(), nullptr);

			    size_t compared_columns = 0;
			    compare_rowgroup_outputs(
			        rowgroup, *expected_rowgroup, rg, expressions, false, &compared_columns, &result, false);
			    total_compared += compared_columns;
		    });
		ASSERT_GT(table_result.total_columns, 0U);
		ASSERT_EQ(table_result.total_columns, expected_total_columns);
		compared_any = compared_any || (total_compared > 0);
	}

	if (!compared_any) {
		GTEST_SKIP() << "No comparable columns across table for reader validation.";
	}
}
