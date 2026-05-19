// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/tests/reader_test.cu
// ────────────────────────────────────────────────────────
#include "engine/materialization/pinned_d2h.cuh"
#include "engine/operators/rowgroup.cuh"
#include "engine/table/table.cuh"
#include "format/reader.cuh"
#include "fls/connection.hpp"
#include "fls/expression/data_type.hpp"
#include "fls/expression/rpn.hpp"
#include "fls/reader/table_reader.hpp"
#include "fls/table/rowgroup.hpp"
#include "codecs/encodings/all.cuh"
#include "galp/galp.hpp"
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <variant>

namespace {

std::filesystem::path pick_fls_file() {
	const char* env_path = std::getenv("FLS_READER_TEST_FILE");
	if (env_path && std::filesystem::exists(env_path)) {
		return std::filesystem::path(env_path);
	}

	const std::filesystem::path galp_root = FLS_GALP_SOURCE_DIR;
	const std::filesystem::path repo_root = galp_root.parent_path();

	const std::filesystem::path candidate1 = repo_root / "data/fls/galp-test/data.fls";
	if (std::filesystem::exists(candidate1)) {
		return candidate1;
	}

	const std::filesystem::path candidate2 = galp_root / "data/fls/galp-test/data.fls";
	if (std::filesystem::exists(candidate2)) {
		return candidate2;
	}

	return {};
}

std::filesystem::path make_partial_rowgroup_fls_fixture() {
	const std::filesystem::path root = std::filesystem::path {GALP_TEST_DATA_DIR} / "partial_rowgroup_public_span";
	std::filesystem::remove_all(root);
	std::filesystem::create_directories(root);

	const auto csv_path    = root / "generated.csv";
	const auto schema_path = root / "schema.json";
	const auto fls_path    = root / "data.fls";

	{
		std::ofstream schema(schema_path);
		schema << R"({"columns":[{"name":"value","type":"FLS_I08"}]})";
	}
	{
		std::ofstream csv(csv_path);
		for (size_t row = 0; row < 1030U; ++row) {
			csv << (row % 100U) << '\n';
		}
	}

	fastlanes::Connection writer;
	writer.set_n_vectors_per_rowgroup(1)
	    .force_schema_pool({fastlanes::OperatorToken::EXP_UNCOMPRESSED_I08})
	    .read_csv(root)
	    .to_fls(fls_path);
	return fls_path;
}

size_t get_n_values(const galp::format::HostColumnVariant& host) {
	return std::visit([](auto&& col) { return col.get_n_values(); }, host);
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
	    fastlanes::OperatorToken::EXP_FREQUENCY_I16,
	    fastlanes::OperatorToken::EXP_CROSS_RLE_I08,
	    fastlanes::OperatorToken::EXP_FFOR_SLPATCH_I16,
	    fastlanes::OperatorToken::EXP_CONSTANT_I08,
	    fastlanes::OperatorToken::EXP_DICT_I08_FFOR_SLPATCH_U08,
	    fastlanes::OperatorToken::EXP_UNCOMPRESSED_I08,
	    fastlanes::OperatorToken::EXP_FFOR_I08,
	    fastlanes::OperatorToken::EXP_FFOR_I16,
	    fastlanes::OperatorToken::EXP_DICT_I08_FFOR_U08,
	    fastlanes::OperatorToken::EXP_DICT_I08_U08,
	    fastlanes::OperatorToken::EXP_DICT_I16_FFOR_U16,
	    fastlanes::OperatorToken::EXP_DICT_I16_FFOR_U08,
	    fastlanes::OperatorToken::EXP_DICT_I16_FFOR_SLPATCH_U16,
	    fastlanes::OperatorToken::EXP_DICT_I16_FFOR_SLPATCH_U08,
	    fastlanes::OperatorToken::EXP_RLE_I08_U16,
	    fastlanes::OperatorToken::EXP_RLE_I16_U16,
	};
}

TEST(Materialize, KickPinnedD2HPreservesZeroLengthEntries) {
	galp::runtime::ExecutionWorkset workset {};
	auto&                           batch = workset.buffers.host_batches.get<int8_t>();
	batch.device_exprs.emplace_back();
	batch.device_exprs.back().plan     = galp::execution::PlanKind::UNCOMPRESSED;
	batch.device_exprs.back().n_values = 0;
	batch.device_exprs.back().out      = nullptr;
	batch.output_offsets.push_back(0);
	batch.expr_indices.push_back(0);

	auto pending = galp::runtime::kick_pinned_d2h_materialize(workset);
	EXPECT_TRUE(batch.device_exprs.empty());
	ASSERT_TRUE(pending.active);
	ASSERT_EQ(pending.entries.size(), 1U);
	EXPECT_EQ(pending.entries[0].global_expr_index, 0U);
	EXPECT_EQ(pending.entries[0].n_values, 0U);

	galp::execution::RowgroupData result {};
	result.columns.resize(1);
	galp::runtime::finalize_pinned_d2h_materialize(
	    pending, [&](const size_t global_expr_index) -> galp::execution::MaterializedColumn* {
		    if (global_expr_index >= result.columns.size()) {
			    return nullptr;
		    }
		    auto& slot = result.columns[global_expr_index];
		    if (!slot.has_value()) {
			    slot.emplace();
		    }
		    return &(*slot);
	    });

	ASSERT_TRUE(result.columns[0].has_value());
	EXPECT_EQ(result.columns[0]->meta.column_index, 0U);
	EXPECT_EQ(result.columns[0]->meta.value_count, 0U);
	EXPECT_EQ(result.columns[0]->meta.value_type, galp::format::DataType::I8);
	std::visit([](const auto& ptr) { EXPECT_NE(ptr.get(), nullptr); }, result.columns[0]->values);
	EXPECT_FALSE(pending.active);
	EXPECT_TRUE(pending.entries.empty());
}

TEST(Materialize, KickPinnedD2HCanDeferWorksetStateClear) {
	galp::runtime::ExecutionWorkset workset {};
	auto&                           batch = workset.buffers.host_batches.get<int8_t>();
	batch.device_exprs.emplace_back();
	batch.device_exprs.back().plan     = galp::execution::PlanKind::UNCOMPRESSED;
	batch.device_exprs.back().n_values = 0;
	batch.device_exprs.back().out      = nullptr;
	batch.output_offsets.push_back(0);
	batch.expr_indices.push_back(0);

	auto pending = galp::runtime::kick_pinned_d2h_materialize(workset, /*clear_workset_state=*/false);
	EXPECT_FALSE(batch.device_exprs.empty());
	ASSERT_TRUE(pending.active);
	ASSERT_EQ(pending.entries.size(), 1U);

	galp::runtime::clear_materialize_workset_state(workset);
	galp::runtime::discard_pinned_d2h_materialize(pending);
	EXPECT_TRUE(batch.device_exprs.empty());
	EXPECT_FALSE(pending.active);
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

void compare_rowgroup_outputs(const galp::format::Rowgroup&                    rowgroup,
                              const fastlanes::Rowgroup&                       expected_rowgroup,
                              const fastlanes::RowgroupDescriptor*             rg,
                              const std::vector<galp::expression::Expression>& expressions,
                              bool                                             verbose,
                              size_t*                                          compared_columns_out,
                              const galp::execution::RowgroupData*             precomputed = nullptr) {
	ASSERT_NE(rg, nullptr);
	ASSERT_NE(rg->m_column_descriptors(), nullptr);
	ASSERT_EQ(rowgroup.columns.size(), rg->m_column_descriptors()->size());
	ASSERT_EQ(rowgroup.n_vecs, static_cast<size_t>(rg->m_n_vec()));

	const size_t expected_rows = static_cast<size_t>(expected_rowgroup.RowCount());
	ASSERT_GE(rowgroup.n_values, expected_rows);

	galp::execution::RowgroupData        local_result;
	const galp::execution::RowgroupData* rowgroup_result_ptr = precomputed;
	if (rowgroup_result_ptr == nullptr) {
		local_result        = galp::execution::decompress_rowgroup(expressions);
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
		const auto  n_values   = get_n_values(rowgroup.columns[i].host);

		std::ostringstream trace;
		trace << "col=" << i << " name=" << col_name << " dtype=" << dtype_name << " token=" << token_str
		      << " n_values=" << n_values << " n_vecs=" << rowgroup.n_vecs << " expected_rows=" << expected_rows;
		if (rowgroup.columns[i].skip_decompress) {
			trace << " skip_decompress=1";
		}
		SCOPED_TRACE(trace.str());
		if (verbose) {
			std::cerr << "[ReaderTest] " << trace.str() << "\n";
		}

		if (rowgroup.columns[i].skip_decompress) {
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
	}

	if (compared_columns_out) {
		*compared_columns_out = compared_columns;
	}
}

} // namespace

TEST(Reader, UnsupportedFormatErrorCarriesContext) {
	const galp::UnsupportedFormatError error("EXP_UNSUPPORTED", 7, 11, "col");
	EXPECT_EQ(error.token(), "EXP_UNSUPPORTED");
	EXPECT_EQ(error.rowgroup_index(), 7U);
	EXPECT_EQ(error.column_index(), 11U);
	EXPECT_EQ(error.column_name(), "col");
	const std::string message = error.what();
	EXPECT_NE(message.find("EXP_UNSUPPORTED"), std::string::npos);
	EXPECT_NE(message.find("rowgroup=7"), std::string::npos);
	EXPECT_NE(message.find("column=11"), std::string::npos);
}

TEST(Reader, UnsupportedTokenContractIsDeterministic) {
	try {
		galp::format::throw_unsupported_zero_copy_token(fastlanes::OperatorToken::EXP_ALP_DBL, 2, 3, "dbl");
		FAIL() << "Expected galp::UnsupportedFormatError";
	} catch (const galp::UnsupportedFormatError& e) {
		EXPECT_EQ(e.token(), "EXP_ALP_DBL");
		EXPECT_EQ(e.rowgroup_index(), 2U);
		EXPECT_EQ(e.column_index(), 3U);
		EXPECT_EQ(e.column_name(), "dbl");
		const std::string message = e.what();
		EXPECT_NE(message.find("EXP_ALP_DBL"), std::string::npos);
		EXPECT_NE(message.find("rowgroup=2"), std::string::npos);
		EXPECT_NE(message.find("column=3"), std::string::npos);
	} catch (const std::exception& e) { FAIL() << "Expected galp::UnsupportedFormatError, got: " << e.what(); }
}

TEST(Reader, UnsupportedTokensSurfaceStructuredError) {
	const std::filesystem::path              galp_root  = FLS_GALP_SOURCE_DIR;
	const std::filesystem::path              repo_root  = galp_root.parent_path();
	const std::vector<std::filesystem::path> candidates = {
	    repo_root / "data/fls/cifar/data.fls",
	    repo_root / "data/fls/celebA/data.fls",
	    repo_root / "data/fls/lfwa/data.fls",
	    repo_root / "data/fls/imagenet-64/image.fls",
	    repo_root / "data/fls/tiny-imagenet/data.fls",
	    repo_root / "data/fls/svhn/data.fls",
	};

	bool saw_existing_fixture = false;
	for (const auto& fls_path : candidates) {
		if (!std::filesystem::exists(fls_path)) {
			continue;
		}
		saw_existing_fixture = true;

		const auto  td_handle = galp::format::detail::load_table_descriptor(fls_path);
		const auto* td        = td_handle.Get();
		ASSERT_NE(td, nullptr);
		ASSERT_NE(td->m_rowgroup_descriptors(), nullptr);

		galp::format::FlsReader rdr(fls_path);
		for (uint32_t rg_idx = 0; rg_idx < td->m_rowgroup_descriptors()->size(); ++rg_idx) {
			try {
				(void)rdr.read_rowgroup_zero_copy_materialized(rg_idx);
			} catch (const galp::UnsupportedFormatError& e) {
				EXPECT_EQ(e.rowgroup_index(), static_cast<size_t>(rg_idx));
				EXPECT_FALSE(e.token().empty());
				return;
			} catch (const std::exception& e) { FAIL() << "Expected galp::UnsupportedFormatError, got: " << e.what(); }
		}
	}

	if (!saw_existing_fixture) {
		GTEST_SKIP() << "No FLS fixtures found for unsupported-format contract test.";
	}
	GTEST_SKIP() << "FLS fixtures did not contain an unsupported operator token.";
}

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

	const auto  td_handle = galp::format::detail::load_table_descriptor(fls_path);
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

	galp::format::FlsReader rdr(fls_path);
	auto                    rowgroup    = rdr.read_rowgroup(0);
	auto                    expressions = galp::expression::assemble(rowgroup);

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

	galp::format::FlsReader rdr(fls_path);
	const size_t            expected_rowgroups = rdr.rowgroup_count();
	ASSERT_GT(expected_rowgroups, 0U);

	const auto supported = supported_tokens();
	auto       conn      = fastlanes::connect();
	ASSERT_TRUE(conn != nullptr);
	auto table_reader = conn->read_fls(fls_path);
	ASSERT_TRUE(table_reader != nullptr);

	const auto  td_handle = galp::format::detail::load_table_descriptor(fls_path);
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

	size_t              expected_total_columns = 0;
	std::vector<size_t> expected_column_counts;
	for (uint32_t rg_idx = 0; rg_idx < td->m_rowgroup_descriptors()->size(); ++rg_idx) {
		if (!should_decompress(rg_idx)) {
			continue;
		}
		const auto* rg = td->m_rowgroup_descriptors()->Get(rg_idx);
		ASSERT_NE(rg, nullptr);
		expected_total_columns += rg->m_column_descriptors()->size();
		expected_column_counts.push_back(rg->m_column_descriptors()->size());
	}

	bool compared_any = false;
	for (const auto scope : {galp::execution::TableDecompressionScope::PerRowgroup,
	                         galp::execution::TableDecompressionScope::WholeTable}) {
		SCOPED_TRACE(scope == galp::execution::TableDecompressionScope::PerRowgroup ? "PerRowgroup" : "WholeTable");
		size_t                                    total_compared = 0;
		galp::execution::TableDecompressionConfig cfg {};
		cfg.scope = scope;

		const auto table_result = galp::execution::decompress_table(
		    fls_path,
		    cfg,
		    should_decompress,
		    [&](size_t                                           rg_idx,
		        galp::format::Rowgroup&                          rowgroup,
		        const std::vector<galp::expression::Expression>& expressions,
		        const galp::execution::RowgroupData&             result) {
			    const auto* rg = td->m_rowgroup_descriptors()->Get(static_cast<uint32_t>(rg_idx));
			    ASSERT_NE(rg, nullptr);

			    auto rowgroup_reader = table_reader->get_rowgroup_reader(static_cast<fastlanes::n_t>(rg_idx));
			    ASSERT_TRUE(rowgroup_reader != nullptr);
			    auto expected_rowgroup = rowgroup_reader->materialize();
			    ASSERT_NE(expected_rowgroup.get(), nullptr);

			    size_t compared_columns = 0;
			    compare_rowgroup_outputs(
			        rowgroup, *expected_rowgroup, rg, expressions, false, &compared_columns, &result);
			    total_compared += compared_columns;
		    });
		ASSERT_GT(table_result.total_columns, 0U);
		ASSERT_EQ(table_result.total_columns, expected_total_columns);
		ASSERT_EQ(table_result.column_counts, expected_column_counts);
		compared_any = compared_any || (total_compared > 0);
	}

	if (!compared_any) {
		GTEST_SKIP() << "No comparable columns across table for reader validation.";
	}
}

TEST(Reader, DecompressTableCallbackThrowLeavesPipelineReusable) {
	const auto fls_path = pick_fls_file();
	if (fls_path.empty()) {
		GTEST_SKIP() << "No .fls file found for reader test (set FLS_READER_TEST_FILE).";
	}

	int        device_count = 0;
	const auto cuda_status  = cudaGetDeviceCount(&device_count);
	if (cuda_status != cudaSuccess || device_count <= 0) {
		GTEST_SKIP() << "CUDA device not available for reader test.";
	}

	galp::format::FlsReader rdr(fls_path);
	ASSERT_GT(rdr.rowgroup_count(), 0U);

	const auto  td_handle = galp::format::detail::load_table_descriptor(fls_path);
	const auto* td        = td_handle.Get();
	ASSERT_NE(td, nullptr);
	ASSERT_NE(td->m_rowgroup_descriptors(), nullptr);

	const auto supported         = supported_tokens();
	auto       should_decompress = [&](const size_t rg_idx) {
        if (rg_idx >= td->m_rowgroup_descriptors()->size()) {
            return false;
        }
        const auto* rg = td->m_rowgroup_descriptors()->Get(static_cast<uint32_t>(rg_idx));
        if (rg == nullptr) {
            return false;
        }
        std::vector<fastlanes::OperatorToken> unsupported;
        return rowgroup_supported(rg, supported, unsupported);
	};

	bool has_supported_rowgroup = false;
	for (size_t rg_idx = 0; rg_idx < rdr.rowgroup_count(); ++rg_idx) {
		has_supported_rowgroup = has_supported_rowgroup || should_decompress(rg_idx);
	}
	if (!has_supported_rowgroup) {
		GTEST_SKIP() << "No supported rowgroups in file for callback exception cleanup test.";
	}

	galp::execution::TableDecompressionConfig cfg {};
	cfg.scope                       = galp::execution::TableDecompressionScope::WholeTable;
	cfg.streaming_target_rowgroups  = 1;
	cfg.streaming_target_work_items = 1;

	size_t throwing_callbacks = 0;
	EXPECT_THROW(galp::execution::decompress_table(fls_path,
	                                               cfg,
	                                               should_decompress,
	                                               [&](size_t,
	                                                   galp::format::Rowgroup&,
	                                                   const std::vector<galp::expression::Expression>&,
	                                                   const galp::execution::RowgroupData&) {
		                                               ++throwing_callbacks;
		                                               throw std::runtime_error("intentional callback failure");
	                                               }),
	             std::runtime_error);
	EXPECT_GT(throwing_callbacks, 0U);

	size_t     retry_callbacks = 0;
	const auto retry_result    = galp::execution::decompress_table(fls_path,
                                                                cfg,
                                                                should_decompress,
                                                                [&](size_t,
                                                                    galp::format::Rowgroup&,
                                                                    const std::vector<galp::expression::Expression>&,
                                                                    const galp::execution::RowgroupData& result) {
                                                                    ++retry_callbacks;
                                                                    EXPECT_FALSE(result.columns.empty());
                                                                });
	EXPECT_GT(retry_callbacks, 0U);
	EXPECT_GT(retry_result.total_columns, 0U);
}

TEST(Reader, PublicNoWriteDecompressReturnsMetadata) {
	const auto fls_path = pick_fls_file();
	if (fls_path.empty()) {
		GTEST_SKIP() << "No .fls file found for reader test (set FLS_READER_TEST_FILE).";
	}

	int        device_count = 0;
	const auto cuda_status  = cudaGetDeviceCount(&device_count);
	if (cuda_status != cudaSuccess || device_count <= 0) {
		GTEST_SKIP() << "CUDA device not available for reader test.";
	}

	const auto  td_handle = galp::format::detail::load_table_descriptor(fls_path);
	const auto* td        = td_handle.Get();
	ASSERT_NE(td, nullptr);
	const auto* rowgroups = td->m_rowgroup_descriptors();
	ASSERT_NE(rowgroups, nullptr);
	ASSERT_GT(rowgroups->size(), 0U);

	const auto          supported              = supported_tokens();
	size_t              expected_total_columns = 0;
	std::vector<size_t> expected_column_counts;
	for (uint32_t rg_idx = 0; rg_idx < rowgroups->size(); ++rg_idx) {
		const auto* rg = rowgroups->Get(rg_idx);
		ASSERT_NE(rg, nullptr);

		std::vector<fastlanes::OperatorToken> unsupported;
		if (!rowgroup_supported(rg, supported, unsupported)) {
			GTEST_SKIP() << "Sample contains rowgroups unsupported by public table decompression.";
		}

		const auto* columns = rg->m_column_descriptors();
		ASSERT_NE(columns, nullptr);
		expected_total_columns += columns->size();
		expected_column_counts.push_back(columns->size());
	}

	galp::Reader            reader(fls_path);
	galp::DecompressOptions options {};
	options.write_output = false;

	for (const auto scope : {galp::TableDecompressionScope::PerRowgroup, galp::TableDecompressionScope::WholeTable}) {
		SCOPED_TRACE(scope == galp::TableDecompressionScope::PerRowgroup ? "PerRowgroup" : "WholeTable");
		options.scope = scope;

		galp::Table table;
		ASSERT_NO_THROW({ table = reader.decompress(options); });
		EXPECT_EQ(table.rowgroup_count(), static_cast<size_t>(rowgroups->size()));
		EXPECT_EQ(table.total_columns(), expected_total_columns);
		EXPECT_EQ(table.rowgroup_column_counts(), expected_column_counts);
	}
}

TEST(Reader, PublicWriteOutputDataAccessReturnsSpans) {
	const auto fls_path = pick_fls_file();
	if (fls_path.empty()) {
		GTEST_SKIP() << "No .fls file found for reader test (set FLS_READER_TEST_FILE).";
	}

	int        device_count = 0;
	const auto cuda_status  = cudaGetDeviceCount(&device_count);
	if (cuda_status != cudaSuccess || device_count <= 0) {
		GTEST_SKIP() << "CUDA device not available for reader test.";
	}

	const auto  td_handle = galp::format::detail::load_table_descriptor(fls_path);
	const auto* td        = td_handle.Get();
	ASSERT_NE(td, nullptr);
	const auto* rowgroups = td->m_rowgroup_descriptors();
	ASSERT_NE(rowgroups, nullptr);
	ASSERT_GT(rowgroups->size(), 0U);

	const auto supported = supported_tokens();
	for (uint32_t rg_idx = 0; rg_idx < rowgroups->size(); ++rg_idx) {
		const auto* rg = rowgroups->Get(rg_idx);
		ASSERT_NE(rg, nullptr);
		std::vector<fastlanes::OperatorToken> unsupported;
		if (!rowgroup_supported(rg, supported, unsupported)) {
			GTEST_SKIP() << "Sample contains rowgroups unsupported by public table decompression.";
		}
	}

	galp::Reader            reader(fls_path);
	galp::DecompressOptions options {};
	options.write_output = true;
	options.scope        = galp::TableDecompressionScope::PerRowgroup;

	galp::Table table;
	ASSERT_NO_THROW({ table = reader.decompress(options); });
	ASSERT_EQ(table.rowgroup_count(), static_cast<size_t>(rowgroups->size()));

	const auto* rg_desc = rowgroups->Get(0);
	ASSERT_NE(rg_desc, nullptr);
	ASSERT_NE(rg_desc->m_column_descriptors(), nullptr);
	auto connection = fastlanes::connect();
	ASSERT_TRUE(connection != nullptr);
	auto table_reader    = connection->read_fls(fls_path);
	auto rowgroup_reader = table_reader->get_rowgroup_reader(0);
	ASSERT_TRUE(rowgroup_reader != nullptr);
	auto expected_rowgroup = rowgroup_reader->materialize();
	ASSERT_NE(expected_rowgroup.get(), nullptr);
	const size_t expected_rows = static_cast<size_t>(expected_rowgroup->RowCount());

	const auto rg_view = table.rowgroup(0);
	ASSERT_EQ(rg_view.column_count(), static_cast<size_t>(rg_desc->m_column_descriptors()->size()));

	bool compared = false;
	for (size_t col_idx = 0; col_idx < rg_view.column_count(); ++col_idx) {
		const auto col_view = rg_view.column(col_idx);
		if (auto* col =
		        std::get_if<fastlanes::up<fastlanes::col_i08>>(&expected_rowgroup->internal_rowgroup[col_idx])) {
			ASSERT_EQ(col_view.type(), galp::DataType::I8);
			const auto values = col_view.values<int8_t>();
			EXPECT_THROW((void)col_view.values<int16_t>(), std::bad_variant_access);
			ASSERT_EQ(values.size(), expected_rows);
			const auto& expected = (*col)->data;
			for (size_t row = 0; row < expected_rows; ++row) {
				ASSERT_EQ(values[row], expected[row]) << "column=" << col_idx << " row=" << row;
			}
			compared = true;
			break;
		}
		if (auto* col =
		        std::get_if<fastlanes::up<fastlanes::col_i16>>(&expected_rowgroup->internal_rowgroup[col_idx])) {
			ASSERT_EQ(col_view.type(), galp::DataType::I16);
			const auto values = col_view.values<int16_t>();
			EXPECT_THROW((void)col_view.values<int8_t>(), std::bad_variant_access);
			ASSERT_EQ(values.size(), expected_rows);
			const auto& expected = (*col)->data;
			for (size_t row = 0; row < expected_rows; ++row) {
				ASSERT_EQ(values[row], expected[row]) << "column=" << col_idx << " row=" << row;
			}
			compared = true;
			break;
		}
		if (auto* col =
		        std::get_if<fastlanes::up<fastlanes::u08_col_t>>(&expected_rowgroup->internal_rowgroup[col_idx])) {
			ASSERT_EQ(col_view.type(), galp::DataType::I8);
			const auto values = col_view.values<int8_t>();
			EXPECT_THROW((void)col_view.values<int16_t>(), std::bad_variant_access);
			ASSERT_EQ(values.size(), expected_rows);
			const auto& expected = (*col)->data;
			for (size_t row = 0; row < expected_rows; ++row) {
				ASSERT_EQ(static_cast<uint8_t>(values[row]), expected[row]) << "column=" << col_idx << " row=" << row;
			}
			compared = true;
			break;
		}
	}

	ASSERT_TRUE(compared) << "No i8/i16 column was available to validate public span access.";
}

TEST(Reader, PublicWriteOutputUsesLogicalTupleCountForPartialRowgroups) {
	int        device_count = 0;
	const auto cuda_status  = cudaGetDeviceCount(&device_count);
	if (cuda_status != cudaSuccess || device_count <= 0) {
		GTEST_SKIP() << "CUDA device not available for reader test.";
	}

	const auto fls_path = make_partial_rowgroup_fls_fixture();

	galp::Reader            reader(fls_path);
	galp::DecompressOptions options {};
	options.write_output = true;
	options.scope        = galp::TableDecompressionScope::PerRowgroup;

	galp::Table table;
	ASSERT_NO_THROW({ table = reader.decompress(options); });
	ASSERT_EQ(table.rowgroup_count(), 2U);

	const auto final_rowgroup = table.rowgroup(1);
	ASSERT_EQ(final_rowgroup.column_count(), 1U);
	const auto column = final_rowgroup.column(0);
	ASSERT_EQ(column.type(), galp::DataType::I8);
	EXPECT_EQ(column.size(), 6U);
	const auto values = column.values<int8_t>();
	ASSERT_EQ(values.size(), 6U);
	for (size_t row = 0; row < values.size(); ++row) {
		EXPECT_EQ(values[row], static_cast<int8_t>((1024U + row) % 100U));
	}
}
