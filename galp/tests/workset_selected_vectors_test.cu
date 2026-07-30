#include "engine/workset/append.cuh"
#include "galp/config.hpp"
#if GALP_WITH_JPEG_DCT
#include "galp/jpeg_dct_diagnostics.hpp"
#include "jpeg/jpeg_dct_cuda_internal.cuh"
#endif
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

TEST(WorksetSelectedVectors, BuildsExplicitCompactChunks) {
	const auto vec_values = static_cast<size_t>(galp::codec::consts::VALUES_PER_VECTOR);

	galp::execution::Column column;
	column.name = "selected_i8";
	column.host = galp::codec::host::CONSTANTColumn<int8_t> {12U * vec_values, 7};

	galp::runtime::ExecutionWorkset workset;
	galp::memory::DeviceArena       arena(nullptr);
	galp::execution::ExecutionConfig cfg;
	cfg.write_out        = true;
	cfg.unpack_n_vectors = 4;

	const std::vector<uint32_t> selected_vectors {0, 8};
	galp::runtime::append_column_to_workset(workset,
	                                        column,
	                                        cfg,
	                                        /*materialize_expr_index=*/11,
	                                        arena,
	                                        nullptr,
	                                        /*emit_typed_work_items=*/true,
	                                        /*register_backing=*/false,
	                                        &selected_vectors,
	                                        cfg.unpack_n_vectors);

	const auto& batch = workset.buffers.host_batches.get<int8_t>();
	ASSERT_EQ(batch.device_exprs.size(), 1U);
	ASSERT_EQ(batch.expr_indices.size(), 1U);
	ASSERT_EQ(batch.output_offsets.size(), 1U);
	ASSERT_EQ(batch.work_items.size(), 2U);
	EXPECT_TRUE(batch.work_items_explicit);
	EXPECT_EQ(galp::runtime::count_work_items(workset), 2U);
	EXPECT_EQ(galp::runtime::count_expr_work_items(workset), 2U);
	EXPECT_EQ(batch.expr_indices[0], 11U);
	EXPECT_EQ(batch.device_exprs[0].n_values, 12U * vec_values);
	EXPECT_EQ(batch.device_exprs[0].output_n_values, 8U * vec_values);
	EXPECT_EQ(workset.outputs.used_bytes, 8U * vec_values * sizeof(int8_t));
	EXPECT_EQ(batch.work_items[0].vector_index, 0U);
	EXPECT_EQ(batch.work_items[0].output_vector_index, 0U);
	EXPECT_EQ(batch.work_items[1].vector_index, 8U);
	EXPECT_EQ(batch.work_items[1].output_vector_index, 4U);
	EXPECT_EQ(batch.work_items[0].type, galp::execution::TypeTag::I8);
	EXPECT_EQ(batch.work_items[1].type, galp::execution::TypeTag::I8);
}

TEST(WorksetSelectedVectors, RejectsTailChunkOverrun) {
	const auto vec_values = static_cast<size_t>(galp::codec::consts::VALUES_PER_VECTOR);

	galp::execution::Column column;
	column.name = "selected_i8";
	column.host = galp::codec::host::CONSTANTColumn<int8_t> {3U * vec_values, 7};

	galp::runtime::ExecutionWorkset workset;
	galp::memory::DeviceArena       arena(nullptr);
	galp::execution::ExecutionConfig cfg;
	cfg.write_out        = true;
	cfg.unpack_n_vectors = 2;

	const std::vector<uint32_t> selected_vectors {2};
	EXPECT_THROW(galp::runtime::append_column_to_workset(workset,
	                                                     column,
	                                                     cfg,
	                                                     /*materialize_expr_index=*/0,
	                                                     arena,
	                                                     nullptr,
	                                                     /*emit_typed_work_items=*/true,
	                                                     /*register_backing=*/false,
	                                                     &selected_vectors,
	                                                     cfg.unpack_n_vectors),
		             std::out_of_range);
}

TEST(WorksetSelectedVectors, BuildsFullDecodeChunksForMultiVectorUnpack) {
	const auto vec_values = static_cast<size_t>(galp::codec::consts::VALUES_PER_VECTOR);

	galp::execution::Column column;
	column.name = "full_i8";
	column.host = galp::codec::host::CONSTANTColumn<int8_t> {12U * vec_values, 42};

	galp::runtime::ExecutionWorkset workset;
	galp::memory::DeviceArena       arena(nullptr);
	galp::execution::ExecutionConfig cfg;
	cfg.write_out        = true;
	cfg.unpack_n_vectors = 4;

	galp::runtime::append_column_to_workset(workset,
	                                        column,
	                                        cfg,
	                                        /*materialize_expr_index=*/5,
	                                        arena,
	                                        nullptr,
	                                        /*emit_typed_work_items=*/true,
	                                        /*register_backing=*/false);

	const auto& batch = workset.buffers.host_batches.get<int8_t>();
	ASSERT_EQ(batch.device_exprs.size(), 1U);
	ASSERT_EQ(batch.work_items.size(), 3U);
	EXPECT_FALSE(batch.work_items_explicit);
	EXPECT_EQ(batch.device_exprs[0].output_n_values, 12U * vec_values);
	EXPECT_EQ(batch.work_items[0].vector_index, 0U);
	EXPECT_EQ(batch.work_items[0].output_vector_index, 0U);
	EXPECT_EQ(batch.work_items[1].vector_index, 4U);
	EXPECT_EQ(batch.work_items[1].output_vector_index, 4U);
	EXPECT_EQ(batch.work_items[2].vector_index, 8U);
	EXPECT_EQ(batch.work_items[2].output_vector_index, 8U);
}

TEST(WorksetSelectedVectors, BuildsFullDecodeScalarTailForMultiVectorUnpack) {
	const auto vec_values = static_cast<size_t>(galp::codec::consts::VALUES_PER_VECTOR);

	galp::execution::Column column;
	column.name = "tail_i8";
	column.host = galp::codec::host::CONSTANTColumn<int8_t> {10U * vec_values, 42};

	galp::runtime::ExecutionWorkset workset;
	galp::memory::DeviceArena       arena(nullptr);
	galp::execution::ExecutionConfig cfg;
	cfg.write_out        = true;
	cfg.unpack_n_vectors = 4;

	galp::runtime::append_column_to_workset(workset,
	                                        column,
	                                        cfg,
	                                        /*materialize_expr_index=*/5,
	                                        arena,
	                                        nullptr,
	                                        /*emit_typed_work_items=*/true,
	                                        /*register_backing=*/false);

	const auto& batch = workset.buffers.host_batches.get<int8_t>();
	ASSERT_EQ(batch.device_exprs.size(), 1U);
	ASSERT_EQ(batch.work_items.size(), 2U);
	ASSERT_EQ(batch.scalar_tail_work_items.size(), 2U);
	EXPECT_EQ(batch.expr_indices[0], 5U);
	EXPECT_EQ(batch.device_exprs[0].output_n_values, 10U * vec_values);
	EXPECT_EQ(batch.work_items[0].vector_index, 0U);
	EXPECT_EQ(batch.work_items[0].output_vector_index, 0U);
	EXPECT_EQ(batch.work_items[1].vector_index, 4U);
	EXPECT_EQ(batch.work_items[1].output_vector_index, 4U);
	EXPECT_EQ(batch.scalar_tail_work_items[0].vector_index, 8U);
	EXPECT_EQ(batch.scalar_tail_work_items[0].output_vector_index, 8U);
	EXPECT_EQ(batch.scalar_tail_work_items[1].vector_index, 9U);
	EXPECT_EQ(batch.scalar_tail_work_items[1].output_vector_index, 9U);
	EXPECT_EQ(galp::runtime::count_work_items(workset), 4U);
}

#if GALP_WITH_JPEG_DCT
TEST(WorksetSelectedVectors, JpegDctColumnProjectionAppendsOnlySelectedColumnsAndAliases) {
	const auto vec_values = static_cast<size_t>(galp::codec::consts::VALUES_PER_VECTOR);

	galp::execution::Rowgroup rowgroup;
	rowgroup.n_values = 2U * vec_values;
	rowgroup.n_vecs   = 2U;
	rowgroup.n_tuples = 2U * vec_values;
	rowgroup.columns.resize(galp::jpeg::detail::kJpegDctCoefficientCount);
	for (size_t coeff = 0; coeff < rowgroup.columns.size(); ++coeff) {
		auto& column = rowgroup.columns[coeff];
		column.name  = "dct_" + std::to_string(coeff);
		column.host = galp::codec::host::CONSTANTColumn<int8_t> {rowgroup.n_values, static_cast<int8_t>(coeff)};
	}
	rowgroup.columns[9].alias_of        = 3U;
	rowgroup.columns[9].skip_decompress = true;

	galp::runtime::ExecutionWorkset workset;
	workset.buffers.chunk_arena = std::make_unique<galp::memory::DeviceArena>(nullptr);
	galp::execution::ExecutionConfig cfg;
	cfg.write_out        = true;
	cfg.unpack_n_vectors = 1;

	const std::vector<uint8_t> selected_coefficients {0U, 5U, 9U};
	galp::jpeg::detail::append_jpeg_rowgroup_columns(workset,
	                                                 rowgroup,
	                                                 cfg,
	                                                 /*expr_index_base=*/100U,
	                                                 selected_coefficients,
	                                                 /*selected_vectors=*/nullptr);

	const auto& batch = workset.buffers.host_batches.get<int8_t>();
	ASSERT_EQ(batch.device_exprs.size(), 3U);
	ASSERT_EQ(batch.expr_indices.size(), 3U);
	EXPECT_EQ(batch.expr_indices[0], 100U);
	EXPECT_EQ(batch.expr_indices[1], 103U);
	EXPECT_EQ(batch.expr_indices[2], 105U);
	EXPECT_EQ(workset.outputs.used_bytes, 3U * rowgroup.n_values * sizeof(int8_t));
}

TEST(WorksetSelectedVectors, JpegDctFusedProjectionPlanCoversPrefixListAndAliasSelections) {
	galp::execution::Rowgroup rowgroup;
	rowgroup.columns.resize(galp::jpeg::detail::kJpegDctCoefficientCount);
	rowgroup.columns[9].alias_of        = 3U;
	rowgroup.columns[9].skip_decompress = true;

	const std::vector<galp::jpeg::detail::JpegDctDeviceGatherItem> blocks {
	    galp::jpeg::detail::JpegDctDeviceGatherItem {7U, 11U, 2U},
	    galp::jpeg::detail::JpegDctDeviceGatherItem {7U, 19U, 3U}};

	const auto make_projection_items_for_blocks =
	    [](const std::vector<galp::jpeg::detail::JpegDctDeviceGatherItem>& block_items,
	       const std::vector<uint8_t>& selected_coefficients) {
		    std::vector<galp::jpeg::detail::JpegDctDeviceProjectionItem> projection_items;
		    projection_items.reserve(block_items.size() * selected_coefficients.size());
		    for (const auto& block_item : block_items) {
			    for (size_t coeff_slot = 0; coeff_slot < selected_coefficients.size(); ++coeff_slot) {
				    const auto logical_coeff = selected_coefficients[coeff_slot];
				    projection_items.push_back(galp::jpeg::detail::JpegDctDeviceProjectionItem {
				        block_item.rowgroup_index,
				        block_item.row_in_rowgroup,
				        block_item.output_block_index,
				        static_cast<uint16_t>(coeff_slot),
				        logical_coeff,
				        logical_coeff});
			    }
		    }
		    return projection_items;
	    };

	const std::vector<uint8_t> prefix_coefficients {0U, 1U, 2U};
	const auto                 prefix_projection = make_projection_items_for_blocks(blocks, prefix_coefficients);
	ASSERT_EQ(prefix_projection.size(), blocks.size() * prefix_coefficients.size());
	EXPECT_EQ(prefix_projection[0].output_block_index, 2U);
	EXPECT_EQ(prefix_projection[0].rowgroup_index, 7U);
	EXPECT_EQ(prefix_projection[0].row_in_rowgroup, 11U);
	EXPECT_EQ(prefix_projection[0].selected_coefficient_slot, 0U);
	EXPECT_EQ(prefix_projection[0].logical_coefficient_id, 0U);
	const auto resolved_prefix =
	    galp::jpeg::detail::resolve_projection_physical_columns(rowgroup, prefix_projection);
	EXPECT_EQ(resolved_prefix.active_physical_coefficients, prefix_coefficients);

	const std::vector<uint8_t> list_coefficients {5U, 9U, 2U};
	const auto                 list_projection = make_projection_items_for_blocks(blocks, list_coefficients);
	const auto resolved_list   = galp::jpeg::detail::resolve_projection_physical_columns(rowgroup, list_projection);
	ASSERT_EQ(resolved_list.items.size(), blocks.size() * list_coefficients.size());
	EXPECT_EQ(resolved_list.items[1].selected_coefficient_slot, 1U);
	EXPECT_EQ(resolved_list.items[1].logical_coefficient_id, 9U);
	EXPECT_EQ(resolved_list.items[1].physical_coefficient_column_id, 3U);
	EXPECT_EQ(resolved_list.active_physical_coefficients, (std::vector<uint8_t> {2U, 3U, 5U}));

	std::vector<int16_t> compact(list_coefficients.size());
	std::array<int16_t, galp::jpeg::detail::kJpegDctCoefficientCount> physical_values {};
	for (size_t coeff = 0; coeff < physical_values.size(); ++coeff) {
		physical_values[coeff] = static_cast<int16_t>(coeff);
	}
	for (const auto& item : resolved_list.items) {
		if (item.output_block_index != 2U) {
			continue;
		}
		compact[galp::jpeg::detail::selected_dct_output_offset(
		    0U, item.selected_coefficient_slot, list_coefficients.size())] =
		    physical_values[item.physical_coefficient_column_id];
	}
	EXPECT_EQ(compact, (std::vector<int16_t> {5, 3, 2}));

	const std::vector<uint8_t> alias_only_coefficients {9U};
	const auto                 alias_only_projection = make_projection_items_for_blocks(blocks, alias_only_coefficients);
	const auto resolved_alias_only =
	    galp::jpeg::detail::resolve_projection_physical_columns(rowgroup, alias_only_projection);
	ASSERT_EQ(resolved_alias_only.items.size(), blocks.size());
	EXPECT_EQ(resolved_alias_only.items[0].selected_coefficient_slot, 0U);
	EXPECT_EQ(resolved_alias_only.items[0].logical_coefficient_id, 9U);
	EXPECT_EQ(resolved_alias_only.items[0].physical_coefficient_column_id, 3U);
	EXPECT_EQ(resolved_alias_only.active_physical_coefficients, (std::vector<uint8_t> {3U}));
}
#endif
