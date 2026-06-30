#include "engine/workset/append.cuh"
#include <gtest/gtest.h>

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

TEST(WorksetSelectedVectors, RejectsFullDecodeTailChunkOverrun) {
	const auto vec_values = static_cast<size_t>(galp::codec::consts::VALUES_PER_VECTOR);

	galp::execution::Column column;
	column.name = "tail_i8";
	column.host = galp::codec::host::CONSTANTColumn<int8_t> {10U * vec_values, 42};

	galp::runtime::ExecutionWorkset workset;
	galp::memory::DeviceArena       arena(nullptr);
	galp::execution::ExecutionConfig cfg;
	cfg.write_out        = true;
	cfg.unpack_n_vectors = 4;

	EXPECT_THROW(galp::runtime::append_column_to_workset(workset,
	                                                     column,
	                                                     cfg,
	                                                     /*materialize_expr_index=*/5,
	                                                     arena,
	                                                     nullptr,
	                                                     /*emit_typed_work_items=*/true,
	                                                     /*register_backing=*/false),
	             std::out_of_range);
}
