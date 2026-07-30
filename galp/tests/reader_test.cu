// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/tests/reader_test.cu
// ────────────────────────────────────────────────────────
#include "engine/materialization/pinned_d2h.cuh"
#include "engine/materialization/metadata.cuh"
#include "engine/materialization/selected_vector_compactor.hpp"
#include "engine/operators/column.cuh"
#include "engine/operators/rowgroup.cuh"
#include "engine/pipeline/pipeline.cuh"
#include "engine/table/table.cuh"
#include "engine/workset/append.cuh"
#include "engine/workset/upload.cuh"
#include "cuda/launch/launch.cuh"
#include "cuda/memory/pinned_host_pool.cuh"
#include "format/reader.cuh"
#include "fls/connection.hpp"
#include "fls/expression/data_type.hpp"
#include "fls/expression/rpn.hpp"
#include "fls/reader/table_reader.hpp"
#include "fls/table/memory_table.hpp"
#include "fls/table/rowgroup.hpp"
#include "codecs/encodings/all.cuh"
#include "galp/galp.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <variant>

namespace {

bool cuda_available_for_reader_tests();

TEST(PinnedHostPool, ReusesBestFitAndBoundsReleasedMemory) {
	if (!cuda_available_for_reader_tests()) {
		GTEST_SKIP() << "CUDA device not available for pinned-host pool test.";
	}
	galp::memory::PinnedHostPool pool(/*cache_limit_bytes=*/8192, /*max_reuse_slack_bytes=*/4096);
	void* const                 first  = pool.alloc(4096);
	void* const                 second = pool.alloc(4096);
	ASSERT_NE(first, nullptr);
	ASSERT_NE(second, nullptr);
	EXPECT_NE(first, second);

	auto stats = pool.stats();
	EXPECT_EQ(stats.in_use_bytes, 8192U);
	EXPECT_EQ(stats.cached_bytes, 0U);
	EXPECT_EQ(stats.cuda_allocation_count, 2U);

	pool.release(first);
	void* const reused = pool.alloc(2048);
	EXPECT_EQ(reused, first);
	stats = pool.stats();
	EXPECT_EQ(stats.in_use_bytes, 8192U);
	EXPECT_EQ(stats.cached_bytes, 0U);
	EXPECT_EQ(stats.cuda_allocation_count, 2U);

	pool.release(second);
	pool.release(reused);
	stats = pool.stats();
	EXPECT_EQ(stats.in_use_bytes, 0U);
	EXPECT_EQ(stats.cached_bytes, 8192U);

	void* const larger = pool.alloc(8192);
	ASSERT_NE(larger, nullptr);
	EXPECT_EQ(pool.stats().cuda_allocation_count, 3U);
	pool.release(larger);
	stats = pool.stats();
	EXPECT_EQ(stats.cached_bytes, 8192U);
	EXPECT_LE(stats.cached_bytes, 8192U);

	pool.set_cache_limit_bytes(4096);
	EXPECT_EQ(pool.stats().cached_bytes, 0U);
}

class ScopedEnvironmentVariable {
public:
	ScopedEnvironmentVariable(std::string name, const char* const value)
	    : name_(std::move(name)) {
		if (const char* const existing = std::getenv(name_.c_str()); existing != nullptr) {
			had_value_ = true;
			old_value_ = existing;
		}
		if ((value == nullptr ? ::unsetenv(name_.c_str()) : ::setenv(name_.c_str(), value, 1)) != 0) {
			throw std::runtime_error("failed to update test environment variable " + name_);
		}
	}

	~ScopedEnvironmentVariable() {
		if (had_value_) {
			(void)::setenv(name_.c_str(), old_value_.c_str(), 1);
		} else {
			(void)::unsetenv(name_.c_str());
		}
	}

	ScopedEnvironmentVariable(const ScopedEnvironmentVariable&)            = delete;
	ScopedEnvironmentVariable& operator=(const ScopedEnvironmentVariable&) = delete;

private:
	std::string name_;
	std::string old_value_;
	bool        had_value_ = false;
};

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

std::filesystem::path make_sparse_vector_read_fixture() {
	const std::filesystem::path root =
	    std::filesystem::path {GALP_TEST_DATA_DIR} / "sparse_vector_physical_ranges";
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
		for (size_t row = 0; row < 8U * galp::codec::consts::VALUES_PER_VECTOR; ++row) {
			csv << static_cast<int>((row * 17U + row / 1024U) % 101U) - 50 << '\n';
		}
	}
	fastlanes::Connection writer;
	writer.set_n_vectors_per_rowgroup(8)
	    .force_schema_pool({fastlanes::OperatorToken::EXP_UNCOMPRESSED_I08})
	    .read_csv(root)
	    .to_fls(fls_path);
	return fls_path;
}

std::filesystem::path make_sparse_vector_bundle_fixture() {
	const std::filesystem::path root =
	    std::filesystem::path {GALP_TEST_DATA_DIR} / "sparse_vector_bundle_ranges";
	std::filesystem::remove_all(root);
	std::filesystem::create_directories(root);

	const auto csv_path    = root / "generated.csv";
	const auto schema_path = root / "schema.json";
	const auto fls_path    = root / "data.fls";
	{
		std::ofstream schema(schema_path);
		schema << R"({"columns":[)";
		for (size_t column = 0; column < 8U; ++column) {
			if (column != 0U) {
				schema << ',';
			}
			schema << R"({"name":"c)" << column << R"(","type":"FLS_I08"})";
		}
		schema << "]}";
	}
	{
		std::ofstream csv(csv_path);
		for (size_t row = 0; row < 8U * galp::codec::consts::VALUES_PER_VECTOR; ++row) {
			for (size_t column = 0; column < 8U; ++column) {
				if (column != 0U) {
					csv << '|';
				}
				csv << static_cast<int>((row * 17U + column * 29U + row / 1024U) % 101U) - 50;
			}
			csv << '\n';
		}
	}
	fastlanes::Connection writer;
	writer.set_n_vectors_per_rowgroup(8)
	    .force_schema_pool({fastlanes::OperatorToken::EXP_UNCOMPRESSED_I08})
	    .read_csv(root)
	    .to_fls(fls_path);
	return fls_path;
}

size_t get_n_values(const galp::format::HostColumnVariant& host) {
	return std::visit([](auto&& col) { return col.get_n_values(); }, host);
}

TEST(Reader, SparseVectorReadUsesPhysicalSegmentRangesAndReportsFallback) {
	const auto fls_path = make_sparse_vector_read_fixture();
	galp::format::FlsReader reader(fls_path);
	ASSERT_EQ(reader.rowgroup_count(), 1U);
	std::string capability_reason;
	ASSERT_TRUE(reader.sparse_vector_read_supported(0, &capability_reason)) << capability_reason;

	galp::format::ZeroCopyReadTiming sparse_timing {};
	auto sparse = reader.read_rowgroup_zero_copy_selected_vectors(0, {1U, 6U}, &sparse_timing);
	galp::format::ZeroCopyReadTiming full_timing {};
	auto full = reader.read_rowgroup_zero_copy(0, &full_timing);
	const auto external_bytes = reader.rowgroup_storage_bytes(0);
	auto external_backing = std::make_shared<std::vector<std::byte>>(external_bytes);
	galp::format::ZeroCopyReadTiming external_timing {};
	auto external = reader.read_rowgroup_zero_copy_into(0,
	                                                   std::static_pointer_cast<void>(external_backing),
	                                                   external_backing->data(),
	                                                   external_backing->size(),
	                                                   /*backing_is_pinned=*/true,
	                                                   &external_timing);
	EXPECT_TRUE(external.backing_is_pinned);
	EXPECT_TRUE(external_timing.used_pinned_backing);
	EXPECT_EQ(external_timing.storage_bytes, full_timing.storage_bytes);
	auto external_rowgroup = reader.materialize_zero_copy_rowgroup(std::move(external));
	ASSERT_FALSE(external_rowgroup.columns.empty());
	for (const auto& column : external_rowgroup.columns) {
		if (!column.skip_decompress) {
			EXPECT_TRUE(column.host_owned_by_backing);
			EXPECT_TRUE(column.backing_is_pinned);
			EXPECT_EQ(column.backing_base, external_backing->data());
			EXPECT_EQ(column.backing_bytes, external_bytes);
		}
	}
	galp::execution::free_rowgroup(external_rowgroup);
	ASSERT_TRUE(sparse_timing.sparse_read_supported);
	ASSERT_TRUE(sparse_timing.used_sparse_read);
	EXPECT_TRUE(sparse_timing.sparse_fallback_reason.empty());
	EXPECT_GT(sparse_timing.pread_count, 1U);
	EXPECT_LT(sparse_timing.storage_bytes, sparse_timing.full_storage_bytes);
	EXPECT_EQ(sparse_timing.full_storage_bytes, full_timing.storage_bytes);

	const auto* columns = sparse.rowgroup_descriptor->m_column_descriptors();
	ASSERT_NE(columns, nullptr);
	ASSERT_EQ(columns->size(), 1U);
	const auto* segments = columns->Get(0)->segment_descriptors();
	ASSERT_NE(segments, nullptr);
	ASSERT_GT(segments->size(), 0U);
	for (flatbuffers::uoffset_t segment_index = 0; segment_index < segments->size(); ++segment_index) {
		const auto* descriptor = segments->Get(segment_index);
		auto sparse_segment = fastlanes::make_segment_view(sparse.backing_span, *descriptor);
		auto full_segment   = fastlanes::make_segment_view(full.backing_span, *descriptor);
		for (const uint32_t vector : {1U, 6U}) {
			sparse_segment.PointTo(vector);
			full_segment.PointTo(vector);
			ASSERT_EQ(sparse_segment.Size(), full_segment.Size());
			EXPECT_EQ(std::memcmp(sparse_segment.data, full_segment.data, sparse_segment.Size()), 0);
		}
	}

	galp::format::ZeroCopyReadTiming fallback_timing {};
	auto fallback = reader.read_rowgroup_zero_copy_selected_vectors(
	    0, {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U}, &fallback_timing);
	(void)fallback;
	EXPECT_TRUE(fallback_timing.sparse_read_supported);
	EXPECT_FALSE(fallback_timing.used_sparse_read);
	EXPECT_EQ(fallback_timing.sparse_fallback_reason, "selected-vectors-cover-full-rowgroup");
	EXPECT_EQ(fallback_timing.storage_bytes, fallback_timing.full_storage_bytes);
	EXPECT_EQ(fallback_timing.pread_count, 1U);
}

TEST(Reader, RowgroupOnlyModeSkipsSparseIndexAndPreservesFullReads) {
	const auto fls_path = make_sparse_vector_read_fixture();
	galp::format::FlsReaderOptions options;
	options.load_column_names                 = false;
	options.enable_sparse_vector_reads         = false;
	options.build_shared_zero_copy_schema_plan = false;
	galp::format::FlsReader reader(fls_path, options);
	ASSERT_EQ(reader.rowgroup_count(), 1U);
	EXPECT_FALSE(reader.has_sparse_vector_bundle());
	std::string capability_reason;
	EXPECT_FALSE(reader.sparse_vector_read_supported(0U, &capability_reason));
	EXPECT_EQ(capability_reason, "sparse-access-index-disabled");
	EXPECT_THROW((void)reader.compile_sparse_vector_read_plan(0U, {1U}), std::runtime_error);

	galp::format::ZeroCopyReadTiming timing {};
	auto zero_copy = reader.read_rowgroup_zero_copy(0U, &timing);
	EXPECT_EQ(timing.pread_count, 1U);
	EXPECT_EQ(timing.storage_bytes, reader.rowgroup_storage_bytes(0U));
	auto rowgroup = reader.materialize_zero_copy_rowgroup(std::move(zero_copy));
	ASSERT_FALSE(rowgroup.columns.empty());
	galp::execution::free_rowgroup(rowgroup);
}

TEST(Reader, CompiledSparseVectorReadPlanMatchesSourceReadsAndIsReaderBound) {
	const auto fls_path = make_sparse_vector_read_fixture();
	galp::format::FlsReader reader(fls_path);
	const auto plan = reader.compile_sparse_vector_read_plan(0, {6U, 1U, 6U});
	EXPECT_FALSE(plan.empty());
	EXPECT_EQ(plan.rowgroup_index(), 0U);
	EXPECT_EQ(plan.selected_vector_count(), 2U);
	EXPECT_TRUE(plan.uses_sparse_read());
	EXPECT_FALSE(plan.uses_packed_device_scatter());
	EXPECT_EQ(plan.full_storage_bytes(), reader.rowgroup_storage_bytes(0U));
	EXPECT_EQ(plan.backend(), galp::format::SparseVectorReadPlan::Backend::kSourceRanges);
	EXPECT_GT(plan.estimated_pread_count(), 0U);

	galp::format::ZeroCopyReadTiming compiled_timing {};
	auto compiled = reader.read_rowgroup_zero_copy_compiled(plan, &compiled_timing);
	galp::format::ZeroCopyReadTiming reference_timing {};
	auto reference = reader.read_rowgroup_zero_copy_selected_vectors(0, {1U, 6U}, &reference_timing);
	ASSERT_EQ(compiled.backing_span.size(), reference.backing_span.size());
	EXPECT_EQ(std::memcmp(compiled.backing_span.data(),
	                      reference.backing_span.data(),
	                      compiled.backing_span.size()),
	          0);
	EXPECT_EQ(compiled_timing.storage_bytes, plan.storage_bytes());
	EXPECT_EQ(compiled_timing.storage_bytes, reference_timing.storage_bytes);
	EXPECT_EQ(compiled_timing.pread_count, reference_timing.pread_count);
	EXPECT_EQ(compiled_timing.pread_count, plan.estimated_pread_count());

	galp::format::FlsReader other_reader(fls_path);
	EXPECT_THROW(other_reader.read_rowgroup_zero_copy_compiled(plan), std::invalid_argument);
}

TEST(Reader, SparseVectorBundlePreservesSegmentsAndCollapsesPhysicalReads) {
	const ScopedEnvironmentVariable bundle_policy("GALP_VECTOR_BUNDLE_READ_POLICY", nullptr);
	const auto fls_path = make_sparse_vector_bundle_fixture();
	galp::format::FlsReader exact_reader(fls_path);
	galp::format::ZeroCopyReadTiming exact_timing {};
	auto exact = exact_reader.read_rowgroup_zero_copy_selected_vectors(0, {1U, 3U, 6U}, &exact_timing);
	ASSERT_TRUE(exact_timing.used_sparse_read);
	ASSERT_FALSE(exact_timing.used_vector_bundle_read);

	const auto bundle_path = galp::format::sparse_vector_bundle_path(fls_path);
	galp::format::write_sparse_vector_bundle(fls_path, bundle_path);
	galp::format::FlsReader bundled_reader(fls_path);
	ASSERT_TRUE(bundled_reader.has_sparse_vector_bundle());
	galp::format::ZeroCopyReadTiming bundle_timing {};
	auto bundled = bundled_reader.read_rowgroup_zero_copy_selected_vectors(0, {1U, 3U, 6U}, &bundle_timing);
	ASSERT_TRUE(bundle_timing.used_sparse_read);
	ASSERT_TRUE(bundle_timing.used_vector_bundle_read);
	// The immutable entrypoint/shared prefix is cached when the reader opens,
	// so runtime I/O consists only of the three disjoint selected-vector runs.
	EXPECT_EQ(bundle_timing.pread_count, 3U);
	EXPECT_LT(bundle_timing.pread_count, exact_timing.pread_count);
	EXPECT_EQ(bundle_timing.storage_bytes, exact_timing.storage_bytes);
	EXPECT_EQ(bundle_timing.full_storage_bytes, exact_timing.full_storage_bytes);

	const auto* columns = bundled.rowgroup_descriptor->m_column_descriptors();
	ASSERT_NE(columns, nullptr);
	for (flatbuffers::uoffset_t column_index = 0; column_index < columns->size(); ++column_index) {
		const auto* segments = columns->Get(column_index)->segment_descriptors();
		ASSERT_NE(segments, nullptr);
		for (flatbuffers::uoffset_t segment_index = 0; segment_index < segments->size(); ++segment_index) {
			const auto* descriptor = segments->Get(segment_index);
			auto bundle_segment = fastlanes::make_segment_view(bundled.backing_span, *descriptor);
			auto exact_segment  = fastlanes::make_segment_view(exact.backing_span, *descriptor);
			for (const uint32_t vector : {1U, 3U, 6U}) {
				bundle_segment.PointTo(vector);
				exact_segment.PointTo(vector);
				ASSERT_EQ(bundle_segment.Size(), exact_segment.Size());
				EXPECT_EQ(std::memcmp(bundle_segment.data, exact_segment.data, bundle_segment.Size()), 0);
			}
		}
	}
}

TEST(Reader, CompiledSparseVectorBundlePlansMatchLogicalPackedAndEnvelopeReads) {
	const auto fls_path = make_sparse_vector_bundle_fixture();
	const auto bundle_path = galp::format::sparse_vector_bundle_path(fls_path);
	galp::format::write_sparse_vector_bundle(fls_path, bundle_path);
	galp::format::FlsReader reader(fls_path);
	const std::vector<uint32_t> selected {1U, 3U, 6U};
	const auto expect_selected_segments_equal = [&](const galp::format::ZeroCopyRowgroup& actual,
	                                                const galp::format::ZeroCopyRowgroup& expected) {
		const auto* columns = actual.rowgroup_descriptor->m_column_descriptors();
		ASSERT_NE(columns, nullptr);
		for (flatbuffers::uoffset_t column_index = 0; column_index < columns->size(); ++column_index) {
			const auto* segments = columns->Get(column_index)->segment_descriptors();
			ASSERT_NE(segments, nullptr);
			for (flatbuffers::uoffset_t segment_index = 0; segment_index < segments->size(); ++segment_index) {
				const auto* descriptor = segments->Get(segment_index);
				auto actual_segment = fastlanes::make_segment_view(actual.backing_span, *descriptor);
				auto expected_segment = fastlanes::make_segment_view(expected.backing_span, *descriptor);
				for (const auto vector : selected) {
					actual_segment.PointTo(vector);
					expected_segment.PointTo(vector);
					ASSERT_EQ(actual_segment.Size(), expected_segment.Size());
					EXPECT_EQ(std::memcmp(actual_segment.data, expected_segment.data, actual_segment.Size()), 0)
					    << "column=" << column_index << " segment=" << segment_index << " vector=" << vector;
				}
			}
		}
	};

	const auto logical_plan = reader.compile_sparse_vector_read_plan(0, selected);
	auto compiled_logical = reader.read_rowgroup_zero_copy_compiled(logical_plan);
	auto reference_logical = reader.read_rowgroup_zero_copy_selected_vectors(0, selected);
	expect_selected_segments_equal(compiled_logical, reference_logical);

	const auto packed_plan = reader.compile_sparse_vector_read_plan(0, selected, true);
	EXPECT_TRUE(packed_plan.uses_packed_device_scatter());
	auto compiled_packed = reader.read_rowgroup_zero_copy_compiled(packed_plan);
	auto reference_packed = reader.read_rowgroup_zero_copy_selected_vectors_packed(0, selected);
	ASSERT_NE(compiled_packed.packed_device_payload, nullptr);
	ASSERT_NE(reference_packed.packed_device_payload, nullptr);
	const auto& compiled_payload = *compiled_packed.packed_device_payload;
	const auto& reference_payload = *reference_packed.packed_device_payload;
	ASSERT_EQ(compiled_payload.ranges.size(), reference_payload.ranges.size());
	for (size_t range_index = 0; range_index < compiled_payload.ranges.size(); ++range_index) {
		EXPECT_EQ(compiled_payload.ranges[range_index].packed_offset,
		          reference_payload.ranges[range_index].packed_offset);
		EXPECT_EQ(compiled_payload.ranges[range_index].logical_offset,
		          reference_payload.ranges[range_index].logical_offset);
		EXPECT_EQ(compiled_payload.ranges[range_index].size,
		          reference_payload.ranges[range_index].size);
	}
	ASSERT_EQ(compiled_payload.packed_bytes, reference_payload.packed_bytes);
	EXPECT_EQ(std::memcmp(compiled_payload.packed_data,
	                      reference_payload.packed_data,
	                      compiled_payload.packed_bytes),
	          0);

	const ScopedEnvironmentVariable envelope_policy("GALP_VECTOR_BUNDLE_READ_POLICY", "envelope");
	const auto envelope_plan = reader.compile_sparse_vector_read_plan(0, selected);
	EXPECT_EQ(envelope_plan.backend(), galp::format::SparseVectorReadPlan::Backend::kBundleEnvelope);
	EXPECT_EQ(envelope_plan.estimated_pread_count(), 1U);
	galp::format::ZeroCopyReadTiming envelope_timing {};
	auto compiled_envelope = reader.read_rowgroup_zero_copy_compiled(envelope_plan, &envelope_timing);
	auto reference_envelope = reader.read_rowgroup_zero_copy_selected_vectors(0, selected);
	EXPECT_TRUE(envelope_timing.used_vector_bundle_envelope_read);
	EXPECT_EQ(envelope_timing.pread_count, 1U);
	expect_selected_segments_equal(compiled_envelope, reference_envelope);
}

TEST(Reader, SparseVectorBundlePackedDeviceRangesRebuildSelectedSegments) {
	const auto fls_path = make_sparse_vector_bundle_fixture();
	const auto bundle_path = galp::format::sparse_vector_bundle_path(fls_path);
	galp::format::write_sparse_vector_bundle(fls_path, bundle_path);
	galp::format::FlsReader reader(fls_path);
	galp::format::ZeroCopyReadTiming packed_timing {};
	auto packed = reader.read_rowgroup_zero_copy_selected_vectors_packed(0, {1U, 3U, 6U}, &packed_timing);
	auto full   = reader.read_rowgroup_zero_copy(0);
	ASSERT_TRUE(packed_timing.used_vector_bundle_read);
	ASSERT_NE(packed.packed_device_payload, nullptr);
	const auto& payload = *packed.packed_device_payload;
	std::vector<std::byte> rebuilt(payload.logical_bytes, std::byte {0});
	for (const auto& range : payload.ranges) {
		ASSERT_LE(range.packed_offset + range.size, payload.packed_bytes);
		ASSERT_LE(range.logical_offset + range.size, payload.logical_bytes);
		std::memcpy(rebuilt.data() + range.logical_offset, payload.packed_data + range.packed_offset, range.size);
	}
	const fastlanes::span<std::byte> rebuilt_span {rebuilt.data(), rebuilt.size()};
	const auto* columns = packed.rowgroup_descriptor->m_column_descriptors();
	ASSERT_NE(columns, nullptr);
	for (flatbuffers::uoffset_t column_index = 0; column_index < columns->size(); ++column_index) {
		const auto* segments = columns->Get(column_index)->segment_descriptors();
		ASSERT_NE(segments, nullptr);
		for (flatbuffers::uoffset_t segment_index = 0; segment_index < segments->size(); ++segment_index) {
			const auto* descriptor = segments->Get(segment_index);
			auto rebuilt_segment = fastlanes::make_segment_view(rebuilt_span, *descriptor);
			auto full_segment    = fastlanes::make_segment_view(full.backing_span, *descriptor);
			for (const uint32_t vector : {1U, 3U, 6U}) {
				rebuilt_segment.PointTo(vector);
				full_segment.PointTo(vector);
				ASSERT_EQ(rebuilt_segment.Size(), full_segment.Size());
				EXPECT_EQ(std::memcmp(rebuilt_segment.data, full_segment.data, rebuilt_segment.Size()), 0);
			}
		}
	}
}

TEST(Reader, SparseVectorBundlePackedDeviceRangesDecodeSelectedVectorsOnGpu) {
	if (!cuda_available_for_reader_tests()) {
		GTEST_SKIP() << "CUDA device not available.";
	}
	const auto fls_path = make_sparse_vector_bundle_fixture();
	const auto bundle_path = galp::format::sparse_vector_bundle_path(fls_path);
	galp::format::write_sparse_vector_bundle(fls_path, bundle_path);
	galp::format::FlsReader reader(fls_path);
	auto zero_copy = reader.read_rowgroup_zero_copy_selected_vectors_packed(0, {1U, 3U, 6U});
	auto rowgroup  = reader.materialize_zero_copy_rowgroup(std::move(zero_copy));
	auto expressions = galp::expression::assemble(rowgroup);

	galp::runtime::ExecutionWorkset      workset {};
	galp::runtime::ExecutionWorksetGuard guard(workset);
	galp::execution::ExecutionConfig     config {};
	config.unpack_n_vectors = 1U;
	config.launch_strategy  = galp::execution::LaunchStrategy::MixedDispatch;
	config.write_out        = true;
	const std::vector<uint32_t> selected_vectors {1U, 3U, 6U};
	galp::runtime::append_rowgroup_columns_selected_vectors(workset, rowgroup, config, selected_vectors);
	galp::runtime::upload_workset(workset, config);
	galp::runtime::run_workset(workset, 1U, config);
	const auto result = galp::runtime::materialize_workset(workset, expressions, config);
	ASSERT_EQ(result.columns.size(), 8U);
	for (size_t column = 0; column < result.columns.size(); ++column) {
		ASSERT_TRUE(result.columns[column].has_value());
		const auto& output = std::get<std::shared_ptr<int8_t[]>>(result.columns[column]->values);
		for (size_t selected_index = 0; selected_index < selected_vectors.size(); ++selected_index) {
			for (size_t offset = 0; offset < galp::codec::consts::VALUES_PER_VECTOR; ++offset) {
				const size_t source_row = static_cast<size_t>(selected_vectors[selected_index]) *
				                              galp::codec::consts::VALUES_PER_VECTOR +
				                          offset;
				const auto expected = static_cast<int8_t>(
				    static_cast<int>((source_row * 17U + column * 29U + source_row / 1024U) % 101U) - 50);
				ASSERT_EQ(output[selected_index * galp::codec::consts::VALUES_PER_VECTOR + offset], expected)
				    << "column=" << column << " selected_index=" << selected_index << " offset=" << offset;
			}
		}
	}
	galp::execution::free_rowgroup(rowgroup);
}

TEST(Reader, SparseVectorBundleEnvelopeUsesOneReadAndPreservesSegments) {
	const auto fls_path = make_sparse_vector_bundle_fixture();
	galp::format::FlsReader exact_reader(fls_path);
	galp::format::ZeroCopyReadTiming exact_timing {};
	auto exact = exact_reader.read_rowgroup_zero_copy_selected_vectors(0, {1U, 3U, 6U}, &exact_timing);
	ASSERT_TRUE(exact_timing.used_sparse_read);
	ASSERT_FALSE(exact_timing.used_vector_bundle_read);

	const auto bundle_path = galp::format::sparse_vector_bundle_path(fls_path);
	galp::format::write_sparse_vector_bundle(fls_path, bundle_path);
	const ScopedEnvironmentVariable bundle_policy("GALP_VECTOR_BUNDLE_READ_POLICY", "envelope");
	galp::format::FlsReader bundled_reader(fls_path);
	galp::format::ZeroCopyReadTiming bundle_timing {};
	auto bundled = bundled_reader.read_rowgroup_zero_copy_selected_vectors(0, {1U, 3U, 6U}, &bundle_timing);
	ASSERT_TRUE(bundle_timing.used_sparse_read);
	ASSERT_TRUE(bundle_timing.used_vector_bundle_read);
	ASSERT_TRUE(bundle_timing.used_vector_bundle_envelope_read);
	EXPECT_EQ(bundle_timing.pread_count, 1U);
	EXPECT_GT(bundle_timing.storage_bytes, exact_timing.storage_bytes);
	EXPECT_LT(bundle_timing.storage_bytes, bundle_timing.full_storage_bytes);
	EXPECT_EQ(bundle_timing.full_storage_bytes, exact_timing.full_storage_bytes);

	const auto* columns = bundled.rowgroup_descriptor->m_column_descriptors();
	ASSERT_NE(columns, nullptr);
	for (flatbuffers::uoffset_t column_index = 0; column_index < columns->size(); ++column_index) {
		const auto* segments = columns->Get(column_index)->segment_descriptors();
		ASSERT_NE(segments, nullptr);
		for (flatbuffers::uoffset_t segment_index = 0; segment_index < segments->size(); ++segment_index) {
			const auto* descriptor = segments->Get(segment_index);
			auto bundle_segment = fastlanes::make_segment_view(bundled.backing_span, *descriptor);
			auto exact_segment  = fastlanes::make_segment_view(exact.backing_span, *descriptor);
			for (const uint32_t vector : {1U, 3U, 6U}) {
				bundle_segment.PointTo(vector);
				exact_segment.PointTo(vector);
				ASSERT_EQ(bundle_segment.Size(), exact_segment.Size());
				EXPECT_EQ(std::memcmp(bundle_segment.data, exact_segment.data, bundle_segment.Size()), 0);
			}
		}
	}
}

TEST(Reader, SparseVectorCapabilityRejectsUnsupportedNestedChildOperator) {
	flatbuffers::FlatBufferBuilder builder;
	const std::vector<fastlanes::OperatorToken> child_tokens {fastlanes::OperatorToken::EXP_NULL_I16};
	const auto child_rpn = fastlanes::CreateRPNDirect(builder, &child_tokens);
	const auto child = fastlanes::CreateColumnDescriptor(
	    builder, fastlanes::DataType::INT16, child_rpn);
	const auto children = builder.CreateVector(std::vector<flatbuffers::Offset<fastlanes::ColumnDescriptor>> {child});
	const std::vector<fastlanes::OperatorToken> parent_tokens {fastlanes::OperatorToken::EXP_UNCOMPRESSED_I08};
	const auto parent_rpn = fastlanes::CreateRPNDirect(builder, &parent_tokens);
	const auto parent = fastlanes::CreateColumnDescriptor(
	    builder, fastlanes::DataType::INT8, parent_rpn, 0, 0, children);
	builder.Finish(parent);

	const auto* descriptor = flatbuffers::GetRoot<fastlanes::ColumnDescriptor>(builder.GetBufferPointer());
	ASSERT_NE(descriptor, nullptr);
	std::string reason;
	EXPECT_FALSE(galp::format::detail::validate_sparse_column_operators(*descriptor, &reason));
	EXPECT_EQ(reason, "operator-sparse-read-unsupported:EXP_NULL_I16");
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
	std::unordered_set<fastlanes::OperatorToken> out;
	for (const auto& capability : galp::expression::kOperatorCapabilities) {
		if (capability.gpu_supported) {
			out.insert(capability.token);
		}
	}
	return out;
}

bool cuda_available_for_reader_tests() {
	int        device_count = 0;
	const auto cuda_status  = cudaGetDeviceCount(&device_count);
	return cuda_status == cudaSuccess && device_count > 0;
}

struct RecordingTableExecutionObserver : galp::runtime::NoopTableExecutionObserver {
	std::vector<size_t> read_rowgroups;
	std::vector<bool>   read_from_prefetch;

	void on_rowgroup_read(const galp::runtime::RowgroupReadResult& result, const bool from_prefetch) {
		read_rowgroups.push_back(result.rowgroup_index);
		read_from_prefetch.push_back(from_prefetch);
	}
};

template <typename T>
typename galp::codec::utils::same_width_uint<T>::type to_column_bits(const T value) {
	typename galp::codec::utils::same_width_uint<T>::type out {};
	std::memcpy(&out, &value, sizeof(T));
	return out;
}

template <typename T>
galp::codec::host::CROSSRLEColumn<T> make_test_cross_rle_column(const std::vector<T>&        values,
                                                                const std::vector<uint32_t>& lengths,
                                                                const size_t                n_values) {
	using UIntT = typename galp::codec::utils::same_width_uint<T>::type;
	if (values.size() != lengths.size()) {
		throw std::runtime_error("test CROSS_RLE values/lengths mismatch");
	}

	const size_t n_runs = values.size();
	auto*        raw_values = n_runs == 0 ? nullptr : new UIntT[n_runs];
	auto*        raw_lengths = n_runs == 0 ? nullptr : new uint32_t[n_runs];
	auto*        raw_positions = n_runs == 0 ? nullptr : new uint32_t[n_runs];

	uint32_t pos = 0;
	for (size_t i = 0; i < n_runs; ++i) {
		raw_values[i]    = to_column_bits(values[i]);
		raw_lengths[i]   = lengths[i];
		raw_positions[i] = pos;
		pos += lengths[i];
	}

	const size_t n_vecs = galp::codec::utils::get_n_vecs_from_size(n_values);
	auto*        offsets = new uint32_t[n_vecs + 1];
	size_t       cur = 0;
	size_t       run = 0;
	for (size_t vec = 0; vec < n_vecs; ++vec) {
		const size_t target_start = vec * galp::codec::consts::VALUES_PER_VECTOR;
		while (run < n_runs && cur + lengths[run] <= target_start) {
			cur += lengths[run];
			++run;
		}
		offsets[vec] = static_cast<uint32_t>(run);
	}
	offsets[n_vecs] = static_cast<uint32_t>(n_runs);

	return galp::codec::host::CROSSRLEColumn<T> {n_values,
	                                             n_runs,
	                                             raw_values,
	                                             raw_lengths,
	                                             offsets,
	                                             raw_positions};
}

template <typename T>
void expect_cross_rle_decompresses(const std::vector<T>&        run_values,
                                   const std::vector<uint32_t>& run_lengths,
                                   const std::vector<T>&        expected) {
	if (!cuda_available_for_reader_tests()) {
		GTEST_SKIP() << "CUDA device not available for CROSS_RLE dispatch test.";
	}

	galp::execution::Rowgroup rowgroup {};
	rowgroup.n_values = expected.size();
	rowgroup.n_vecs   = galp::codec::utils::get_n_vecs_from_size(expected.size());
	rowgroup.n_tuples = expected.size();
	rowgroup.columns.push_back(galp::execution::Column {
	    "value",
	    std::is_same_v<T, int8_t> ? fastlanes::OperatorToken::EXP_CROSS_RLE_I08
	                              : fastlanes::OperatorToken::EXP_CROSS_RLE_I16,
	    make_test_cross_rle_column(run_values, run_lengths, expected.size())});

	auto expressions = galp::expression::assemble(rowgroup);
	ASSERT_EQ(expressions.size(), 1U);
	const auto result = galp::execution::decompress_rowgroup(expressions);
	ASSERT_EQ(result.columns.size(), 1U);
	ASSERT_TRUE(result.columns[0].has_value());

	const auto& values = result.columns[0]->values;
	ASSERT_TRUE(std::holds_alternative<std::shared_ptr<T[]>>(values));
	const auto out = std::get<std::shared_ptr<T[]>>(values);
	ASSERT_NE(out, nullptr);
	for (size_t i = 0; i < expected.size(); ++i) {
		EXPECT_EQ(out[i], expected[i]) << "row=" << i;
	}
}

std::filesystem::path make_cross_rle_i16_fls_fixture() {
	const std::filesystem::path root = std::filesystem::path {GALP_TEST_DATA_DIR} / "cross_rle_i16";
	std::filesystem::remove_all(root);
	std::filesystem::create_directories(root);

	const auto csv_path    = root / "generated.csv";
	const auto schema_path = root / "schema.json";
	const auto fls_path    = root / "data.fls";

	{
		std::ofstream schema(schema_path);
		schema << R"({"columns":[{"name":"value","type":"FLS_I16"}]})";
	}
	{
		std::ofstream csv(csv_path);
		for (int row = 0; row < 1536; ++row) {
			int value = 0;
			if (row < 256) {
				value = -32768;
			} else if (row < 512) {
				value = -7;
			} else if (row < 1024) {
				value = 4096;
			} else {
				value = 32767;
			}
			csv << value << '\n';
		}
	}

	fastlanes::Connection writer;
	writer.set_n_vectors_per_rowgroup(2)
	    .force_schema_pool({fastlanes::OperatorToken::EXP_CROSS_RLE_I16})
	    .read_csv(root)
	    .to_fls(fls_path);
	return fls_path;
}

std::filesystem::path make_forced_i16_fls_fixture(const fastlanes::OperatorToken token,
                                                  const std::string&              label,
                                                  const std::vector<int16_t>&     values) {
	const std::filesystem::path root = std::filesystem::path {GALP_TEST_DATA_DIR} / label;
	std::filesystem::remove_all(root);
	std::filesystem::create_directories(root);

	const auto fls_path = root / "data.fls";
	const std::array<fastlanes::MemoryColumn, 1> columns {
	    fastlanes::MemoryColumn {"value", std::span<const int16_t>(values.data(), values.size())}};
	const fastlanes::MemoryTable table {std::span<const fastlanes::MemoryColumn>(columns)};
	fastlanes::MemoryTableOptions options;
	options.n_vectors_per_rowgroup = 70;
	// Forced schemas intentionally bypass wizard statistics. Constants need
	// the wizard's max-value metadata, whereas uncompressed is safely forced.
	if (token != fastlanes::OperatorToken::EXP_CONSTANT_I16) {
		options.force_schema  = true;
		options.forced_schema = {token};
	}
	fastlanes::write_memory_table_to_fls(table, fls_path, options);
	return fls_path;
}

void expect_forced_i16_gpu_roundtrip(const fastlanes::OperatorToken token,
                                     const std::string&              label,
                                     const std::vector<int16_t>&     expected) {
	const auto fls_path = make_forced_i16_fls_fixture(token, label, expected);

	const auto  td_handle = galp::format::detail::load_table_descriptor(fls_path);
	const auto* td        = td_handle.Get();
	ASSERT_NE(td, nullptr);
	ASSERT_NE(td->m_rowgroup_descriptors(), nullptr);
	ASSERT_EQ(td->m_rowgroup_descriptors()->size(), 1U);
	const auto* rg = td->m_rowgroup_descriptors()->Get(0);
	ASSERT_NE(rg, nullptr);
	ASSERT_GT(rg->m_n_vec(), 64U);
	ASSERT_NE(rg->m_column_descriptors(), nullptr);
	ASSERT_EQ(rg->m_column_descriptors()->size(), 1U);
	const auto* col = rg->m_column_descriptors()->Get(0);
	ASSERT_NE(col, nullptr);
	ASSERT_NE(col->encoding_rpn(), nullptr);
	ASSERT_NE(col->encoding_rpn()->operator_tokens(), nullptr);
	ASSERT_EQ(col->encoding_rpn()->operator_tokens()->size(), 1U);
	ASSERT_EQ(col->encoding_rpn()->operator_tokens()->Get(0), token);

	galp::format::FlsReader reader(fls_path);
	auto                    rowgroup = reader.read_rowgroup(0);
	ASSERT_EQ(rowgroup.n_tuples, expected.size());
	ASSERT_EQ(rowgroup.columns.size(), 1U);
	EXPECT_EQ(rowgroup.columns[0].token, token);
	if (token == fastlanes::OperatorToken::EXP_CONSTANT_I16) {
		EXPECT_TRUE((std::holds_alternative<galp::codec::host::CONSTANTColumn<int16_t>>(rowgroup.columns[0].host)));
	} else {
		EXPECT_TRUE((std::holds_alternative<galp::codec::host::BPColumn<int16_t>>(rowgroup.columns[0].host)));
	}

	auto expressions = galp::expression::assemble(rowgroup);
	ASSERT_EQ(expressions.size(), 1U);
	galp::execution::ExecutionConfig config {};
	config.unpack_n_vectors = 1;
	config.write_out        = true;
	const auto result       = galp::execution::decompress_rowgroup(expressions, config);
	ASSERT_EQ(result.columns.size(), 1U);
	ASSERT_TRUE(result.columns[0].has_value());
	ASSERT_EQ(result.columns[0]->meta.value_count, rowgroup.n_values);
	const auto& output = std::get<std::shared_ptr<int16_t[]>>(result.columns[0]->values);
	ASSERT_NE(output, nullptr);
	for (size_t row = 0; row < expected.size(); ++row) {
		ASSERT_EQ(output[row], expected[row]) << "token=" << fastlanes::token_to_string(token) << " row=" << row;
	}
	for (size_t row = expected.size(); row < rowgroup.n_values; ++row) {
		ASSERT_EQ(output[row], expected.back())
		    << "padded token=" << fastlanes::token_to_string(token) << " row=" << row;
	}
}

struct ExternalDictI16Fixture {
	std::filesystem::path   root;
	std::filesystem::path   fls_path;
	std::vector<int16_t>    source;
	std::vector<int16_t>    mapped;
	fastlanes::OperatorToken expected_ref_token;
};

ExternalDictI16Fixture make_external_dict_i16_fixture(const size_t cardinality, const std::string& label) {
	if (cardinality == 0 || cardinality > 1024) {
		throw std::invalid_argument("external dictionary fixture cardinality is outside the test range");
	}
	const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
	ExternalDictI16Fixture fixture;
	fixture.root = std::filesystem::temp_directory_path() /
	               ("galp_external_dict_i16_" + label + "_" + std::to_string(suffix));
	fixture.fls_path = fixture.root / "data.fls";
	fixture.expected_ref_token = cardinality <= 256U ? fastlanes::OperatorToken::EXP_DICT_I16_U08
	                                                : fastlanes::OperatorToken::EXP_DICT_I16_U16;
	std::filesystem::create_directories(fixture.root);

	const size_t n_values = 12U * galp::codec::consts::VALUES_PER_VECTOR;
	fixture.source.resize(n_values);
	fixture.mapped.resize(n_values);
	for (size_t row = 0; row < n_values; ++row) {
		const size_t rank    = row % cardinality;
		const size_t mapping = (rank * 37U) % cardinality;
		fixture.source[row]  = static_cast<int16_t>(static_cast<int>(rank) - static_cast<int>(cardinality / 2U));
		fixture.mapped[row]  = static_cast<int16_t>(static_cast<int>(mapping) * 41 - 15000);
	}
	{
		std::ofstream schema(fixture.root / "schema.json");
		schema << R"({"columns":[{"name":"source","type":"FLS_I16"},{"name":"mapped","type":"FLS_I16"}]})";
	}
	{
		std::ofstream csv(fixture.root / "generated.csv");
		for (size_t row = 0; row < n_values; ++row) {
			csv << fixture.source[row] << '|' << fixture.mapped[row] << '\n';
		}
	}
	fastlanes::Connection writer;
	writer.set_n_vectors_per_rowgroup(12).read_csv(fixture.root).to_fls(fixture.fls_path);
	return fixture;
}

void set_external_dict_source(galp::execution::Column& column, const uint32_t source_index) {
	bool updated = false;
	std::visit(
	    [&](auto& payload) {
		    using PayloadT = std::decay_t<decltype(payload)>;
		    if constexpr (std::is_same_v<PayloadT, galp::codec::host::DICTREFColumn<int16_t, uint8_t>> ||
		                  std::is_same_v<PayloadT, galp::codec::host::DICTREFColumn<int16_t, uint16_t>>) {
			    payload.index_column_index = source_index;
			    updated                     = true;
		    }
	    },
	    column.host);
	if (!updated) {
		throw std::runtime_error("test expected an unresolved I16 external dictionary");
	}
}

galp::execution::Column make_test_alias_column(const size_t n_values, const size_t target, const std::string& name) {
	galp::execution::Column alias {};
	alias.name            = name;
	alias.token           = fastlanes::OperatorToken::EXP_EQUAL;
	alias.host            = galp::codec::host::CONSTANTColumn<int16_t> {n_values, 0};
	alias.skip_decompress = true;
	alias.alias_of        = target;
	return alias;
}

TEST(ExternalDictionaryI16, Full) {
	for (const auto& [cardinality, label] :
	     std::array<std::pair<size_t, const char*>, 2> {{{61U, "u8"}, {300U, "u16"}}}) {
		auto fixture = make_external_dict_i16_fixture(cardinality, label);
		SCOPED_TRACE(label);

		const auto  descriptor_handle = galp::format::detail::load_table_descriptor(fixture.fls_path);
		const auto* descriptor        = descriptor_handle.Get();
		ASSERT_NE(descriptor, nullptr);
		ASSERT_NE(descriptor->m_rowgroup_descriptors(), nullptr);
		ASSERT_EQ(descriptor->m_rowgroup_descriptors()->size(), 1U);
		const auto* rowgroup_descriptor = descriptor->m_rowgroup_descriptors()->Get(0);
		ASSERT_NE(rowgroup_descriptor, nullptr);
		ASSERT_NE(rowgroup_descriptor->m_column_descriptors(), nullptr);
		ASSERT_EQ(rowgroup_descriptor->m_column_descriptors()->size(), 2U);
		const auto* ref_descriptor = rowgroup_descriptor->m_column_descriptors()->Get(1);
		ASSERT_NE(ref_descriptor, nullptr);
		ASSERT_NE(ref_descriptor->encoding_rpn(), nullptr);
		ASSERT_EQ(ref_descriptor->encoding_rpn()->operator_tokens()->Get(0), fixture.expected_ref_token);

		galp::format::FlsReader reader(fixture.fls_path);
		auto rowgroup = reader.read_rowgroup_zero_copy_materialized(0);
		ASSERT_EQ(rowgroup.columns.size(), 2U);
		ASSERT_TRUE(rowgroup.backing_storage);
		EXPECT_EQ(rowgroup.columns[1].token, fixture.expected_ref_token);

		const uint16_t* borrowed_keys = nullptr;
		if (fixture.expected_ref_token == fastlanes::OperatorToken::EXP_DICT_I16_U08) {
			const auto& dict_ref =
			    std::get<galp::codec::host::DICTREFColumn<int16_t, uint8_t>>(rowgroup.columns[1].host);
			EXPECT_FALSE(dict_ref.keys.owns());
			borrowed_keys = dict_ref.keys.get();
		} else {
			const auto& dict_ref =
			    std::get<galp::codec::host::DICTREFColumn<int16_t, uint16_t>>(rowgroup.columns[1].host);
			EXPECT_FALSE(dict_ref.keys.owns());
			borrowed_keys = dict_ref.keys.get();
		}
		ASSERT_NE(borrowed_keys, nullptr);

		auto expressions = galp::expression::assemble(rowgroup);
		galp::execution::resolve_dict_refs(expressions);
		EXPECT_FALSE(galp::execution::has_unresolved_dict_ref(rowgroup.columns[1].host));
		EXPECT_TRUE(rowgroup.columns[1].token == fastlanes::OperatorToken::EXP_DICT_I16_FFOR_U08 ||
		            rowgroup.columns[1].token == fastlanes::OperatorToken::EXP_DICT_I16_FFOR_SLPATCH_U08 ||
		            rowgroup.columns[1].token == fastlanes::OperatorToken::EXP_DICT_I16_FFOR_U16 ||
		            rowgroup.columns[1].token == fastlanes::OperatorToken::EXP_DICT_I16_FFOR_SLPATCH_U16);
		std::visit(
		    [&](const auto& column) {
			    using ColumnT = std::decay_t<decltype(column)>;
			    if constexpr (std::is_same_v<ColumnT, galp::codec::host::DICTFFORColumn<int16_t, uint8_t>> ||
			                  std::is_same_v<ColumnT, galp::codec::host::DICTFFORColumn<int16_t, uint16_t>> ||
			                  std::is_same_v<ColumnT, galp::codec::host::DICTSLPATCHColumn<int16_t, uint8_t>> ||
			                  std::is_same_v<ColumnT, galp::codec::host::DICTSLPATCHColumn<int16_t, uint16_t>>) {
				    EXPECT_EQ(column.keys.get(), borrowed_keys);
				    EXPECT_FALSE(column.keys.owns());
			    }
		    },
		    rowgroup.columns[1].host);

		if (cuda_available_for_reader_tests()) {
			galp::execution::ExecutionConfig config {};
			config.unpack_n_vectors = 4;
			config.write_out        = true;
			const auto result       = galp::execution::decompress_rowgroup(expressions, config);
			ASSERT_EQ(result.columns.size(), 2U);
			ASSERT_TRUE(result.columns[1].has_value());
			const auto& output = std::get<std::shared_ptr<int16_t[]>>(result.columns[1]->values);
			for (size_t row = 0; row < fixture.mapped.size(); ++row) {
				ASSERT_EQ(output[row], fixture.mapped[row]) << "row=" << row;
			}
		}
		galp::execution::free_rowgroup(rowgroup);
		std::filesystem::remove_all(fixture.root);
	}
}

TEST(ExternalDictionaryI16, AliasChain) {
	auto fixture = make_external_dict_i16_fixture(61U, "alias_chain");
	galp::format::FlsReader reader(fixture.fls_path);
	auto rowgroup = reader.read_rowgroup_zero_copy_materialized(0);
	ASSERT_EQ(rowgroup.columns.size(), 2U);
	set_external_dict_source(rowgroup.columns[1], 2U);
	rowgroup.columns.push_back(make_test_alias_column(rowgroup.n_values, 3U, "alias_1"));
	rowgroup.columns.push_back(make_test_alias_column(rowgroup.n_values, 0U, "alias_2"));

	auto expressions = galp::expression::assemble(rowgroup);
	EXPECT_NO_THROW(galp::execution::resolve_dict_refs(expressions));
	EXPECT_FALSE(galp::execution::has_unresolved_dict_ref(rowgroup.columns[1].host));
	EXPECT_EQ(rowgroup.columns[1].token, fastlanes::OperatorToken::EXP_DICT_I16_FFOR_U08);

	galp::execution::free_rowgroup(rowgroup);
	std::filesystem::remove_all(fixture.root);
}

TEST(ExternalDictionaryI16, DependencyErrors) {
	auto fixture = make_external_dict_i16_fixture(300U, "dependency_errors");
	galp::format::FlsReader reader(fixture.fls_path);
	const auto expect_error = [](auto&& action, const std::string& expected_fragment) {
		try {
			action();
			FAIL() << "expected DICTREF resolution to fail";
		} catch (const std::exception& error) {
			EXPECT_NE(std::string(error.what()).find(expected_fragment), std::string::npos) << error.what();
		}
	};

	{
		auto rowgroup = reader.read_rowgroup_zero_copy_materialized(0);
		set_external_dict_source(rowgroup.columns[1], 99U);
		auto expressions = galp::expression::assemble(rowgroup);
		expect_error([&] { galp::execution::resolve_dict_refs(expressions); }, "rowgroup has 2 columns");
		galp::execution::free_rowgroup(rowgroup);
	}
	{
		auto rowgroup  = reader.read_rowgroup_zero_copy_materialized(0);
		auto expressions = galp::expression::assemble(rowgroup);
		expressions[0].column = nullptr;
		expect_error([&] { galp::execution::resolve_dict_refs(expressions); }, "source column 0 is missing");
		galp::execution::free_rowgroup(rowgroup);
	}
	{
		auto rowgroup = reader.read_rowgroup_zero_copy_materialized(0);
		rowgroup.columns[0].host = galp::codec::host::CONSTANTColumn<int16_t> {rowgroup.n_values, 0};
		rowgroup.columns[0].token = fastlanes::OperatorToken::EXP_CONSTANT_I16;
		auto expressions = galp::expression::assemble(rowgroup);
		expect_error([&] { galp::execution::resolve_dict_refs(expressions); }, "expects a 16-bit BP/FFOR/SLPATCH");
		galp::execution::free_rowgroup(rowgroup);
	}

	std::filesystem::remove_all(fixture.root);
}

TEST(ExternalDictionaryI16, CycleDetection) {
	auto fixture = make_external_dict_i16_fixture(61U, "cycle");
	galp::format::FlsReader reader(fixture.fls_path);
	auto rowgroup = reader.read_rowgroup_zero_copy_materialized(0);
	set_external_dict_source(rowgroup.columns[1], 2U);
	rowgroup.columns.push_back(make_test_alias_column(rowgroup.n_values, 1U, "cycle_alias"));
	auto expressions = galp::expression::assemble(rowgroup);
	try {
		galp::execution::resolve_dict_refs(expressions);
		FAIL() << "expected a DICTREF dependency cycle";
	} catch (const std::exception& error) {
		EXPECT_NE(std::string(error.what()).find("DICTREF dependency cycle detected: 1 -> 2 -> 1"), std::string::npos)
		    << error.what();
	}
	galp::execution::free_rowgroup(rowgroup);
	std::filesystem::remove_all(fixture.root);
}

TEST(ExternalDictionaryI16, SelectedVectors) {
	if (!cuda_available_for_reader_tests()) {
		GTEST_SKIP() << "CUDA device not available.";
	}
	auto fixture = make_external_dict_i16_fixture(300U, "selected_vectors");
	galp::format::FlsReader reader(fixture.fls_path);
	auto rowgroup   = reader.read_rowgroup_zero_copy_materialized(0);
	auto expressions = galp::expression::assemble(rowgroup);
	ASSERT_TRUE(galp::execution::has_unresolved_dict_ref(rowgroup.columns[1].host));

	galp::runtime::ExecutionWorkset      workset {};
	galp::runtime::ExecutionWorksetGuard guard(workset);
	galp::execution::ExecutionConfig     config {};
	config.unpack_n_vectors = 2;
	config.launch_strategy  = galp::execution::LaunchStrategy::MixedDispatch;
	config.write_out        = true;
	const std::vector<uint32_t> selected_vectors {1U, 6U, 10U};
	galp::runtime::append_rowgroup_columns_selected_vectors(workset, rowgroup, config, selected_vectors);
	EXPECT_FALSE(galp::execution::has_unresolved_dict_ref(rowgroup.columns[1].host));
	galp::runtime::upload_workset(workset, config);
	galp::runtime::run_workset(workset, 1U, config);
	const auto result = galp::runtime::materialize_workset(workset, expressions, config);
	ASSERT_EQ(result.columns.size(), 2U);
	ASSERT_TRUE(result.columns[1].has_value());
	const auto& output = std::get<std::shared_ptr<int16_t[]>>(result.columns[1]->values);
	for (size_t chunk = 0; chunk < selected_vectors.size(); ++chunk) {
		for (size_t row = 0; row < 2U * galp::codec::consts::VALUES_PER_VECTOR; ++row) {
			const size_t source = static_cast<size_t>(selected_vectors[chunk]) *
			                          galp::codec::consts::VALUES_PER_VECTOR +
			                      row;
			const size_t destination = chunk * 2U * galp::codec::consts::VALUES_PER_VECTOR + row;
			ASSERT_EQ(output[destination], fixture.mapped[source]) << "chunk=" << chunk << " row=" << row;
		}
	}

	galp::execution::free_rowgroup(rowgroup);
	std::filesystem::remove_all(fixture.root);
}

TEST(Reader, CapabilityTableIsUniqueAndCoversNewI16Tokens) {
	std::unordered_set<fastlanes::OperatorToken> seen;
	for (const auto& capability : galp::expression::kOperatorCapabilities) {
		EXPECT_TRUE(seen.insert(capability.token).second) << fastlanes::token_to_string(capability.token);
		EXPECT_EQ(galp::expression::capability_for_token(capability.token), &capability);
		EXPECT_EQ(galp::expression::is_supported_token(capability.token), capability.gpu_supported);
		EXPECT_EQ(galp::expression::is_sparse_read_supported_token(capability.token),
		          capability.sparse_read_supported);
	}
	EXPECT_TRUE(galp::expression::is_supported_token(fastlanes::OperatorToken::EXP_CONSTANT_I16));
	EXPECT_TRUE(galp::expression::is_supported_token(fastlanes::OperatorToken::EXP_UNCOMPRESSED_I16));
	EXPECT_TRUE(galp::expression::is_supported_token(fastlanes::OperatorToken::EXP_DELTA_I08));
	EXPECT_TRUE(galp::expression::is_supported_token(fastlanes::OperatorToken::EXP_DELTA_I16));
	EXPECT_FALSE(galp::expression::is_supported_token(fastlanes::OperatorToken::EXP_NULL_I16));
	EXPECT_FALSE(galp::expression::is_sparse_read_supported_token(fastlanes::OperatorToken::EXP_NULL_I16));
	EXPECT_TRUE(galp::expression::is_sparse_read_supported_token(fastlanes::OperatorToken::EXP_UNCOMPRESSED_I08));
	EXPECT_TRUE(galp::expression::is_supported_token(fastlanes::OperatorToken::EXP_DICT_I16_U08));
	EXPECT_TRUE(galp::expression::is_supported_token(fastlanes::OperatorToken::EXP_DICT_I16_U16));
}

TEST(Reader, ConstantI16GpuRoundtripHandlesLargePartialRowgroup) {
	if (!cuda_available_for_reader_tests()) {
		GTEST_SKIP() << "CUDA device not available.";
	}
	const std::vector<int16_t> expected(65U * galp::codec::consts::VALUES_PER_VECTOR + 7U, -12345);
	expect_forced_i16_gpu_roundtrip(
	    fastlanes::OperatorToken::EXP_CONSTANT_I16, "constant_i16_large_partial", expected);
}

TEST(Reader, UncompressedI16GpuRoundtripHandlesLargePartialRowgroup) {
	if (!cuda_available_for_reader_tests()) {
		GTEST_SKIP() << "CUDA device not available.";
	}
	std::vector<int16_t> expected(65U * galp::codec::consts::VALUES_PER_VECTOR + 7U);
	for (size_t row = 0; row < expected.size(); ++row) {
		expected[row] = static_cast<int16_t>((row * 251U + row / 17U) & 0xFFFFU);
	}
	expect_forced_i16_gpu_roundtrip(
	    fastlanes::OperatorToken::EXP_UNCOMPRESSED_I16, "uncompressed_i16_large_partial", expected);
}

template <typename T>
std::filesystem::path make_forced_delta_fls_fixture(const fastlanes::OperatorToken token,
	                                                const std::string&              label,
	                                                const std::vector<T>&           values) {
	const std::filesystem::path root = std::filesystem::path {GALP_TEST_DATA_DIR} / label;
	std::filesystem::remove_all(root);
	std::filesystem::create_directories(root);
	const auto fls_path = root / "data.fls";
	const std::array<fastlanes::MemoryColumn, 1> columns {
	    fastlanes::MemoryColumn {"value", std::span<const T>(values.data(), values.size())}};
	const fastlanes::MemoryTable table {std::span<const fastlanes::MemoryColumn>(columns)};
	fastlanes::MemoryTableOptions options;
	options.n_vectors_per_rowgroup = 70;
	options.force_schema           = true;
	options.forced_schema          = {token};
	fastlanes::write_memory_table_to_fls(table, fls_path, options);
	return fls_path;
}

TEST(ExternalDictionaryI16, RawIndexPayloadsAndOwnedKeys) {
	const auto run_case = [&]<typename SourceT, typename IndexT>(
	                          const fastlanes::OperatorToken source_token,
	                          const fastlanes::OperatorToken resolved_token,
	                          const std::string&              label,
	                          const size_t                    key_count,
	                          const std::vector<SourceT>&     indexes) {
		const auto fls_path = make_forced_delta_fls_fixture(source_token, label, indexes);
		galp::format::FlsReader reader(fls_path);
		auto rowgroup = reader.read_rowgroup_zero_copy_materialized(0);
		ASSERT_EQ(rowgroup.columns.size(), 1U);

		auto* keys = new uint16_t[key_count];
		std::vector<int16_t> expected_keys(key_count);
		for (size_t index = 0; index < key_count; ++index) {
			expected_keys[index] = static_cast<int16_t>(-15000 + static_cast<int>(index) * 41);
			keys[index]          = to_column_bits(expected_keys[index]);
		}
		const auto* original_keys = keys;
		galp::execution::Column target {};
		target.name  = "external";
		target.token = std::is_same_v<IndexT, uint8_t> ? fastlanes::OperatorToken::EXP_DICT_I16_U08
		                                              : fastlanes::OperatorToken::EXP_DICT_I16_U16;
		target.host = galp::codec::host::DICTREFColumn<int16_t, IndexT> {
		    rowgroup.n_values, 0U, keys, key_count};
		rowgroup.columns.push_back(std::move(target));

		auto expressions = galp::expression::assemble(rowgroup);
		galp::execution::resolve_dict_refs(expressions);
		EXPECT_EQ(rowgroup.columns[1].token, resolved_token);
		std::visit(
		    [&](const auto& column) {
			    using ColumnT = std::decay_t<decltype(column)>;
			    if constexpr (std::is_same_v<ColumnT, galp::codec::host::DICTFFORColumn<int16_t, IndexT>> ||
			                  std::is_same_v<ColumnT, galp::codec::host::DICTSLPATCHColumn<int16_t, IndexT>>) {
				    EXPECT_EQ(column.keys.get(), original_keys);
				    EXPECT_TRUE(column.keys.owns());
			    }
		    },
		    rowgroup.columns[1].host);

		if (cuda_available_for_reader_tests()) {
			galp::execution::ExecutionConfig config {};
			config.unpack_n_vectors = 1;
			config.write_out        = true;
			const auto result       = galp::execution::decompress_rowgroup(expressions, config);
			ASSERT_EQ(result.columns.size(), 2U);
			ASSERT_TRUE(result.columns[1].has_value());
			const auto& output = std::get<std::shared_ptr<int16_t[]>>(result.columns[1]->values);
			for (size_t row = 0; row < indexes.size(); ++row) {
				const auto index = static_cast<size_t>(static_cast<IndexT>(indexes[row]));
				ASSERT_LT(index, expected_keys.size());
				ASSERT_EQ(output[row], expected_keys[index]) << "row=" << row;
			}
		}
		galp::execution::free_rowgroup(rowgroup);
	};

	const size_t n_values = 4U * galp::codec::consts::VALUES_PER_VECTOR;
	std::vector<int8_t> bp_i8(n_values);
	std::vector<int16_t> ffor_i16(n_values);
	std::vector<int16_t> slpatch_i16(n_values);
	for (size_t row = 0; row < n_values; ++row) {
		bp_i8[row]      = static_cast<int8_t>(row % 61U);
		ffor_i16[row]   = static_cast<int16_t>(row % 300U);
		slpatch_i16[row] = static_cast<int16_t>((row % 257U == 0U) ? 299U : (row % 7U));
	}
	run_case.template operator()<int8_t, uint8_t>(fastlanes::OperatorToken::EXP_UNCOMPRESSED_I08,
	                                               fastlanes::OperatorToken::EXP_DICT_I16_FFOR_U08,
	                                               "external_raw_bp_i8",
	                                               61U,
	                                               bp_i8);
	run_case.template operator()<int16_t, uint16_t>(fastlanes::OperatorToken::EXP_FFOR_I16,
	                                                 fastlanes::OperatorToken::EXP_DICT_I16_FFOR_U16,
	                                                 "external_raw_ffor_i16",
	                                                 300U,
	                                                 ffor_i16);
	run_case.template operator()<int16_t, uint16_t>(fastlanes::OperatorToken::EXP_FFOR_SLPATCH_I16,
	                                                 fastlanes::OperatorToken::EXP_DICT_I16_FFOR_SLPATCH_U16,
	                                                 "external_raw_slpatch_i16",
	                                                 300U,
	                                                 slpatch_i16);
}

template <typename T>
void expect_forced_delta_cpu_gpu_roundtrip(const fastlanes::OperatorToken token,
	                                       const std::string&              label,
	                                       const std::vector<T>&           expected) {
	const auto fls_path = make_forced_delta_fls_fixture<T>(token, label, expected);
	const auto handle   = galp::format::detail::load_table_descriptor(fls_path);
	const auto* table   = handle.Get();
	ASSERT_NE(table, nullptr);
	ASSERT_NE(table->m_rowgroup_descriptors(), nullptr);
	ASSERT_EQ(table->m_rowgroup_descriptors()->size(), 1U);
	const auto* rg = table->m_rowgroup_descriptors()->Get(0);
	ASSERT_NE(rg, nullptr);
	ASSERT_GT(rg->m_n_vec(), 64U);
	ASSERT_NE(rg->m_column_descriptors(), nullptr);
	const auto* column = rg->m_column_descriptors()->Get(0);
	ASSERT_NE(column, nullptr);
	ASSERT_EQ(column->encoding_rpn()->operator_tokens()->Get(0), token);
	ASSERT_EQ(column->encoding_rpn()->operand_tokens()->size(), 4U);

	// FastLanes CPU is the format oracle; compare it bit-for-bit before
	// checking every GPU launch configuration.
	auto connection = fastlanes::connect();
	auto table_reader = connection->read_fls(fls_path);
	auto rowgroup_reader = table_reader->get_rowgroup_reader(0);
	auto cpu_rowgroup = rowgroup_reader->materialize();
	const auto* cpu_column = std::get_if<fastlanes::up<fastlanes::TypedCol<T>>>(&cpu_rowgroup->internal_rowgroup[0]);
	ASSERT_NE(cpu_column, nullptr);
	ASSERT_NE(cpu_column->get(), nullptr);
	for (size_t row = 0; row < expected.size(); ++row) {
		ASSERT_EQ((*cpu_column)->data[row], expected[row]) << "CPU row=" << row;
	}

	galp::format::FlsReader reader(fls_path);
	auto                    rowgroup = reader.read_rowgroup(0);
	ASSERT_EQ(rowgroup.n_tuples, expected.size());
	ASSERT_EQ(rowgroup.columns.size(), 1U);
	EXPECT_TRUE((std::holds_alternative<galp::codec::host::DELTAColumn<T>>(rowgroup.columns[0].host)));
	auto expressions = galp::expression::assemble(rowgroup);
	ASSERT_EQ(expressions.size(), 1U);
	for (const auto decoder :
	     {galp::execution::DeltaDecoder::Stateful, galp::execution::DeltaDecoder::Register}) {
		SCOPED_TRACE(decoder == galp::execution::DeltaDecoder::Register ? "register" : "stateful");
		galp::execution::ExecutionConfig standalone_config {};
		standalone_config.unpack_n_vectors = 4;
		standalone_config.write_out        = true;
		standalone_config.delta_decoder    = decoder;
		const auto standalone = galp::execution::decompress(expressions[0], standalone_config);
		const auto& standalone_output = std::get<std::shared_ptr<T[]>>(standalone);
		for (size_t row = 0; row < expected.size(); ++row) {
			ASSERT_EQ(standalone_output[row], expected[row]) << "standalone GPU row=" << row;
		}
		const auto standalone_hot = galp::execution::decompress(expressions[0], standalone_config);
		const auto& standalone_hot_output = std::get<std::shared_ptr<T[]>>(standalone_hot);
		for (size_t row = 0; row < expected.size(); ++row) {
			ASSERT_EQ(standalone_hot_output[row], expected[row]) << "hot standalone GPU row=" << row;
		}

		for (const unsigned unpack_n_vectors : {1U, 2U, 4U}) {
			for (const auto strategy :
			     {galp::execution::LaunchStrategy::MixedDispatch, galp::execution::LaunchStrategy::TypedBatches}) {
				SCOPED_TRACE(std::string("unpack=") + std::to_string(unpack_n_vectors) +
				             (strategy == galp::execution::LaunchStrategy::MixedDispatch ? " mixed" : " typed"));
				galp::execution::ExecutionConfig config {};
				config.unpack_n_vectors = unpack_n_vectors;
				config.launch_strategy  = strategy;
				config.delta_decoder    = decoder;
				config.write_out        = true;
				const auto result       = galp::execution::decompress_rowgroup(expressions, config);
				ASSERT_EQ(result.columns.size(), 1U);
				ASSERT_TRUE(result.columns[0].has_value());
				const auto& output = std::get<std::shared_ptr<T[]>>(result.columns[0]->values);
				ASSERT_NE(output, nullptr);
				for (size_t row = 0; row < expected.size(); ++row) {
					ASSERT_EQ(output[row], expected[row]) << "GPU row=" << row;
				}
			}
		}

		galp::runtime::ExecutionWorkset selected_workset {};
		galp::runtime::ExecutionWorksetGuard selected_guard(selected_workset);
		galp::execution::ExecutionConfig selected_config {};
		selected_config.unpack_n_vectors = 4;
		selected_config.launch_strategy  = galp::execution::LaunchStrategy::MixedDispatch;
		selected_config.delta_decoder    = decoder;
		selected_config.write_out        = true;
		const std::vector<uint32_t> selected_vectors {1U, 17U, 61U};
		galp::runtime::append_rowgroup_columns_selected_vectors(
		    selected_workset, rowgroup, selected_config, selected_vectors);
		galp::runtime::upload_workset(selected_workset, selected_config);
		galp::runtime::run_workset(selected_workset, 1, selected_config);
		const auto selected_result =
		    galp::runtime::materialize_workset(selected_workset, expressions, selected_config);
		ASSERT_EQ(selected_result.columns.size(), 1U);
		ASSERT_TRUE(selected_result.columns[0].has_value());
		const auto& selected_output = std::get<std::shared_ptr<T[]>>(selected_result.columns[0]->values);
		for (size_t chunk = 0; chunk < selected_vectors.size(); ++chunk) {
			for (size_t row = 0; row < 4U * galp::codec::consts::VALUES_PER_VECTOR; ++row) {
				const size_t expected_row =
				    static_cast<size_t>(selected_vectors[chunk]) * galp::codec::consts::VALUES_PER_VECTOR + row;
				const size_t output_row = chunk * 4U * galp::codec::consts::VALUES_PER_VECTOR + row;
				ASSERT_EQ(selected_output[output_row], expected[expected_row])
				    << "selected chunk=" << chunk << " row=" << row;
			}
		}
	}
}

TEST(Reader, CompactAllVectorsMatchesOriginalRowgroupOnGpu) {
	if (!cuda_available_for_reader_tests()) {
		GTEST_SKIP() << "CUDA device not available.";
	}
	const char* path_env = std::getenv("GALP_COMPACT_TEST_FILE");
	if (path_env == nullptr || !std::filesystem::exists(path_env)) {
		GTEST_SKIP() << "GALP_COMPACT_TEST_FILE does not name an FLS file.";
	}
	size_t rowgroup_index = 0U;
	if (const char* rowgroup_env = std::getenv("GALP_COMPACT_TEST_ROWGROUP");
	    rowgroup_env != nullptr && *rowgroup_env != '\0') {
		rowgroup_index = std::stoull(rowgroup_env);
	}
	galp::format::FlsReader reader(path_env);
	ASSERT_LT(rowgroup_index, reader.rowgroup_count());
	auto original = reader.read_rowgroup_zero_copy_materialized(rowgroup_index);
	const size_t original_n_values = original.n_values;
	auto original_expressions = galp::expression::assemble(original);
	galp::execution::ExecutionConfig config {};
	config.unpack_n_vectors = 1U;
	config.write_out        = true;
	const auto expected = galp::execution::decompress_rowgroup(original_expressions, config);

	auto compact = reader.read_rowgroup_zero_copy_materialized(rowgroup_index);
	auto resolved = galp::expression::assemble(compact);
	galp::execution::resolve_dict_refs(resolved);
	std::vector<uint32_t> vectors(compact.n_vecs);
	std::iota(vectors.begin(), vectors.end(), uint32_t {0});
	galp::runtime::compact_selected_vectors(compact, vectors);
	auto compact_expressions = galp::expression::assemble(compact);
	const auto actual = galp::execution::decompress_rowgroup(compact_expressions, config);

	ASSERT_EQ(actual.columns.size(), expected.columns.size());
	for (size_t column = 0; column < expected.columns.size(); ++column) {
		ASSERT_TRUE(expected.columns[column].has_value()) << "column=" << column;
		ASSERT_TRUE(actual.columns[column].has_value()) << "column=" << column;
		std::visit(
		    [&](const auto& expected_ptr) {
			    using PtrT = std::decay_t<decltype(expected_ptr)>;
			    const auto& actual_ptr = std::get<PtrT>(actual.columns[column]->values);
			    for (size_t row = 0; row < original_n_values; ++row) {
				    ASSERT_EQ(actual_ptr[row], expected_ptr[row]) << "column=" << column << " row=" << row;
			    }
		    },
		    expected.columns[column]->values);
	}

	for (uint32_t source_vector = 0; source_vector < original.n_vecs; ++source_vector) {
		auto single = reader.read_rowgroup_zero_copy_materialized(rowgroup_index);
		auto single_resolved = galp::expression::assemble(single);
		galp::execution::resolve_dict_refs(single_resolved);
		galp::runtime::compact_selected_vectors(single, std::vector<uint32_t> {source_vector});
		auto single_expressions = galp::expression::assemble(single);
		const auto single_actual = galp::execution::decompress_rowgroup(single_expressions, config);
		const size_t source_begin = static_cast<size_t>(source_vector) * galp::codec::consts::VALUES_PER_VECTOR;
		const size_t compare_count = std::min<size_t>(galp::codec::consts::VALUES_PER_VECTOR,
		                                              original_n_values - source_begin);
		for (size_t column = 0; column < expected.columns.size(); ++column) {
			ASSERT_TRUE(single_actual.columns[column].has_value()) << "column=" << column;
			std::visit(
			    [&](const auto& expected_ptr) {
				    using PtrT = std::decay_t<decltype(expected_ptr)>;
				    const auto& actual_ptr = std::get<PtrT>(single_actual.columns[column]->values);
				    for (size_t row = 0; row < compare_count; ++row) {
					    ASSERT_EQ(actual_ptr[row], expected_ptr[source_begin + row])
					        << "source_vector=" << source_vector << " column=" << column << " row=" << row;
				    }
			    },
			    expected.columns[column]->values);
		}
	}

	if (reader.has_sparse_vector_bundle()) {
		std::vector<uint32_t> sparse_vectors;
		if (const char* vectors_env = std::getenv("GALP_COMPACT_TEST_VECTORS");
		    vectors_env != nullptr && *vectors_env != '\0') {
			std::istringstream input(vectors_env);
			std::string token;
			while (std::getline(input, token, ',')) {
				sparse_vectors.push_back(static_cast<uint32_t>(std::stoul(token)));
			}
		} else {
			for (uint32_t vector = 0; vector < original.n_vecs; ++vector) {
				if (vector % 3U != 2U) {
					sparse_vectors.push_back(vector);
				}
			}
		}
		const ScopedEnvironmentVariable mirror("GALP_VECTOR_BUNDLE_DEVICE_SCATTER_CPU_MIRROR", "1");
		const ScopedEnvironmentVariable mirror_limit(
		    "GALP_VECTOR_BUNDLE_DEVICE_SCATTER_CPU_MIRROR_MAX_BYTES", "1536");
		auto sparse = reader.materialize_zero_copy_rowgroup(
		    reader.read_rowgroup_zero_copy_selected_vectors_packed(rowgroup_index, sparse_vectors));
		auto sparse_resolved = galp::expression::assemble(sparse);
		galp::execution::resolve_dict_refs(sparse_resolved);
		galp::runtime::compact_selected_vectors(sparse, sparse_vectors);
		auto sparse_expressions = galp::expression::assemble(sparse);
		const auto sparse_actual = galp::execution::decompress_rowgroup(sparse_expressions, config);
		for (size_t column = 0; column < expected.columns.size(); ++column) {
			ASSERT_TRUE(sparse_actual.columns[column].has_value()) << "column=" << column;
			std::visit(
			    [&](const auto& expected_ptr) {
				    using PtrT = std::decay_t<decltype(expected_ptr)>;
				    const auto& actual_ptr = std::get<PtrT>(sparse_actual.columns[column]->values);
				    for (size_t selected_index = 0; selected_index < sparse_vectors.size(); ++selected_index) {
					    const size_t source_begin = static_cast<size_t>(sparse_vectors[selected_index]) *
					                                galp::codec::consts::VALUES_PER_VECTOR;
					    const size_t compare_count = std::min<size_t>(
					        galp::codec::consts::VALUES_PER_VECTOR, original_n_values - source_begin);
					    for (size_t row = 0; row < compare_count; ++row) {
						    ASSERT_EQ(actual_ptr[selected_index * galp::codec::consts::VALUES_PER_VECTOR + row],
						              expected_ptr[source_begin + row])
						        << "source_vector=" << sparse_vectors[selected_index] << " column=" << column
						        << " row=" << row;
					    }
				    }
			    },
			    expected.columns[column]->values);
		}
	}
}

TEST(Reader, DeltaI08CpuGpuRoundtripHandlesLargePartialRowgroup) {
	if (!cuda_available_for_reader_tests()) {
		GTEST_SKIP() << "CUDA device not available.";
	}
	std::vector<int8_t> expected(65U * galp::codec::consts::VALUES_PER_VECTOR + 7U);
	int8_t             value = -101;
	for (size_t row = 0; row < expected.size(); ++row) {
		value         = static_cast<int8_t>(value + static_cast<int8_t>((row % 9U) - 4));
		expected[row] = value;
	}
	expect_forced_delta_cpu_gpu_roundtrip(
	    fastlanes::OperatorToken::EXP_DELTA_I08, "delta_i08_large_partial", expected);
}

TEST(Reader, DeltaI16CpuGpuRoundtripHandlesLargePartialRowgroup) {
	if (!cuda_available_for_reader_tests()) {
		GTEST_SKIP() << "CUDA device not available.";
	}
	std::vector<int16_t> expected(65U * galp::codec::consts::VALUES_PER_VECTOR + 7U);
	int16_t              value = -30000;
	for (size_t row = 0; row < expected.size(); ++row) {
		value         = static_cast<int16_t>(value + static_cast<int16_t>((row % 31U) - 15));
		expected[row] = value;
	}
	expect_forced_delta_cpu_gpu_roundtrip(
	    fastlanes::OperatorToken::EXP_DELTA_I16, "delta_i16_large_partial", expected);
}

TEST(DeltaDecode, SelectedVectors) {
	if (!cuda_available_for_reader_tests()) {
		GTEST_SKIP() << "CUDA device not available.";
	}
	constexpr size_t vector_count = 10U;
	const size_t     n_values     = vector_count * galp::codec::consts::VALUES_PER_VECTOR;
	std::vector<int8_t> expected(n_values);
	uint8_t             bits = 241U;
	for (size_t row = 0; row < n_values; ++row) {
		bits = static_cast<uint8_t>(bits + static_cast<uint8_t>((row * 5U + 3U) & 0x1FU));
		std::memcpy(&expected[row], &bits, sizeof(bits));
	}

	const auto fls_path = make_forced_delta_fls_fixture(
	    fastlanes::OperatorToken::EXP_DELTA_I08, "delta_decode_selected_vectors", expected);
	galp::format::FlsReader reader(fls_path);
	auto                    rowgroup   = reader.read_rowgroup(0);
	auto                    expressions = galp::expression::assemble(rowgroup);

	galp::runtime::ExecutionWorkset      workset {};
	galp::runtime::ExecutionWorksetGuard guard(workset);
	galp::execution::ExecutionConfig     config {};
	config.unpack_n_vectors = 2;
	config.launch_strategy  = galp::execution::LaunchStrategy::MixedDispatch;
	config.write_out        = true;
	const std::vector<uint32_t> selected_vectors {1U, 7U};
	galp::runtime::append_rowgroup_columns_selected_vectors(workset, rowgroup, config, selected_vectors);
	galp::runtime::upload_workset(workset, config);
	galp::runtime::run_workset(workset, 1U, config);
	const auto result = galp::runtime::materialize_workset(workset, expressions, config);
	ASSERT_EQ(result.columns.size(), 1U);
	ASSERT_TRUE(result.columns[0].has_value());
	const auto& output = std::get<std::shared_ptr<int8_t[]>>(result.columns[0]->values);
	for (size_t chunk = 0; chunk < selected_vectors.size(); ++chunk) {
		for (size_t row = 0; row < 2U * galp::codec::consts::VALUES_PER_VECTOR; ++row) {
			const size_t source = static_cast<size_t>(selected_vectors[chunk]) *
			                          galp::codec::consts::VALUES_PER_VECTOR +
			                      row;
			const size_t destination = chunk * 2U * galp::codec::consts::VALUES_PER_VECTOR + row;
			ASSERT_EQ(output[destination], expected[source]) << "chunk=" << chunk << " row=" << row;
		}
	}
}

TEST(DeltaDecode, VectorTail) {
	if (!cuda_available_for_reader_tests()) {
		GTEST_SKIP() << "CUDA device not available.";
	}
	constexpr size_t vector_count = 7U;
	const size_t     n_values     = vector_count * galp::codec::consts::VALUES_PER_VECTOR;
	std::vector<int16_t> expected(n_values);
	uint16_t              bits = 65500U;
	for (size_t row = 0; row < n_values; ++row) {
		bits = static_cast<uint16_t>(bits + static_cast<uint16_t>((row * 29U + 7U) & 0x1FFU));
		std::memcpy(&expected[row], &bits, sizeof(bits));
	}

	const auto fls_path = make_forced_delta_fls_fixture(
	    fastlanes::OperatorToken::EXP_DELTA_I16, "delta_decode_vector_tail", expected);
	galp::format::FlsReader reader(fls_path);
	auto                    rowgroup   = reader.read_rowgroup(0);
	auto                    expressions = galp::expression::assemble(rowgroup);
	ASSERT_EQ(rowgroup.n_vecs, vector_count);
	galp::execution::ExecutionConfig config {};
	config.unpack_n_vectors = 4;
	config.launch_strategy  = galp::execution::LaunchStrategy::MixedDispatch;
	config.write_out        = true;
	const auto result       = galp::execution::decompress_rowgroup(expressions, config);
	ASSERT_EQ(result.columns.size(), 1U);
	ASSERT_TRUE(result.columns[0].has_value());
	const auto& output = std::get<std::shared_ptr<int16_t[]>>(result.columns[0]->values);
	for (size_t row = 0; row < n_values; ++row) {
		ASSERT_EQ(output[row], expected[row]) << "row=" << row;
	}
}

TEST(Reader, DeltaI08I16ShareOneMixedWorksetAndPreserveColumnOrder) {
	if (!cuda_available_for_reader_tests()) {
		GTEST_SKIP() << "CUDA device not available.";
	}
	const size_t n_values = 8U * galp::codec::consts::VALUES_PER_VECTOR;
	std::vector<int8_t>  expected_i8(n_values);
	std::vector<int16_t> expected_i16(n_values);
	for (size_t row = 0; row < n_values; ++row) {
		expected_i8[row]  = static_cast<int8_t>((row * 7U + row / 13U) & 0xFFU);
		expected_i16[row] = static_cast<int16_t>((row * 251U + row / 17U) & 0xFFFFU);
	}

	const auto root = std::filesystem::path {GALP_TEST_DATA_DIR} / "delta_i08_i16_mixed";
	std::filesystem::remove_all(root);
	std::filesystem::create_directories(root);
	const auto fls_path = root / "data.fls";
	const std::array<fastlanes::MemoryColumn, 2> columns {
	    fastlanes::MemoryColumn {"i8", std::span<const int8_t>(expected_i8)},
	    fastlanes::MemoryColumn {"i16", std::span<const int16_t>(expected_i16)}};
	fastlanes::MemoryTableOptions options;
	options.n_vectors_per_rowgroup = 12;
	options.force_schema           = true;
	options.forced_schema          = {fastlanes::OperatorToken::EXP_DELTA_I08,
	                                  fastlanes::OperatorToken::EXP_DELTA_I16};
	fastlanes::write_memory_table_to_fls(
	    fastlanes::MemoryTable {std::span<const fastlanes::MemoryColumn>(columns)}, fls_path, options);

	galp::format::FlsReader reader(fls_path);
	auto                    rowgroup = reader.read_rowgroup(0);
	auto                    expressions = galp::expression::assemble(rowgroup);
	ASSERT_EQ(expressions.size(), 2U);

	for (const auto decoder :
	     {galp::execution::DeltaDecoder::Stateful, galp::execution::DeltaDecoder::Register}) {
		SCOPED_TRACE(decoder == galp::execution::DeltaDecoder::Register ? "register" : "stateful");
		galp::runtime::ExecutionWorkset workset {};
		galp::runtime::ExecutionWorksetGuard guard(workset);
		galp::execution::ExecutionConfig config {};
		config.unpack_n_vectors = 4;
		config.launch_strategy  = galp::execution::LaunchStrategy::MixedDispatch;
		config.delta_decoder    = decoder;
		config.write_out        = true;
		galp::runtime::append_expressions(workset, expressions, config);
		galp::runtime::upload_workset(workset, config);
		ASSERT_FALSE(workset.slots.mixed.empty());
		bool saw_i8  = false;
		bool saw_i16 = false;
		for (const auto& slot : workset.slots.mixed) {
			for (const auto& work : {slot.first, slot.second}) {
				if (!galp::execution::is_valid_work_item(work)) {
					continue;
				}
				saw_i8  = saw_i8 || work.type == galp::execution::TypeTag::I8;
				saw_i16 = saw_i16 || work.type == galp::execution::TypeTag::I16;
			}
		}
		EXPECT_TRUE(saw_i8);
		EXPECT_TRUE(saw_i16);
		size_t launches = 0;
		galp::runtime::run_workset(workset, 1, config, nullptr, &launches);
		EXPECT_EQ(launches, 1U);
		const auto result = galp::runtime::materialize_workset(workset, expressions, config);
		ASSERT_EQ(result.columns.size(), 2U);
		ASSERT_TRUE(result.columns[0].has_value());
		ASSERT_TRUE(result.columns[1].has_value());
		const auto& out_i8  = std::get<std::shared_ptr<int8_t[]>>(result.columns[0]->values);
		const auto& out_i16 = std::get<std::shared_ptr<int16_t[]>>(result.columns[1]->values);
		for (size_t row = 0; row < n_values; ++row) {
			ASSERT_EQ(out_i8[row], expected_i8[row]) << "i8 row=" << row;
			ASSERT_EQ(out_i16[row], expected_i16[row]) << "i16 row=" << row;
		}
	}
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

TEST(Reader, CrossRleI16TokenMaterializesAsI16Column) {
	const auto fls_path = make_cross_rle_i16_fls_fixture();

	const auto  td_handle = galp::format::detail::load_table_descriptor(fls_path);
	const auto* td        = td_handle.Get();
	ASSERT_NE(td, nullptr);
	ASSERT_NE(td->m_rowgroup_descriptors(), nullptr);
	ASSERT_GT(td->m_rowgroup_descriptors()->size(), 0U);
	const auto* rg = td->m_rowgroup_descriptors()->Get(0);
	ASSERT_NE(rg, nullptr);
	ASSERT_NE(rg->m_column_descriptors(), nullptr);
	ASSERT_EQ(rg->m_column_descriptors()->size(), 1U);
	const auto* col_desc = rg->m_column_descriptors()->Get(0);
	ASSERT_NE(col_desc, nullptr);
	ASSERT_NE(col_desc->encoding_rpn(), nullptr);
	ASSERT_NE(col_desc->encoding_rpn()->operator_tokens(), nullptr);
	ASSERT_EQ(col_desc->encoding_rpn()->operator_tokens()->Get(0), fastlanes::OperatorToken::EXP_CROSS_RLE_I16);

	galp::format::FlsReader rdr(fls_path);
	auto                    rowgroup = rdr.read_rowgroup(0);
	ASSERT_EQ(rowgroup.columns.size(), 1U);
	EXPECT_EQ(rowgroup.columns[0].token, fastlanes::OperatorToken::EXP_CROSS_RLE_I16);
	EXPECT_TRUE((std::holds_alternative<galp::codec::host::CROSSRLEColumn<int16_t>>(rowgroup.columns[0].host)));
	EXPECT_TRUE(galp::expression::is_supported_token(fastlanes::OperatorToken::EXP_CROSS_RLE_I16));
	EXPECT_NO_THROW({
		auto expressions = galp::expression::assemble(rowgroup);
		EXPECT_EQ(expressions.size(), 1U);
	});
}

TEST(Reader, CrossRleI16DispatchMatchesI08EquivalentRuns) {
	const std::vector<uint32_t> lengths {3, 5, 2, 6};
	expect_cross_rle_decompresses<int8_t>(
	    {1, -2, -2, 7}, lengths, {1, 1, 1, -2, -2, -2, -2, -2, -2, -2, 7, 7, 7, 7, 7, 7});
	expect_cross_rle_decompresses<int16_t>(
	    {1, -2, -2, 7}, lengths, {1, 1, 1, -2, -2, -2, -2, -2, -2, -2, 7, 7, 7, 7, 7, 7});
}

TEST(Reader, CrossRleI16DispatchHandlesBoundaryValuesAndMultipleRuns) {
	expect_cross_rle_decompresses<int16_t>(
	    {std::numeric_limits<int16_t>::min(), -1, 0, std::numeric_limits<int16_t>::max()},
	    {2, 3, 4, 7},
	    {std::numeric_limits<int16_t>::min(),
	     std::numeric_limits<int16_t>::min(),
	     -1,
	     -1,
	     -1,
	     0,
	     0,
	     0,
	     0,
	     std::numeric_limits<int16_t>::max(),
	     std::numeric_limits<int16_t>::max(),
	     std::numeric_limits<int16_t>::max(),
	     std::numeric_limits<int16_t>::max(),
	     std::numeric_limits<int16_t>::max(),
	     std::numeric_limits<int16_t>::max(),
	     std::numeric_limits<int16_t>::max()});
}

TEST(Reader, CrossRleI16DispatchHandlesSingleRun) {
	expect_cross_rle_decompresses<int16_t>({-12345}, {16}, std::vector<int16_t>(16, -12345));
}

TEST(Reader, CrossRleI16DispatchHandlesEmptyInput) {
	expect_cross_rle_decompresses<int16_t>({}, {}, {});
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

TEST(Reader, RowgroupPrefetchQueueHonorsExplicitSchedule) {
	const auto fls_path = make_partial_rowgroup_fls_fixture();

	auto reader = std::make_shared<galp::format::FlsReader>(fls_path, /*load_column_names=*/false);
	ASSERT_EQ(reader->rowgroup_count(), 2U);

	galp::runtime::RowgroupPrefetchQueue queue(reader,
	                                           std::vector<size_t> {1U},
	                                           /*depth=*/2,
	                                           /*num_workers=*/1);
	auto                                 result = queue.pop();
	EXPECT_EQ(result.rowgroup_index, 1U);
	EXPECT_EQ(result.rowgroup.n_tuples, 6U);
	galp::execution::free_rowgroup(result.rowgroup);
}

TEST(Reader, WholeTablePrefetchBuildsPredicateScheduleBeforeRead) {
	if (!cuda_available_for_reader_tests()) {
		GTEST_SKIP() << "CUDA device not available for reader test.";
	}

	const auto fls_path = make_partial_rowgroup_fls_fixture();

	galp::runtime::TableExecutionRequest request {};
	request.config.scope                       = galp::execution::TableDecompressionScope::WholeTable;
	request.config.enable_rowgroup_prefetch    = true;
	request.config.prefetch_depth              = 2;
	request.config.prefetch_workers            = 1;
	request.config.streaming_target_rowgroups  = 1;
	request.config.streaming_target_work_items = 1;
	request.config.execution.write_out         = false;
	request.materialize_results                = false;
	request.load_column_names                  = false;

	RecordingTableExecutionObserver observer {};
	std::vector<size_t>             callbacks;
	auto                            result = galp::runtime::execute_table_pipeline(
        fls_path,
        request,
        [](const size_t rowgroup_index) { return rowgroup_index == 1U; },
        [&](const size_t                                  rowgroup_index,
            galp::format::Rowgroup&,
            const std::vector<galp::expression::Expression>&,
            const galp::execution::RowgroupData*) { callbacks.push_back(rowgroup_index); },
        observer);

	const std::vector<size_t> expected_rowgroups {1U};
	const std::vector<bool>   expected_from_prefetch {true};
	EXPECT_EQ(observer.read_rowgroups, expected_rowgroups);
	EXPECT_EQ(observer.read_from_prefetch, expected_from_prefetch);
	EXPECT_EQ(callbacks, expected_rowgroups);
	EXPECT_EQ(result.rowgroups, 1U);
}

TEST(Reader, MultiVectorUnpackHandlesPartialTailRowgroup) {
	if (!cuda_available_for_reader_tests()) {
		GTEST_SKIP() << "CUDA device not available.";
	}

	const auto fls_path = make_partial_rowgroup_fls_fixture();

	galp::format::FlsReader rdr(fls_path);
	ASSERT_EQ(rdr.rowgroup_count(), 2U);

	const auto td_handle = galp::format::detail::load_table_descriptor(fls_path);
	const auto* td       = td_handle.Get();
	ASSERT_NE(td, nullptr);
	const auto* rowgroups = td->m_rowgroup_descriptors();
	ASSERT_NE(rowgroups, nullptr);
	const auto* rg_desc = rowgroups->Get(1);
	ASSERT_NE(rg_desc, nullptr);
	ASSERT_NE(static_cast<size_t>(rg_desc->m_n_vec()) % 4U, 0U);

	auto rowgroup    = rdr.read_rowgroup(1);
	auto expressions = galp::expression::assemble(rowgroup);

	galp::execution::ExecutionConfig cfg_u1 {};
	cfg_u1.unpack_n_vectors = 1;
	cfg_u1.write_out        = true;
	const auto result_u1    = galp::execution::decompress_rowgroup(expressions, cfg_u1);

	for (const auto strategy :
	     {galp::execution::LaunchStrategy::MixedDispatch, galp::execution::LaunchStrategy::TypedBatches}) {
		SCOPED_TRACE(strategy == galp::execution::LaunchStrategy::MixedDispatch ? "mixed_u4_tail" : "typed_u4_tail");

		galp::execution::ExecutionConfig cfg_u4 {};
		cfg_u4.unpack_n_vectors = 4;
		cfg_u4.launch_strategy  = strategy;
		cfg_u4.write_out        = true;
		const auto result_u4    = galp::execution::decompress_rowgroup(expressions, cfg_u4);

		ASSERT_EQ(result_u1.columns.size(), result_u4.columns.size());
		ASSERT_FALSE(result_u1.columns.empty());
		ASSERT_TRUE(result_u1.columns[0].has_value());
		ASSERT_TRUE(result_u4.columns[0].has_value());

		const size_t n_vals = result_u1.columns[0]->meta.value_count;
		ASSERT_EQ(n_vals, result_u4.columns[0]->meta.value_count);
		const auto& ptr_u1 = std::get<std::shared_ptr<int8_t[]>>(result_u1.columns[0]->values);
		const auto& ptr_u4 = std::get<std::shared_ptr<int8_t[]>>(result_u4.columns[0]->values);
		for (size_t row = 0; row < n_vals; ++row) {
			EXPECT_EQ(ptr_u1[row], ptr_u4[row]) << "row=" << row;
		}
	}

}

TEST(Reader, MultiVectorUnpackMatchesSingleVector) {
	const auto fls_path = pick_fls_file();
	if (fls_path.empty()) {
		GTEST_SKIP() << "No .fls file found (set FLS_READER_TEST_FILE).";
	}
	if (!cuda_available_for_reader_tests()) {
		GTEST_SKIP() << "CUDA device not available.";
	}

	galp::format::FlsReader rdr(fls_path);
	const size_t            n_rowgroups = rdr.rowgroup_count();
	ASSERT_GT(n_rowgroups, 0U);

	const auto supported = supported_tokens();
	const auto td_handle = galp::format::detail::load_table_descriptor(fls_path);
	const auto* td       = td_handle.Get();
	ASSERT_NE(td, nullptr);

	size_t compared_total = 0;

	for (size_t rg_idx = 0; rg_idx < n_rowgroups; ++rg_idx) {
		const auto* rg_desc = td->m_rowgroup_descriptors()->Get(static_cast<uint32_t>(rg_idx));
		if (!rg_desc) continue;
		std::vector<fastlanes::OperatorToken> unsup;
		if (!rowgroup_supported(rg_desc, supported, unsup)) continue;

		auto rowgroup    = rdr.read_rowgroup(static_cast<uint32_t>(rg_idx));
		auto expressions = galp::expression::assemble(rowgroup);

		galp::execution::ExecutionConfig cfg_u1 {};
		cfg_u1.unpack_n_vectors = 1;
		cfg_u1.write_out        = true;
		const auto result_u1    = galp::execution::decompress_rowgroup(expressions, cfg_u1);

		for (const auto strategy :
		     {galp::execution::LaunchStrategy::MixedDispatch, galp::execution::LaunchStrategy::TypedBatches}) {
			SCOPED_TRACE(strategy == galp::execution::LaunchStrategy::MixedDispatch ? "mixed_u4" : "typed_u4");

			galp::execution::ExecutionConfig cfg_u4 {};
			cfg_u4.unpack_n_vectors = 4;
			cfg_u4.launch_strategy  = strategy;
			cfg_u4.write_out        = true;
			const auto result_u4    = galp::execution::decompress_rowgroup(expressions, cfg_u4);

			ASSERT_EQ(result_u1.columns.size(), result_u4.columns.size());
			for (size_t col = 0; col < result_u1.columns.size(); ++col) {
				if (!result_u1.columns[col].has_value()) continue;
				ASSERT_TRUE(result_u4.columns[col].has_value())
				    << "rg=" << rg_idx << " col=" << col << " missing in unpack=4";

				const size_t n_vals = result_u1.columns[col]->meta.value_count;
				std::visit(
				    [&](auto& ptr_u1) {
					    using T      = std::remove_pointer_t<decltype(ptr_u1.get())>;
					    auto& ptr_u4 = std::get<std::shared_ptr<T[]>>(result_u4.columns[col]->values);
					    for (size_t row = 0; row < n_vals; ++row) {
						    if (ptr_u1[row] != ptr_u4[row]) {
							    ADD_FAILURE() << "rg=" << rg_idx << " col=" << col << " row=" << row
							                  << " u1=" << static_cast<int64_t>(ptr_u1[row])
							                  << " u4=" << static_cast<int64_t>(ptr_u4[row]);
							    return;
						    }
					    }
					    ++compared_total;
				    },
				    result_u1.columns[col]->values);
			}
		}
	}

	EXPECT_GT(compared_total, 0U) << "No columns were compared.";
}
