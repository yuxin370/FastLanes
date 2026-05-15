// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/runtime/table/helpers.cu
// ────────────────────────────────────────────────────────
#include "engine/runtime/table/helpers.cuh"
#include "engine/execution/rowgroup.cuh"
#include "engine/runtime/materialize/metadata.cuh"
#include "engine/runtime/materialize/pinned_d2h.cuh"
#include "engine/runtime/workset/append.cuh"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace galp::runtime::detail {

TableChunkState::~TableChunkState() noexcept {
	cleanup_noexcept();
}

void TableChunkState::cleanup_noexcept() noexcept {
	try {
		if (run.active) {
			runtime::wait_workset_async(run);
		}
	} catch (const std::exception& e) {
		std::fprintf(stderr, "TableChunkState cleanup: wait_workset_async failed: %s\n", e.what());
	}
	runtime::discard_pinned_d2h_materialize(pending_materialize);
	try {
		runtime::release_workset(workset);
	} catch (const std::exception& e) {
		std::fprintf(stderr, "TableChunkState cleanup: release_workset failed: %s\n", e.what());
	}
	for (auto& pending : rowgroups) {
		try {
			galp::execution::free_rowgroup(pending.rowgroup);
		} catch (const std::exception& e) {
			std::fprintf(stderr, "TableChunkState cleanup: free_rowgroup failed: %s\n", e.what());
		}
	}
	rowgroups.clear();
	expr_locations.clear();
	submitted = false;
}

void reset_chunk(TableChunkState& chunk) {
	chunk.rowgroups.clear();
	chunk.expr_locations.clear();
	chunk.work_items     = 0;
	chunk.active_columns = 0;
	chunk.launch_grid    = 0;
	chunk.launches       = 0;
	chunk.submitted      = false;
}

size_t data_type_size(const fastlanes::DataType dt) {
	using fastlanes::DataType;
	switch (dt) {
	case DataType::INT8:
	case DataType::UINT8:
	case DataType::BOOLEAN:
		return 1;
	case DataType::INT16:
	case DataType::UINT16:
		return 2;
	case DataType::INT32:
	case DataType::UINT32:
	case DataType::FLOAT:
	case DataType::DATE:
		return 4;
	case DataType::INT64:
	case DataType::UINT64:
	case DataType::DOUBLE:
	case DataType::TIMESTAMP:
		return 8;
	default:
		return 0;
	}
}

size_t rowgroup_logical_bytes(const fastlanes::RowgroupDescriptor* rg) {
	if (!rg) {
		return 0;
	}

	size_t      bytes = 0;
	const auto* cols  = rg->m_column_descriptors();
	if (!cols) {
		return 0;
	}

	for (const auto* col_desc : *cols) {
		if (!col_desc) {
			continue;
		}
		const size_t element_size = data_type_size(col_desc->data_type());
		if (element_size == 0) {
			continue;
		}
		bytes += static_cast<size_t>(rg->m_n_tuples()) * element_size;
	}
	return bytes;
}

size_t count_active_columns(const std::vector<galp::expression::Expression>& expressions) {
	return runtime::count_active_columns(expressions);
}

size_t max_rowgroup_storage_bytes(galp::format::FlsReader& rdr, const size_t start, const size_t end) {
	size_t max_bytes = 0;
	for (size_t rowgroup_index = start; rowgroup_index < end; ++rowgroup_index) {
		max_bytes = std::max(max_bytes, rdr.rowgroup_storage_bytes(rowgroup_index));
	}
	return max_bytes;
}

void check_rowgroup_index(const size_t n_rowgroups, const std::optional<size_t>& rowgroup) {
	if (rowgroup.has_value() && *rowgroup >= n_rowgroups) {
		throw std::out_of_range("rowgroup index out of range");
	}
}

void validate_table_request(const TableExecutionRequest& request) {
	if (request.config.streaming_target_work_items == 0) {
		throw std::invalid_argument("streaming_target_work_items must be > 0");
	}
	if (request.config.prefetch_depth == 0) {
		throw std::invalid_argument("prefetch_depth must be > 0");
	}
}

bool use_whole_table_pipeline(const TableExecutionRequest& request) {
	return !request.rowgroup.has_value() && request.config.scope == TableDecompressionScope::WholeTable;
}

size_t choose_prefetch_workers(const size_t requested, const size_t rowgroup_count) {
	if (requested != 0) {
		return std::max<size_t>(1, requested);
	}
	if (rowgroup_count >= 128) {
		return 4;
	}
	if (rowgroup_count >= 32) {
		return 3;
	}
	return 2;
}

std::optional<size_t> env_size_value(const char* name) {
	const char* raw = std::getenv(name);
	if (raw == nullptr || *raw == '\0') {
		return std::nullopt;
	}
	char*                    end   = nullptr;
	const unsigned long long value = std::strtoull(raw, &end, 10);
	if (end == raw) {
		return std::nullopt;
	}
	if (value > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
		return std::numeric_limits<size_t>::max();
	}
	return static_cast<size_t>(value);
}

bool env_mode_is(const char* raw, const char* a, const char* b, const char* c) {
	return raw != nullptr &&
	       (std::strcmp(raw, a) == 0 || std::strcmp(raw, b) == 0 || (c[0] != '\0' && std::strcmp(raw, c) == 0));
}

size_t choose_pinned_prewarm_slots(const size_t pooled_slots,
                                   const size_t total_rowgroups,
                                   const size_t max_rowgroups_per_chunk,
                                   const size_t prefetch_depth_slots,
                                   const size_t active_io_workers) {
	const size_t full_prefetch_slots = prefetch_depth_slots + active_io_workers + 2U;
	const size_t full_slots = std::min({pooled_slots, max_rowgroups_per_chunk + full_prefetch_slots, total_rowgroups});
	if (full_slots == 0) {
		return 0;
	}

	if (const auto requested = env_size_value("GALP_PINNED_ROWGROUP_PREWARM_SLOTS"); requested.has_value()) {
		return std::min(*requested, full_slots);
	}

	const char* mode = std::getenv("GALP_PINNED_ROWGROUP_PREWARM_MODE");
	if (env_mode_is(mode, "full", "eager", "all")) {
		return full_slots;
	}
	if (env_mode_is(mode, "off", "none", "lazy")) {
		return 0;
	}

	const size_t consumer_slots = std::min(max_rowgroups_per_chunk, total_rowgroups);
	const size_t worker_slots   = std::max<size_t>(1, active_io_workers);
	const size_t target_slots   = consumer_slots + worker_slots + prefetch_depth_slots;
	return std::min(full_slots, std::max<size_t>(1, target_slots));
}

size_t choose_direct_pinned_prewarm_slots(const size_t pooled_slots,
                                          const size_t total_rowgroups,
                                          const size_t max_rowgroups_per_chunk) {
	if (pooled_slots == 0 || total_rowgroups == 0 || max_rowgroups_per_chunk == 0) {
		return 0;
	}

	const size_t chunk_slots = std::min(max_rowgroups_per_chunk, total_rowgroups);
	const size_t double_buffer_slots =
	    chunk_slots > std::numeric_limits<size_t>::max() / 2U ? std::numeric_limits<size_t>::max() : 2U * chunk_slots;
	const size_t live_slots = std::min({pooled_slots, total_rowgroups, double_buffer_slots});
	if (live_slots == 0) {
		return 0;
	}

	if (const auto requested = env_size_value("GALP_PINNED_ROWGROUP_PREWARM_SLOTS"); requested.has_value()) {
		return std::min(*requested, live_slots);
	}

	const char* mode = std::getenv("GALP_PINNED_ROWGROUP_PREWARM_MODE");
	if (env_mode_is(mode, "off", "none", "lazy")) {
		return 0;
	}

	return live_slots;
}

RowgroupReadResult read_rowgroup(galp::format::FlsReader&                         rdr,
                                 const size_t                                     rowgroup_index,
                                 const std::shared_ptr<PinnedRowgroupBufferPool>& pinned_pool) {
	RowgroupReadResult                    result {};
	const auto                            read_start = std::chrono::steady_clock::now();
	galp::format::ZeroCopyRowgroup        zero_copy {};
	galp::format::ZeroCopyReadTiming      io_timing {};
	std::chrono::steady_clock::time_point file_read_start {};
	const size_t                          storage_bytes = rdr.rowgroup_storage_bytes(rowgroup_index);
	result.rowgroup_index                               = rowgroup_index;
	result.storage_bytes                                = storage_bytes;
	if (pinned_pool) {
		const auto acquire_start = std::chrono::steady_clock::now();
		auto       lease         = pinned_pool->acquire(storage_bytes);
		const auto acquire_end   = std::chrono::steady_clock::now();
		result.timing.pinned_acquire_ms =
		    std::chrono::duration<double, std::milli>(acquire_end - acquire_start).count();
		file_read_start = acquire_end;
		zero_copy       = rdr.read_rowgroup_zero_copy_into(rowgroup_index,
                                                     std::move(lease.owner),
                                                     lease.data,
                                                     lease.capacity,
                                                     /*backing_is_pinned=*/true,
                                                     &io_timing);
	} else {
		file_read_start = std::chrono::steady_clock::now();
		zero_copy       = rdr.read_rowgroup_zero_copy(rowgroup_index, &io_timing);
	}
	const auto file_read_end   = std::chrono::steady_clock::now();
	const auto build_start     = std::chrono::steady_clock::now();
	result.rowgroup            = rdr.materialize_zero_copy_rowgroup(std::move(zero_copy));
	const auto build_end       = std::chrono::steady_clock::now();
	result.timing.file_read_ms = std::chrono::duration<double, std::milli>(file_read_end - file_read_start).count();
	result.timing.rowgroup_build_ms       = std::chrono::duration<double, std::milli>(build_end - build_start).count();
	result.timing.pread_ms                = io_timing.pread_ms;
	result.timing.zero_copy_view_setup_ms = io_timing.zero_copy_view_setup_ms;
	result.timing.timeline.pread_start    = io_timing.pread_start;
	result.timing.timeline.pread_end      = io_timing.pread_end;
	result.timing.timeline.read_submit =
	    (io_timing.pread_start == std::chrono::steady_clock::time_point {}) ? file_read_start : io_timing.pread_start;
	result.timing.read_ms                  = std::chrono::duration<double, std::milli>(build_end - read_start).count();
	result.timing.timeline.read_start      = read_start;
	result.timing.timeline.file_read_start = file_read_start;
	result.timing.timeline.file_read_end   = file_read_end;
	result.timing.timeline.build_start     = build_start;
	result.timing.timeline.build_end       = build_end;
	result.timing.timeline.ready_push      = build_end;
	result.timing.timeline.consumer_wait_start = build_end;
	result.timing.timeline.consumer_wait_end   = build_end;
	result.timing.timeline.consumer_pop        = build_end;
	result.timing.timeline.read_end            = build_end;
	return result;
}

} // namespace galp::runtime::detail
