// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/benchmark/table.cu
// ────────────────────────────────────────────────────────
#include "engine/benchmark/table.cuh"
#include "engine/execution/internal/materialize.cuh"
#include "engine/execution/internal/streaming_pipeline.cuh"
#include "engine/execution/rowgroup.cuh"
#include "engine/expression.cuh"
#include "engine/reader.cuh"
#include "fls/cor/lyt/buf.hpp"
#include "fls/file/file_footer.hpp"
#include "fls/file/file_header.hpp"
#include "fls/footer/datatype_generated.h"
#include "fls/footer/table_descriptor.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace dispatch {
namespace {

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

fastlanes::TableDescriptorHandle load_table_descriptor(const std::filesystem::path& file_path) {
	fastlanes::FileHeader header {};
	fastlanes::FileFooter footer {};

	fastlanes::FileHeader::Load(header, file_path);
	fastlanes::FileFooter::Load(footer, file_path);

	if (header.settings.inline_footer) {
		return fastlanes::TableDescriptorHandle::FromFileSlice(
		    file_path, footer.table_descriptor_offset, footer.table_descriptor_size, /*verify=*/true);
	}

	return fastlanes::TableDescriptorHandle::FromFile(file_path.parent_path() / "table_descriptor.fbb",
	                                                  /*verify=*/true);
}

size_t rowgroup_bytes(const fastlanes::RowgroupDescriptor* rg) {
	if (!rg) {
		return 0;
	}
	size_t      rg_bytes = 0;
	const auto* cols     = rg->m_column_descriptors();
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
		rg_bytes += static_cast<size_t>(rg->m_n_tuples()) * element_size;
	}
	return rg_bytes;
}

void check_rowgroup_index(const size_t n_rowgroups, const std::optional<size_t>& rowgroup) {
	if (rowgroup.has_value() && *rowgroup >= n_rowgroups) {
		throw std::out_of_range("rowgroup index out of range");
	}
}

size_t count_active_columns(const std::vector<expr::Expression>& expressions) {
	return runtime::count_active_columns(expressions);
}

struct StreamingBenchmarkChunk {
	runtime::ExecutionWorkset  workset {};
	runtime::AsyncWorksetRun   run {};
	struct PendingRowgroup {
		reader::Rowgroup              rowgroup {};
		std::vector<expr::Expression> expressions;
		size_t                        active_columns = 0;
	};
	std::vector<PendingRowgroup> pending_rowgroups;
	size_t                     work_items  = 0;
	size_t                     rowgroups   = 0;
	size_t                     active_columns = 0;
	size_t                     launch_grid = 0;
	size_t                     launches    = 0;
	bool                       submitted   = false;
};

struct RowgroupReadResult {
	reader::Rowgroup rowgroup {};
	double           read_ms = 0.0;
};

class RowgroupPrefetchQueue {
public:
	RowgroupPrefetchQueue(const std::filesystem::path& file_path,
	                      const size_t                start,
	                      const size_t                end,
	                      const bool                  use_zero_copy_parse,
	                      const size_t                depth)
	    : depth_(std::max<size_t>(1, depth))
	    , worker_([this, file_path, start, end, use_zero_copy_parse]() {
		    try {
			    reader::reader rdr(file_path);
			    for (size_t rg_idx = start; rg_idx < end; ++rg_idx) {
				    const auto read_start = std::chrono::steady_clock::now();
				    auto       rowgroup   = use_zero_copy_parse ? rdr.read_rowgroup_zero_copy_materialized(rg_idx)
				                                                : rdr.read_rowgroup(rg_idx);
				    const auto read_end = std::chrono::steady_clock::now();
				    Prefetched prefetched {
				        std::move(rowgroup),
				        std::chrono::duration<double, std::milli>(read_end - read_start).count(),
				    };

				    std::unique_lock<std::mutex> lock(mutex_);
				    cv_not_full_.wait(lock, [&]() { return stop_ || queue_.size() < depth_; });
				    if (stop_) {
					    return;
				    }
				    queue_.push_back(std::move(prefetched));
				    cv_not_empty_.notify_one();
			    }
		    } catch (...) {
			    std::lock_guard<std::mutex> lock(mutex_);
			    error_ = std::current_exception();
		    }

		    std::lock_guard<std::mutex> lock(mutex_);
		    done_ = true;
		    cv_not_empty_.notify_all();
	    }) {
	}

	~RowgroupPrefetchQueue() {
		{
			std::lock_guard<std::mutex> lock(mutex_);
			stop_ = true;
		}
		cv_not_full_.notify_all();
		cv_not_empty_.notify_all();
		if (worker_.joinable()) {
			worker_.join();
		}
	}

	RowgroupReadResult pop() {
		const auto         wait_start = std::chrono::steady_clock::now();
		std::exception_ptr pending_error;
		Prefetched         item;
		bool               underflow = false;
		{
			std::unique_lock<std::mutex> lock(mutex_);
			cv_not_empty_.wait(lock, [&]() { return stop_ || !queue_.empty() || done_ || error_ != nullptr; });
			const auto wait_end = std::chrono::steady_clock::now();
			wait_ms_ += std::chrono::duration<double, std::milli>(wait_end - wait_start).count();

			if (error_ != nullptr) {
				pending_error = error_;
			} else if (queue_.empty()) {
				underflow = true;
			} else {
				item = std::move(queue_.front());
				queue_.pop_front();
			}
		}
		cv_not_full_.notify_one();

		if (pending_error) {
			std::rethrow_exception(pending_error);
		}
		if (underflow) {
			throw std::runtime_error("rowgroup prefetch queue underflow");
		}
		return RowgroupReadResult {std::move(item.rowgroup), item.read_ms};
	}

	// Only safe to call from the consumer thread (the one that owns pop()).
	double wait_ms() const {
		return wait_ms_;
	}

private:
	struct Prefetched {
		reader::Rowgroup rowgroup {};
		double           read_ms = 0.0;
	};

	size_t                      depth_     = 1;
	std::thread                 worker_;
	mutable std::mutex          mutex_;
	std::condition_variable     cv_not_empty_;
	std::condition_variable     cv_not_full_;
	std::deque<Prefetched>      queue_;
	std::exception_ptr          error_;
	bool                        done_      = false;
	bool                        stop_      = false;
	double                      wait_ms_ = 0.0;
};

void reset_streaming_chunk(StreamingBenchmarkChunk& chunk) {
	chunk.pending_rowgroups.clear();
	chunk.work_items  = 0;
	chunk.rowgroups   = 0;
	chunk.active_columns = 0;
	chunk.launch_grid = 0;
	chunk.launches    = 0;
	chunk.submitted   = false;
}

void accumulate_rowgroup_stats(TableBenchmarkResult& result,
                               const size_t          rg_columns,
                               const size_t          rg_vectors,
                               const size_t          bytes,
                               const size_t          payload_arena_bytes,
                               const double          read_rowgroup_ms,
                               const double          assemble_expr_ms,
                               const double          append_expr_ms,
                               const double          upload_workset_ms,
                               const double          kernel_ms,
                               const double          release_device_ms,
                               const double          free_rowgroup_ms,
                               const size_t          launch_grid,
                               const size_t          launches) {
	result.read_rowgroup_ms += read_rowgroup_ms;
	result.assemble_expr_ms += assemble_expr_ms;
	result.append_expr_ms += append_expr_ms;
	result.upload_workset_ms += upload_workset_ms;
	result.kernel_ms += kernel_ms;
	result.release_device_ms += release_device_ms;
	result.free_rowgroup_ms += free_rowgroup_ms;
	result.total_launches += launches;
	result.total_launch_grid += launch_grid * launches;
	result.total_columns += rg_columns;
	result.total_items += rg_vectors;
	result.total_bytes += bytes;
	result.total_payload_arena_bytes += payload_arena_bytes;
	++result.total_rgs;
}

} // namespace

TableBenchmarkResult benchmark_table(const std::filesystem::path& fls_path, const TableBenchmarkConfig& cfg) {
	TableBenchmarkResult out {};
	out.samples = cfg.samples;

	reader::reader rdr(fls_path);
	const size_t   n_rowgroups = rdr.rowgroup_count();
	check_rowgroup_index(n_rowgroups, cfg.rowgroup);

	size_t start = 0;
	size_t end   = n_rowgroups;
	if (cfg.rowgroup.has_value()) {
		start = *cfg.rowgroup;
		end   = start + 1;
	}

	const auto  td_handle = load_table_descriptor(fls_path);
	const auto* td        = td_handle.Get();
	if (!td) {
		throw std::runtime_error("failed to load table descriptor");
	}

	const auto wall_start = std::chrono::steady_clock::now();
	const bool whole_table = !cfg.rowgroup.has_value() && cfg.aggregation_scope == AggregationScope::WholeTable;
	if (cfg.streaming_target_work_items == 0) {
		throw std::invalid_argument("streaming_target_work_items must be > 0");
	}
	if (cfg.prefetch_depth == 0) {
		throw std::invalid_argument("prefetch_depth must be > 0");
	}
	const size_t stream_target_work_items = cfg.streaming_target_work_items;
	const size_t stream_target_rowgroups  = cfg.streaming_target_rowgroups;
	const bool   use_rowgroup_prefetch    = whole_table && cfg.enable_rowgroup_prefetch;

	if (!whole_table) {
		for (size_t rg_idx = start; rg_idx < end; ++rg_idx) {
			const auto* rg    = td->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rg_idx));
			const size_t bytes = rowgroup_bytes(rg);

			const auto read_start = std::chrono::steady_clock::now();
			auto       rowgroup   = cfg.use_zero_copy_parse ? rdr.read_rowgroup_zero_copy_materialized(rg_idx)
			                                                : rdr.read_rowgroup(rg_idx);
			const auto read_end = std::chrono::steady_clock::now();
			const double read_rowgroup_ms = std::chrono::duration<double, std::milli>(read_end - read_start).count();

			const auto assemble_start = std::chrono::steady_clock::now();
			auto       expressions    = expr::assemble(rowgroup);
			const auto assemble_end   = std::chrono::steady_clock::now();
			const double assemble_expr_ms =
			    std::chrono::duration<double, std::milli>(assemble_end - assemble_start).count();

			const size_t rg_columns = count_active_columns(expressions);
			const size_t rg_vectors = rg_columns * rowgroup.n_vecs;

			runtime::ExecutionWorkset      workset {};
			runtime::ExecutionWorksetGuard guard(workset);
			const auto append_start = std::chrono::steady_clock::now();
			runtime::append_expressions(workset, expressions, cfg.execution);
			const auto append_end     = std::chrono::steady_clock::now();
			const double append_expr_ms = std::chrono::duration<double, std::milli>(append_end - append_start).count();

			const auto upload_start = std::chrono::steady_clock::now();
			runtime::upload_workset(workset);
			const auto upload_end       = std::chrono::steady_clock::now();
			const double upload_workset_ms =
			    std::chrono::duration<double, std::milli>(upload_end - upload_start).count();

			size_t       rg_launch_grid = 0;
			size_t       rg_launches    = 0;
			const double kernel_ms =
			    runtime::run_workset(workset, cfg.samples, cfg.execution, &rg_launch_grid, &rg_launches, true);

			const auto release_start = std::chrono::steady_clock::now();
			guard.dismiss();
			runtime::release_workset(workset);
			const auto release_end = std::chrono::steady_clock::now();
			const double release_device_ms = std::chrono::duration<double, std::milli>(release_end - release_start).count();

			const auto free_start = std::chrono::steady_clock::now();
			free_rowgroup(rowgroup);
			const auto free_end = std::chrono::steady_clock::now();
			const double free_rowgroup_ms = std::chrono::duration<double, std::milli>(free_end - free_start).count();

			accumulate_rowgroup_stats(out,
			                          rg_columns,
			                          rg_vectors,
			                          bytes,
			                          workset.payload_arena_bytes,
			                          read_rowgroup_ms,
			                          assemble_expr_ms,
			                          append_expr_ms,
			                          upload_workset_ms,
			                          kernel_ms,
			                          release_device_ms,
			                          free_rowgroup_ms,
			                          rg_launch_grid,
			                          rg_launches);
		}
		const auto wall_end = std::chrono::steady_clock::now();
		out.end_to_end_ms   = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
		return out;
	}

	std::unique_ptr<RowgroupPrefetchQueue> prefetch_queue;
	if (use_rowgroup_prefetch) {
		prefetch_queue = std::make_unique<RowgroupPrefetchQueue>(
		    fls_path, start, end, cfg.use_zero_copy_parse, cfg.prefetch_depth);
	}

	const auto fetch_rowgroup = [&](const size_t rg_idx) -> reader::Rowgroup {
		RowgroupReadResult result {};
		if (prefetch_queue) {
			result = prefetch_queue->pop();
			++out.prefetched_rowgroups;
		} else {
			const auto read_start = std::chrono::steady_clock::now();
			result.rowgroup       = cfg.use_zero_copy_parse ? rdr.read_rowgroup_zero_copy_materialized(rg_idx)
			                                                : rdr.read_rowgroup(rg_idx);
			const auto read_end = std::chrono::steady_clock::now();
			result.read_ms      = std::chrono::duration<double, std::milli>(read_end - read_start).count();
		}
		out.read_rowgroup_ms += result.read_ms;
		return std::move(result.rowgroup);
	};

	if (!cfg.enable_streaming) {
		runtime::ExecutionWorkset   workset {};
		std::vector<StreamingBenchmarkChunk::PendingRowgroup> pending_rowgroups;
		size_t                      chunk_work_items = 0;
		size_t                      chunk_rowgroups  = 0;
		size_t                      chunk_active_columns = 0;
		bool                        did_warmup       = false;

		const auto run_sync_chunk = [&]() {
			if (chunk_rowgroups == 0) {
				return;
			}

			runtime::begin_workset_chunk_arena(workset, chunk_active_columns);
			const auto append_start = std::chrono::steady_clock::now();
			for (auto& pending : pending_rowgroups) {
				runtime::append_expressions(workset, pending.expressions, cfg.execution);
			}
			const auto append_end = std::chrono::steady_clock::now();
			out.append_expr_ms += std::chrono::duration<double, std::milli>(append_end - append_start).count();

			const auto upload_start = std::chrono::steady_clock::now();
			runtime::upload_workset(workset);
			const auto upload_end = std::chrono::steady_clock::now();
			out.upload_workset_ms += std::chrono::duration<double, std::milli>(upload_end - upload_start).count();
			out.total_payload_arena_bytes += workset.payload_arena_bytes;

			size_t chunk_launch_grid = 0;
			size_t chunk_launches    = 0;
			const bool warmup_once   = !did_warmup;
			out.kernel_ms += runtime::run_workset(
			    workset, cfg.samples, cfg.execution, &chunk_launch_grid, &chunk_launches, warmup_once);
			did_warmup = true;
			out.total_launches += chunk_launches;
			out.total_launch_grid += chunk_launch_grid * chunk_launches;

			const auto release_start = std::chrono::steady_clock::now();
			runtime::release_workset(workset);
			const auto release_end = std::chrono::steady_clock::now();
			out.release_device_ms += std::chrono::duration<double, std::milli>(release_end - release_start).count();

			for (auto& pending : pending_rowgroups) {
				const auto free_start = std::chrono::steady_clock::now();
				free_rowgroup(pending.rowgroup);
				const auto free_end = std::chrono::steady_clock::now();
				out.free_rowgroup_ms += std::chrono::duration<double, std::milli>(free_end - free_start).count();
			}
			pending_rowgroups.clear();
			chunk_work_items = 0;
			chunk_rowgroups  = 0;
			chunk_active_columns = 0;
		};

		for (size_t rg_idx = start; rg_idx < end; ++rg_idx) {
			const auto* rg     = td->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rg_idx));
			const size_t bytes = rowgroup_bytes(rg);

			auto rowgroup = fetch_rowgroup(rg_idx);

			const auto assemble_start = std::chrono::steady_clock::now();
			auto       expressions    = expr::assemble(rowgroup);
			const auto assemble_end   = std::chrono::steady_clock::now();
			out.assemble_expr_ms += std::chrono::duration<double, std::milli>(assemble_end - assemble_start).count();

			const size_t rg_columns = count_active_columns(expressions);
			const size_t rg_vectors = rg_columns * rowgroup.n_vecs;

			chunk_work_items += rg_vectors;
			++chunk_rowgroups;
			chunk_active_columns += rg_columns;
			pending_rowgroups.push_back(
			    StreamingBenchmarkChunk::PendingRowgroup {std::move(rowgroup), std::move(expressions), rg_columns});

			out.total_columns += rg_columns;
			out.total_items += rg_vectors;
			out.total_bytes += bytes;
			++out.total_rgs;

			const bool reach_items     = chunk_work_items >= stream_target_work_items;
				const bool reach_rowgroups = (stream_target_rowgroups > 0) && (chunk_rowgroups >= stream_target_rowgroups);
				if (reach_items || reach_rowgroups) {
					run_sync_chunk();
				}
			}

		run_sync_chunk();
		if (prefetch_queue) {
			out.prefetch_wait_ms += prefetch_queue->wait_ms();
		}

		const auto wall_end = std::chrono::steady_clock::now();
		out.end_to_end_ms   = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
		return out;
	}

	runtime::StreamingDoubleBuffer<StreamingBenchmarkChunk> pipeline {};
	bool                                                    did_warmup = false;

	const auto submit_chunk = [&](StreamingBenchmarkChunk& chunk) {
		if (chunk.rowgroups == 0) {
			return;
		}
		runtime::begin_workset_chunk_arena(chunk.workset, chunk.active_columns);
		const auto append_start = std::chrono::steady_clock::now();
		for (auto& pending : chunk.pending_rowgroups) {
			runtime::append_expressions(chunk.workset, pending.expressions, cfg.execution);
		}
		const auto append_end = std::chrono::steady_clock::now();
		out.append_expr_ms += std::chrono::duration<double, std::milli>(append_end - append_start).count();

		const auto upload_start = std::chrono::steady_clock::now();
		runtime::upload_workset(chunk.workset);
		const auto upload_end = std::chrono::steady_clock::now();
		out.upload_workset_ms += std::chrono::duration<double, std::milli>(upload_end - upload_start).count();
		out.total_payload_arena_bytes += chunk.workset.payload_arena_bytes;

		const bool warmup_once = !did_warmup;
		chunk.run = runtime::run_workset_async(
		    chunk.workset, cfg.samples, cfg.execution, &chunk.launch_grid, &chunk.launches, warmup_once);
		chunk.submitted = true;
		did_warmup = true;
	};

	const auto consume_chunk = [&](StreamingBenchmarkChunk& chunk) {
		if (chunk.rowgroups == 0) {
			return;
		}
		if (chunk.submitted) {
			runtime::wait_workset_async(chunk.run);
			out.kernel_ms += chunk.run.elapsed_ms;
			out.total_launches += chunk.launches;
			out.total_launch_grid += chunk.launch_grid * chunk.launches;
		}

		const auto release_start = std::chrono::steady_clock::now();
		runtime::release_workset(chunk.workset, /*preserve_resources=*/true);
		const auto release_end = std::chrono::steady_clock::now();
		out.release_device_ms += std::chrono::duration<double, std::milli>(release_end - release_start).count();

		for (auto& pending : chunk.pending_rowgroups) {
			const auto free_start = std::chrono::steady_clock::now();
			free_rowgroup(pending.rowgroup);
			const auto free_end = std::chrono::steady_clock::now();
			out.free_rowgroup_ms += std::chrono::duration<double, std::milli>(free_end - free_start).count();
		}

		reset_streaming_chunk(chunk);
	};

	for (size_t rg_idx = start; rg_idx < end; ++rg_idx) {
		const auto* rg    = td->m_rowgroup_descriptors()->Get(static_cast<flatbuffers::uoffset_t>(rg_idx));
		const size_t bytes = rowgroup_bytes(rg);

		auto rowgroup = fetch_rowgroup(rg_idx);

		const auto assemble_start = std::chrono::steady_clock::now();
		auto       expressions    = expr::assemble(rowgroup);
		const auto assemble_end   = std::chrono::steady_clock::now();
		out.assemble_expr_ms += std::chrono::duration<double, std::milli>(assemble_end - assemble_start).count();

		auto&        chunk      = pipeline.build_chunk();
		const size_t rg_columns = count_active_columns(expressions);
		const size_t rg_vectors = rg_columns * rowgroup.n_vecs;

		chunk.work_items += rg_vectors;
		++chunk.rowgroups;
		chunk.active_columns += rg_columns;

		out.total_columns += rg_columns;
		out.total_items += rg_vectors;
		out.total_bytes += bytes;
		++out.total_rgs;
		chunk.pending_rowgroups.push_back(
		    StreamingBenchmarkChunk::PendingRowgroup {std::move(rowgroup), std::move(expressions), rg_columns});

		const bool reach_items     = chunk.work_items >= stream_target_work_items;
		const bool reach_rowgroups = (stream_target_rowgroups > 0) && (chunk.rowgroups >= stream_target_rowgroups);
		if (reach_items || reach_rowgroups) {
			pipeline.submit_build_and_rotate(submit_chunk, consume_chunk);
		}
	}

	pipeline.flush(submit_chunk, consume_chunk);
	if (prefetch_queue) {
		out.prefetch_wait_ms += prefetch_queue->wait_ms();
	}

	const auto wall_end = std::chrono::steady_clock::now();
	out.end_to_end_ms    = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
	return out;
}

} // namespace dispatch
