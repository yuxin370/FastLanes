// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/encoder/encoder.cpp
// ────────────────────────────────────────────────────────
#include "fls/encoder/encoder.hpp"
#include "fls/common/alias.hpp" // for up, n_t
#include "fls/connection.hpp"   // for Connection
#include "fls/cor/lyt/buf.hpp"  // for Buf
#include "fls/encoder/validate_options.hpp"
#include "fls/expression/expression_executor.hpp" // for ExprExecutor
#include "fls/expression/interpreter.hpp"         // for Interpreter
#include "fls/expression/physical_expression.hpp" // for PhysicalExpr
#include "fls/file/file_header.hpp"
#include "fls/footer/rowgroup_descriptor_generated.h"
#include "fls/io/file.hpp" // for File
#include "fls/std/filesystem.hpp"
#include "fls/std/vector.hpp"     // for vector
#include "fls/table/rowgroup.hpp" // for Rowgroup
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <fls/io/io.hpp>
#include <memory> // for unique_ptr
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace fastlanes {

namespace {

void encode_rowgroup(const Rowgroup& rowgroup, RowgroupDescriptorT& rowgroup_descriptor, Buf& buf) {
	vector<uint8_t> helper_buffer(sizeof(entry_point_t) * rowgroup_descriptor.m_n_vec);

	for (auto& column_descriptor : rowgroup_descriptor.m_column_descriptors) {
		InterpreterState state;
		auto physical_expr_up = Interpreter::Encoding::Interpret(*column_descriptor, rowgroup.internal_rowgroup, state);

		for (n_t vec_idx {0}; vec_idx < rowgroup_descriptor.m_n_vec; ++vec_idx) {
			physical_expr_up->PointTo(vec_idx);
			ExprExecutor::execute(*physical_expr_up, vec_idx);
		}

		physical_expr_up->Finalize();
		physical_expr_up->Flush(buf, *column_descriptor, helper_buffer.data());
	}
}

struct EncodedRowgroup {
	up<Buf>             payload;
	RowgroupDescriptorT descriptor;
};

void append_rowgroup(io& output, const Buf& payload, RowgroupDescriptorT& descriptor, n_t& offset) {
	IO::append(output, payload);
	descriptor.m_size   = payload.Size();
	descriptor.m_offset = offset;
	offset += descriptor.m_size;
}

struct ParallelEncodingState {
	explicit ParallelEncodingState(const n_t rowgroup_count, const n_t buffer_count)
	    : results(static_cast<std::size_t>(rowgroup_count)) {
		available_buffers.reserve(static_cast<std::size_t>(buffer_count));
		for (n_t buffer_idx = 0; buffer_idx < buffer_count; ++buffer_idx) {
			available_buffers.push_back(make_unique<Buf>(64U * 1024U));
		}
	}

	// Called with mutex held. In budgeted mode, admit the next rowgroup in order;
	// an empty queue can accept one oversized payload to guarantee progress.
	bool can_publish(n_t index, n_t bytes, n_t limit) const {
		return limit == 0 || (index == next_publish &&
		                      (buffered_bytes == 0 || (buffered_bytes <= limit && bytes <= limit - buffered_bytes)));
	}

	bool publish(n_t index, up<EncodedRowgroup> encoded, n_t byte_limit) {
		const n_t bytes = encoded->payload->Size();
		{
			std::unique_lock lock(mutex);
			cv.wait(lock, [&] { return cancelled || can_publish(index, bytes, byte_limit); });
			if (cancelled) {
				return false;
			}
			buffered_bytes += bytes;
			peak_bytes     = std::max(peak_bytes, buffered_bytes);
			results[index] = std::move(encoded);
			if (byte_limit != 0) {
				++next_publish;
			}
		}
		cv.notify_all();
		return true;
	}

	std::mutex                  mutex;
	std::condition_variable     cv;
	vector<up<EncodedRowgroup>> results;
	vector<up<Buf>>             available_buffers;
	n_t                         next_claim     = 0;
	n_t                         next_publish   = 0;
	n_t                         next_commit    = 0;
	n_t                         buffered_bytes = 0;
	n_t                         peak_bytes     = 0;
	n_t                         peak_rowgroups = 0;
	bool                        cancelled      = false;
	std::exception_ptr          first_exception;
};

void save_first_exception(ParallelEncodingState& state, std::exception_ptr exception) {
	std::lock_guard<std::mutex> lock(state.mutex);
	if (!state.first_exception) {
		state.first_exception = std::move(exception);
	}
	state.cancelled = true;
	state.cv.notify_all();
}

EncodingStats
encode_serial(const Table& table, TableDescriptorT& table_descriptor, io& file_io, const EncodingOptions& options) {
	Buf buf; // TODO[memory pool]

	n_t cur_rowgroup_offset {sizeof(FileHeader)};

	EncodingStats stats;
	stats.requested_worker_count      = options.worker_count;
	stats.effective_worker_count      = table.get_n_rowgroups() == 0 ? 0 : 1;
	stats.encoded_rowgroups           = table.get_n_rowgroups();
	stats.resolved_inflight_rowgroups = table.get_n_rowgroups() == 0 ? 0 : 1;
	stats.resolved_rowgroups_per_task = table.get_n_rowgroups() == 0 ? 0 : 1;

	for (n_t rowgroup_idx {0}; rowgroup_idx < table.get_n_rowgroups(); ++rowgroup_idx) {
		auto&       rowgroup_descriptor = *table_descriptor.m_rowgroup_descriptors[rowgroup_idx];
		const auto& rowgroup            = *table.m_rowgroups[rowgroup_idx];

		encode_rowgroup(rowgroup, rowgroup_descriptor, buf);

		stats.peak_inflight_rowgroups   = 1;
		stats.peak_pending_output_bytes = std::max(stats.peak_pending_output_bytes, buf.Size());
		append_rowgroup(file_io, buf, rowgroup_descriptor, cur_rowgroup_offset);
		buf.Reset();
	}
	table_descriptor.m_table_binary_size = cur_rowgroup_offset;
	return stats;
}

EncodingStats
encode_parallel(const Table& table, TableDescriptorT& table_descriptor, io& file_io, const EncodingOptions& options) {
	const n_t     rowgroup_count = table.get_n_rowgroups();
	EncodingStats stats;
	stats.requested_worker_count = options.worker_count;
	stats.encoded_rowgroups      = rowgroup_count;
	if (rowgroup_count == 0) {
		table_descriptor.m_table_binary_size = sizeof(FileHeader);
		return stats;
	}

	const n_t worker_count    = std::min(options.worker_count, rowgroup_count);
	const n_t window          = std::min(rowgroup_count,
                                options.max_inflight_rowgroups == 0 ? std::max<n_t>(1, worker_count * 2)
	                                                                         : options.max_inflight_rowgroups);
	const n_t adaptive_chunk  = std::max<n_t>(1, rowgroup_count / std::max<n_t>(1, worker_count * 4));
	const n_t requested_chunk = options.rowgroups_per_task == 0 ? adaptive_chunk : options.rowgroups_per_task;
	const n_t chunk_size      = std::min(requested_chunk, std::max<n_t>(1, window / worker_count));

	stats.effective_worker_count      = worker_count;
	stats.resolved_inflight_rowgroups = window;
	stats.resolved_rowgroups_per_task = chunk_size;

	ParallelEncodingState state(rowgroup_count, window);
	vector<std::thread>   workers;
	workers.reserve(static_cast<std::size_t>(worker_count));
	n_t cur_rowgroup_offset {sizeof(FileHeader)};

	const auto worker = [&]() {
		try {
			while (true) {
				n_t chunk_begin = 0;
				n_t chunk_end   = 0;
				{
					std::unique_lock<std::mutex> lock(state.mutex);
					state.cv.wait(lock, [&]() {
						return state.cancelled || state.next_claim == rowgroup_count ||
						       state.next_claim < std::min(rowgroup_count, state.next_commit + window);
					});
					if (state.cancelled || state.next_claim == rowgroup_count) {
						return;
					}

					chunk_begin      = state.next_claim;
					chunk_end        = std::min({rowgroup_count, chunk_begin + chunk_size, state.next_commit + window});
					state.next_claim = chunk_end;
					state.peak_rowgroups = std::max(state.peak_rowgroups, state.next_claim - state.next_commit);
				}

				for (n_t rowgroup_idx = chunk_begin; rowgroup_idx < chunk_end; ++rowgroup_idx) {
					{
						std::lock_guard<std::mutex> lock(state.mutex);
						if (state.cancelled) {
							return;
						}
					}

					auto encoded = make_unique<EncodedRowgroup>();
					{
						std::unique_lock<std::mutex> lock(state.mutex);
						state.cv.wait(lock, [&]() { return state.cancelled || !state.available_buffers.empty(); });
						if (state.cancelled) {
							return;
						}
						encoded->payload = std::move(state.available_buffers.back());
						state.available_buffers.pop_back();
					}
					encoded->descriptor = *table_descriptor.m_rowgroup_descriptors[rowgroup_idx];
					encode_rowgroup(*table.m_rowgroups[rowgroup_idx], encoded->descriptor, *encoded->payload);
					if (!state.publish(rowgroup_idx, std::move(encoded), options.max_pending_output_bytes)) {
						return;
					}
				}
			}
		} catch (...) { save_first_exception(state, std::current_exception()); }
	};

	// Only this coordinator joins these threads, and it cannot be one of them.
	// With joinable() checked, neither invalid-thread nor self-join is possible.
	const auto join_workers = [&]() noexcept {
		for (auto& thread : workers) {
			if (thread.joinable()) {
				thread.join();
			}
		}
	};

	try {
		for (n_t worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
			workers.emplace_back(worker);
		}
		for (n_t rowgroup_idx = 0; rowgroup_idx < rowgroup_count; ++rowgroup_idx) {
			up<EncodedRowgroup> encoded;
			{
				std::unique_lock<std::mutex> lock(state.mutex);
				state.cv.wait(lock, [&]() { return state.cancelled || state.results[rowgroup_idx] != nullptr; });
				if (state.cancelled) {
					break;
				}
				encoded = std::move(state.results[rowgroup_idx]);
			}

			append_rowgroup(file_io, *encoded->payload, encoded->descriptor, cur_rowgroup_offset);
			*table_descriptor.m_rowgroup_descriptors[rowgroup_idx] = std::move(encoded->descriptor);

			const auto payload_size = encoded->payload->Size();
			encoded->payload->Reset();
			{
				std::lock_guard<std::mutex> lock(state.mutex);
				state.buffered_bytes -= payload_size;
				state.available_buffers.push_back(std::move(encoded->payload));
				++state.next_commit;
			}
			state.cv.notify_all();
		}
	} catch (...) { save_first_exception(state, std::current_exception()); }

	join_workers();
	if (state.first_exception) {
		std::rethrow_exception(state.first_exception);
	}

	table_descriptor.m_table_binary_size = cur_rowgroup_offset;
	stats.peak_inflight_rowgroups        = state.peak_rowgroups;
	stats.peak_pending_output_bytes      = state.peak_bytes;
	return stats;
}

} // namespace

void Encoder::encode(const Connection& connection, const path& file_path) {
	(void)encode(connection, file_path, EncodingOptions {});
}

EncodingStats Encoder::encode(const Connection& connection, const path& file_path, const EncodingOptions& options) {
	detail::validate_encoding_options(options);
	io file_io = make_unique<File>(file_path);
	if (options.worker_count == 1) {
		return encode_serial(*connection.m_table, *connection.m_table_descriptor, file_io, options);
	}

	// Keep descriptor publication transactional as well as file publication. A
	// worker failure cannot publish a partially encoded set of descriptors.
	// This does not make the preceding in-place preparation safe to repeat.
	auto working_descriptor        = make_unique<TableDescriptorT>(*connection.m_table_descriptor);
	auto stats                     = encode_parallel(*connection.m_table, *working_descriptor, file_io, options);
	*connection.m_table_descriptor = std::move(*working_descriptor);
	return stats;
}

} // namespace fastlanes
