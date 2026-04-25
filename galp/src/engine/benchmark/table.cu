// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/benchmark/table.cu
// ────────────────────────────────────────────────────────
#include "engine/benchmark/table.cuh"
#include "engine/execution/internal/table_pipeline.cuh"
#include "flsgpu/memory/cuda_macros.cuh"
#include <chrono>
#include <mutex>

namespace dispatch {
namespace {

__global__ void benchmark_warmup_kernel() {
}

void warmup_cuda_runtime_once() {
	static std::once_flag once;
	std::call_once(once, []() {
		CUDA_SAFE_CALL(cudaFree(0));

		cudaStream_t stream {};
		cudaEvent_t  done {};
		CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
		CUDA_SAFE_CALL(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));

		for (int i = 0; i < 32; ++i) {
			benchmark_warmup_kernel<<<1, 1, 0, stream>>>();
		}
		CUDA_SAFE_CALL(cudaGetLastError());
		CUDA_SAFE_CALL(cudaEventRecord(done, stream));
		CUDA_SAFE_CALL(cudaEventSynchronize(done));
		CUDA_SAFE_CALL(cudaEventDestroy(done));
		CUDA_SAFE_CALL(cudaStreamDestroy(stream));
	});
}

struct BenchmarkObserver {
	TableBenchmarkResult& out;

	void on_rowgroup_read(const runtime::RowgroupReadResult& result, const bool from_prefetch) {
		out.read_rowgroup_ms += result.read_ms;
		out.file_read_ms += result.file_read_ms;
		out.rowgroup_build_ms += result.rowgroup_build_ms;
		if (from_prefetch) {
			++out.prefetched_rowgroups;
		}
	}

	void on_assemble_expr(const double ms) {
		out.assemble_expr_ms += ms;
	}

	void on_rowgroup_stats(const size_t active_columns, const size_t work_items, const size_t logical_bytes) {
		out.total_columns += active_columns;
		out.total_items += work_items;
		out.total_bytes += logical_bytes;
		++out.total_rgs;
	}

	void on_append_expr(const double ms) {
		out.append_expr_ms += ms;
	}

	void on_upload_workset(const double                    ms,
	                       const runtime::UploadBreakdown& breakdown,
	                       const size_t                    payload_arena_bytes,
	                       const size_t                    output_arena_bytes) {
		out.upload_workset_ms += ms;
		out.upload_prep_ms += breakdown.prep_ms;
		out.upload_prep_reset_ms += breakdown.prep_reset_ms;
		out.upload_prep_output_arena_ms += breakdown.prep_output_arena_ms;
		out.upload_prep_bind_ms += breakdown.prep_bind_ms;
		out.upload_prep_slots_ms += breakdown.prep_slots_ms;
		out.upload_arena_pack_ms += breakdown.arena_pack_ms;
		out.upload_layout_ms += breakdown.arena.layout_ms;
		out.upload_alloc_ms += breakdown.arena.alloc_ms;
		out.upload_resolve_ms += breakdown.arena.resolve_ms;
		out.upload_pack_ms += breakdown.arena.pack_ms;
		out.upload_dma_issue_ms += breakdown.arena.dma_issue_ms;
		out.upload_event_ms += breakdown.event_record_ms;
		out.total_payload_arena_bytes += payload_arena_bytes;
		out.total_output_arena_bytes += output_arena_bytes;
	}

	void on_kernel(const double ms, const size_t launch_grid, const size_t launches) {
		out.kernel_ms += ms;
		out.total_launches += launches;
		out.total_launch_grid += launch_grid * launches;
	}

	void on_release_workset(const double ms) {
		out.release_device_ms += ms;
	}

	void on_free_rowgroup(const double ms) {
		out.free_rowgroup_ms += ms;
	}

	void on_prefetch_wait(const double ms) {
		out.prefetch_wait_ms += ms;
	}
};

} // namespace

TableBenchmarkResult benchmark_table(const std::filesystem::path& fls_path, const TableBenchmarkConfig& cfg) {
	warmup_cuda_runtime_once();

	TableBenchmarkResult out {};
	out.samples = cfg.samples;

	runtime::TableExecutionRequest request {};
	request.config              = cfg;
	request.samples             = cfg.samples;
	request.rowgroup            = cfg.rowgroup;
	request.materialize_results = false;
	request.warmup_first_run    = true;

	BenchmarkObserver observer {out};
	const auto        wall_start = std::chrono::steady_clock::now();
	runtime::execute_table_pipeline(
	    fls_path,
	    request,
	    [](size_t) { return true; },
	    [](size_t, reader::Rowgroup&, const std::vector<expr::Expression>&, const RowgroupData*) {},
	    observer);
	const auto wall_end = std::chrono::steady_clock::now();
	out.end_to_end_ms   = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
	return out;
}

} // namespace dispatch
