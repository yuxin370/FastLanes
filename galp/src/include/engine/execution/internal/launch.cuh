// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/include/engine/execution/internal/launch.cuh
// ────────────────────────────────────────────────────────
#ifndef ENGINE_EXECUTION_INTERNAL_LAUNCH_CUH
#define ENGINE_EXECUTION_INTERNAL_LAUNCH_CUH

#include "engine/execution/internal/prepare.cuh"

namespace dispatch::runtime {

inline bool uses_mixed_dispatch(const LaunchStrategy strategy) {
	return strategy == LaunchStrategy::MixedDispatch;
}

template <bool WRITE_OUT>
inline void launch_typed_batches(ExecutionWorkset& workset, cudaStream_t stream) {
	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T            = typename decltype(tag)::type;
		auto& host_batch   = workset.buffers.host_batches.template get<T>();
		auto& device_batch = workset.buffers.device_batches.template get<T>();
		if (device_batch.d_exprs == nullptr || device_batch.d_items == nullptr) {
			return;
		}
		dispatch::detail::launch_batch_no_sync<T, WRITE_OUT>(
		    host_batch, device_batch.d_exprs, device_batch.d_items, device_batch.n_items, stream);
	});
}

template <bool WRITE_OUT>
inline void launch_mixed_dispatch(ExecutionWorkset& workset, cudaStream_t stream) {
	const auto*            exprs_i8  = workset.buffers.device_batches.template get<int8_t>().d_exprs;
	const auto*            exprs_i16 = workset.buffers.device_batches.template get<int16_t>().d_exprs;
	const MixedSlotMapping mapping(workset.slots.mixed.size());
	const dim3             block(MixedSlotMapping::N_THREADS_PER_BLOCK);
	const dim3             grid(mapping.n_blocks());

	kernels::device::decompress_dispatch_mixed<1, 1, WRITE_OUT>
	    <<<grid, block, 0, stream>>>(exprs_i8, exprs_i16, workset.slots.d, workset.slots.mixed.size());
	CUDA_SAFE_CALL(cudaGetLastError());
}

template <LaunchStrategy Strategy, bool WRITE_OUT>
inline void launch_strategy_once(ExecutionWorkset& workset, cudaStream_t stream) {
	if constexpr (Strategy == LaunchStrategy::MixedDispatch) {
		launch_mixed_dispatch<WRITE_OUT>(workset, stream);
	} else {
		launch_typed_batches<WRITE_OUT>(workset, stream);
	}
}

inline size_t typed_launches_per_sample(const ExecutionWorkset& workset) {
	size_t launches = 0;
	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T = typename decltype(tag)::type;
		auto& d = workset.buffers.device_batches.template get<T>();
		if (d.d_exprs != nullptr && d.d_items != nullptr) {
			++launches;
		}
	});
	return launches;
}

inline size_t typed_total_items_per_sample(const ExecutionWorkset& workset) {
	size_t items = 0;
	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T = typename decltype(tag)::type;
		items += workset.buffers.device_batches.template get<T>().n_items;
	});
	return items;
}

inline bool has_any_expr(const ExecutionWorkset& workset) {
	bool has_any = false;
	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T = typename decltype(tag)::type;
		auto& d = workset.buffers.device_batches.template get<T>();
		has_any = has_any || (d.d_exprs != nullptr && d.d_items != nullptr);
	});
	return has_any;
}

struct AsyncWorksetRun {
	cudaStream_t stream = nullptr;
	cudaEvent_t  start  = nullptr;
	cudaEvent_t  stop   = nullptr;
	double       elapsed_ms = 0.0;
	bool         active = false;
};

inline AsyncWorksetRun run_workset_async(ExecutionWorkset&      workset,
                                         const uint32_t         samples,
                                         const ExecutionConfig& cfg,
                                         size_t*                out_grid     = nullptr,
                                         size_t*                out_launches = nullptr,
                                         const bool             warmup       = false) {
	AsyncWorksetRun handle {};
	if (!has_any_expr(workset)) {
		if (out_launches) {
			*out_launches = 0;
		}
		return handle;
	}

	const size_t launches_per_sample = typed_launches_per_sample(workset);
	const size_t total_items         = typed_total_items_per_sample(workset);
	const bool   mixed_dispatch      = uses_mixed_dispatch(cfg.launch_strategy);

	if (mixed_dispatch && (workset.slots.d == nullptr || workset.slots.mixed.empty())) {
		if (out_launches) {
			*out_launches = 0;
		}
		return handle;
	}

	if (out_grid) {
		if (mixed_dispatch) {
			const MixedSlotMapping mapping(workset.slots.mixed.size());
			*out_grid = mapping.n_blocks();
		} else {
			*out_grid = (launches_per_sample > 0) ? (total_items / launches_per_sample) : 0;
		}
	}
	if (out_launches) {
		*out_launches = (mixed_dispatch ? 1 : launches_per_sample) * static_cast<size_t>(samples);
	}

	if (workset.transfer.h2d_stream == nullptr) {
		flsgpu::memory::sync_h2d();
	} else if (!use_async_h2d()) {
		flsgpu::memory::sync_h2d(workset.transfer.h2d_stream);
	}
	handle.stream = ensure_workset_compute_stream(workset);
	CUDA_SAFE_CALL(cudaEventCreate(&handle.start));
	CUDA_SAFE_CALL(cudaEventCreate(&handle.stop));
	handle.active = true;

	if (use_async_h2d() && workset.transfer.h2d_stream != nullptr && workset.transfer.h2d_ready_event != nullptr) {
		CUDA_SAFE_CALL(cudaStreamWaitEvent(handle.stream, workset.transfer.h2d_ready_event, 0));
	}

	if (warmup) {
		if (mixed_dispatch) {
			if (cfg.write_out) {
				launch_strategy_once<LaunchStrategy::MixedDispatch, true>(workset, handle.stream);
			} else {
				launch_strategy_once<LaunchStrategy::MixedDispatch, false>(workset, handle.stream);
			}
		} else {
			if (cfg.write_out) {
				launch_strategy_once<LaunchStrategy::TypedBatches, true>(workset, handle.stream);
			} else {
				launch_strategy_once<LaunchStrategy::TypedBatches, false>(workset, handle.stream);
			}
		}
		CUDA_SAFE_CALL(cudaStreamSynchronize(handle.stream));
	}

	CUDA_SAFE_CALL(cudaEventRecord(handle.start, handle.stream));

	for (uint32_t sample = 0; sample < samples; ++sample) {
		if (mixed_dispatch) {
			if (cfg.write_out) {
				launch_strategy_once<LaunchStrategy::MixedDispatch, true>(workset, handle.stream);
			} else {
				launch_strategy_once<LaunchStrategy::MixedDispatch, false>(workset, handle.stream);
			}
		} else {
			if (cfg.write_out) {
				launch_strategy_once<LaunchStrategy::TypedBatches, true>(workset, handle.stream);
			} else {
				launch_strategy_once<LaunchStrategy::TypedBatches, false>(workset, handle.stream);
			}
		}
	}
	CUDA_SAFE_CALL(cudaEventRecord(handle.stop, handle.stream));

	return handle;
}

inline void wait_workset_async(AsyncWorksetRun& handle) {
	if (!handle.active) {
		return;
	}
	CUDA_SAFE_CALL(cudaEventSynchronize(handle.stop));
	float ms = 0.0f;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, handle.start, handle.stop));
	handle.elapsed_ms = static_cast<double>(ms);
	CUDA_SAFE_CALL(cudaEventDestroy(handle.start));
	CUDA_SAFE_CALL(cudaEventDestroy(handle.stop));
	handle.start = nullptr;
	handle.stop  = nullptr;
	handle.active = false;
}

inline double run_workset(ExecutionWorkset&      workset,
                          const uint32_t         samples,
                          const ExecutionConfig& cfg,
                          size_t*                out_grid     = nullptr,
                          size_t*                out_launches = nullptr,
                          const bool             warmup       = false) {
	auto handle = run_workset_async(workset, samples, cfg, out_grid, out_launches, warmup);
	wait_workset_async(handle);
	return handle.elapsed_ms;
}

} // namespace dispatch::runtime

#endif // ENGINE_EXECUTION_INTERNAL_LAUNCH_CUH
