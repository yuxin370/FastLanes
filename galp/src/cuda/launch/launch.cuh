// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/cuda/launch/launch.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_CUDA_LAUNCH_LAUNCH_CUH
#define GALP_CUDA_LAUNCH_LAUNCH_CUH

#include "engine/dispatch.cuh"
#include "engine/operators/batch_kernel.cuh"
#include "engine/unpack_dispatch.cuh"
#include "engine/workset/model.cuh"
#include "engine/workset/streams.cuh"
#include <chrono>

namespace galp::runtime {

using galp::execution::ExecutionConfig;
using galp::execution::LaunchStrategy;

inline bool uses_mixed_dispatch(const LaunchStrategy strategy) {
	return strategy == LaunchStrategy::MixedDispatch;
}

template <bool WRITE_OUT>
inline void launch_typed_batches(ExecutionWorkset& workset, const ExecutionConfig& cfg, cudaStream_t stream) {
	auto scalar_tail_cfg             = cfg;
	scalar_tail_cfg.unpack_n_vectors = 1;
	galp::execution::for_each_type(galp::execution::SupportedTypes {}, [&](auto tag) {
		using T            = typename decltype(tag)::type;
		auto& host_batch   = workset.buffers.host_batches.template get<T>();
		auto& device_batch = workset.buffers.device_batches.template get<T>();
		if (device_batch.d_exprs == nullptr) {
			return;
		}
		if (device_batch.d_items != nullptr && device_batch.n_items != 0) {
			galp::execution::detail::launch_batch_no_sync<T, WRITE_OUT>(
			    host_batch, device_batch.d_exprs, device_batch.d_items, device_batch.n_items, cfg, stream);
		}
		if (device_batch.d_scalar_tail_items != nullptr && device_batch.n_scalar_tail_items != 0) {
			galp::execution::detail::launch_batch_no_sync<T, WRITE_OUT>(host_batch,
			                                                            device_batch.d_exprs,
			                                                            device_batch.d_scalar_tail_items,
			                                                            device_batch.n_scalar_tail_items,
			                                                            scalar_tail_cfg,
			                                                            stream);
		}
	});
}

template <bool WRITE_OUT>
inline void launch_mixed_slots(const galp::execution::DeviceExpression<int8_t>*  exprs_i8,
                               const galp::execution::DeviceExpression<int16_t>* exprs_i16,
                               const galp::execution::MixedWorkSlot*             slots,
                               const size_t                                      n_slots,
                               const ExecutionConfig&                            cfg,
                               cudaStream_t                                      stream) {
	if (slots == nullptr || n_slots == 0) {
		return;
	}
	const MixedSlotMapping mapping(n_slots);
	const dim3             block(MixedSlotMapping::N_THREADS_PER_BLOCK);
	const dim3             grid(mapping.n_blocks());

	runtime::with_unpack_config(cfg, [&](auto unpack_n_vectors, auto unpack_n_values) {
		constexpr unsigned UNPACK_N_VECTORS = decltype(unpack_n_vectors)::value;
		constexpr unsigned UNPACK_N_VALUES  = decltype(unpack_n_values)::value;
		galp::kernels::device::decompress_dispatch_mixed<UNPACK_N_VECTORS, UNPACK_N_VALUES, WRITE_OUT>
		    <<<grid, block, 0, stream>>>(exprs_i8, exprs_i16, slots, n_slots);
		CUDA_SAFE_CALL(cudaGetLastError());
	});
}

template <bool WRITE_OUT>
inline void launch_mixed_dispatch(ExecutionWorkset& workset, const ExecutionConfig& cfg, cudaStream_t stream) {
	const auto* exprs_i8  = workset.buffers.device_batches.template get<int8_t>().d_exprs;
	const auto* exprs_i16 = workset.buffers.device_batches.template get<int16_t>().d_exprs;
	launch_mixed_slots<WRITE_OUT>(exprs_i8, exprs_i16, workset.slots.d, workset.slots.mixed.size(), cfg, stream);
	auto scalar_tail_cfg             = cfg;
	scalar_tail_cfg.unpack_n_vectors = 1;
	launch_mixed_slots<WRITE_OUT>(exprs_i8,
	                               exprs_i16,
	                               workset.slots.d_scalar_tail,
	                               workset.slots.scalar_tail_mixed.size(),
	                               scalar_tail_cfg,
	                               stream);
}

template <LaunchStrategy Strategy, bool WRITE_OUT>
inline void launch_strategy_once(ExecutionWorkset& workset, const ExecutionConfig& cfg, cudaStream_t stream) {
	if constexpr (Strategy == LaunchStrategy::MixedDispatch) {
		launch_mixed_dispatch<WRITE_OUT>(workset, cfg, stream);
	} else {
		launch_typed_batches<WRITE_OUT>(workset, cfg, stream);
	}
}

inline size_t typed_launches_per_sample(const ExecutionWorkset& workset) {
	size_t launches = 0;
	galp::execution::for_each_type(galp::execution::SupportedTypes {}, [&](auto tag) {
		using T = typename decltype(tag)::type;
		auto& d = workset.buffers.device_batches.template get<T>();
		if (d.d_exprs != nullptr && d.d_items != nullptr) {
			++launches;
		}
		if (d.d_exprs != nullptr && d.d_scalar_tail_items != nullptr) {
			++launches;
		}
	});
	return launches;
}

inline size_t typed_total_items_per_sample(const ExecutionWorkset& workset) {
	size_t items = 0;
	galp::execution::for_each_type(galp::execution::SupportedTypes {}, [&](auto tag) {
		using T = typename decltype(tag)::type;
		const auto& d = workset.buffers.device_batches.template get<T>();
		items += d.n_items + d.n_scalar_tail_items;
	});
	return items;
}

inline bool has_any_expr(const ExecutionWorkset& workset) {
	bool has_any = false;
	galp::execution::for_each_type(galp::execution::SupportedTypes {}, [&](auto tag) {
		using T = typename decltype(tag)::type;
		auto& d = workset.buffers.device_batches.template get<T>();
		has_any = has_any || (d.d_exprs != nullptr);
	});
	return has_any;
}

struct AsyncWorksetRun {
	cudaStream_t             stream                  = nullptr;
	galp::memory::CudaEvent* queued                  = nullptr;
	galp::memory::CudaEvent* start                   = nullptr;
	galp::memory::CudaEvent* stop                    = nullptr;
	double                   elapsed_ms              = 0.0;
	double                   pre_kernel_event_ms     = 0.0;
	double                   event_sync_wall_ms      = 0.0;
	double                   timing_event_create_ms  = 0.0;
	double                   warmup_wall_ms          = 0.0;
	bool                     active                  = false;
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
	const bool   has_mixed_items =
	    workset.slots.d != nullptr && !workset.slots.mixed.empty();
	const bool has_mixed_tail_items =
	    workset.slots.d_scalar_tail != nullptr && !workset.slots.scalar_tail_mixed.empty();
	const size_t mixed_launches_per_sample = (has_mixed_items ? 1U : 0U) + (has_mixed_tail_items ? 1U : 0U);

	if (mixed_dispatch && mixed_launches_per_sample == 0) {
		if (out_launches) {
			*out_launches = 0;
		}
		return handle;
	}

	if (out_grid) {
		if (mixed_dispatch) {
			size_t grid = 0;
			if (has_mixed_items) {
				const MixedSlotMapping mapping(workset.slots.mixed.size());
				grid += mapping.n_blocks();
			}
			if (has_mixed_tail_items) {
				const MixedSlotMapping mapping(workset.slots.scalar_tail_mixed.size());
				grid += mapping.n_blocks();
			}
			*out_grid = grid;
		} else {
			*out_grid = (launches_per_sample > 0) ? (total_items / launches_per_sample) : 0;
		}
	}
	if (out_launches) {
		*out_launches = (mixed_dispatch ? mixed_launches_per_sample : launches_per_sample) * static_cast<size_t>(samples);
	}

	if (!workset.transfer.h2d_stream) {
		galp::memory::sync_h2d();
	} else if (!use_async_h2d()) {
		galp::memory::sync_h2d(workset.transfer.h2d_stream.get());
	}
	handle.stream                 = ensure_workset_compute_stream(workset);
	const auto event_create_start = std::chrono::steady_clock::now();
	handle.queued                 = &workset.transfer.timing_queued_event;
	handle.start                  = &workset.transfer.timing_start_event;
	handle.stop                   = &workset.transfer.timing_stop_event;
	handle.queued->create();
	handle.start->create();
	handle.stop->create();
	const auto event_create_end = std::chrono::steady_clock::now();
	handle.timing_event_create_ms =
	    std::chrono::duration<double, std::milli>(event_create_end - event_create_start).count();
	handle.active = true;
	handle.queued->record(handle.stream);

	if (use_async_h2d() && workset.transfer.h2d_stream && workset.transfer.h2d_ready_event) {
		CUDA_SAFE_CALL(cudaStreamWaitEvent(handle.stream, workset.transfer.h2d_ready_event.get(), 0));
	}

	if (warmup) {
		const auto warmup_start = std::chrono::steady_clock::now();
		if (mixed_dispatch) {
			if (cfg.write_out) {
				launch_strategy_once<LaunchStrategy::MixedDispatch, true>(workset, cfg, handle.stream);
			} else {
				launch_strategy_once<LaunchStrategy::MixedDispatch, false>(workset, cfg, handle.stream);
			}
		} else {
			if (cfg.write_out) {
				launch_strategy_once<LaunchStrategy::TypedBatches, true>(workset, cfg, handle.stream);
			} else {
				launch_strategy_once<LaunchStrategy::TypedBatches, false>(workset, cfg, handle.stream);
			}
		}
		CUDA_SAFE_CALL(cudaStreamSynchronize(handle.stream));
		const auto warmup_end = std::chrono::steady_clock::now();
		handle.warmup_wall_ms = std::chrono::duration<double, std::milli>(warmup_end - warmup_start).count();
	}

	handle.start->record(handle.stream);

	for (uint32_t sample = 0; sample < samples; ++sample) {
		if (mixed_dispatch) {
			if (cfg.write_out) {
				launch_strategy_once<LaunchStrategy::MixedDispatch, true>(workset, cfg, handle.stream);
			} else {
				launch_strategy_once<LaunchStrategy::MixedDispatch, false>(workset, cfg, handle.stream);
			}
		} else {
			if (cfg.write_out) {
				launch_strategy_once<LaunchStrategy::TypedBatches, true>(workset, cfg, handle.stream);
			} else {
				launch_strategy_once<LaunchStrategy::TypedBatches, false>(workset, cfg, handle.stream);
			}
		}
	}
	handle.stop->record(handle.stream);

	return handle;
}

inline bool workset_run_is_ready(const AsyncWorksetRun& handle) {
	if (!handle.active || handle.stop == nullptr) {
		return true;
	}
	const cudaError_t status = cudaEventQuery(handle.stop->get());
	if (status == cudaSuccess) {
		return true;
	}
	if (status == cudaErrorNotReady) {
		return false;
	}
	CUDA_SAFE_CALL(status);
	return false;
}

inline void wait_workset_async(AsyncWorksetRun& handle) {
	if (!handle.active) {
		return;
	}
	const auto sync_start = std::chrono::steady_clock::now();
	handle.stop->synchronize();
	const auto sync_end            = std::chrono::steady_clock::now();
	handle.event_sync_wall_ms      = std::chrono::duration<double, std::milli>(sync_end - sync_start).count();
	handle.pre_kernel_event_ms     = static_cast<double>(handle.start->elapsed_since(*handle.queued));
	handle.elapsed_ms              = static_cast<double>(handle.stop->elapsed_since(*handle.start));
	handle.queued = nullptr;
	handle.start  = nullptr;
	handle.stop   = nullptr;
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

} // namespace galp::runtime

#endif // GALP_CUDA_LAUNCH_LAUNCH_CUH
