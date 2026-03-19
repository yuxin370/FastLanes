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
		auto& host_batch   = workset.host_batches.template get<T>();
		auto& device_batch = workset.device_batches.template get<T>();
		if (!device_batch.d_exprs.has_value() || !device_batch.d_items.has_value()) {
			return;
		}
		dispatch::detail::launch_batch_no_sync<T, WRITE_OUT>(
		    host_batch, device_batch.d_exprs->get(), device_batch.d_items->get(), device_batch.n_items, stream);
	});
}

template <bool WRITE_OUT>
inline void launch_mixed_dispatch(ExecutionWorkset& workset, cudaStream_t stream) {
	const auto*            exprs_i8  = workset.device_batches.template get<int8_t>().d_exprs.has_value()
	                                       ? workset.device_batches.template get<int8_t>().d_exprs->get()
	                                       : nullptr;
	const auto*            exprs_i16 = workset.device_batches.template get<int16_t>().d_exprs.has_value()
	                                       ? workset.device_batches.template get<int16_t>().d_exprs->get()
	                                       : nullptr;
	const MixedSlotMapping mapping(workset.mixed_slots.size());
	const dim3             block(MixedSlotMapping::N_THREADS_PER_BLOCK);
	const dim3             grid(mapping.n_blocks());

	kernels::device::decompress_dispatch_mixed<1, 1, WRITE_OUT>
	    <<<grid, block, 0, stream>>>(exprs_i8, exprs_i16, workset.d_slots->get(), workset.mixed_slots.size());
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
		auto& d = workset.device_batches.template get<T>();
		if (d.d_exprs.has_value() && d.d_items.has_value()) {
			++launches;
		}
	});
	return launches;
}

inline size_t typed_total_items_per_sample(const ExecutionWorkset& workset) {
	size_t items = 0;
	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T = typename decltype(tag)::type;
		items += workset.device_batches.template get<T>().n_items;
	});
	return items;
}

inline bool has_any_expr(const ExecutionWorkset& workset) {
	bool has_any = false;
	dispatch::for_each_type(dispatch::SupportedTypes {}, [&](auto tag) {
		using T = typename decltype(tag)::type;
		auto& d = workset.device_batches.template get<T>();
		has_any = has_any || (d.d_exprs.has_value() && d.d_items.has_value());
	});
	return has_any;
}

inline double run_workset(ExecutionWorkset&      workset,
                          const uint32_t         samples,
                          const ExecutionConfig& cfg,
                          size_t*                out_grid     = nullptr,
                          size_t*                out_launches = nullptr,
                          const bool             warmup       = false) {
	if (!has_any_expr(workset)) {
		if (out_launches) {
			*out_launches = 0;
		}
		return 0.0;
	}

	const size_t launches_per_sample = typed_launches_per_sample(workset);
	const size_t total_items         = typed_total_items_per_sample(workset);
	const bool   mixed_dispatch      = uses_mixed_dispatch(cfg.launch_strategy);

	if (mixed_dispatch && (!workset.d_slots.has_value() || workset.mixed_slots.empty())) {
		if (out_launches) {
			*out_launches = 0;
		}
		return 0.0;
	}

	if (out_grid) {
		if (mixed_dispatch) {
			const MixedSlotMapping mapping(workset.mixed_slots.size());
			*out_grid = mapping.n_blocks();
		} else {
			*out_grid = (launches_per_sample > 0) ? (total_items / launches_per_sample) : 0;
		}
	}
	if (out_launches) {
		*out_launches = (mixed_dispatch ? 1 : launches_per_sample) * static_cast<size_t>(samples);
	}

	flsgpu::memory::sync_h2d();

	cudaStream_t stream {};
	CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	cudaEvent_t start {};
	cudaEvent_t stop {};
	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));

	if (warmup) {
		if (mixed_dispatch) {
			if (cfg.write_out) {
				launch_strategy_once<LaunchStrategy::MixedDispatch, true>(workset, stream);
			} else {
				launch_strategy_once<LaunchStrategy::MixedDispatch, false>(workset, stream);
			}
		} else {
			if (cfg.write_out) {
				launch_strategy_once<LaunchStrategy::TypedBatches, true>(workset, stream);
			} else {
				launch_strategy_once<LaunchStrategy::TypedBatches, false>(workset, stream);
			}
		}
		CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
	}

	CUDA_SAFE_CALL(cudaEventRecord(start, stream));

	for (uint32_t sample = 0; sample < samples; ++sample) {
		if (mixed_dispatch) {
			if (cfg.write_out) {
				launch_strategy_once<LaunchStrategy::MixedDispatch, true>(workset, stream);
			} else {
				launch_strategy_once<LaunchStrategy::MixedDispatch, false>(workset, stream);
			}
		} else {
			if (cfg.write_out) {
				launch_strategy_once<LaunchStrategy::TypedBatches, true>(workset, stream);
			} else {
				launch_strategy_once<LaunchStrategy::TypedBatches, false>(workset, stream);
			}
		}
	}

	CUDA_SAFE_CALL(cudaEventRecord(stop, stream));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));

	float ms = 0.0f;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, start, stop));
	CUDA_SAFE_CALL(cudaEventDestroy(start));
	CUDA_SAFE_CALL(cudaEventDestroy(stop));
	CUDA_SAFE_CALL(cudaStreamDestroy(stream));

	return static_cast<double>(ms);
}

} // namespace dispatch::runtime

#endif // ENGINE_EXECUTION_INTERNAL_LAUNCH_CUH
