// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/engine/dispatch.cu
// ────────────────────────────────────────────────────────
#include "engine/dispatch.cuh"
#include "engine/reader.cuh"
#include "flsgpu/fls.cuh"
#include <cstring>
#include <memory>

namespace dispatch {
namespace detail {

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
using BPUnpacker =
    flsgpu::device::BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, flsgpu::device::BPFunctor<T>>;

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
using FFORUnpacker = flsgpu::device::BitUnpackerStatefulBranchless<T,
                                                                   UNPACK_N_VECTORS,
                                                                   UNPACK_N_VALUES,
                                                                   flsgpu::device::FFORFunctor<T, UNPACK_N_VECTORS>>;

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
using DICTUnpacker = flsgpu::device::BitUnpackerStatefulBranchless<T,
                                                                   UNPACK_N_VECTORS,
                                                                   UNPACK_N_VALUES,
                                                                   flsgpu::device::DICTFunctor<T, UNPACK_N_VECTORS>>;

template <typename T, typename IndexT, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
using DICTUnpackerIdx =
    flsgpu::device::BitUnpackerStatefulBranchlessIdx<T,
                                                     IndexT,
                                                     UNPACK_N_VECTORS,
                                                     UNPACK_N_VALUES,
                                                     flsgpu::device::DICTFunctorIdx<T, IndexT, UNPACK_N_VECTORS>>;

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
using DefaultFREQPatcher = flsgpu::device::StatefulFREQExceptionPatcher<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>;

template <typename T, unsigned UNPACK_N_VECTORS, unsigned UNPACK_N_VALUES>
using DefaultCROSSRLEExpander = flsgpu::device::StatefulCROSSRLEExpander<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>;

template <typename ColumnT>
auto decompress_device(const ColumnT& column, const Config& cfg) -> typename column_value_type<ColumnT>::type* {
	using T = typename column_value_type<ColumnT>::type;

	// For now we only expose a single default configuration.
	constexpr unsigned UNPACK_N_VECTORS = 1;
	constexpr unsigned UNPACK_N_VALUES  = 1;
	(void)cfg;

	if constexpr (std::is_same_v<ColumnT, flsgpu::device::BPColumn<T>>) {
		using DecompressorT = flsgpu::device::BPDecompressor<T,
		                                                     UNPACK_N_VECTORS,
		                                                     BPUnpacker<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		                                                     flsgpu::device::BPColumn<T>>;
		return kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, cfg.n_samples);
	} else if constexpr (std::is_same_v<ColumnT, flsgpu::device::CONSTANTColumn<T>>) {
		using DecompressorT = flsgpu::device::
		    CONSTANTDecompressor<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, flsgpu::device::CONSTANTColumn<T>>;
		return kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, cfg.n_samples);
	} else if constexpr (std::is_same_v<ColumnT, flsgpu::device::FFORColumn<T>>) {
		using DecompressorT = flsgpu::device::FFORDecompressor<T,
		                                                       UNPACK_N_VECTORS,
		                                                       FFORUnpacker<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		                                                       flsgpu::device::FFORColumn<T>>;
		return kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, cfg.n_samples);
	} else if constexpr (std::is_same_v<ColumnT, flsgpu::device::DICTFFORColumn<T, uint8_t>>) {
		using IndexT     = uint8_t;
		using ProcessorT = flsgpu::device::DICTFunctorIdx<T, IndexT, UNPACK_N_VECTORS>;
		using DecompressorT =
		    flsgpu::device::DICTDecompressor<T,
		                                     UNPACK_N_VECTORS,
		                                     DICTUnpackerIdx<T, IndexT, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		                                     flsgpu::device::DICTFFORColumn<T, IndexT>,
		                                     ProcessorT>;
		return kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, cfg.n_samples);
	} else if constexpr (std::is_same_v<ColumnT, flsgpu::device::DICTFFORColumn<T, uint16_t>>) {
		using ProcessorT    = flsgpu::device::DICTFunctor<T, UNPACK_N_VECTORS>;
		using DecompressorT = flsgpu::device::DICTDecompressor<T,
		                                                       UNPACK_N_VECTORS,
		                                                       DICTUnpacker<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		                                                       flsgpu::device::DICTFFORColumn<T, uint16_t>,
		                                                       ProcessorT>;
		return kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, cfg.n_samples);
	} else if constexpr (std::is_same_v<ColumnT, flsgpu::device::DICTSLPATCHColumn<T, uint8_t>>) {
		using IndexT     = uint8_t;
		using ProcessorT = flsgpu::device::DICTFunctorIdx<T, IndexT, UNPACK_N_VECTORS>;
		using UnpackerT  = DICTUnpackerIdx<T, IndexT, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
		using PatcherT =
		    flsgpu::device::StatefulSLPATCHDictExceptionPatcher<T, IndexT, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
		using DecompressorT = flsgpu::device::DICTSLPATCHDecompressor<T,
		                                                              UNPACK_N_VECTORS,
		                                                              UNPACK_N_VALUES,
		                                                              UnpackerT,
		                                                              PatcherT,
		                                                              flsgpu::device::DICTSLPATCHColumn<T, IndexT>,
		                                                              ProcessorT>;
		return kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, cfg.n_samples);
	} else if constexpr (std::is_same_v<ColumnT, flsgpu::device::DICTSLPATCHColumn<T, uint16_t>>) {
		using IndexT     = uint16_t;
		using ProcessorT = flsgpu::device::DICTFunctor<T, UNPACK_N_VECTORS>;
		using UnpackerT =
		    flsgpu::device::BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, ProcessorT>;
		using PatcherT =
		    flsgpu::device::StatefulSLPATCHDictExceptionPatcher<T, IndexT, UNPACK_N_VECTORS, UNPACK_N_VALUES>;
		using DecompressorT = flsgpu::device::DICTSLPATCHDecompressor<T,
		                                                              UNPACK_N_VECTORS,
		                                                              UNPACK_N_VALUES,
		                                                              UnpackerT,
		                                                              PatcherT,
		                                                              flsgpu::device::DICTSLPATCHColumn<T, uint16_t>,
		                                                              ProcessorT>;
		return kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, cfg.n_samples);
	} else if constexpr (std::is_same_v<ColumnT, flsgpu::device::FREQColumn<T>>) {
		using DecompressorT = flsgpu::device::FREQDecompressor<T,
		                                                       UNPACK_N_VECTORS,
		                                                       DefaultFREQPatcher<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		                                                       flsgpu::device::FREQColumn<T>>;
		return kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, cfg.n_samples);
	} else if constexpr (std::is_same_v<ColumnT, flsgpu::device::CROSSRLEColumn<T>>) {
		using DecompressorT =
		    flsgpu::device::CROSSRLEDecompressor<T,
		                                         UNPACK_N_VECTORS,
		                                         DefaultCROSSRLEExpander<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		                                         flsgpu::device::CROSSRLEColumn<T>>;
		return kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, cfg.n_samples);
	} else if constexpr (std::is_same_v<ColumnT, flsgpu::device::SLPATCHColumn<T>>) {
		using DecompressorT = flsgpu::device::SLPATCHDecompressor<
		    T,
		    UNPACK_N_VECTORS,
		    flsgpu::device::BitUnpackerStatefulBranchless<T,
		                                                  UNPACK_N_VECTORS,
		                                                  UNPACK_N_VALUES,
		                                                  flsgpu::device::FFORFunctor<T, UNPACK_N_VECTORS>>,
		    flsgpu::device::StatefulSLPATCHExceptionPatcher<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>,
		    flsgpu::device::SLPATCHColumn<T>>;
		return kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, cfg.n_samples);
	} else if constexpr (std::is_same_v<ColumnT, flsgpu::device::RLEColumn<T, uint16_t>>) {
		using UnpackerT =
		    flsgpu::device::BitUnpackerStatefulBranchless<uint16_t,
		                                                  UNPACK_N_VECTORS,
		                                                  UNPACK_N_VALUES,
		                                                  flsgpu::device::FFORFunctor<uint16_t, UNPACK_N_VECTORS>>;
		using DecompressorT = flsgpu::device::RLEDecompressor<T,
		                                                      uint16_t,
		                                                      UNPACK_N_VECTORS,
		                                                      UNPACK_N_VALUES,
		                                                      UnpackerT,
		                                                      flsgpu::device::RLEColumn<T, uint16_t>>;
		return kernels::host::decompress_column<T, UNPACK_N_VECTORS, UNPACK_N_VALUES, DecompressorT, ColumnT>(
		    column, cfg.n_samples);
	} else {
		static_assert(always_false_v<ColumnT>, "Unsupported column type for dispatch");
		return nullptr;
	}
}

template <typename HostColT>
DecompressResult decompress_common(const HostColT& host_col, const Config& cfg) {
	using T = typename host_value_type<HostColT>::type;

	auto  device_col = host_col.copy_to_device();
	auto* out        = detail::decompress_device(device_col, cfg);
	flsgpu::host::free_column(device_col);

	return make_result<T>(out);
}

template <typename HostColT>
DecompressResult decompress_host(const HostColT& host_col, const PlanKind plan, const Config& cfg) {
	using T = typename host_value_type<HostColT>::type;
	static_assert(is_supported_type_v<T>, "dispatch::decompress only supports int8_t and int16_t columns");

	auto fail = []() -> DecompressResult {
		throw std::runtime_error("dispatch plan/column mismatch");
	};

	switch (plan) {
	case PlanKind::UNCOMPRESSED: {
		if constexpr (std::is_same_v<HostColT, flsgpu::host::BPColumn<T>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::CONSTANT: {
		if constexpr (std::is_same_v<HostColT, flsgpu::host::CONSTANTColumn<T>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::FREQUENCY: {
		if constexpr (std::is_same_v<HostColT, flsgpu::host::FREQColumn<T>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::UNFFOR: {
		if constexpr (std::is_same_v<HostColT, flsgpu::host::FFORColumn<T>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::UNFFOR_SLPATCH: {
		if constexpr (std::is_same_v<HostColT, flsgpu::host::SLPATCHColumn<T>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::DICT_FFOR: {
		if constexpr (std::is_same_v<HostColT, flsgpu::host::DICTFFORColumn<T, uint8_t>> ||
		              std::is_same_v<HostColT, flsgpu::host::DICTFFORColumn<T, uint16_t>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::DICT_FFOR_SLPATCH: {
		if constexpr (std::is_same_v<HostColT, flsgpu::host::DICTSLPATCHColumn<T, uint8_t>> ||
		              std::is_same_v<HostColT, flsgpu::host::DICTSLPATCHColumn<T, uint16_t>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::CROSS_RLE: {
		if constexpr (std::is_same_v<HostColT, flsgpu::host::CROSSRLEColumn<T>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	case PlanKind::RLE: {
		if constexpr (std::is_same_v<HostColT, flsgpu::host::RLEColumn<T, uint16_t>>) {
			return decompress_common(host_col, cfg);
		}
		return fail();
	}
	default:
		return fail();
	}
}

template <typename T>
struct DeviceBatch {
	std::unique_ptr<GPUArray<DeviceExpression<T>>> d_exprs;
	std::unique_ptr<GPUArray<WorkItem>>            d_items;

	bool empty() const {
		return !d_exprs || !d_items;
	}
};

template <typename... Ts>
struct DeviceBatchSet {
	std::tuple<DeviceBatch<Ts>...> batches;

	template <typename T>
	DeviceBatch<T>& get() {
		return std::get<DeviceBatch<T>>(batches);
	}
};

template <typename List>
struct DeviceBatchSetFromList;

template <typename... Ts>
struct DeviceBatchSetFromList<dispatch::TypeList<Ts...>> {
	using type = DeviceBatchSet<Ts...>;
};

template <typename T>
void release_batch(Batch<T>& batch) {
	for (auto& expr : batch.device_exprs) {
		free_device_expr(expr);
	}
}

template <typename T>
void launch_batch_no_sync(const Batch<T>& batch, GPUArray<DeviceExpression<T>>& d_exprs, GPUArray<WorkItem>& d_items) {
	if (batch.device_exprs.empty() || batch.work_items.empty()) {
		return;
	}
	constexpr unsigned UNPACK_N_VECTORS = 1;
	constexpr unsigned UNPACK_N_VALUES  = 1;
	const int          threads          = utils::get_n_lanes<T>();
	const dim3         block(static_cast<unsigned>(threads));
	const dim3         grid(static_cast<unsigned>(batch.work_items.size()));

	kernels::device::decompress_rowgroup<T, UNPACK_N_VECTORS, UNPACK_N_VALUES>
	    <<<grid, block>>>(d_exprs.get(), d_items.get(), batch.work_items.size());
	CUDA_SAFE_CALL(cudaGetLastError());
}

} // namespace detail

namespace {

inline size_t resolve_alias(const std::vector<expr::Expression>& expressions, size_t idx) {
	size_t cur = idx;
	for (;;) {
		const auto* col = expressions[cur].column;
		if (!col || !col->alias_of.has_value()) {
			return cur;
		}
		cur = *col->alias_of;
	}
}

inline DecompressResult clone_result(const DecompressResult& src, const size_t n_values) {
	return std::visit(
	    [&](auto&& ptr) -> DecompressResult {
		    using T  = std::remove_pointer_t<decltype(ptr.get())>;
		    auto out = std::make_unique<T[]>(n_values);
		    std::memcpy(out.get(), ptr.get(), n_values * sizeof(T));
		    return DecompressResult {std::move(out)};
	    },
	    src);
}

} // namespace

DecompressResult decompress(const expr::Expression& expression, const Config& cfg) {
	auto& col = *expression.column;
	return std::visit(
	    [&](auto&& host_col) -> DecompressResult {
		    using HostColT = std::decay_t<decltype(host_col)>;
		    const auto plan =
		        col.skip_decompress ? detail::plan_for_host_col<HostColT>() : plan_for_ops(expression.ops);
		    return detail::decompress_host(host_col, plan, cfg);
	    },
	    col.host);
}

RowgroupDecompressResult decompress_rowgroup(const std::vector<expr::Expression>& expressions, const Config& cfg) {
	(void)cfg;
	RowgroupDecompressResult result;
	result.columns.resize(expressions.size());

	typename detail::BatchSetFromList<SupportedTypes>::type batches;

	for (size_t i = 0; i < expressions.size(); ++i) {
		const auto& expr = expressions[i];
		if (!expr.column || expr.column->skip_decompress) {
			continue;
		}
		const auto plan = plan_for_ops(expr.ops);

		std::visit(
		    [&](auto&& host_col) {
			    using HostColT = std::decay_t<decltype(host_col)>;
			    using T        = typename host_value_type<HostColT>::type;

			    static_assert(is_supported_type_v<T>, "dispatch rowgroup only supports int8_t/int16_t");
			    detail::add_expression_to_batch<T>(i, host_col, plan, batches.template get<T>());
		    },
		    expr.column->host);
	}

	dispatch::for_each_type(SupportedTypes {}, [&](auto tag) {
		using T     = typename decltype(tag)::type;
		auto& batch = batches.template get<T>();
		detail::launch_batch(batch);
		detail::finalize_batch(batch, result);
	});

	for (size_t i = 0; i < expressions.size(); ++i) {
		const auto* col = expressions[i].column;
		if (!col || !col->alias_of.has_value()) {
			continue;
		}
		const size_t src_idx = resolve_alias(expressions, i);
		if (src_idx >= result.columns.size() || !result.columns[src_idx].has_value()) {
			throw std::runtime_error("EXP_EQUAL: source column not decompressed");
		}
		if (result.columns[i].has_value()) {
			continue;
		}
		const size_t n_values =
		    std::visit([](auto&& host_col) -> size_t { return host_col.get_n_values(); }, col->host);
		result.columns[i] = clone_result(*result.columns[src_idx], n_values);
	}

	return result;
}

BenchmarkResult benchmark_rowgroup(const std::vector<expr::Expression>& expressions, const Config& cfg) {
	BenchmarkResult result {};
	result.n_samples = cfg.n_samples;

	typename detail::BatchSetFromList<SupportedTypes>::type       batches;
	typename detail::DeviceBatchSetFromList<SupportedTypes>::type device_batches;

	for (size_t i = 0; i < expressions.size(); ++i) {
		const auto& expr = expressions[i];
		if (!expr.column || expr.column->skip_decompress) {
			continue;
		}
		const auto plan = plan_for_ops(expr.ops);
		++result.n_expressions;

		std::visit(
		    [&](auto&& host_col) {
			    using HostColT = std::decay_t<decltype(host_col)>;
			    using T        = typename host_value_type<HostColT>::type;

			    static_assert(is_supported_type_v<T>, "dispatch rowgroup only supports int8_t/int16_t");
			    result.total_bytes += host_col.get_n_values() * sizeof(T);
			    detail::add_expression_to_batch<T>(i, host_col, plan, batches.template get<T>());
		    },
		    expr.column->host);
	}

	dispatch::for_each_type(SupportedTypes {}, [&](auto tag) {
		using T     = typename decltype(tag)::type;
		auto& batch = batches.template get<T>();
		result.n_work_items += batch.work_items.size();
		if (batch.device_exprs.empty() || batch.work_items.empty()) {
			return;
		}
		auto& dev = device_batches.template get<T>();
		dev.d_exprs =
		    std::make_unique<GPUArray<DeviceExpression<T>>>(batch.device_exprs.size(), batch.device_exprs.data());
		dev.d_items = std::make_unique<GPUArray<WorkItem>>(batch.work_items.size(), batch.work_items.data());
	});

	struct EventPair {
		cudaEvent_t start {};
		cudaEvent_t stop {};
	};

	EventPair ev;
	CUDA_SAFE_CALL(cudaEventCreate(&ev.start));
	CUDA_SAFE_CALL(cudaEventCreate(&ev.stop));

	CUDA_SAFE_CALL(cudaEventRecord(ev.start, 0));

	for (uint32_t sample = 0; sample < cfg.n_samples; ++sample) {
		dispatch::for_each_type(SupportedTypes {}, [&](auto tag) {
			using T     = typename decltype(tag)::type;
			auto& batch = batches.template get<T>();
			auto& dev   = device_batches.template get<T>();
			if (dev.empty()) {
				return;
			}
			detail::launch_batch_no_sync<T>(batch, *dev.d_exprs, *dev.d_items);
		});
	}

	CUDA_SAFE_CALL(cudaEventRecord(ev.stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(ev.stop));

	float ms = 0.0f;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, ev.start, ev.stop));
	CUDA_SAFE_CALL(cudaEventDestroy(ev.start));
	CUDA_SAFE_CALL(cudaEventDestroy(ev.stop));

	result.total_ms = static_cast<double>(ms);
	result.avg_us   = (cfg.n_samples > 0) ? (result.total_ms * 1000.0 / static_cast<double>(cfg.n_samples)) : 0.0;

	dispatch::for_each_type(SupportedTypes {}, [&](auto tag) {
		using T     = typename decltype(tag)::type;
		auto& batch = batches.template get<T>();
		detail::release_batch(batch);
	});

	return result;
}

template DecompressResult detail::decompress_host(const flsgpu::host::BPColumn<int8_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::FFORColumn<int8_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::DICTFFORColumn<int8_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::DICTSLPATCHColumn<int8_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::CONSTANTColumn<int8_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::FREQColumn<int8_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::SLPATCHColumn<int8_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::CROSSRLEColumn<int8_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::RLEColumn<int8_t, uint16_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::BPColumn<int16_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::FFORColumn<int16_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::DICTFFORColumn<int16_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::DICTFFORColumn<int16_t, uint8_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::DICTSLPATCHColumn<int16_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::DICTSLPATCHColumn<int16_t, uint8_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::SLPATCHColumn<int16_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::FREQColumn<int16_t>&, const PlanKind, const Config&);
template DecompressResult
detail::decompress_host(const flsgpu::host::RLEColumn<int16_t, uint16_t>&, const PlanKind, const Config&);

} // namespace dispatch
