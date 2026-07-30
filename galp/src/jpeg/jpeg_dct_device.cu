#include "codecs/consts.cuh"
#include "galp/jpeg_dct_diagnostics.hpp"
#include "cuda/cuda_macros.cuh"
#include "cuda/launch/launch.cuh"
#include "cuda/memory/cuda_raii.cuh"
#include "cuda/memory/device_pool.cuh"
#include "cuda/memory/gpu_array.cuh"
#include "engine/materialization/metadata.cuh"
#include "engine/materialization/selected_vector_compactor.hpp"
#include "engine/operators/rowgroup.cuh"
#include "engine/pipeline/rowgroup_prefetch_queue.cuh"
#include "engine/workset/append.cuh"
#include "engine/workset/upload.cuh"
#include "format/reader.cuh"
#include "jpeg/jpeg_dct_cuda_internal.cuh"
#include "jpeg/jpeg_dct_kernel_launch.cuh"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <functional>
#include <list>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

namespace galp::jpeg {

struct JpegDctDeviceBatch::Impl {
	JpegDctDeviceLayout                        layout = JpegDctDeviceLayout::kImageMajorComponentBlockCoeff;
	int                                        cuda_device = -1;
	std::optional<GPUArray<int16_t>>           coefficients;
	std::optional<GPUArray<int16_t>>           y_coefficients;
	std::optional<GPUArray<int16_t>>           cbcr_coefficients;
	std::optional<GPUArray<float>>             y_accum;
	std::optional<GPUArray<float>>             cbcr_accum;
	std::optional<GPUArray<uint16_t>>          fixed_quant_tables;
	std::optional<GPUArray<float>>             fixed_resize_weight_matrices;
	size_t                                     coefficient_count      = 0;
	size_t                                     y_coefficient_count    = 0;
	size_t                                     cbcr_coefficient_count = 0;
	size_t                                     coefficients_per_block = 64;
	JpegDctGridOutputDataType                  grid_output_data_type  = JpegDctGridOutputDataType::kInt16;
	JpegDctYcbcrDctGridShape            ycbcr_dct_grid_shape {};
	std::vector<JpegDctDeviceImageLayout>      image_layouts;
	std::vector<JpegDctDeviceBlockMetadata>    block_metadata;
	std::vector<JpegDctDeviceRowgroupMetadata> rowgroups;
	std::vector<uint8_t>                       selected_coefficients;
	JpegDctDeviceCacheStats                    cache_stats;
	JpegDctDeviceExecutionStats                execution_stats;
	galp::memory::CudaEvent                    fixed_grid_round_start_event;
	galp::memory::CudaEvent                    completion_event;
	bool                                       completion_synchronized           = false;
	bool                                       fixed_grid_round_timing_finalized = false;
	std::exception_ptr                         completion_error;
	bool                                       completion_error_observed = false;
	bool                                       completion_error_reported = false;

	void finalize_fixed_grid_round_timing() {
		if (fixed_grid_round_timing_finalized || !fixed_grid_round_start_event || !completion_event) {
			return;
		}
		execution_stats.fixed_grid_round_ms +=
		    static_cast<double>(completion_event.elapsed_since(fixed_grid_round_start_event));
		fixed_grid_round_timing_finalized = true;
	}

	void synchronize_completion() {
		if (completion_error) {
			completion_error_observed = true;
			std::rethrow_exception(completion_error);
		}
		try {
			if (completion_event && !completion_synchronized) {
				completion_event.synchronize();
				completion_synchronized = true;
			}
			finalize_fixed_grid_round_timing();
		} catch (...) {
			completion_error = std::current_exception();
			completion_error_observed = true;
			throw;
		}
	}

	void synchronize_completion_noexcept() noexcept {
		if (completion_error_reported || completion_error_observed) {
			return;
		}
		try {
			synchronize_completion();
		} catch (const std::exception& e) {
			std::fprintf(stderr, "JpegDctDeviceBatch completion wait failed: %s\n", e.what());
			completion_error_reported = true;
		} catch (...) {
			std::fprintf(stderr, "JpegDctDeviceBatch completion wait failed with an unknown exception\n");
			completion_error_reported = true;
		}
	}
};

JpegDctDeviceBatch detail::make_failed_device_batch_for_testing(std::string message) {
	auto impl              = std::make_unique<JpegDctDeviceBatch::Impl>();
	impl->completion_error = std::make_exception_ptr(std::runtime_error(std::move(message)));
	return JpegDctDeviceBatch(std::move(impl));
}

JpegDctDeviceBatch::JpegDctDeviceBatch() noexcept = default;

JpegDctDeviceBatch::JpegDctDeviceBatch(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
}

JpegDctDeviceBatch::~JpegDctDeviceBatch() {
	if (impl_) {
		impl_->synchronize_completion_noexcept();
	}
}

JpegDctDeviceBatch::JpegDctDeviceBatch(JpegDctDeviceBatch&&) noexcept = default;

JpegDctDeviceBatch& JpegDctDeviceBatch::operator=(JpegDctDeviceBatch&& other) noexcept {
	if (this != &other) {
		if (impl_) {
			impl_->synchronize_completion_noexcept();
		}
		impl_ = std::move(other.impl_);
	}
	return *this;
}

const int16_t* JpegDctDeviceBatch::device_coefficients() const {
	if (impl_) {
		impl_->synchronize_completion();
	}
	return impl_ && impl_->coefficients.has_value() ? const_cast<GPUArray<int16_t>&>(*impl_->coefficients).get()
	                                                : nullptr;
}

const int16_t* JpegDctDeviceBatch::y_coefficients() const {
	if (impl_) {
		impl_->synchronize_completion();
	}
	return impl_ && impl_->y_coefficients.has_value() ? const_cast<GPUArray<int16_t>&>(*impl_->y_coefficients).get()
	                                                  : nullptr;
}

const int16_t* JpegDctDeviceBatch::cbcr_coefficients() const {
	if (impl_) {
		impl_->synchronize_completion();
	}
	return impl_ && impl_->cbcr_coefficients.has_value() ? const_cast<GPUArray<int16_t>&>(*impl_->cbcr_coefficients).get()
	                                                     : nullptr;
}

const int16_t* JpegDctDeviceBatch::device_coefficients_async() const noexcept {
	return impl_ && impl_->coefficients.has_value() ? const_cast<GPUArray<int16_t>&>(*impl_->coefficients).get()
	                                                : nullptr;
}

const int16_t* JpegDctDeviceBatch::y_coefficients_async() const noexcept {
	return impl_ && impl_->y_coefficients.has_value() ? const_cast<GPUArray<int16_t>&>(*impl_->y_coefficients).get()
	                                                  : nullptr;
}

const int16_t* JpegDctDeviceBatch::cbcr_coefficients_async() const noexcept {
	return impl_ && impl_->cbcr_coefficients.has_value() ? const_cast<GPUArray<int16_t>&>(*impl_->cbcr_coefficients).get()
	                                                     : nullptr;
}

const float* JpegDctDeviceBatch::y_float_coefficients() const {
	if (impl_) {
		impl_->synchronize_completion();
	}
	return impl_ && impl_->grid_output_data_type == JpegDctGridOutputDataType::kFloat32 && impl_->y_accum.has_value()
	           ? const_cast<GPUArray<float>&>(*impl_->y_accum).get()
	           : nullptr;
}

const float* JpegDctDeviceBatch::cbcr_float_coefficients() const {
	if (impl_) {
		impl_->synchronize_completion();
	}
	return impl_ && impl_->grid_output_data_type == JpegDctGridOutputDataType::kFloat32 && impl_->cbcr_accum.has_value()
	           ? const_cast<GPUArray<float>&>(*impl_->cbcr_accum).get()
	           : nullptr;
}

const float* JpegDctDeviceBatch::y_float_coefficients_async() const noexcept {
	return impl_ && impl_->grid_output_data_type == JpegDctGridOutputDataType::kFloat32 && impl_->y_accum.has_value()
	           ? const_cast<GPUArray<float>&>(*impl_->y_accum).get()
	           : nullptr;
}

const float* JpegDctDeviceBatch::cbcr_float_coefficients_async() const noexcept {
	return impl_ && impl_->grid_output_data_type == JpegDctGridOutputDataType::kFloat32 && impl_->cbcr_accum.has_value()
	           ? const_cast<GPUArray<float>&>(*impl_->cbcr_accum).get()
	           : nullptr;
}

JpegDctGridOutputDataType JpegDctDeviceBatch::grid_output_data_type() const noexcept {
	return impl_ ? impl_->grid_output_data_type : JpegDctGridOutputDataType::kInt16;
}

void JpegDctDeviceBatch::synchronize() const {
	if (impl_) {
		impl_->synchronize_completion();
	}
}

size_t JpegDctDeviceBatch::coefficient_count() const noexcept {
	return impl_ ? impl_->coefficient_count : 0;
}

size_t JpegDctDeviceBatch::coefficient_bytes() const noexcept {
	return coefficient_count() * sizeof(int16_t);
}

size_t JpegDctDeviceBatch::y_coefficient_count() const noexcept {
	return impl_ ? impl_->y_coefficient_count : 0;
}

size_t JpegDctDeviceBatch::cbcr_coefficient_count() const noexcept {
	return impl_ ? impl_->cbcr_coefficient_count : 0;
}

size_t JpegDctDeviceBatch::coefficients_per_block() const noexcept {
	return impl_ ? impl_->coefficients_per_block : 64;
}

size_t JpegDctDeviceBatch::block_count() const noexcept {
	if (!impl_) {
		return 0;
	}
	if (!impl_->block_metadata.empty() || impl_->image_layouts.empty()) {
		return impl_->block_metadata.size();
	}
	const auto& last = impl_->image_layouts.back();
	return static_cast<size_t>(last.block_offset) + last.block_count;
}

size_t JpegDctDeviceBatch::image_count() const noexcept {
	return impl_ ? impl_->image_layouts.size() : 0;
}

size_t JpegDctDeviceBatch::rowgroup_count() const noexcept {
	return impl_ ? impl_->rowgroups.size() : 0;
}

int JpegDctDeviceBatch::cuda_device() const noexcept {
	return impl_ ? impl_->cuda_device : -1;
}

JpegDctDeviceCacheStats JpegDctDeviceBatch::cache_stats() const noexcept {
	return cache_stats_ref();
}

JpegDctDeviceExecutionStats JpegDctDeviceBatch::execution_stats() const {
	if (impl_) {
		// GPU stage durations are only available once their timing events complete.
		// Keep submission asynchronous and pay this wait only when callers request
		// the complete statistics snapshot.
		impl_->synchronize_completion();
	}
	return execution_stats_ref();
}

const JpegDctDeviceCacheStats& JpegDctDeviceBatch::cache_stats_ref() const noexcept {
	static const JpegDctDeviceCacheStats empty;
	return impl_ ? impl_->cache_stats : empty;
}

const JpegDctDeviceExecutionStats& JpegDctDeviceBatch::execution_stats_ref() const noexcept {
	static const JpegDctDeviceExecutionStats empty;
	return impl_ ? impl_->execution_stats : empty;
}

JpegDctDeviceLayout JpegDctDeviceBatch::layout() const noexcept {
	return impl_ ? impl_->layout : JpegDctDeviceLayout::kImageMajorComponentBlockCoeff;
}

void* JpegDctDeviceBatch::cuda_completion_event() const noexcept {
	return impl_ && impl_->completion_event ? impl_->completion_event.get() : nullptr;
}

const std::vector<JpegDctDeviceImageLayout>& JpegDctDeviceBatch::image_layouts() const noexcept {
	static const std::vector<JpegDctDeviceImageLayout> empty;
	return impl_ ? impl_->image_layouts : empty;
}

const std::vector<JpegDctDeviceBlockMetadata>& JpegDctDeviceBatch::block_metadata() const noexcept {
	static const std::vector<JpegDctDeviceBlockMetadata> empty;
	return impl_ ? impl_->block_metadata : empty;
}

const std::vector<JpegDctDeviceRowgroupMetadata>& JpegDctDeviceBatch::rowgroups() const noexcept {
	static const std::vector<JpegDctDeviceRowgroupMetadata> empty;
	return impl_ ? impl_->rowgroups : empty;
}

const std::vector<uint8_t>& JpegDctDeviceBatch::selected_coefficients() const noexcept {
	static const std::vector<uint8_t> empty;
	return impl_ ? impl_->selected_coefficients : empty;
}

JpegDctYcbcrDctGridShape JpegDctDeviceBatch::ycbcr_dct_grid_shape() const noexcept {
	return impl_ ? impl_->ycbcr_dct_grid_shape : JpegDctYcbcrDctGridShape {};
}

} // namespace galp::jpeg

namespace galp::jpeg::detail {

struct JpegDctStagedRowgroupRead {
	galp::format::Rowgroup          rowgroup;
	galp::format::ZeroCopyReadTiming io_timing;
	std::shared_ptr<galp::format::FlsReader> reader;
	std::atomic<bool>                consumed {false};
};

class JpegDctHostIoWorkerPool {
public:
	JpegDctHostIoWorkerPool() = default;
	JpegDctHostIoWorkerPool(const JpegDctHostIoWorkerPool&)            = delete;
	JpegDctHostIoWorkerPool& operator=(const JpegDctHostIoWorkerPool&) = delete;

	~JpegDctHostIoWorkerPool() {
		{
			std::lock_guard lock(mutex_);
			stopping_ = true;
		}
		work_ready_.notify_all();
		for (auto& worker : workers_) {
			if (worker.joinable()) {
				worker.join();
			}
		}
	}

	void run(const size_t worker_count, std::function<void()> worker_body) {
		if (worker_count == 0U) {
			return;
		}
		ensure_worker_count(worker_count);

		struct Completion {
			std::mutex              mutex;
			std::condition_variable ready;
			size_t                  pending = 0U;
			std::exception_ptr      error;
		};
		auto completion = std::make_shared<Completion>();
		std::exception_ptr submit_error;
		{
			std::lock_guard lock(mutex_);
			try {
				for (size_t worker = 0; worker < worker_count; ++worker) {
					jobs_.emplace_back([completion, worker_body]() mutable {
						try {
							worker_body();
						} catch (...) {
							std::lock_guard completion_lock(completion->mutex);
							if (!completion->error) {
								completion->error = std::current_exception();
							}
						}
						{
							std::lock_guard completion_lock(completion->mutex);
							--completion->pending;
						}
						completion->ready.notify_one();
					});
					++completion->pending;
				}
			} catch (...) {
				submit_error = std::current_exception();
			}
		}
		work_ready_.notify_all();
		{
			std::unique_lock completion_lock(completion->mutex);
			completion->ready.wait(completion_lock, [&] { return completion->pending == 0U; });
		}
		if (submit_error) {
			std::rethrow_exception(submit_error);
		}
		if (completion->error) {
			std::rethrow_exception(completion->error);
		}
	}

private:
	void ensure_worker_count(const size_t count) {
		std::lock_guard lock(mutex_);
		while (workers_.size() < count) {
			workers_.emplace_back([this] { worker_loop(); });
		}
	}

	void worker_loop() noexcept {
		for (;;) {
			std::function<void()> job;
			{
				std::unique_lock lock(mutex_);
				work_ready_.wait(lock, [&] { return stopping_ || !jobs_.empty(); });
				if (stopping_ && jobs_.empty()) {
					return;
				}
				job = std::move(jobs_.front());
				jobs_.pop_front();
			}
			try {
				job();
			} catch (...) {
				// Every submitted job records its exception in its completion.
				// Keep the persistent worker alive if that wrapper ever regresses.
			}
		}
	}

	std::mutex                       mutex_;
	std::condition_variable          work_ready_;
	std::deque<std::function<void()>> jobs_;
	std::vector<std::thread>          workers_;
	bool                              stopping_ = false;
};

struct JpegDctHostIoContext {
	struct CachedReader {
		std::shared_ptr<galp::format::FlsReader> reader;
		std::list<std::string>::iterator         lru_position;
	};
	struct SparsePlanKey {
		std::string           path;
		const galp::format::FlsReader* reader_identity = nullptr;
		size_t                rowgroup_index = 0U;
		bool                  packed_device_scatter = false;
		bool                  envelope_policy = false;
		std::vector<uint32_t> physical_vectors;

		bool operator==(const SparsePlanKey& other) const noexcept {
			return reader_identity == other.reader_identity && rowgroup_index == other.rowgroup_index &&
			       packed_device_scatter == other.packed_device_scatter &&
			       envelope_policy == other.envelope_policy && path == other.path &&
			       physical_vectors == other.physical_vectors;
		}
	};
	struct SparsePlanKeyHash {
		size_t operator()(const SparsePlanKey& key) const noexcept {
			size_t hash = std::hash<std::string> {}(key.path);
			const auto combine = [&hash](const size_t value) {
				hash ^= value + size_t {0x9e3779b9U} + (hash << 6U) + (hash >> 2U);
			};
			combine(std::hash<const galp::format::FlsReader*> {}(key.reader_identity));
			combine(std::hash<size_t> {}(key.rowgroup_index));
			combine(std::hash<bool> {}(key.packed_device_scatter));
			combine(std::hash<bool> {}(key.envelope_policy));
			for (const auto vector : key.physical_vectors) {
				combine(std::hash<uint32_t> {}(vector));
			}
			return hash;
		}
	};
	struct CachedSparsePlan {
		std::shared_ptr<const galp::format::SparseVectorReadPlan> plan;
		std::shared_ptr<galp::format::FlsReader>    reader;
		std::list<SparsePlanKey>::iterator          lru_position;
	};
	std::shared_ptr<galp::format::FlsReader> reader(const std::filesystem::path& path,
	                                                const bool enable_sparse_vector_reads = true) {
		const auto key = path.lexically_normal().string() +
		                 (enable_sparse_vector_reads ? "#sparse-vector" : "#rowgroup-only");
		{
			std::lock_guard<std::mutex> lock(mutex);
			const auto found = readers.find(key);
			if (found != readers.end()) {
				lru.splice(lru.begin(), lru, found->second.lru_position);
				return found->second.reader;
			}
		}

		galp::format::FlsReaderOptions reader_options;
		reader_options.load_column_names = false;
		reader_options.enable_sparse_vector_reads = enable_sparse_vector_reads;
		reader_options.build_shared_zero_copy_schema_plan = enable_sparse_vector_reads;
		auto opened = std::make_shared<galp::format::FlsReader>(path, reader_options);
		std::lock_guard<std::mutex> lock(mutex);
		if (const auto concurrent = readers.find(key); concurrent != readers.end()) {
			lru.splice(lru.begin(), lru, concurrent->second.lru_position);
			return concurrent->second.reader;
		}
		while (readers.size() >= kCapacity && !lru.empty()) {
			readers.erase(lru.back());
			lru.pop_back();
		}
		lru.push_front(key);
		readers.emplace(key, CachedReader {opened, lru.begin()});
		return opened;
	}

	std::shared_ptr<const galp::format::SparseVectorReadPlan> sparse_plan(
	    const std::filesystem::path& path,
	    const std::shared_ptr<galp::format::FlsReader>& reader,
	    const size_t rowgroup_index,
	    std::vector<uint32_t> physical_vectors,
	    const bool packed_device_scatter) {
		const char* const policy = std::getenv("GALP_VECTOR_BUNDLE_READ_POLICY");
		SparsePlanKey key {path.lexically_normal().string(),
		                   reader.get(),
		                   rowgroup_index,
		                   packed_device_scatter,
		                   !packed_device_scatter && policy != nullptr &&
		                       std::strcmp(policy, "envelope") == 0,
		                   std::move(physical_vectors)};
		{
			std::lock_guard<std::mutex> lock(mutex);
			if (const auto found = sparse_plans.find(key); found != sparse_plans.end()) {
				sparse_plan_lru.splice(sparse_plan_lru.begin(), sparse_plan_lru, found->second.lru_position);
				return found->second.plan;
			}
		}

		auto compiled = std::make_shared<const galp::format::SparseVectorReadPlan>(
		    reader->compile_sparse_vector_read_plan(rowgroup_index, key.physical_vectors, packed_device_scatter));
		std::lock_guard<std::mutex> lock(mutex);
		if (const auto concurrent = sparse_plans.find(key); concurrent != sparse_plans.end()) {
			sparse_plan_lru.splice(
			    sparse_plan_lru.begin(), sparse_plan_lru, concurrent->second.lru_position);
			return concurrent->second.plan;
		}
		while (sparse_plans.size() >= kSparsePlanCapacity && !sparse_plan_lru.empty()) {
			sparse_plans.erase(sparse_plan_lru.back());
			sparse_plan_lru.pop_back();
		}
		sparse_plan_lru.push_front(std::move(key));
		auto [inserted, unused] = sparse_plans.emplace(
		    sparse_plan_lru.front(), CachedSparsePlan {compiled, reader, sparse_plan_lru.begin()});
		(void)unused;
		return inserted->second.plan;
	}

	static constexpr size_t kCapacity = 256U;
	static constexpr size_t kSparsePlanCapacity = 65536U;
	std::mutex mutex;
	std::list<std::string> lru;
	std::unordered_map<std::string, CachedReader> readers;
	std::list<SparsePlanKey> sparse_plan_lru;
	std::unordered_map<SparsePlanKey, CachedSparsePlan, SparsePlanKeyHash> sparse_plans;
	// Declared last so workers stop before reader/plan caches are destroyed.
	JpegDctHostIoWorkerPool worker_pool;
};

using Clock                                       = std::chrono::steady_clock;
constexpr size_t kMinJpegDctScratchBufferCapacity = 1024;

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
	return std::chrono::duration<double, std::milli>(end - start).count();
}

size_t resolve_physical_coefficient_column(const galp::execution::Rowgroup& rowgroup,
                                           const size_t                     logical_coeff_idx) {
	if (logical_coeff_idx >= kJpegDctCoefficientCount || logical_coeff_idx >= rowgroup.columns.size()) {
		throw std::out_of_range("JPEG DCT selected coefficient is outside the rowgroup column range");
	}

	std::array<bool, kJpegDctCoefficientCount> visited {};
	size_t                                     current = logical_coeff_idx;
	for (;;) {
		if (current >= kJpegDctCoefficientCount || current >= rowgroup.columns.size()) {
			throw std::out_of_range("JPEG DCT coefficient alias target is outside the rowgroup column range");
		}
		if (visited[current]) {
			throw std::runtime_error("JPEG DCT coefficient alias cycle detected");
		}
		visited[current] = true;
		const auto& column = rowgroup.columns[current];
		if (column.alias_of.has_value()) {
			current = *column.alias_of;
			continue;
		}
		if (column.skip_decompress) {
			throw std::runtime_error("JPEG DCT coefficient is marked skip_decompress without an alias source");
		}
		return current;
	}
}

JpegDctDeviceResolvedProjection
resolve_projection_physical_columns(const galp::execution::Rowgroup&               rowgroup,
                                    const std::vector<JpegDctDeviceProjectionItem>& projection_items) {
	JpegDctDeviceResolvedProjection resolved;
	resolved.items.reserve(projection_items.size());
	std::array<bool, kJpegDctCoefficientCount> active {};
	for (auto item : projection_items) {
		const auto physical = resolve_physical_coefficient_column(rowgroup, item.logical_coefficient_id);
		item.physical_coefficient_column_id = static_cast<uint8_t>(physical);
		active[physical]                    = true;
		resolved.items.push_back(item);
	}
	for (size_t coeff_idx = 0; coeff_idx < active.size(); ++coeff_idx) {
		if (active[coeff_idx]) {
			resolved.active_physical_coefficients.push_back(static_cast<uint8_t>(coeff_idx));
		}
	}
	return resolved;
}

 struct BoundCoeffColumns {
	std::array<const int8_t*, 64>     columns_i8 {};
	std::array<const int16_t*, 64>    columns_i16 {};
	std::array<DeviceCoeffSource, 64> column_sources {};
};

   struct DecodedRowgroupWork {
	uint32_t                                                 shard_id       = 0;
	uint32_t                                                 rowgroup_index = 0;
	galp::execution::Rowgroup                                rowgroup {};
	std::vector<uint32_t>                                    owned_selected_vectors;
	const std::vector<uint32_t>*                             selected_vectors = nullptr;
	std::vector<uint32_t>                                    owned_decode_vectors;
	const std::vector<uint32_t>*                             decode_vectors = nullptr;
	std::vector<JpegDctDeviceGatherItem>                     owned_gather_items;
	const std::vector<JpegDctDeviceGatherItem>*              gather_items = nullptr;
	std::vector<JpegDctDeviceProjectionItem>                 owned_projection_items;
	const std::vector<JpegDctDeviceProjectionItem>*          projection_items = nullptr;
	std::vector<JpegDctDeviceFixedTransformItem>             owned_fixed_transform_items;
	const std::vector<JpegDctDeviceFixedTransformItem>*      fixed_transform_items = nullptr;
	const std::vector<JpegDctDevicePlanlessImageDescriptor>* planless_images       = nullptr;
	std::vector<uint8_t>                                     active_physical_coefficients;
	size_t                                                   expr_index_base         = 0;
	size_t                                                   logical_rowgroup_n_vecs = 0;
	uint32_t                                                 decode_unpack_n_vectors = kJpegDctDeviceUnpackNVectors;
	bool                                                     decodes_full_rowgroup   = false;
	bool                                                     owns_rowgroup           = false;
	JpegDctDeviceDecodedRowgroupCacheKey                     cache_key {};
	std::unique_ptr<JpegDctDeviceDecodedRowgroupCacheEntry>  cache_entry;

	DecodedRowgroupWork()                                      = default;
	DecodedRowgroupWork(const DecodedRowgroupWork&)            = delete;
	DecodedRowgroupWork& operator=(const DecodedRowgroupWork&) = delete;

	DecodedRowgroupWork(DecodedRowgroupWork&& other) noexcept
	    : shard_id(other.shard_id)
	    , rowgroup_index(other.rowgroup_index)
	    , rowgroup(std::move(other.rowgroup))
	    , owned_selected_vectors(std::move(other.owned_selected_vectors))
	    , selected_vectors(other.selected_vectors == &other.owned_selected_vectors ? &owned_selected_vectors
	                                                                               : other.selected_vectors)
	    , owned_decode_vectors(std::move(other.owned_decode_vectors))
	    , decode_vectors(other.decode_vectors == &other.owned_decode_vectors ? &owned_decode_vectors
	                                                                         : other.decode_vectors)
	    , owned_gather_items(std::move(other.owned_gather_items))
	    , gather_items(other.gather_items == &other.owned_gather_items ? &owned_gather_items : other.gather_items)
	    , owned_projection_items(std::move(other.owned_projection_items))
	    , projection_items(other.projection_items == &other.owned_projection_items ? &owned_projection_items
	                                                                               : other.projection_items)
	    , owned_fixed_transform_items(std::move(other.owned_fixed_transform_items))
	    , fixed_transform_items(other.fixed_transform_items == &other.owned_fixed_transform_items
	                                ? &owned_fixed_transform_items
	                                : other.fixed_transform_items)
	    , planless_images(other.planless_images)
	    , active_physical_coefficients(std::move(other.active_physical_coefficients))
	    , expr_index_base(other.expr_index_base)
	    , logical_rowgroup_n_vecs(other.logical_rowgroup_n_vecs)
	    , decode_unpack_n_vectors(other.decode_unpack_n_vectors)
	    , decodes_full_rowgroup(other.decodes_full_rowgroup)
	    , owns_rowgroup(std::exchange(other.owns_rowgroup, false))
	    , cache_key(other.cache_key)
	    , cache_entry(std::move(other.cache_entry)) {
	}

	DecodedRowgroupWork& operator=(DecodedRowgroupWork&& other) noexcept = delete;

	~DecodedRowgroupWork() {
		reset();
	}

	void reset() {
		if (owns_rowgroup) {
			galp::execution::free_rowgroup(rowgroup);
			owns_rowgroup = false;
		}
	}
};

template <typename T>
struct JpegDctDeviceScratchBuffer {
	T*     data     = nullptr;
	size_t capacity = 0;

	JpegDctDeviceScratchBuffer()                                             = default;
	JpegDctDeviceScratchBuffer(const JpegDctDeviceScratchBuffer&)            = delete;
	JpegDctDeviceScratchBuffer& operator=(const JpegDctDeviceScratchBuffer&) = delete;

	~JpegDctDeviceScratchBuffer() {
		release();
	}

	void release() noexcept {
		try {
			if (data != nullptr) {
				galp::memory::device_free(data);
				data     = nullptr;
				capacity = 0;
			}
		} catch (const std::exception& e) {
			std::fprintf(stderr, "JpegDctDeviceScratchBuffer destructor: %s\n", e.what());
		}
	}

	static size_t growth_capacity(const size_t count) {
		size_t next = kMinJpegDctScratchBufferCapacity;
		while (next < count) {
			if (next > std::numeric_limits<size_t>::max() / 2U) {
				return count;
			}
			next *= 2U;
		}
		return next;
	}

	[[nodiscard]] bool needs_reallocation(const size_t count) const noexcept {
		return count > capacity;
	}

	void upload(const T* host, const size_t count, cudaStream_t stream, JpegDctDeviceExecutionStats& stats) {
		if (count == 0) {
			return;
		}
		if (count > capacity) {
			if (data != nullptr) {
				galp::memory::device_free(data);
			}
			const size_t alloc_count = growth_capacity(count);
			if (alloc_count > std::numeric_limits<size_t>::max() / sizeof(T)) {
				throw std::overflow_error("JPEG DCT scratch buffer allocation size overflow");
			}
			data     = stream != nullptr
			               ? reinterpret_cast<T*>(galp::memory::device_malloc_on_stream(alloc_count * sizeof(T), stream))
			               : reinterpret_cast<T*>(galp::memory::device_malloc(alloc_count * sizeof(T)));
			capacity = alloc_count;
			++stats.scratch_allocation_count;
		}
		galp::memory::device_memcpy_h2d_async(data, host, count * sizeof(T), stream);
		++stats.scratch_upload_count;
	}
};

struct JpegDctDeviceScratch {
	galp::memory::CudaStream                                               cache_hit_stream;
	galp::memory::CudaEvent                                                cached_gather_start;
	galp::memory::CudaEvent                                                cached_gather_done;
	galp::memory::CudaEvent                                                decoded_batch_gather_done;
	galp::memory::CudaStream                                               fixed_grid_round_stream;
	galp::memory::CudaStream                                               transform_stream;
	galp::memory::CudaEvent                                                decode_to_transform_event;
	galp::runtime::ExecutionWorkset                                        decode_workset;
	JpegDctDeviceScratchBuffer<DeviceCoeffBinding>                         column_bindings;
	JpegDctDeviceScratchBuffer<JpegDctDeviceProjectionBatchItem>           batch_projection_items;
	JpegDctDeviceScratchBuffer<JpegDctDeviceFixedTransformBatchItem>       batch_fixed_transform_items;
	JpegDctDeviceScratchBuffer<JpegDctDevicePlanlessImageDescriptor>       planless_image_descriptors;
	JpegDctDeviceScratchBuffer<uint32_t>                                   planless_vector_remap;
	JpegDctDeviceScratchBuffer<uint32_t>                                   fixed_transform_group_offsets;
	JpegDctDeviceScratchBuffer<JpegDctDeviceMaterializeBatchItem>          batch_materialize_items;
	JpegDctDeviceScratchBuffer<JpegDctDeviceCachedGatherBatchItem>         cached_gather_items;
	JpegDctDeviceScratchBuffer<JpegDctDeviceCachedFixedTransformBatchItem> cached_fixed_transform_items;
	JpegDctDeviceScratchBuffer<JpegDctDeviceDecodedGatherBatchItem>        decoded_gather_items;
	std::vector<DeviceCoeffBinding>                                        host_column_bindings;
	std::vector<BoundCoeffColumns>                                         host_bound_sources;
	std::vector<JpegDctDeviceProjectionBatchItem>                          host_projection_items;
	std::vector<JpegDctDeviceFixedTransformBatchItem>                      host_fixed_transform_items;
	std::vector<JpegDctDeviceFixedTransformBatchItem>                      host_ordered_fixed_transform_items;
	std::vector<JpegDctDevicePlanlessImageDescriptor>                      host_planless_image_descriptors;
	std::vector<uint32_t>                                                  host_planless_vector_remap;
	std::vector<uint32_t>                                                  host_workset_fixed_transform_item_order;
	std::vector<uint32_t>                                                  host_workset_fixed_transform_group_offsets;
	std::vector<uint32_t>                                                  host_workset_fixed_transform_item_groups;
	std::vector<uint32_t>                                                  host_workset_fixed_transform_group_cursors;
	std::unordered_map<size_t, uint32_t>                                   host_workset_fixed_transform_group_lookup;
	std::vector<JpegDctDeviceMaterializeBatchItem>                         host_materialize_items;
	std::vector<JpegDctDeviceDecodedGatherBatchItem>                       host_decoded_gather_items;
	std::vector<JpegDctDeviceCachedGatherBatchItem>                        host_cached_gather_items;
	std::vector<JpegDctDeviceCachedFixedTransformBatchItem>                host_cached_fixed_transform_items;
	std::deque<std::vector<JpegDctDeviceCachedGatherBatchItem>>            host_cached_gather_uploads;
	std::deque<std::vector<JpegDctDeviceCachedFixedTransformBatchItem>>    host_cached_fixed_transform_uploads;
	std::vector<DecodedRowgroupWork>                                       host_pending_works;
	std::shared_ptr<galp::runtime::PinnedRowgroupBufferPool>               rowgroup_prefetch_pinned_pool;
	size_t                                                                 rowgroup_prefetch_pinned_pool_slots = 0;
	bool                                                                   cached_gather_in_flight             = false;
	bool                                                                   cached_fixed_transform_in_flight    = false;
	int                                                                    direct_dct_stream_priority          = 0;
	int                                                                    cuda_least_stream_priority          = 0;
	int                                                                    cuda_greatest_stream_priority       = 0;
	bool                                                                   direct_dct_low_priority_streams     = false;
	size_t                                                                 transform_blocks_per_launch         = 0;
	size_t                                                                 transform_ctas_per_launch           = 0;
	bool                                                                   planless_transform_resources_ready  = false;
	size_t                                                                 planless_transform_registers_per_thread = 0;
	size_t                                                                 planless_transform_static_shared_bytes_per_cta = 0;
	size_t                                                                 planless_transform_local_bytes_per_thread = 0;
	size_t                                                                 planless_transform_max_active_ctas_per_sm = 0;
	size_t                                                                 cuda_max_threads_per_sm             = 0;
	size_t                                                                 cuda_warp_size                      = 0;
	std::list<std::string>                                                 fls_reader_lru;
	struct CachedFlsReaderEntry {
		std::shared_ptr<galp::format::FlsReader> reader;
		std::list<std::string>::iterator         lru_position;
	};
	std::unordered_map<std::string, CachedFlsReaderEntry> fls_readers;

	std::shared_ptr<galp::format::FlsReader> fls_reader(const std::filesystem::path& path,
	                                                    const bool enable_sparse_vector_reads = true) {
		const auto key = path.lexically_normal().string() +
		                 (enable_sparse_vector_reads ? "#sparse-vector" : "#rowgroup-only");
		const auto found = fls_readers.find(key);
		if (found != fls_readers.end()) {
			fls_reader_lru.splice(fls_reader_lru.begin(), fls_reader_lru, found->second.lru_position);
			return found->second.reader;
		}

		constexpr size_t kFlsReaderCacheCapacity = 64;
		if (fls_readers.size() >= kFlsReaderCacheCapacity) {
			fls_readers.erase(fls_reader_lru.back());
			fls_reader_lru.pop_back();
		}
		galp::format::FlsReaderOptions reader_options;
		reader_options.load_column_names = false;
		reader_options.enable_sparse_vector_reads = enable_sparse_vector_reads;
		reader_options.build_shared_zero_copy_schema_plan = enable_sparse_vector_reads;
		auto reader = std::make_shared<galp::format::FlsReader>(path, reader_options);
		fls_reader_lru.push_front(key);
		fls_readers.emplace(key, CachedFlsReaderEntry {reader, fls_reader_lru.begin()});
		return reader;
	}

	~JpegDctDeviceScratch() {
		try {
			if (cached_gather_in_flight) {
				cached_gather_done.synchronize();
				if (cache_hit_stream) {
					galp::memory::complete_h2d(cache_hit_stream.get());
				}
				cached_gather_in_flight = false;
			}
			// Free device scratch buffers while their allocation streams are still alive.
			// The workset release below may destroy the stream these were allocated on;
			// freeing after that would pass a stale stream to cudaFreeAsync.
			column_bindings.release();
			batch_projection_items.release();
			batch_fixed_transform_items.release();
			planless_image_descriptors.release();
			planless_vector_remap.release();
			fixed_transform_group_offsets.release();
			batch_materialize_items.release();
			cached_gather_items.release();
			cached_fixed_transform_items.release();
			decoded_gather_items.release();
			galp::runtime::release_workset(
			    decode_workset, /*preserve_resources=*/false, /*h2d_already_complete=*/false);
		} catch (const std::exception& e) { std::fprintf(stderr, "JpegDctDeviceScratch cleanup: %s\n", e.what()); }
	}

	cudaStream_t stream_for_cache_hit() {
		if (!cache_hit_stream) {
			cache_hit_stream.create_with_priority(cudaStreamNonBlocking, direct_dct_stream_priority);
		}
		return cache_hit_stream.get();
	}

	cudaStream_t stream_for_fixed_grid_rounding() {
		if (!fixed_grid_round_stream) {
			fixed_grid_round_stream.create_with_priority(cudaStreamNonBlocking, direct_dct_stream_priority);
		}
		return fixed_grid_round_stream.get();
	}

	cudaStream_t stream_for_transform() {
		if (!transform_stream) {
			transform_stream.create_with_priority(cudaStreamNonBlocking, direct_dct_stream_priority);
		}
		return transform_stream.get();
	}

	void configure_scheduling(const bool   use_low_priority,
	                          const size_t blocks_per_launch,
	                          const size_t ctas_per_launch) {
		CUDA_SAFE_CALL(cudaDeviceGetStreamPriorityRange(
		    &cuda_least_stream_priority, &cuda_greatest_stream_priority));
		direct_dct_low_priority_streams       = use_low_priority;
		direct_dct_stream_priority            = use_low_priority ? cuda_least_stream_priority : 0;
		transform_blocks_per_launch           = blocks_per_launch;
		transform_ctas_per_launch             = ctas_per_launch;
		decode_workset.transfer.stream_priority = direct_dct_stream_priority;
	}

	int actual_stream_priority(const cudaStream_t stream) const {
		if (stream == nullptr) {
			return std::numeric_limits<int>::max();
		}
		int actual_priority = 0;
		CUDA_SAFE_CALL(cudaStreamGetPriority(stream, &actual_priority));
		return actual_priority;
	}

	void ensure_cached_gather_events() {
		cached_gather_start.create();
		cached_gather_done.create();
	}

	void ensure_decoded_batch_events() {
		decoded_batch_gather_done.create();
	}
};

// Reader-owned tail fence for the shared scratch/cache execution context. It
// is separate from the batch's public completion event so a later submission
// can safely reuse the context even when the earlier batch remains alive and
// the caller never invokes a synchronous accessor.
struct JpegDctDeviceExecutionFence {
	galp::memory::CudaEvent completion_event;
	std::exception_ptr      completion_error;
	bool                    pending = false;

	void wait_before_reuse() {
		if (completion_error) {
			std::rethrow_exception(completion_error);
		}
		if (!pending) {
			return;
		}
		try {
			completion_event.synchronize();
			pending = false;
		} catch (...) {
			completion_error = std::current_exception();
			throw;
		}
	}

	void record(const cudaStream_t stream) {
		completion_event.create_with_flags(cudaEventDisableTiming);
		completion_event.record(stream);
		pending = true;
	}

	void mark_complete() noexcept {
		pending = false;
	}

	void mark_failed(std::exception_ptr error) noexcept {
		completion_error = std::move(error);
		pending          = false;
	}
};

void JpegDctDeviceScratchDeleter::operator()(JpegDctDeviceScratch* scratch) const noexcept {
	delete scratch;
}

JpegDctDeviceScratchPtr make_jpeg_dct_device_scratch() {
	return JpegDctDeviceScratchPtr(new JpegDctDeviceScratch());
}

void JpegDctHostIoContextDeleter::operator()(JpegDctHostIoContext* context) const noexcept {
	delete context;
}

JpegDctHostIoContextPtr make_jpeg_dct_host_io_context() {
	return JpegDctHostIoContextPtr(new JpegDctHostIoContext());
}

void JpegDctDeviceExecutionFenceDeleter::operator()(JpegDctDeviceExecutionFence* fence) const noexcept {
	delete fence;
}

JpegDctDeviceExecutionFencePtr make_jpeg_dct_device_execution_fence() {
	return JpegDctDeviceExecutionFencePtr(new JpegDctDeviceExecutionFence());
}

void wait_jpeg_dct_device_execution_fence(JpegDctDeviceExecutionFence& fence) {
	fence.wait_before_reuse();
}

void JpegDctDeviceDecodedRowgroupCacheDeleter::operator()(JpegDctDeviceDecodedRowgroupCache* cache) const noexcept {
	delete cache;
}

JpegDctDeviceDecodedRowgroupCachePtr make_jpeg_dct_device_cache() {
	return JpegDctDeviceDecodedRowgroupCachePtr(new JpegDctDeviceDecodedRowgroupCache());
}

void set_jpeg_dct_device_cache_capacity(JpegDctDeviceDecodedRowgroupCache& cache, const size_t bytes) {
	cache.set_capacity(bytes);
}

namespace {

constexpr unsigned kPlanlessTransformThreadsPerCta        = 64U;
constexpr unsigned kLimitedPlanlessTransformCtasPerLaunch = 64U;

struct FixedTransformPlanView {
	const std::vector<uint32_t>* item_order    = nullptr;
	const std::vector<uint32_t>* group_offsets = nullptr;
};

size_t fixed_transform_item_count(const std::vector<DecodedRowgroupWork>& works) {
	size_t count = 0;
	for (const auto& work : works) {
		if (work.fixed_transform_items != nullptr) {
			count += work.fixed_transform_items->size();
		}
	}
	return count;
}

FixedTransformPlanView fixed_transform_plan_for_workset(
    const std::vector<uint32_t>* fixed_transform_item_order,
    const std::vector<uint32_t>* fixed_transform_group_offsets,
    const size_t                 source_item_offset,
    const size_t                 source_item_count,
    JpegDctDeviceScratch&        scratch) {
	if (fixed_transform_item_order == nullptr && fixed_transform_group_offsets == nullptr) {
		return {};
	}
	if (fixed_transform_item_order == nullptr || fixed_transform_group_offsets == nullptr) {
		throw std::runtime_error("JPEG DCT deterministic fixed-transform plan is incomplete");
	}
	const auto& plan_order   = *fixed_transform_item_order;
	const auto& plan_offsets = *fixed_transform_group_offsets;
	if (plan_order.empty()) {
		if (!plan_offsets.empty() || source_item_offset != 0U || source_item_count != 0U) {
			throw std::runtime_error("JPEG DCT deterministic fixed-transform plan is inconsistent");
		}
		return {};
	}
	if (plan_offsets.size() < 2U || plan_offsets.front() != 0U || plan_offsets.back() != plan_order.size() ||
	    !std::is_sorted(plan_offsets.begin(), plan_offsets.end())) {
		throw std::runtime_error("JPEG DCT deterministic fixed-transform group offsets are invalid");
	}
	if (source_item_offset > plan_order.size() || source_item_count > plan_order.size() - source_item_offset) {
		throw std::runtime_error("JPEG DCT deterministic fixed-transform workset exceeds the plan");
	}
	if (source_item_count == 0U) {
		return {};
	}
	if (source_item_offset == 0U && source_item_count == plan_order.size()) {
		return {fixed_transform_item_order, fixed_transform_group_offsets};
	}
	if (source_item_count > std::numeric_limits<uint32_t>::max()) {
		throw std::runtime_error("JPEG DCT fixed-transform workset exceeds deterministic index range");
	}

	// The prepared permutation maps plan-wide source order to plan-wide groups.
	// A workset is a contiguous source slice, but its global grouped indices need
	// not be contiguous when request order differs from shard/rowgroup order.
	// Repack represented groups in first-source order while retaining source order
	// within each group (the global stable sort uses that same within-group order).
	auto& workset_order         = scratch.host_workset_fixed_transform_item_order;
	auto& workset_offsets       = scratch.host_workset_fixed_transform_group_offsets;
	auto& workset_item_groups   = scratch.host_workset_fixed_transform_item_groups;
	auto& workset_group_cursors = scratch.host_workset_fixed_transform_group_cursors;
	auto& workset_group_lookup  = scratch.host_workset_fixed_transform_group_lookup;
	workset_order.resize(source_item_count);
	workset_item_groups.resize(source_item_count);
	workset_offsets.clear();
	workset_offsets.push_back(0U);
	workset_group_lookup.clear();
	workset_group_lookup.reserve(std::min(source_item_count, plan_offsets.size() - 1U));
	for (size_t source_index = 0; source_index < source_item_count; ++source_index) {
		const auto global_ordered_index = plan_order[source_item_offset + source_index];
		if (global_ordered_index >= plan_order.size()) {
			throw std::runtime_error("JPEG DCT deterministic fixed-transform permutation is invalid");
		}
		const auto upper = std::upper_bound(plan_offsets.begin(), plan_offsets.end(), global_ordered_index);
		if (upper == plan_offsets.begin() || upper == plan_offsets.end()) {
			throw std::runtime_error("JPEG DCT deterministic fixed-transform group lookup failed");
		}
		const auto global_group = static_cast<size_t>(std::distance(plan_offsets.begin(), upper) - 1);
		auto [group_it, inserted] =
		    workset_group_lookup.emplace(global_group, static_cast<uint32_t>(workset_group_lookup.size()));
		if (inserted) {
			workset_offsets.push_back(0U);
		}
		const auto local_group            = group_it->second;
		workset_item_groups[source_index] = local_group;
		++workset_offsets[static_cast<size_t>(local_group) + 1U];
	}
	for (size_t group = 1; group < workset_offsets.size(); ++group) {
		workset_offsets[group] += workset_offsets[group - 1U];
	}
	workset_group_cursors.assign(workset_offsets.begin(), workset_offsets.end() - 1);
	for (size_t source_index = 0; source_index < source_item_count; ++source_index) {
		workset_order[source_index] = workset_group_cursors[workset_item_groups[source_index]]++;
	}
	return {&workset_order, &workset_offsets};
}

 void record_planless_transform_resources(JpegDctDeviceScratch& scratch, JpegDctDeviceExecutionStats& stats) {
	if (!scratch.planless_transform_resources_ready) {
		cudaFuncAttributes attributes {};
		CUDA_SAFE_CALL(cudaFuncGetAttributes(&attributes, transformed_dct_grid_planless_kernel));
		int max_active_ctas_per_sm = 0;
		CUDA_SAFE_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
		    &max_active_ctas_per_sm, transformed_dct_grid_planless_kernel, kPlanlessTransformThreadsPerCta, 0));
		int device = 0;
		int max_threads_per_sm = 0;
		int warp_size = 0;
		CUDA_SAFE_CALL(cudaGetDevice(&device));
		CUDA_SAFE_CALL(cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device));
		CUDA_SAFE_CALL(cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, device));
		if (attributes.numRegs <= 0 || max_active_ctas_per_sm <= 0 || max_threads_per_sm <= 0 || warp_size <= 0) {
			throw std::runtime_error("invalid CUDA planless transform kernel resource attributes");
		}
		scratch.planless_transform_registers_per_thread = static_cast<size_t>(attributes.numRegs);
		scratch.planless_transform_static_shared_bytes_per_cta = attributes.sharedSizeBytes;
		scratch.planless_transform_local_bytes_per_thread = attributes.localSizeBytes;
		scratch.planless_transform_max_active_ctas_per_sm = static_cast<size_t>(max_active_ctas_per_sm);
		scratch.cuda_max_threads_per_sm = static_cast<size_t>(max_threads_per_sm);
		scratch.cuda_warp_size = static_cast<size_t>(warp_size);
		scratch.planless_transform_resources_ready = true;
	}
	stats.planless_transform_registers_per_thread = scratch.planless_transform_registers_per_thread;
	stats.planless_transform_static_shared_bytes_per_cta =
	    scratch.planless_transform_static_shared_bytes_per_cta;
	stats.planless_transform_local_bytes_per_thread = scratch.planless_transform_local_bytes_per_thread;
	stats.planless_transform_threads_per_cta = kPlanlessTransformThreadsPerCta;
	stats.planless_transform_max_active_ctas_per_sm = scratch.planless_transform_max_active_ctas_per_sm;
	stats.cuda_max_threads_per_sm = scratch.cuda_max_threads_per_sm;
	stats.cuda_warp_size = scratch.cuda_warp_size;
}

 template <typename Fn>
void for_each_selected_coefficient(const std::vector<uint8_t>&             selected_coefficients,
                                   const JpegDctCoefficientSelectionShape& selection_shape,
                                   Fn&&                                    fn) {
	if (selection_shape.is_contiguous_prefix()) {
		for (size_t coeff_idx = 0; coeff_idx < selection_shape.count; ++coeff_idx) {
			fn(coeff_idx);
		}
		return;
	}
	for (const auto coeff_idx : selected_coefficients) {
		fn(static_cast<size_t>(coeff_idx));
	}
}

BoundCoeffColumns bind_coeff_columns(const galp::execution::Rowgroup&        rowgroup,
                                     const galp::runtime::ExecutionWorkset&  workset,
                                     const std::vector<uint8_t>&             coefficients_to_bind,
                                     const JpegDctCoefficientSelectionShape& binding_selection_shape,
                                     const size_t                            expr_index_base = 0) {
	if (rowgroup.columns.size() < 64) {
		throw std::runtime_error("JPEG DCT device batch requires at least 64 logical coefficient columns");
	}

	BoundCoeffColumns bound;
	{
		const auto& batch = workset.buffers.host_batches.template get<int8_t>();
		for (size_t expr_idx = 0; expr_idx < batch.device_exprs.size(); ++expr_idx) {
			const auto materialize_idx = batch.expr_indices[expr_idx];
			if (materialize_idx < expr_index_base || materialize_idx >= expr_index_base + bound.columns_i16.size()) {
				continue;
			}
			const auto logical_idx            = materialize_idx - expr_index_base;
			bound.columns_i8[logical_idx]     = batch.device_exprs[expr_idx].out;
			bound.column_sources[logical_idx] = DeviceCoeffSource::kI8;
		}
	}
	{
		const auto& batch = workset.buffers.host_batches.template get<int16_t>();
		for (size_t expr_idx = 0; expr_idx < batch.device_exprs.size(); ++expr_idx) {
			const auto materialize_idx = batch.expr_indices[expr_idx];
			if (materialize_idx < expr_index_base || materialize_idx >= expr_index_base + bound.columns_i16.size()) {
				continue;
			}
			const auto logical_idx            = materialize_idx - expr_index_base;
			bound.columns_i16[logical_idx]    = batch.device_exprs[expr_idx].out;
			bound.column_sources[logical_idx] = DeviceCoeffSource::kI16;
		}
	}

	const auto resolve_column = [&](const auto& self, const size_t coeff_idx) -> DeviceCoeffSource {
		if (coeff_idx >= bound.column_sources.size()) {
			throw std::runtime_error("JPEG DCT device batch coefficient alias index is outside 64-column DCT block");
		}
		if (bound.column_sources[coeff_idx] != DeviceCoeffSource::kMissing) {
			return bound.column_sources[coeff_idx];
		}
		const auto& column = rowgroup.columns[coeff_idx];
		if (column.alias_of.has_value()) {
			const auto source               = self(self, *column.alias_of);
			bound.column_sources[coeff_idx] = source;
			if (source == DeviceCoeffSource::kI8) {
				bound.columns_i8[coeff_idx] = bound.columns_i8[*column.alias_of];
			} else if (source == DeviceCoeffSource::kI16) {
				bound.columns_i16[coeff_idx] = bound.columns_i16[*column.alias_of];
			}
			return source;
		}
		throw std::runtime_error("JPEG DCT device batch could not bind a coefficient column to device output");
	};
	for_each_selected_coefficient(coefficients_to_bind, binding_selection_shape, [&](const size_t coeff_idx) {
		(void)resolve_column(resolve_column, coeff_idx);
	});
	return bound;
}

void append_compact_column_bindings(const BoundCoeffColumns&       source,
                                    const std::vector<uint8_t>&    active_physical_coefficients,
                                    std::vector<DeviceCoeffBinding>& column_bindings,
                                    std::array<uint32_t, kJpegDctCoefficientCount>& binding_index_by_physical) {
	binding_index_by_physical.fill(std::numeric_limits<uint32_t>::max());
	for (const auto physical_coeff : active_physical_coefficients) {
		const auto coeff_idx = static_cast<size_t>(physical_coeff);
		if (coeff_idx >= source.column_sources.size()) {
			throw std::runtime_error("JPEG DCT physical coefficient column is outside binding range");
		}
		if (source.column_sources[coeff_idx] == DeviceCoeffSource::kMissing) {
			throw std::runtime_error("JPEG DCT projection references an unbound physical coefficient column");
		}
		const auto binding_index                 = static_cast<uint32_t>(column_bindings.size());
		binding_index_by_physical[coeff_idx]     = binding_index;
		column_bindings.push_back(DeviceCoeffBinding {
		    source.columns_i8[coeff_idx], source.columns_i16[coeff_idx], source.column_sources[coeff_idx]});
	}
}

void append_dense_column_bindings(const BoundCoeffColumns&                        source,
                                  std::vector<DeviceCoeffBinding>&                column_bindings,
                                  std::array<uint32_t, kJpegDctCoefficientCount>& binding_index_by_physical) {
	binding_index_by_physical.fill(std::numeric_limits<uint32_t>::max());
	const auto binding_base = static_cast<uint32_t>(column_bindings.size());
	for (size_t coeff_idx = 0; coeff_idx < kJpegDctCoefficientCount; ++coeff_idx) {
		binding_index_by_physical[coeff_idx] = binding_base + static_cast<uint32_t>(coeff_idx);
		column_bindings.push_back(DeviceCoeffBinding {
		    source.columns_i8[coeff_idx], source.columns_i16[coeff_idx], source.column_sources[coeff_idx]});
	}
}

void project_planless_transformed_dct_grid_batch(const std::vector<BoundCoeffColumns>&   sources,
                                                 const std::vector<DecodedRowgroupWork>& works,
                                                 const uint16_t*                         quant_tables,
                                                 const float*                            phase_matrices,
                                                 const JpegDctGridTransformSpec&         transform,
                                                 float*                                  y_accum,
                                                 float*                                  cbcr_accum,
                                                 JpegDctDeviceScratch&                   scratch,
                                                 JpegDctDeviceExecutionStats&            stats,
                                                 const size_t                            transform_blocks_per_launch,
                                                 const size_t                            transform_ctas_per_launch,
                                                 cudaStream_t                            stream) {
	if (sources.empty() || works.empty()) {
		return;
	}
	if (quant_tables == nullptr || sources.size() != works.size()) {
		throw std::runtime_error("JPEG DCT planless transform has invalid sources or quantization tables");
	}
	auto& column_bindings = scratch.host_column_bindings;
	auto& images          = scratch.host_planless_image_descriptors;
	auto& vector_remap    = scratch.host_planless_vector_remap;
	column_bindings.clear();
	images.clear();
	vector_remap.clear();
	column_bindings.reserve(works.size() * kJpegDctCoefficientCount);
	size_t image_count = 0;
	for (const auto& work : works) {
		if (work.planless_images != nullptr) {
			image_count += work.planless_images->size();
		}
	}
	images.reserve(image_count);
	for (size_t source_idx = 0; source_idx < works.size(); ++source_idx) {
		const auto                                     binding_base = static_cast<uint32_t>(column_bindings.size());
		std::array<uint32_t, kJpegDctCoefficientCount> binding_index_by_physical {};
		append_dense_column_bindings(sources[source_idx], column_bindings, binding_index_by_physical);
		const auto* source_images = works[source_idx].planless_images;
		if (source_images == nullptr) {
			continue;
		}
		uint32_t remap_base = std::numeric_limits<uint32_t>::max();
		if (!works[source_idx].decodes_full_rowgroup) {
			if (works[source_idx].selected_vectors == nullptr || works[source_idx].selected_vectors->empty()) {
				throw std::runtime_error("JPEG DCT planless selected decode is missing its vector remap");
			}
			const size_t logical_n_vecs = works[source_idx].logical_rowgroup_n_vecs;
			if (logical_n_vecs == 0U || vector_remap.size() > std::numeric_limits<uint32_t>::max() ||
			    logical_n_vecs > std::numeric_limits<uint32_t>::max() - vector_remap.size()) {
				throw std::runtime_error("JPEG DCT planless vector remap exceeds uint32 range");
			}
			remap_base = static_cast<uint32_t>(vector_remap.size());
			auto source_remap = build_logical_to_compact_vector_remap(
			    *works[source_idx].selected_vectors, logical_n_vecs, works[source_idx].decode_unpack_n_vectors);
			vector_remap.insert(vector_remap.end(), source_remap.begin(), source_remap.end());
		}
		for (auto image : *source_images) {
			image.binding_base      = binding_base;
			image.vector_remap_base = remap_base;
			images.push_back(image);
		}
	}
	if (images.empty()) {
		return;
	}
	record_planless_transform_resources(scratch, stats);
	const uint64_t blocks_per_image =
	    static_cast<uint64_t>(transform.y_output_width_blocks) * transform.y_output_height_blocks +
	    2U * static_cast<uint64_t>(transform.cbcr_output_width_blocks) * transform.cbcr_output_height_blocks;
	const uint64_t output_blocks = blocks_per_image * images.size();
	if (output_blocks > std::numeric_limits<unsigned>::max()) {
		throw std::runtime_error("JPEG DCT planless transform grid exceeds CUDA launch range");
	}
	scratch.column_bindings.upload(column_bindings.data(), column_bindings.size(), stream, stats);
	if (!vector_remap.empty()) {
		scratch.planless_vector_remap.upload(vector_remap.data(), vector_remap.size(), stream, stats);
	}
	scratch.planless_image_descriptors.upload(images.data(), images.size(), stream, stats);
	const uint64_t launch_output_limit = transform_blocks_per_launch == 0
	                                         ? output_blocks
	                                         : std::min<uint64_t>(output_blocks, transform_blocks_per_launch);
	for (uint64_t offset = 0; offset < output_blocks; offset += launch_output_limit) {
		const auto launch_output_blocks = std::min<uint64_t>(launch_output_limit, output_blocks - offset);
		const auto limited_cta_limit    = transform_ctas_per_launch == 0
		                                      ? static_cast<uint64_t>(kLimitedPlanlessTransformCtasPerLaunch)
		                                      : static_cast<uint64_t>(transform_ctas_per_launch);
		const auto launch_ctas          = static_cast<unsigned>(
            transform_blocks_per_launch == 0 ? launch_output_blocks
                                             : std::min<uint64_t>(launch_output_blocks, limited_cta_limit));
		transformed_dct_grid_planless_kernel<<<dim3(launch_ctas), dim3(kPlanlessTransformThreadsPerCta), 0, stream>>>(
		    scratch.column_bindings.data,
		    scratch.planless_image_descriptors.data,
		    vector_remap.empty() ? nullptr : scratch.planless_vector_remap.data,
		    images.size(),
		    offset,
		    launch_output_blocks,
		    quant_tables,
		    phase_matrices,
		    transform.y_output_width_blocks,
		    transform.y_output_height_blocks,
		    transform.cbcr_output_width_blocks,
		    transform.cbcr_output_height_blocks,
		    transform.clamp_min,
		    transform.clamp_max,
		    y_accum,
		    cbcr_accum);
		CUDA_SAFE_CALL(cudaGetLastError());
		++stats.materialize_kernel_launch_count;
		++stats.planless_transform_kernel_launch_count;
		stats.planless_transform_max_blocks_per_launch =
		    std::max(stats.planless_transform_max_blocks_per_launch, static_cast<size_t>(launch_ctas));
		stats.planless_transform_max_output_blocks_per_launch =
		    std::max(stats.planless_transform_max_output_blocks_per_launch, static_cast<size_t>(launch_output_blocks));
	}
	stats.planless_image_descriptor_count += images.size();
	stats.planless_transform_output_block_count += output_blocks;
	stats.device_mapping_fused = true;
}

void project_transformed_dct_grid_batch(const std::vector<BoundCoeffColumns>&   sources,
                                        const std::vector<DecodedRowgroupWork>& works,
                                        const std::vector<uint32_t>*            fixed_transform_item_order,
                                        const std::vector<uint32_t>*            fixed_transform_group_offsets,
                                        const uint16_t*                         quant_tables,
                                        const float*                            resize_weight_matrices,
                                        const JpegDctGridTransformSpec&         transform,
                                        float*                                  y_accum,
                                        float*                                  cbcr_accum,
                                        JpegDctDeviceScratch&                   scratch,
                                        JpegDctDeviceExecutionStats&            stats,
                                        cudaStream_t                            stream) {
	if (sources.empty() || works.empty()) {
		return;
	}
	if (quant_tables == nullptr || resize_weight_matrices == nullptr) {
		throw std::runtime_error("JPEG DCT fixed transform requires uploaded quantization tables and resize weights");
	}
	if (sources.size() != works.size()) {
		throw std::runtime_error("JPEG DCT fixed transform source/work count mismatch");
	}
	auto& column_bindings       = scratch.host_column_bindings;
	auto& source_order_items    = scratch.host_fixed_transform_items;
	auto& ordered_items         = scratch.host_ordered_fixed_transform_items;
	column_bindings.clear();
	source_order_items.clear();
	ordered_items.clear();

	size_t total_items = 0;
	for (const auto& work : works) {
		if (work.fixed_transform_items != nullptr) {
			total_items += work.fixed_transform_items->size();
		}
	}
	const bool deterministic = fixed_transform_item_order != nullptr && fixed_transform_group_offsets != nullptr &&
	                           !fixed_transform_item_order->empty();
	if (deterministic &&
	    (fixed_transform_item_order->size() != total_items || fixed_transform_group_offsets->size() < 2U ||
	     fixed_transform_group_offsets->front() != 0U || fixed_transform_group_offsets->back() != total_items)) {
		throw std::runtime_error("JPEG DCT deterministic fixed-transform plan does not match decoded items");
	}
	auto& transform_items = deterministic ? ordered_items : source_order_items;
	column_bindings.reserve(works.size() * kJpegDctCoefficientCount);
	if (deterministic) {
		transform_items.resize(total_items);
	} else {
		transform_items.reserve(total_items);
	}
	size_t flat_item_index = 0;
	for (size_t source_idx = 0; source_idx < works.size(); ++source_idx) {
		const auto& source = sources[source_idx];
		const auto& work   = works[source_idx];
		std::array<uint32_t, kJpegDctCoefficientCount> binding_index_by_physical {};
		const auto binding_base = static_cast<uint32_t>(column_bindings.size());
		append_dense_column_bindings(source, column_bindings, binding_index_by_physical);
		if (work.fixed_transform_items == nullptr) {
			continue;
		}
		for (const auto& item : *work.fixed_transform_items) {
			const auto device_item =
			    JpegDctDeviceFixedTransformBatchItem {binding_base,
			                                          item.row_in_rowgroup,
			                                          item.output_block_index,
			                                          item.component,
			                                          static_cast<uint8_t>(item.zigzag_columns ? 1U : 0U),
			                                          item.x_factor,
			                                          item.y_factor,
			                                          item.x_subblock,
			                                          item.y_subblock,
			                                          static_cast<uint8_t>(item.x_upsample ? 1U : 0U),
			                                          static_cast<uint8_t>(item.y_upsample ? 1U : 0U),
			                                          item.x_up_factor,
			                                          item.y_up_factor,
			                                          item.x_down_factor,
			                                          item.y_down_factor,
			                                          item.quant_table_index,
			                                          item.x_weight_matrix_index,
			                                          item.y_weight_matrix_index};
			if (deterministic) {
				const auto ordered_index = (*fixed_transform_item_order)[flat_item_index];
				if (ordered_index >= transform_items.size()) {
					throw std::runtime_error("JPEG DCT deterministic fixed-transform permutation is out of range");
				}
				transform_items[ordered_index] = device_item;
			} else {
				transform_items.push_back(device_item);
			}
			++flat_item_index;
		}
	}
	if (transform_items.empty()) {
		return;
	}
	// The deterministic grouped kernel uses one thread per coefficient in a
	// composed factor-2 16x16 source block.  Its generic fallback keeps only the
	// first 64 lanes active while all lanes participate in barriers.
	constexpr unsigned kGroupedThreads = 256;
	constexpr unsigned kSourceThreads  = 64;
	scratch.column_bindings.upload(column_bindings.data(), column_bindings.size(), stream, stats);
	scratch.batch_fixed_transform_items.upload(transform_items.data(), transform_items.size(), stream, stats);
	const dim3 block(deterministic ? kGroupedThreads : kSourceThreads);
	if (deterministic) {
		scratch.fixed_transform_group_offsets.upload(
		    fixed_transform_group_offsets->data(), fixed_transform_group_offsets->size(), stream, stats);
		const size_t group_count = fixed_transform_group_offsets->size() - 1U;
		const dim3   grid(static_cast<unsigned>(group_count));
		transformed_dct_grid_grouped_kernel<<<grid, block, 0, stream>>>(scratch.column_bindings.data,
		                                                                scratch.batch_fixed_transform_items.data,
		                                                                scratch.fixed_transform_group_offsets.data,
		                                                                group_count,
		                                                                quant_tables,
		                                                                resize_weight_matrices,
		                                                                transform.clamp_min,
		                                                                transform.clamp_max,
		                                                                y_accum,
		                                                                cbcr_accum);
	} else {
		const dim3 grid(static_cast<unsigned>(transform_items.size()));
		transformed_dct_grid_sources_kernel<<<grid, block, 0, stream>>>(scratch.column_bindings.data,
		                                                                scratch.batch_fixed_transform_items.data,
		                                                                transform_items.size(),
		                                                                quant_tables,
		                                                                resize_weight_matrices,
		                                                                transform.clamp_min,
		                                                                transform.clamp_max,
		                                                                y_accum,
		                                                                cbcr_accum);
	}
	CUDA_SAFE_CALL(cudaGetLastError());
	++stats.materialize_kernel_launch_count;
	stats.fixed_transform_item_count += transform_items.size();
}

    void gather_cached_rowgroups_batch(const std::vector<JpegDctDeviceCachedGatherBatchItem>& items,
	                                   int16_t*                                               output,
	                                   JpegDctDeviceScratch&                                  scratch,
	                                   JpegDctDeviceExecutionStats&                           stats,
	                                   cudaStream_t                                           stream) {
	if (items.empty()) {
		return;
	}

	scratch.cached_gather_items.upload(items.data(), items.size(), stream, stats);
	constexpr unsigned kThreads = 256;
	const size_t       total    = items.size() * 64U;
	const dim3         block(kThreads);
	const dim3         grid(static_cast<unsigned>((total + kThreads - 1U) / kThreads));
	gather_cached_dct_blocks_batch_kernel<<<grid, block, 0, stream>>>(
	    scratch.cached_gather_items.data, items.size(), output);
	CUDA_SAFE_CALL(cudaGetLastError());
		++stats.gather_kernel_launch_count;
		++stats.cached_gather_kernel_launch_count;
	}

	void gather_decoded_rowgroup_batch(const std::vector<BoundCoeffColumns>& sources,
	                                   const std::vector<DecodedRowgroupWork>& works,
	                                   int16_t* output,
	                                   JpegDctDeviceScratch& scratch,
	                                   JpegDctDeviceExecutionStats& stats,
	                                   cudaStream_t stream) {
		if (sources.empty() || works.empty()) {
			return;
		}
		if (sources.size() != works.size()) {
			throw std::runtime_error("JPEG DCT decoded gather source/work count mismatch");
		}

		auto& column_bindings = scratch.host_column_bindings;
		auto& gather_items    = scratch.host_decoded_gather_items;
		column_bindings.clear();
		gather_items.clear();

		size_t total_gather_items = 0;
		for (const auto& work : works) {
			if (work.gather_items != nullptr) {
				total_gather_items += work.gather_items->size();
			}
		}
		column_bindings.reserve(works.size() * kJpegDctCoefficientCount);
		gather_items.reserve(total_gather_items);

		for (size_t source_idx = 0; source_idx < works.size(); ++source_idx) {
			const auto& source = sources[source_idx];
			const auto& work   = works[source_idx];
			std::array<uint32_t, kJpegDctCoefficientCount> binding_index_by_physical {};
			append_dense_column_bindings(source, column_bindings, binding_index_by_physical);
			if (work.gather_items == nullptr) {
				continue;
			}
			if (source_idx > std::numeric_limits<uint32_t>::max()) {
				throw std::runtime_error("JPEG DCT decoded gather source index overflow");
			}
			for (const auto& item : *work.gather_items) {
				gather_items.push_back(JpegDctDeviceDecodedGatherBatchItem {
				    static_cast<uint32_t>(source_idx), item.row_in_rowgroup, item.output_block_index});
			}
		}
		if (gather_items.empty()) {
			return;
		}

		constexpr unsigned kThreads = 256;
		scratch.column_bindings.upload(column_bindings.data(), column_bindings.size(), stream, stats);
		scratch.decoded_gather_items.upload(gather_items.data(), gather_items.size(), stream, stats);
		const size_t total = gather_items.size() * kJpegDctCoefficientCount;
		const dim3   block(kThreads);
		const dim3   grid(static_cast<unsigned>((total + kThreads - 1U) / kThreads));
		gather_decoded_dct_blocks_batch_kernel<<<grid, block, 0, stream>>>(
		    scratch.column_bindings.data, scratch.decoded_gather_items.data, gather_items.size(), output);
		CUDA_SAFE_CALL(cudaGetLastError());
		++stats.gather_kernel_launch_count;
		stats.gather_item_count += gather_items.size();
		stats.decoded_gather_item_count += gather_items.size();
	}

	void project_decoded_rowgroup_batch(const std::vector<BoundCoeffColumns>& sources,
	                                    const std::vector<DecodedRowgroupWork>& works,
                                    const size_t coefficients_per_block,
                                    const bool use_dense_bindings,
                                    int16_t* output,
                                    JpegDctDeviceScratch& scratch,
                                    JpegDctDeviceExecutionStats& stats,
                                    cudaStream_t stream) {
	if (sources.empty() || works.empty() || coefficients_per_block == 0) {
		return;
	}
	if (sources.size() != works.size()) {
		throw std::runtime_error("JPEG DCT projection source/work count mismatch");
	}

	auto& column_bindings  = scratch.host_column_bindings;
	auto& projection_items = scratch.host_projection_items;
	column_bindings.clear();
	projection_items.clear();
	const auto build_start = Clock::now();

	size_t total_projection_items = 0;
	size_t total_active_columns   = 0;
	for (const auto& work : works) {
		if (work.projection_items != nullptr) {
			total_projection_items += work.projection_items->size();
		}
		total_active_columns += use_dense_bindings ? kJpegDctCoefficientCount : work.active_physical_coefficients.size();
	}
	column_bindings.reserve(total_active_columns);
	projection_items.reserve(total_projection_items);

	for (size_t source_idx = 0; source_idx < works.size(); ++source_idx) {
		const auto& source = sources[source_idx];
		const auto& work   = works[source_idx];
		std::array<uint32_t, kJpegDctCoefficientCount> binding_index_by_physical {};
		if (use_dense_bindings) {
			append_dense_column_bindings(source, column_bindings, binding_index_by_physical);
		} else {
			append_compact_column_bindings(
			    source, work.active_physical_coefficients, column_bindings, binding_index_by_physical);
		}
		if (work.projection_items == nullptr) {
			continue;
		}
		for (const auto& item : *work.projection_items) {
			const auto physical_coeff = static_cast<size_t>(item.physical_coefficient_column_id);
			if (physical_coeff >= binding_index_by_physical.size() ||
			    binding_index_by_physical[physical_coeff] == std::numeric_limits<uint32_t>::max()) {
				throw std::runtime_error("JPEG DCT projection references a physical column that was not bound");
			}
			projection_items.push_back(JpegDctDeviceProjectionBatchItem {
			    binding_index_by_physical[physical_coeff],
			    item.row_in_rowgroup,
			    item.output_block_index,
			    item.selected_coefficient_slot,
			    item.logical_coefficient_id,
			    item.physical_coefficient_column_id,
			    item.output_coefficient_id,
			    item.output_grid_tensor});
		}
	}
	if (projection_items.empty()) {
		stats.projection_item_build_ms += elapsed_ms(build_start, Clock::now());
		return;
	}
	stats.projection_item_build_ms += elapsed_ms(build_start, Clock::now());

	constexpr unsigned kThreads = 256;
	scratch.column_bindings.upload(column_bindings.data(), column_bindings.size(), stream, stats);
	scratch.batch_projection_items.upload(projection_items.data(), projection_items.size(), stream, stats);
	const dim3 block(kThreads);
	const dim3 grid(static_cast<unsigned>((projection_items.size() + kThreads - 1U) / kThreads));
	project_dct_coefficients_batch_kernel<<<grid, block, 0, stream>>>(
	    scratch.column_bindings.data,
	    scratch.batch_projection_items.data,
	    projection_items.size(),
	    coefficients_per_block,
	    output);
	CUDA_SAFE_CALL(cudaGetLastError());
	++stats.materialize_kernel_launch_count;
	stats.projection_item_count += projection_items.size();
	stats.decoded_projection_item_count += projection_items.size();
	stats.jpeg_dct_projection_items_materialized += projection_items.size();
}

void project_decoded_ycbcr_grid_batch(const std::vector<BoundCoeffColumns>&   sources,
                                      const std::vector<DecodedRowgroupWork>& works,
                                      const bool                              use_dense_bindings,
                                      int16_t*                                y_output,
                                      int16_t*                                cbcr_output,
                                      float*                                  y_accum,
                                      float*                                  cbcr_accum,
                                      JpegDctDeviceScratch&                   scratch,
                                      JpegDctDeviceExecutionStats&            stats,
                                      cudaStream_t                            stream) {
	if (sources.empty() || works.empty()) {
		return;
	}
	if (sources.size() != works.size()) {
		throw std::runtime_error("JPEG DCT YCbCr-grid projection source/work count mismatch");
	}

	auto& column_bindings  = scratch.host_column_bindings;
	auto& projection_items = scratch.host_projection_items;
	column_bindings.clear();
	projection_items.clear();
	const auto build_start = Clock::now();

	size_t total_projection_items = 0;
	size_t total_active_columns   = 0;
	for (const auto& work : works) {
		if (work.projection_items != nullptr) {
			total_projection_items += work.projection_items->size();
		}
		total_active_columns += use_dense_bindings ? kJpegDctCoefficientCount : work.active_physical_coefficients.size();
	}
	column_bindings.reserve(total_active_columns);
	projection_items.reserve(total_projection_items);

	for (size_t source_idx = 0; source_idx < works.size(); ++source_idx) {
		const auto& source = sources[source_idx];
		const auto& work   = works[source_idx];
		std::array<uint32_t, kJpegDctCoefficientCount> binding_index_by_physical {};
		if (use_dense_bindings) {
			append_dense_column_bindings(source, column_bindings, binding_index_by_physical);
		} else {
			append_compact_column_bindings(
			    source, work.active_physical_coefficients, column_bindings, binding_index_by_physical);
		}
		if (work.projection_items == nullptr) {
			continue;
		}
		for (const auto& item : *work.projection_items) {
			const auto physical_coeff = static_cast<size_t>(item.physical_coefficient_column_id);
			if (physical_coeff >= binding_index_by_physical.size() ||
			    binding_index_by_physical[physical_coeff] == std::numeric_limits<uint32_t>::max()) {
				throw std::runtime_error("JPEG DCT YCbCr-grid projection references an unbound physical column");
			}
			projection_items.push_back(JpegDctDeviceProjectionBatchItem {
			    binding_index_by_physical[physical_coeff],
			    item.row_in_rowgroup,
			    item.output_block_index,
			    item.selected_coefficient_slot,
			    item.logical_coefficient_id,
			    item.physical_coefficient_column_id,
			    item.output_coefficient_id,
			    item.output_grid_tensor,
			    item.weight});
		}
	}
	if (projection_items.empty()) {
		stats.projection_item_build_ms += elapsed_ms(build_start, Clock::now());
		return;
	}
	stats.projection_item_build_ms += elapsed_ms(build_start, Clock::now());

	constexpr unsigned kThreads = 256;
	scratch.column_bindings.upload(column_bindings.data(), column_bindings.size(), stream, stats);
	scratch.batch_projection_items.upload(projection_items.data(), projection_items.size(), stream, stats);
	const dim3 block(kThreads);
	const dim3 grid(static_cast<unsigned>((projection_items.size() + kThreads - 1U) / kThreads));
	project_dct_ycbcr_grid_batch_kernel<<<grid, block, 0, stream>>>(
	    scratch.column_bindings.data,
	    scratch.batch_projection_items.data,
	    projection_items.size(),
	    y_output,
	    cbcr_output,
	    y_accum,
	    cbcr_accum);
	CUDA_SAFE_CALL(cudaGetLastError());
	++stats.materialize_kernel_launch_count;
	++stats.project_decoded_ycbcr_grid_launch_count;
	stats.projection_item_count += projection_items.size();
	stats.decoded_projection_item_count += projection_items.size();
	stats.jpeg_dct_projection_items_materialized += projection_items.size();
}

void materialize_dense_rowgroup_batch(const std::vector<JpegDctDeviceMaterializeBatchItem>& items,
                                      JpegDctDeviceScratch&                                 scratch,
                                      JpegDctDeviceExecutionStats&                          stats,
                                      cudaStream_t                                          stream) {
	if (items.empty()) {
		return;
	}
	// Reuses the column pointer/source scratch uploaded for the preceding batch projection on this stream.
	constexpr unsigned kThreads = 256;
	scratch.batch_materialize_items.upload(items.data(), items.size(), stream, stats);
	uint32_t max_rows = 0;
	for (const auto& item : items) {
		max_rows = std::max(max_rows, item.row_count);
	}
	if (max_rows == 0) {
		return;
	}
	const size_t total = static_cast<size_t>(max_rows) * 64U;
	const dim3   block(kThreads);
	const dim3   grid(static_cast<unsigned>((total + kThreads - 1U) / kThreads), static_cast<unsigned>(items.size()));
	materialize_dense_dct_rowgroup_batch_kernel<<<grid, block, 0, stream>>>(
	    scratch.column_bindings.data, scratch.batch_materialize_items.data, items.size());
	CUDA_SAFE_CALL(cudaGetLastError());
	++stats.materialize_kernel_launch_count;
}

void round_fixed_ycbcr_grid_outputs(float*                          y_accum,
                                    float*                          cbcr_accum,
                                    int16_t*                        y_output,
                                    int16_t*                        cbcr_output,
                                    const size_t                    y_count,
                                    const size_t                    cbcr_count,
                                    JpegDctDeviceScratch&           scratch,
                                    JpegDctDeviceExecutionStats&    stats,
                                    galp::memory::CudaEvent&        timing_start_event,
                                    galp::memory::CudaEvent&        completion_event,
                                    const JpegDctGridTransformSpec& transform) {
	constexpr unsigned kThreads     = 256;
	const bool         float_output = transform.output_data_type == JpegDctGridOutputDataType::kFloat32;
	const bool         launch_y     = y_accum != nullptr && y_count != 0 && (float_output || y_output != nullptr);
	const bool launch_cbcr = cbcr_accum != nullptr && cbcr_count != 0 && (float_output || cbcr_output != nullptr);
	if (!launch_y && !launch_cbcr) {
		return;
	}
	const size_t       active_y_count    = launch_y ? y_count : 0U;
	const size_t       active_cbcr_count = launch_cbcr ? cbcr_count : 0U;
	const size_t       total_count       = active_y_count + active_cbcr_count;
	const cudaStream_t stream            = scratch.stream_for_fixed_grid_rounding();
	timing_start_event.create();
	completion_event.create();
	timing_start_event.record(stream);
	const dim3 block(kThreads);
	const dim3 grid(static_cast<unsigned>((total_count + kThreads - 1U) / kThreads));
	if (float_output) {
		round_affine_dct_grid_accum_pair_kernel<<<grid, block, 0, stream>>>(
		    y_accum, active_y_count, cbcr_accum, active_cbcr_count, transform.output_add, transform.output_scale);
	} else {
		round_dct_grid_accum_pair_kernel<<<grid, block, 0, stream>>>(
		    y_accum, active_y_count, cbcr_accum, active_cbcr_count, y_output, cbcr_output);
	}
	CUDA_SAFE_CALL(cudaGetLastError());
	++stats.materialize_kernel_launch_count;
	++stats.fixed_grid_finalize_kernel_launch_count;
	completion_event.record(stream);
	++stats.fixed_grid_round_event_handoff_count;
}

size_t decoded_rowgroup_bytes(const size_t rows) {
	return rows * 64U * sizeof(int16_t);
}

void refresh_runtime_policy_summary(JpegDctDeviceExecutionStats& stats) {
	if (stats.runtime_policy_selected_rowgroups == 0 && stats.runtime_policy_full_rowgroups == 0) {
		stats.runtime_policy_decision = "none";
	} else if (stats.runtime_policy_selected_rowgroups != 0 && stats.runtime_policy_full_rowgroups != 0) {
		stats.runtime_policy_decision = "mixed";
	} else if (stats.runtime_policy_selected_rowgroups != 0) {
		stats.runtime_policy_decision = "selected-vector";
	} else {
		stats.runtime_policy_decision = "full-rowgroup";
	}
	stats.runtime_policy_reason = "selected=" + std::to_string(stats.runtime_policy_selected_rowgroups) +
	                              ",full=" + std::to_string(stats.runtime_policy_full_rowgroups) +
	                              ",tail_full=" + std::to_string(stats.runtime_policy_tail_full_rowgroups) +
	                              ",ratio_full=" + std::to_string(stats.runtime_policy_ratio_full_rowgroups) +
	                              ",low_saving_full=" + std::to_string(stats.runtime_policy_low_saving_full_rowgroups) +
	                              ",forced_full=" + std::to_string(stats.runtime_policy_forced_full_rowgroups) +
	                              ",forced_selected=" + std::to_string(stats.runtime_policy_forced_selected_rowgroups) +
	                              ",max_selected_ratio=" + std::to_string(kMaxSelectedVectorRatioForPushdown) +
	                              ",min_saved_vectors=" + std::to_string(kMinSavedVectorsForPushdown);
}

void record_runtime_policy(const JpegDctRuntimePolicyResult policy, JpegDctDeviceExecutionStats& stats) {
	if (policy.decision == JpegDctRuntimePolicyDecision::kSelectedVectors) {
		++stats.runtime_policy_selected_rowgroups;
	} else {
		++stats.runtime_policy_full_rowgroups;
	}
	switch (policy.reason) {
	case JpegDctRuntimePolicyReason::kCropSavesEnoughVectors:
		break;
	case JpegDctRuntimePolicyReason::kTailChunkWouldOverrun:
		++stats.runtime_policy_tail_full_rowgroups;
		break;
	case JpegDctRuntimePolicyReason::kSelectedCoversMostVectors:
		++stats.runtime_policy_ratio_full_rowgroups;
		break;
	case JpegDctRuntimePolicyReason::kSavingsTooSmall:
		++stats.runtime_policy_low_saving_full_rowgroups;
		break;
	case JpegDctRuntimePolicyReason::kForcedFullRowgroup:
		++stats.runtime_policy_forced_full_rowgroups;
		break;
	case JpegDctRuntimePolicyReason::kForcedSelectedVectors:
		++stats.runtime_policy_forced_selected_rowgroups;
		break;
	}
}

DecodedRowgroupWork
prepare_decoded_rowgroup_work_from_materialized(galp::execution::Rowgroup               rowgroup,
                                                uint32_t                                shard_id,
                                                const JpegDctDeviceRowgroupPlan&        rowgroup_plan,
                                                const std::vector<uint8_t>&             selected_coefficients,
                                                const JpegDctCoefficientSelectionShape& selection_shape,
                                                bool                                    force_projection,
                                                unsigned                                decode_unpack_n_vectors,
                                                JpegDctDeviceDecodedRowgroupCache*      cache,
                                                JpegDctDeviceExecutionStats&            execution_stats);

struct DecodedRowgroupReadResult {
	galp::execution::Rowgroup         rowgroup {};
	galp::format::ZeroCopyReadTiming io_timing {};
};

galp::format::ZeroCopyRowgroup read_full_rowgroup_zero_copy(
    galp::format::FlsReader& rdr,
    const size_t             rowgroup_index,
    const bool               use_pinned_backing,
    galp::format::ZeroCopyReadTiming* const timing) {
	if (!use_pinned_backing) {
		return rdr.read_rowgroup_zero_copy(rowgroup_index, timing);
	}

	const size_t storage_bytes = rdr.rowgroup_storage_bytes(rowgroup_index);
	void*        pinned_raw    = nullptr;
	try {
		pinned_raw = galp::memory::DevicePool::instance().alloc_pinned(storage_bytes);
	} catch (const galp::memory::CudaError& error) {
		if (error.code() != cudaErrorMemoryAllocation && error.code() != cudaErrorNotSupported &&
		    error.code() != cudaErrorNoDevice && error.code() != cudaErrorInsufficientDriver &&
		    error.code() != cudaErrorInitializationError && error.code() != cudaErrorSystemDriverMismatch) {
			throw;
		}
		return rdr.read_rowgroup_zero_copy(rowgroup_index, timing);
	}
	if (pinned_raw == nullptr) {
		throw std::runtime_error("JPEG DCT pinned rowgroup allocation returned null");
	}
	auto pinned_owner = std::shared_ptr<void>(pinned_raw, [](void* ptr) {
		if (ptr == nullptr) {
			return;
		}
		try {
			galp::memory::DevicePool::instance().release_pinned(ptr);
		} catch (const std::exception& e) {
			std::fprintf(stderr, "JPEG DCT pinned rowgroup release failed: %s\n", e.what());
		}
	});
	return rdr.read_rowgroup_zero_copy_into(rowgroup_index,
	                                        std::move(pinned_owner),
	                                        reinterpret_cast<std::byte*>(pinned_raw),
	                                        storage_bytes,
	                                        /*backing_is_pinned=*/true,
	                                        timing);
}

DecodedRowgroupReadResult read_decoded_rowgroup(galp::format::FlsReader&         rdr,
                                                const JpegDctDeviceRowgroupPlan& rowgroup_plan,
	                                            const unsigned                   decode_unpack_n_vectors,
	                                            const bool                       use_pinned_backing,
	                                            const galp::format::SparseVectorReadPlan* compiled_sparse_plan) {
	DecodedRowgroupReadResult result {};
	galp::format::ZeroCopyRowgroup zero_copy {};
	const bool can_issue_sparse_read =
	    rowgroup_plan.has_vector_plan && rowgroup_plan.sparse_storage_read &&
	    rowgroup_plan.runtime_policy.decision == JpegDctRuntimePolicyDecision::kSelectedVectors &&
	    decode_unpack_n_vectors == kJpegDctDeviceUnpackNVectors && !rowgroup_plan.selected_vectors.empty();
	if (can_issue_sparse_read) {
		if (compiled_sparse_plan != nullptr) {
			zero_copy = rdr.read_rowgroup_zero_copy_compiled(*compiled_sparse_plan, &result.io_timing);
		} else {
			auto physical_vectors = expand_selected_decode_chunks(
			    rowgroup_plan.selected_vectors, rowgroup_plan.full_vector_count, decode_unpack_n_vectors);
			const bool use_packed_device_scatter =
			    std::getenv("GALP_VECTOR_BUNDLE_DEVICE_SCATTER") != nullptr;
			zero_copy = use_packed_device_scatter
			                ? rdr.read_rowgroup_zero_copy_selected_vectors_packed(
			                      rowgroup_plan.rowgroup_index, physical_vectors, &result.io_timing)
			                : rdr.read_rowgroup_zero_copy_selected_vectors(
			                      rowgroup_plan.rowgroup_index, physical_vectors, &result.io_timing);
		}
	} else {
		if (compiled_sparse_plan != nullptr) {
			throw std::invalid_argument("compiled sparse read plan supplied for a full-rowgroup read");
		}
		zero_copy = read_full_rowgroup_zero_copy(
		    rdr, rowgroup_plan.rowgroup_index, use_pinned_backing, &result.io_timing);
	}
	result.rowgroup = rdr.materialize_zero_copy_rowgroup(std::move(zero_copy));
	return result;
}

DecodedRowgroupReadResult take_staged_or_read_decoded_rowgroup(
	galp::format::FlsReader&         rdr,
	const JpegDctDeviceRowgroupPlan& rowgroup_plan,
	const unsigned                   decode_unpack_n_vectors) {
	if (!rowgroup_plan.staged_read) {
		return read_decoded_rowgroup(rdr,
		                             rowgroup_plan,
		                             decode_unpack_n_vectors,
		                             /*use_pinned_backing=*/false,
		                             /*compiled_sparse_plan=*/nullptr);
	}
	if (rowgroup_plan.staged_read->consumed.exchange(true, std::memory_order_acq_rel)) {
		throw std::runtime_error("JPEG DCT staged rowgroup was consumed more than once");
	}
	DecodedRowgroupReadResult result;
	result.rowgroup   = std::move(rowgroup_plan.staged_read->rowgroup);
	result.io_timing = std::move(rowgroup_plan.staged_read->io_timing);
	return result;
}

void compile_jpeg_dct_device_batch_io_impl(JpegDctDeviceBatchPlan& plan, JpegDctHostIoContext& context) {
	if (plan.cache_enabled || !plan.uses_planless_fixed_transform) {
		return;
	}
	unsigned batch_unpack_n_vectors = kJpegDctDeviceUnpackNVectors;
	for (const auto& shard : *plan.shards) {
		for (const auto& rowgroup : shard.rowgroups) {
			batch_unpack_n_vectors =
			    constrain_jpeg_dct_batch_unpack_n_vectors(batch_unpack_n_vectors, rowgroup);
		}
	}
	const bool packed_device_scatter = std::getenv("GALP_VECTOR_BUNDLE_DEVICE_SCATTER") != nullptr;
	for (auto& shard : *plan.shards) {
		for (auto& rowgroup : shard.rowgroups) {
			const auto* source_path = rowgroup.source_fls_path == nullptr ? shard.fls_path : rowgroup.source_fls_path;
			if (source_path == nullptr) {
				throw std::runtime_error("JPEG DCT compiled I/O plan has no source path");
			}
			const bool can_compile_sparse_read =
			    rowgroup.has_vector_plan &&
			    (rowgroup.sparse_storage_read || rowgroup.automatic_sparse_storage_candidate) &&
			    rowgroup.runtime_policy.decision == JpegDctRuntimePolicyDecision::kSelectedVectors &&
			    batch_unpack_n_vectors == kJpegDctDeviceUnpackNVectors && !rowgroup.selected_vectors.empty();
			if (!rowgroup.prepared_reader) {
				rowgroup.prepared_reader = context.reader(*source_path, can_compile_sparse_read);
			}
			if (can_compile_sparse_read && !rowgroup.compiled_sparse_read_plan) {
				auto physical_vectors = expand_selected_decode_chunks(
				    rowgroup.selected_vectors, rowgroup.full_vector_count, batch_unpack_n_vectors);
				auto candidate = context.sparse_plan(*source_path,
				                                     rowgroup.prepared_reader,
				                                     rowgroup.rowgroup_index,
				                                     std::move(physical_vectors),
				                                     packed_device_scatter);
				if (rowgroup.automatic_sparse_storage_candidate) {
					const auto storage_policy = choose_jpeg_dct_sparse_storage_policy(
					    {candidate->full_storage_bytes(),
					     candidate->storage_bytes(),
					     candidate->estimated_pread_count(),
					     candidate->uses_sparse_read()});
					++plan.automatic_sparse_storage_candidate_rowgroup_count;
					plan.automatic_sparse_storage_full_bytes += candidate->full_storage_bytes();
					plan.automatic_sparse_storage_candidate_bytes += candidate->storage_bytes();
					plan.automatic_sparse_storage_candidate_pread_count += candidate->estimated_pread_count();
					plan.automatic_sparse_storage_full_estimated_ns += storage_policy.full_estimated_ns;
					plan.automatic_sparse_storage_candidate_estimated_ns += storage_policy.sparse_estimated_ns;
					rowgroup.sparse_storage_read = candidate->uses_sparse_read() && storage_policy.use_sparse_read;
					if (rowgroup.sparse_storage_read) {
						++plan.automatic_sparse_storage_selected_rowgroup_count;
					} else {
						++plan.automatic_sparse_storage_rejected_rowgroup_count;
					}
				}
				if (rowgroup.sparse_storage_read) {
					rowgroup.compiled_sparse_read_plan = std::move(candidate);
				}
			}
		}
	}
}

void stage_jpeg_dct_device_batch_io_impl(JpegDctDeviceBatchPlan& plan, JpegDctHostIoContext& context) {
	if (plan.host_io_staged) {
		return;
	}
	// Cache hit classification mutates the device-resident decoded-rowgroup
	// cache and therefore remains part of ordered submission. The compact
	// planless path deliberately disables that cache, so all of its storage I/O
	// can be completed before entering the CUDA submission queue.
	if (plan.cache_enabled || !plan.uses_planless_fixed_transform) {
		plan.host_io_staged = true;
		return;
	}
	const auto staging_start = Clock::now();

	struct ReadTask {
		JpegDctDeviceRowgroupPlan* rowgroup = nullptr;
		uint32_t                   shard_id = std::numeric_limits<uint32_t>::max();
		std::filesystem::path      path;
	};
	std::vector<ReadTask> tasks;
	for (auto& shard : *plan.shards) {
		for (auto& rowgroup : shard.rowgroups) {
			const auto source_shard_id = rowgroup.source_shard_id == std::numeric_limits<uint32_t>::max()
			                                 ? shard.shard_id
			                                 : rowgroup.source_shard_id;
			const auto* source_path = rowgroup.source_fls_path == nullptr ? shard.fls_path : rowgroup.source_fls_path;
			if (source_path == nullptr) {
				throw std::runtime_error("JPEG DCT staged I/O plan has no source path");
			}
			tasks.push_back(ReadTask {&rowgroup, source_shard_id, *source_path});
		}
	}
	if (tasks.empty()) {
		plan.host_io_staged = true;
		plan.host_io_staging_ms = elapsed_ms(staging_start, Clock::now());
		return;
	}
	unsigned batch_unpack_n_vectors = kJpegDctDeviceUnpackNVectors;
	for (const auto& task : tasks) {
		batch_unpack_n_vectors =
		    constrain_jpeg_dct_batch_unpack_n_vectors(batch_unpack_n_vectors, *task.rowgroup);
	}
	std::vector<std::shared_ptr<galp::format::FlsReader>> readers(tasks.size());
	std::vector<std::shared_ptr<JpegDctStagedRowgroupRead>> staged(tasks.size());
	std::vector<std::exception_ptr> errors(tasks.size());
	std::atomic<size_t> next_task {0U};
	const size_t worker_count = std::min(
	    tasks.size(), std::max<size_t>(1U, plan.rowgroup_prefetch.workers));
	context.worker_pool.run(worker_count, [&]() {
			while (true) {
				const auto task_index = next_task.fetch_add(1U, std::memory_order_relaxed);
				if (task_index >= tasks.size()) {
					return;
				}
				try {
					const auto& rowgroup_plan = *tasks[task_index].rowgroup;
					auto compiled_sparse_plan = rowgroup_plan.compiled_sparse_read_plan;
					const bool can_issue_sparse_read =
					    rowgroup_plan.has_vector_plan && rowgroup_plan.sparse_storage_read &&
					    rowgroup_plan.runtime_policy.decision == JpegDctRuntimePolicyDecision::kSelectedVectors &&
					    batch_unpack_n_vectors == kJpegDctDeviceUnpackNVectors &&
					    !rowgroup_plan.selected_vectors.empty();
					readers[task_index] = rowgroup_plan.prepared_reader
					                          ? rowgroup_plan.prepared_reader
					                          : context.reader(tasks[task_index].path, can_issue_sparse_read);
					if (can_issue_sparse_read && !compiled_sparse_plan) {
						auto physical_vectors = expand_selected_decode_chunks(rowgroup_plan.selected_vectors,
						                                                      rowgroup_plan.full_vector_count,
						                                                      batch_unpack_n_vectors);
						const bool packed_device_scatter =
						    std::getenv("GALP_VECTOR_BUNDLE_DEVICE_SCATTER") != nullptr;
						compiled_sparse_plan = context.sparse_plan(tasks[task_index].path,
						                                                   readers[task_index],
						                                                   rowgroup_plan.rowgroup_index,
						                                                   std::move(physical_vectors),
						                                                   packed_device_scatter);
					}
					auto result = read_decoded_rowgroup(*readers[task_index],
					                                    rowgroup_plan,
					                                    batch_unpack_n_vectors,
					                                    /*use_pinned_backing=*/true,
					                                    compiled_sparse_plan.get());
					auto entry       = std::make_shared<JpegDctStagedRowgroupRead>();
					entry->rowgroup   = std::move(result.rowgroup);
					entry->io_timing  = std::move(result.io_timing);
					entry->reader     = readers[task_index];
					staged[task_index] = std::move(entry);
				} catch (...) {
					errors[task_index] = std::current_exception();
				}
			}
		});
	for (const auto& error : errors) {
		if (error) {
			std::rethrow_exception(error);
		}
	}
	for (size_t task_index = 0; task_index < tasks.size(); ++task_index) {
		tasks[task_index].rowgroup->staged_read = std::move(staged[task_index]);
	}
	plan.host_io_staged           = true;
	plan.host_io_staged_rowgroups = tasks.size();
	plan.host_io_staging_ms       = elapsed_ms(staging_start, Clock::now());
}

void record_decoded_rowgroup_read(const galp::format::ZeroCopyReadTiming& io_timing,
	                              JpegDctDeviceExecutionStats&             execution_stats) {
	execution_stats.rowgroup_storage_bytes_read += io_timing.storage_bytes;
	execution_stats.compressed_payload_bytes_read += io_timing.storage_bytes;
	execution_stats.full_compressed_payload_bytes += io_timing.full_storage_bytes;
	execution_stats.pread_count += io_timing.pread_count;
	if (io_timing.used_pinned_backing) {
		++execution_stats.pinned_rowgroup_read_count;
		execution_stats.pinned_rowgroup_read_bytes += io_timing.storage_bytes;
	}
	if (io_timing.used_vector_bundle_read) {
		++execution_stats.vector_bundle_rowgroup_count;
		execution_stats.vector_bundle_pread_count += io_timing.pread_count;
		execution_stats.vector_bundle_envelope_rowgroup_count +=
		    io_timing.used_vector_bundle_envelope_read ? 1U : 0U;
	}
	execution_stats.sparse_read_supported = execution_stats.sparse_read_supported || io_timing.sparse_read_supported;
	if (!io_timing.sparse_fallback_reason.empty()) {
		++execution_stats.sparse_read_fallback_rowgroup_count;
		execution_stats.sparse_read_fallback_reason = io_timing.sparse_fallback_reason;
	}
}

DecodedRowgroupWork prepare_decoded_rowgroup_work(galp::format::FlsReader&                rdr,
                                                  const uint32_t                          shard_id,
                                                  const JpegDctDeviceRowgroupPlan&        rowgroup_plan,
                                                  const std::vector<uint8_t>&             selected_coefficients,
                                                  const JpegDctCoefficientSelectionShape& selection_shape,
                                                  const bool                              force_projection,
                                                  const unsigned                          decode_unpack_n_vectors,
                                                  JpegDctDeviceDecodedRowgroupCache*      cache,
                                                  JpegDctDeviceExecutionStats&            execution_stats) {
	const auto sync_read_start = Clock::now();
	auto       read_result = take_staged_or_read_decoded_rowgroup(rdr, rowgroup_plan, decode_unpack_n_vectors);
	const auto                       sync_read_end = Clock::now();
	execution_stats.sync_rowgroup_read_ms += elapsed_ms(sync_read_start, sync_read_end);
	record_decoded_rowgroup_read(read_result.io_timing, execution_stats);

	return prepare_decoded_rowgroup_work_from_materialized(std::move(read_result.rowgroup),
	                                                       shard_id,
	                                                       rowgroup_plan,
	                                                       selected_coefficients,
	                                                       selection_shape,
	                                                       force_projection,
	                                                       decode_unpack_n_vectors,
	                                                       cache,
	                                                       execution_stats);
}

DecodedRowgroupWork
prepare_decoded_rowgroup_work_from_materialized(galp::execution::Rowgroup               rowgroup,
                                                const uint32_t                          shard_id,
                                                const JpegDctDeviceRowgroupPlan&        rowgroup_plan,
                                                const std::vector<uint8_t>&             selected_coefficients,
                                                const JpegDctCoefficientSelectionShape& selection_shape,
                                                const bool                              force_projection,
                                                const unsigned                          decode_unpack_n_vectors,
                                                JpegDctDeviceDecodedRowgroupCache*      cache,
                                                JpegDctDeviceExecutionStats&            execution_stats) {
	DecodedRowgroupWork work;
	work.shard_id       = shard_id;
	work.rowgroup_index = rowgroup_plan.rowgroup_index;
	work.cache_key      = JpegDctDeviceDecodedRowgroupCacheKey {shard_id, rowgroup_plan.rowgroup_index};
	work.rowgroup       = std::move(rowgroup);
	work.logical_rowgroup_n_vecs = work.rowgroup.n_vecs;
	work.owns_rowgroup  = true;
	// JPEG's projection/selected-vector path appends columns directly instead
	// of using append_expressions(), so resolve external dictionaries here.
	// This keeps DICTREF strictly host-side and guarantees that cache misses,
	// selected-vector decodes, and full-rowgroup decodes all upload local plans.
	{
		auto expressions = galp::expression::assemble(work.rowgroup);
		galp::execution::resolve_dict_refs(expressions);
	}
	if (work.rowgroup.columns.size() < 64) {
		throw std::runtime_error("JPEG DCT FLS rowgroup has fewer than 64 coefficient columns");
	}

	galp::execution::ExecutionConfig cfg;
	cfg.write_out                                    = true;
	cfg.unpack_n_vectors                             = std::max(1U, decode_unpack_n_vectors);
	work.decode_unpack_n_vectors                     = cfg.unpack_n_vectors;
	size_t                     selected_vector_count = 0;
	JpegDctRuntimePolicyResult policy {};
	const bool                 can_reuse_vector_plan = rowgroup_plan.has_vector_plan &&
	                                   rowgroup_plan.full_vector_count == work.rowgroup.n_vecs &&
	                                   cfg.unpack_n_vectors == kJpegDctDeviceUnpackNVectors;
	if (can_reuse_vector_plan) {
		work.selected_vectors = &rowgroup_plan.selected_vectors;
		selected_vector_count = rowgroup_plan.selected_vector_count;
		policy                = rowgroup_plan.runtime_policy;
	} else {
		work.owned_selected_vectors =
		    selected_decode_vectors(rowgroup_plan.items, work.rowgroup.n_vecs, cfg.unpack_n_vectors);
		work.selected_vectors = &work.owned_selected_vectors;
		const bool selected_chunks_fit =
		    selected_decode_chunks_fit(*work.selected_vectors, work.rowgroup.n_vecs, cfg.unpack_n_vectors);
		selected_vector_count =
		    selected_decode_vector_count(*work.selected_vectors, work.rowgroup.n_vecs, cfg.unpack_n_vectors);
		policy = choose_jpeg_dct_runtime_policy(selected_vector_count, work.rowgroup.n_vecs, selected_chunks_fit);
	}
	record_runtime_policy(policy, execution_stats);
	work.decodes_full_rowgroup = policy.decision == JpegDctRuntimePolicyDecision::kFullRowgroup;
	if (work.decodes_full_rowgroup && work.rowgroup.n_vecs % cfg.unpack_n_vectors != 0U) {
		throw std::runtime_error("JPEG DCT batch unpack width is incompatible with a full rowgroup");
	}
	if (work.decodes_full_rowgroup) {
		work.gather_items = &rowgroup_plan.items;
	} else if (can_reuse_vector_plan && !rowgroup_plan.selected_gather_items.empty()) {
		work.gather_items = &rowgroup_plan.selected_gather_items;
	} else {
		work.owned_gather_items =
		    remap_items_to_selected_vectors(rowgroup_plan.items, *work.selected_vectors, cfg.unpack_n_vectors);
		work.gather_items = &work.owned_gather_items;
	}
	if (!rowgroup_plan.planless_images.empty()) {
		work.active_physical_coefficients = selected_coefficients;
		work.planless_images              = &rowgroup_plan.planless_images;
	} else if (!rowgroup_plan.fixed_transform_items.empty()) {
		work.active_physical_coefficients = selected_coefficients;
		if (work.decodes_full_rowgroup) {
			work.fixed_transform_items = &rowgroup_plan.fixed_transform_items;
		} else if (can_reuse_vector_plan && !rowgroup_plan.selected_fixed_transform_items.empty()) {
			work.fixed_transform_items = &rowgroup_plan.selected_fixed_transform_items;
		} else {
			work.owned_fixed_transform_items = remap_fixed_transform_items_to_selected_vectors(
			    rowgroup_plan.fixed_transform_items, *work.selected_vectors, cfg.unpack_n_vectors);
			work.fixed_transform_items = &work.owned_fixed_transform_items;
		}
	} else if (selection_shape.kind == JpegDctCoefficientSelectionKind::kAll && !force_projection) {
		work.active_physical_coefficients = selected_coefficients;
	} else {
		const std::vector<JpegDctDeviceProjectionItem>* logical_projection_items = nullptr;
		if (work.decodes_full_rowgroup) {
			logical_projection_items = &rowgroup_plan.projection_items;
		} else if (can_reuse_vector_plan && !rowgroup_plan.selected_projection_items.empty()) {
			logical_projection_items = &rowgroup_plan.selected_projection_items;
		} else {
			work.owned_projection_items = remap_projection_items_to_selected_vectors(
			    rowgroup_plan.projection_items, *work.selected_vectors, cfg.unpack_n_vectors);
			logical_projection_items = &work.owned_projection_items;
		}
		if (logical_projection_items == nullptr || logical_projection_items->empty()) {
			throw std::runtime_error("JPEG DCT rowgroup projection plan is missing");
		}
		auto resolved_projection    = resolve_projection_physical_columns(work.rowgroup, *logical_projection_items);
		work.owned_projection_items = std::move(resolved_projection.items);
		work.projection_items       = &work.owned_projection_items;
		work.active_physical_coefficients = std::move(resolved_projection.active_physical_coefficients);
	}
	const size_t actual_selected_vector_count =
	    work.decodes_full_rowgroup ? work.rowgroup.n_vecs : selected_vector_count;
	execution_stats.selected_vector_count += actual_selected_vector_count;
	if (!work.decodes_full_rowgroup && work.rowgroup.packed_device_payload != nullptr &&
	    std::getenv("GALP_VECTOR_BUNDLE_COMPACT_SELECTED") != nullptr) {
		galp::runtime::compact_selected_vectors(work.rowgroup, *work.selected_vectors);
		work.owned_decode_vectors.resize(work.rowgroup.n_vecs);
		std::iota(work.owned_decode_vectors.begin(), work.owned_decode_vectors.end(), uint32_t {0});
		work.decode_vectors = &work.owned_decode_vectors;
	}
	if (cache != nullptr && cache->capacity > 0 && work.decodes_full_rowgroup) {
		const auto rowgroup_bytes = decoded_rowgroup_bytes(work.rowgroup.n_tuples);
		if (rowgroup_bytes <= cache->capacity && work.rowgroup.n_tuples <= std::numeric_limits<uint32_t>::max()) {
			work.cache_entry              = std::make_unique<JpegDctDeviceDecodedRowgroupCacheEntry>();
			work.cache_entry->rows        = static_cast<uint32_t>(work.rowgroup.n_tuples);
			work.cache_entry->bytes       = rowgroup_bytes;
			work.cache_entry->last_access = ++cache->clock;
			work.cache_entry->blocks.emplace(work.rowgroup.n_tuples * 64U);
		}
	}
	return work;
}

void finish_cached_gather_after_wait(JpegDctDeviceScratch& scratch, JpegDctDeviceExecutionStats& execution_stats) {
	if (!scratch.cached_gather_in_flight) {
		return;
	}
	const auto elapsed_ms = static_cast<double>(scratch.cached_gather_done.elapsed_since(scratch.cached_gather_start));
	if (scratch.cached_fixed_transform_in_flight) {
		execution_stats.fixed_transform_ms += elapsed_ms;
	} else {
		execution_stats.cached_gather_ms += elapsed_ms;
		execution_stats.gather_ms += elapsed_ms;
	}
	if (scratch.cache_hit_stream) {
		galp::memory::complete_h2d(scratch.cache_hit_stream.get());
	}
	scratch.host_cached_gather_uploads.clear();
	scratch.host_cached_fixed_transform_uploads.clear();
	scratch.cached_gather_in_flight          = false;
	scratch.cached_fixed_transform_in_flight = false;
}

void drain_cached_gather(JpegDctDeviceScratch& scratch, JpegDctDeviceExecutionStats& execution_stats) {
	if (!scratch.cached_gather_in_flight) {
		return;
	}
	// Drain only when cached-gather scratch must be reused or a legacy default-stream caller needs host ordering.
	scratch.cached_gather_done.synchronize();
	++execution_stats.internal_sync_count;
	++execution_stats.cached_gather_sync_count;
	finish_cached_gather_after_wait(scratch, execution_stats);
}

void make_stream_wait_for_cached_gather(cudaStream_t                 stream,
                                        JpegDctDeviceScratch&        scratch,
                                        JpegDctDeviceExecutionStats& execution_stats) {
	if (!scratch.cached_gather_in_flight) {
		return;
	}
	if (stream == nullptr) {
		drain_cached_gather(scratch, execution_stats);
		return;
	}
	CUDA_SAFE_CALL(cudaStreamWaitEvent(stream, scratch.cached_gather_done.get(), 0));
	++execution_stats.cached_gather_event_handoff_count;
}

void finish_workset_run_after_stream_tail_wait(galp::runtime::AsyncWorksetRun& run) {
	if (!run.active) {
		return;
	}
	if (run.queued != nullptr && run.start != nullptr) {
		run.pre_kernel_event_ms = static_cast<double>(run.start->elapsed_since(*run.queued));
	}
	if (run.start != nullptr && run.stop != nullptr) {
		run.elapsed_ms = static_cast<double>(run.stop->elapsed_since(*run.start));
	}
	run.queued = nullptr;
	run.start  = nullptr;
	run.stop   = nullptr;
	run.active = false;
}

void release_completed_workset(galp::runtime::ExecutionWorkset&      workset,
                               galp::runtime::ExecutionWorksetGuard& guard,
                               const bool                            preserve_resources) {
	galp::runtime::release_workset(workset, preserve_resources, true);
	guard.dismiss();
}

void execute_decoded_rowgroup_batch(std::vector<DecodedRowgroupWork>&       works,
                                    const std::vector<uint32_t>*            fixed_transform_item_order,
                                    const std::vector<uint32_t>*            fixed_transform_group_offsets,
                                    int16_t*                                output,
                                    int16_t*                                y_output,
                                    int16_t*                                cbcr_output,
                                    float*                                  y_accum,
                                    float*                                  cbcr_accum,
                                    const uint16_t*                         fixed_quant_tables,
                                    const float*                            fixed_resize_weight_matrices,
                                    const JpegDctGridTransformSpec&         grid_transform,
                                    const std::vector<uint8_t>&             selected_coefficients,
                                    const JpegDctCoefficientSelectionShape& selection_shape,
                                    const bool                              output_ycbcr_dct_grid,
                                    const bool                              output_transformed_dct_grid,
                                    JpegDctDeviceDecodedRowgroupCache*      cache,
                                    JpegDctDeviceCacheStats&                batch_cache_stats,
                                    JpegDctDeviceExecutionStats&            execution_stats,
                                    JpegDctDeviceScratch&                   scratch) {
	if (works.empty()) {
		return;
	}

	galp::execution::ExecutionConfig cfg;
	cfg.write_out        = true;
	cfg.unpack_n_vectors = works.front().decode_unpack_n_vectors;
	for (const auto& work : works) {
		if (work.decode_unpack_n_vectors != cfg.unpack_n_vectors) {
			throw std::runtime_error("JPEG DCT decode batch mixed incompatible unpack widths");
		}
	}
	auto&                                workset = scratch.decode_workset;
	galp::runtime::ExecutionWorksetGuard guard(workset);
	const auto                           build_start = Clock::now();
	galp::runtime::reserve_batch_expr_storage(workset, works.size() * kJpegDctCoefficientCount);
	workset.buffers.pending_device_scatters.reserve(works.size());
	size_t packed_scatter_range_count = 0U;
	for (const auto& work : works) {
		if (work.rowgroup.packed_device_payload) {
			packed_scatter_range_count += work.rowgroup.packed_device_payload->ranges.size();
		}
	}
	workset.buffers.device_scatter_copies.reserve(packed_scatter_range_count);
	for (size_t idx = 0; idx < works.size(); ++idx) {
		auto& work                          = works[idx];
		work.expr_index_base                = idx * 64U;
		const auto physical_selection_shape = classify_coefficient_selection(work.active_physical_coefficients);
		append_jpeg_rowgroup_columns(workset,
		                             work.rowgroup,
		                             cfg,
		                             work.expr_index_base,
		                             work.active_physical_coefficients,
		                             physical_selection_shape,
		                             work.decodes_full_rowgroup
		                                 ? nullptr
		                                 : (work.decode_vectors != nullptr ? work.decode_vectors : work.selected_vectors));
	}
	const auto build_end = Clock::now();
	execution_stats.workset_build_ms += elapsed_ms(build_start, build_end);
	const auto upload_start     = Clock::now();
	const auto upload_breakdown = galp::runtime::upload_workset(workset, cfg);
	const auto upload_end       = Clock::now();
	execution_stats.workset_upload_ms += elapsed_ms(upload_start, upload_end);
	execution_stats.workset_upload_prep_ms += upload_breakdown.prep_ms;
	execution_stats.workset_upload_arena_ms += upload_breakdown.arena_upload_ms;
	execution_stats.workset_upload_arena_pack_ms += upload_breakdown.arena_pack_ms;
	execution_stats.workset_upload_arena_layout_ms += upload_breakdown.arena.layout_ms;
	execution_stats.workset_upload_arena_alloc_ms += upload_breakdown.arena.alloc_ms;
	execution_stats.workset_upload_arena_resolve_ms += upload_breakdown.arena.resolve_ms;
	execution_stats.workset_upload_dma_issue_ms += upload_breakdown.arena.dma_issue_ms;
	execution_stats.workset_upload_event_record_ms += upload_breakdown.event_record_ms;
	execution_stats.workset_upload_dma_bytes += upload_breakdown.arena.dma_bytes;
	execution_stats.workset_upload_dma_count += upload_breakdown.arena.dma_count;
	++execution_stats.workset_count;
	++execution_stats.workset_upload_count;
	size_t launches = 0;
	auto   run      = galp::runtime::run_workset_async(workset, 1, cfg, nullptr, &launches);
	execution_stats.decode_kernel_launch_count += launches;
	if (galp::runtime::use_async_h2d() && workset.transfer.h2d_stream && workset.transfer.h2d_ready_event) {
		++execution_stats.copy_to_decode_event_handoff_count;
	}
	const cudaStream_t decode_stream = run.stream;

	auto& sources = scratch.host_bound_sources;
	sources.clear();
	sources.reserve(works.size());
	const bool materializes_dense_cache =
	    cache != nullptr && cache->capacity > 0 && selects_all_coefficients(selected_coefficients);
	const bool uses_decoded_gather =
	    selection_shape.kind == JpegDctCoefficientSelectionKind::kAll && !output_ycbcr_dct_grid;
	const bool batch_has_expanded_fixed_transform =
	    output_transformed_dct_grid && std::any_of(works.begin(), works.end(), [](const auto& work) {
		    return work.fixed_transform_items != nullptr && !work.fixed_transform_items->empty();
	    });
	const bool batch_has_planless_fixed_transform =
	    output_transformed_dct_grid && std::any_of(works.begin(), works.end(), [](const auto& work) {
		    return work.planless_images != nullptr && !work.planless_images->empty();
	    });
	if (batch_has_expanded_fixed_transform && batch_has_planless_fixed_transform) {
		throw std::runtime_error("JPEG DCT workset mixed expanded and planless fixed transforms");
	}
	const bool batch_has_fixed_transform = batch_has_expanded_fixed_transform || batch_has_planless_fixed_transform;
	cudaStream_t materialize_stream = decode_stream;
	if (batch_has_fixed_transform) {
		scratch.decode_to_transform_event.create_with_flags(cudaEventDisableTiming);
		scratch.decode_to_transform_event.record(decode_stream);
		materialize_stream = scratch.stream_for_transform();
		CUDA_SAFE_CALL(cudaStreamWaitEvent(materialize_stream, scratch.decode_to_transform_event.get(), 0));
		++execution_stats.decode_to_transform_event_handoff_count;
	}
	make_stream_wait_for_cached_gather(materialize_stream, scratch, execution_stats);
	const bool use_dense_bindings        = materializes_dense_cache || uses_decoded_gather || batch_has_fixed_transform;
	for (size_t source_idx = 0; source_idx < works.size(); ++source_idx) {
		const auto& work = works[source_idx];
		if (use_dense_bindings) {
			sources.push_back(bind_coeff_columns(
			    work.rowgroup, workset, selected_coefficients, selection_shape, work.expr_index_base));
		} else {
			const auto physical_selection_shape = classify_coefficient_selection(work.active_physical_coefficients);
			sources.push_back(bind_coeff_columns(work.rowgroup,
			                                     workset,
			                                     work.active_physical_coefficients,
			                                     physical_selection_shape,
			                                     work.expr_index_base));
		}
	}
	if (batch_has_planless_fixed_transform) {
		project_planless_transformed_dct_grid_batch(sources,
		                                            works,
		                                            fixed_quant_tables,
		                                            fixed_resize_weight_matrices,
		                                            grid_transform,
		                                            y_accum,
		                                            cbcr_accum,
		                                            scratch,
		                                            execution_stats,
		                                            scratch.transform_blocks_per_launch,
		                                            scratch.transform_ctas_per_launch,
		                                            materialize_stream);
	} else if (batch_has_expanded_fixed_transform) {
		project_transformed_dct_grid_batch(sources,
		                                   works,
		                                   fixed_transform_item_order,
		                                   fixed_transform_group_offsets,
		                                   fixed_quant_tables,
		                                   fixed_resize_weight_matrices,
		                                   grid_transform,
		                                   y_accum,
		                                   cbcr_accum,
		                                   scratch,
		                                   execution_stats,
		                                   materialize_stream);
	} else if (uses_decoded_gather) {
		gather_decoded_rowgroup_batch(sources, works, output, scratch, execution_stats, materialize_stream);
	} else if (output_ycbcr_dct_grid) {
		project_decoded_ycbcr_grid_batch(sources,
		                                 works,
		                                 materializes_dense_cache,
		                                 y_output,
		                                 cbcr_output,
		                                 y_accum,
		                                 cbcr_accum,
		                                 scratch,
		                                 execution_stats,
		                                 materialize_stream);
	} else {
		project_decoded_rowgroup_batch(sources,
		                               works,
		                               selected_coefficients.size(),
		                               materializes_dense_cache,
		                               output,
		                               scratch,
		                               execution_stats,
		                               materialize_stream);
	}
	if (materializes_dense_cache) {
		auto& materialize_items = scratch.host_materialize_items;
		materialize_items.clear();
		materialize_items.reserve(works.size());
		for (size_t source_idx = 0; source_idx < works.size(); ++source_idx) {
			auto& work = works[source_idx];
			if (work.cache_entry) {
				if (work.rowgroup.n_tuples > std::numeric_limits<uint32_t>::max()) {
					throw std::runtime_error("JPEG DCT rowgroup is too large for device cache materialization");
				}
				const auto row_count = static_cast<uint32_t>(work.rowgroup.n_tuples);
				materialize_items.push_back(JpegDctDeviceMaterializeBatchItem {
				    static_cast<uint32_t>(source_idx), row_count, work.cache_entry->blocks->get()});
			}
		}
		materialize_dense_rowgroup_batch(materialize_items, scratch, execution_stats, materialize_stream);
	}
	scratch.ensure_decoded_batch_events();
	scratch.decoded_batch_gather_done.record(materialize_stream);

	// Rowgroup metadata, workset output arena, and scratch are reused after this batch.
	// Wait only for the projection completion event; batch-level workset ownership can remove this later.
	scratch.decoded_batch_gather_done.synchronize();
	++execution_stats.internal_sync_count;
	++execution_stats.decoded_batch_sync_count;
	if (decode_stream != nullptr) {
		galp::memory::complete_h2d(decode_stream);
	}
	if (materialize_stream != nullptr && materialize_stream != decode_stream) {
		galp::memory::complete_h2d(materialize_stream);
	}
	finish_cached_gather_after_wait(scratch, execution_stats);
	if (run.stop != nullptr) {
		const auto elapsed_ms = static_cast<double>(scratch.decoded_batch_gather_done.elapsed_since(*run.stop));
		if (uses_decoded_gather) {
			execution_stats.decoded_gather_ms += elapsed_ms;
			execution_stats.gather_ms += elapsed_ms;
		} else if (batch_has_fixed_transform) {
			execution_stats.fixed_transform_ms += elapsed_ms;
		} else {
			execution_stats.decoded_projection_ms += elapsed_ms;
			execution_stats.projection_ms += elapsed_ms;
		}
	}
	finish_workset_run_after_stream_tail_wait(run);
	execution_stats.decode_ms += run.elapsed_ms;

	if (cache != nullptr && cache->capacity > 0) {
		for (auto& work : works) {
			cache->insert_ready_entry(work.cache_key, std::move(work.cache_entry), batch_cache_stats);
		}
	}
	works.clear();
	release_completed_workset(workset, guard, /*preserve_resources=*/true);
}

void execute_cached_rowgroup_hits(const std::vector<JpegDctDeviceCachedGatherBatchItem>& items,
                                  int16_t*                                               output,
                                  JpegDctDeviceExecutionStats&                           execution_stats,
                                  JpegDctDeviceScratch&                                  scratch) {
	if (items.empty()) {
		return;
	}
	if (scratch.cached_gather_in_flight && scratch.cached_fixed_transform_in_flight) {
		drain_cached_gather(scratch, execution_stats);
	}
	// Cached gathers use one stream. If the item scratch already has capacity, the next upload is
	// stream-ordered after the previous gather and does not need a host wait. Growing the buffer may
	// free the old device pointer, so that path still drains first.
	if (scratch.cached_gather_in_flight && scratch.cached_gather_items.needs_reallocation(items.size())) {
		drain_cached_gather(scratch, execution_stats);
	}
	auto& upload_items = scratch.host_cached_gather_uploads.emplace_back(items.begin(), items.end());
	execution_stats.gather_item_count += items.size();
	execution_stats.cached_gather_item_count += items.size();
	const cudaStream_t stream = scratch.stream_for_cache_hit();
	scratch.ensure_cached_gather_events();
	if (!scratch.cached_gather_in_flight) {
		scratch.cached_gather_start.record(stream);
		scratch.cached_fixed_transform_in_flight = false;
	}
	gather_cached_rowgroups_batch(upload_items, output, scratch, execution_stats, stream);
	scratch.cached_gather_done.record(stream);
	scratch.cached_gather_in_flight = true;
}

void execute_cached_fixed_transform_hits(
    const std::vector<JpegDctDeviceCachedFixedTransformBatchItem>& items,
    const uint16_t*                                                quant_tables,
    const float*                                                   resize_weight_matrices,
    const JpegDctGridTransformSpec&                                transform,
    float*                                                         y_accum,
    float*                                                         cbcr_accum,
    JpegDctDeviceExecutionStats&                                   execution_stats,
    JpegDctDeviceScratch&                                          scratch) {
	if (items.empty()) {
		return;
	}
	if (scratch.cached_gather_in_flight && !scratch.cached_fixed_transform_in_flight) {
		drain_cached_gather(scratch, execution_stats);
	}
	if (scratch.cached_gather_in_flight &&
	    scratch.cached_fixed_transform_items.needs_reallocation(items.size())) {
		drain_cached_gather(scratch, execution_stats);
	}
	auto& upload_items = scratch.host_cached_fixed_transform_uploads.emplace_back(items.begin(), items.end());
	const auto stream = scratch.stream_for_cache_hit();
	scratch.ensure_cached_gather_events();
	if (!scratch.cached_gather_in_flight) {
		scratch.cached_gather_start.record(stream);
		scratch.cached_fixed_transform_in_flight = true;
	}
	scratch.cached_fixed_transform_items.upload(
	    upload_items.data(), upload_items.size(), stream, execution_stats);
	constexpr unsigned kThreads = 64;
	const dim3 block(kThreads);
	const dim3 grid(static_cast<unsigned>(upload_items.size()));
	transformed_dct_grid_cached_kernel<<<grid, block, 0, stream>>>(
	    scratch.cached_fixed_transform_items.data,
	    upload_items.size(),
	    quant_tables,
	    resize_weight_matrices,
	    transform.clamp_min,
	    transform.clamp_max,
	    y_accum,
	    cbcr_accum);
	CUDA_SAFE_CALL(cudaGetLastError());
	++execution_stats.materialize_kernel_launch_count;
	execution_stats.fixed_transform_item_count += upload_items.size();
	execution_stats.cached_gather_item_count += upload_items.size();
	scratch.cached_gather_done.record(stream);
	scratch.cached_gather_in_flight = true;
}

void execute_unified_image_major_plan(
    const std::vector<JpegDctDeviceShardPlan>&      shards,
	const std::vector<uint32_t>*                     fixed_transform_item_order,
	const std::vector<uint32_t>*                     fixed_transform_group_offsets,
    int16_t*                                        output,
    int16_t*                                        y_output,
    int16_t*                                        cbcr_output,
    float*                                          y_accum,
    float*                                          cbcr_accum,
    const uint16_t*                                 fixed_quant_tables,
    const float*                                    fixed_resize_weight_matrices,
    const JpegDctGridTransformSpec&                 grid_transform,
    const bool                                      output_ycbcr_dct_grid,
    const bool                                      output_transformed_dct_grid,
    const std::vector<uint8_t>&                     selected_coefficients,
    const JpegDctCoefficientSelectionShape&         selection_shape,
	JpegDctDeviceDecodedRowgroupCache*               cache,
    JpegDctDeviceCacheStats&                        batch_cache_stats,
    JpegDctDeviceExecutionStats&                    execution_stats,
	JpegDctDeviceScratch&                           scratch,
	const size_t                                    decode_batch_rowgroups,
	const JpegDctDeviceRowgroupPrefetchConfig&      rowgroup_read_parallelism,
	std::vector<JpegDctDeviceCachedGatherBatchItem>& cached_pending,
	std::vector<JpegDctDeviceCachedFixedTransformBatchItem>& cached_fixed_pending) {
	auto& pending = scratch.host_pending_works;
	pending.clear();
	size_t rowgroup_count = 0;
	for (const auto& shard : shards) {
		rowgroup_count += shard.rowgroups.size();
	}
	pending.reserve(rowgroup_count);
	const size_t effective_decode_batch_rowgroups =
	    decode_batch_rowgroups == 0 ? kDefaultJpegDctDecodeBatchRowgroups : decode_batch_rowgroups;

	struct MissPlanRef {
		const JpegDctDeviceRowgroupPlan* rowgroup = nullptr;
		uint32_t                         shard_id = std::numeric_limits<uint32_t>::max();
		const std::filesystem::path*     fls_path = nullptr;
	};
	std::vector<MissPlanRef> misses;
	misses.reserve(rowgroup_count);

	// Classify the entire cross-shard batch before any miss can insert into and
	// evict from the cache. Hit buffers are gathered first; all remaining misses
	// are then safe to aggregate into worksets independent of physical shard.
	for (const auto& shard : shards) {
		for (const auto& rowgroup_plan : shard.rowgroups) {
			const auto source_shard_id = rowgroup_plan.source_shard_id == std::numeric_limits<uint32_t>::max()
			                                 ? shard.shard_id
			                                 : rowgroup_plan.source_shard_id;
			const auto* source_fls_path =
			    rowgroup_plan.source_fls_path == nullptr ? shard.fls_path : rowgroup_plan.source_fls_path;
			const auto key = JpegDctDeviceDecodedRowgroupCacheKey {source_shard_id, rowgroup_plan.rowgroup_index};
			if (cache != nullptr && cache->capacity > 0) {
				auto it = cache->entries.find(key);
				if (it != cache->entries.end() && it->second->blocks.has_value()) {
					it->second->last_access = ++cache->clock;
					++batch_cache_stats.hits;
					const auto* dense = it->second->blocks->get();
					if (output_transformed_dct_grid) {
						cached_fixed_pending.reserve(
						    cached_fixed_pending.size() + rowgroup_plan.fixed_transform_items.size());
						for (const auto& item : rowgroup_plan.fixed_transform_items) {
							cached_fixed_pending.push_back(JpegDctDeviceCachedFixedTransformBatchItem {
							    dense,
							    JpegDctDeviceFixedTransformBatchItem {0U,
							                                          item.row_in_rowgroup,
							                                          item.output_block_index,
							                                          item.component,
							                                          static_cast<uint8_t>(item.zigzag_columns ? 1U : 0U),
							                                          item.x_factor,
							                                          item.y_factor,
							                                          item.x_subblock,
							                                          item.y_subblock,
							                                          static_cast<uint8_t>(item.x_upsample ? 1U : 0U),
							                                          static_cast<uint8_t>(item.y_upsample ? 1U : 0U),
							                                          item.x_up_factor,
							                                          item.y_up_factor,
							                                          item.x_down_factor,
							                                          item.y_down_factor,
							                                          item.quant_table_index,
							                                          item.x_weight_matrix_index,
							                                          item.y_weight_matrix_index}});
						}
					} else {
						cached_pending.reserve(cached_pending.size() + rowgroup_plan.items.size());
						for (const auto& item : rowgroup_plan.items) {
							cached_pending.push_back(
							    JpegDctDeviceCachedGatherBatchItem {dense, item.row_in_rowgroup, item.output_block_index});
						}
					}
					continue;
				}
				++batch_cache_stats.misses;
			}
			misses.push_back(MissPlanRef {&rowgroup_plan, source_shard_id, source_fls_path});
		}
	}

	execute_cached_rowgroup_hits(cached_pending, output, execution_stats, scratch);
	execute_cached_fixed_transform_hits(cached_fixed_pending,
	                                    fixed_quant_tables,
	                                    fixed_resize_weight_matrices,
	                                    grid_transform,
	                                    y_accum,
	                                    cbcr_accum,
	                                    execution_stats,
	                                    scratch);
	cached_pending.clear();
	cached_fixed_pending.clear();
	unsigned batch_unpack_n_vectors = kJpegDctDeviceUnpackNVectors;
	for (const auto miss : misses) {
		if (miss.rowgroup == nullptr) {
			throw std::runtime_error("unified JPEG DCT cache miss plan is invalid");
		}
		batch_unpack_n_vectors =
		    constrain_jpeg_dct_batch_unpack_n_vectors(batch_unpack_n_vectors, *miss.rowgroup);
	}

	size_t fixed_transform_source_item_offset = 0;
	const auto flush_pending = [&]() {
		if (pending.empty()) {
			return;
		}
		const auto workset_item_count = fixed_transform_item_count(pending);
		const auto workset_fixed_transform_plan =
		    fixed_transform_plan_for_workset(fixed_transform_item_order,
		                                     fixed_transform_group_offsets,
		                                     fixed_transform_source_item_offset,
		                                     workset_item_count,
		                                     scratch);
		execute_decoded_rowgroup_batch(pending,
		                               workset_fixed_transform_plan.item_order,
		                               workset_fixed_transform_plan.group_offsets,
		                               output,
		                               y_output,
		                               cbcr_output,
		                               y_accum,
		                               cbcr_accum,
		                               fixed_quant_tables,
		                               fixed_resize_weight_matrices,
		                               grid_transform,
		                               selected_coefficients,
		                               selection_shape,
		                               output_ycbcr_dct_grid,
		                               output_transformed_dct_grid,
		                               cache,
		                               batch_cache_stats,
		                               execution_stats,
		                               scratch);
		fixed_transform_source_item_offset += workset_item_count;
	};

	const size_t read_worker_count = rowgroup_read_parallelism.enabled
	                                     ? std::min(rowgroup_read_parallelism.workers, misses.size())
	                                     : 1U;
	if (read_worker_count <= 1U) {
		uint32_t current_shard_id = std::numeric_limits<uint32_t>::max();
		bool     current_sparse_vector_reads = false;
		std::shared_ptr<galp::format::FlsReader> rdr;
		for (const auto miss : misses) {
			if (miss.rowgroup == nullptr || miss.fls_path == nullptr ||
			    miss.shard_id == std::numeric_limits<uint32_t>::max()) {
				throw std::runtime_error("unified JPEG DCT cache miss plan is invalid");
			}
			const bool miss_sparse_vector_reads = miss.rowgroup->sparse_storage_read;
			if (miss.rowgroup->staged_read) {
				rdr = miss.rowgroup->staged_read->reader;
			} else if (!rdr || current_shard_id != miss.shard_id ||
			           current_sparse_vector_reads != miss_sparse_vector_reads) {
				current_shard_id            = miss.shard_id;
				current_sparse_vector_reads = miss_sparse_vector_reads;
				rdr = scratch.fls_reader(*miss.fls_path, miss_sparse_vector_reads);
			}
			if (!rdr) {
				throw std::runtime_error("JPEG DCT staged rowgroup has no bound reader");
			}
			auto work = prepare_decoded_rowgroup_work(*rdr,
			                                          miss.shard_id,
			                                          *miss.rowgroup,
			                                          selected_coefficients,
			                                          selection_shape,
			                                          output_ycbcr_dct_grid,
			                                          batch_unpack_n_vectors,
			                                          cache,
			                                          execution_stats);
			pending.push_back(std::move(work));
			if (pending.size() >= effective_decode_batch_rowgroups) {
				flush_pending();
			}
		}
	} else {
		std::vector<std::shared_ptr<galp::format::FlsReader>> readers;
		readers.reserve(misses.size());
		for (const auto miss : misses) {
			if (miss.rowgroup == nullptr || miss.fls_path == nullptr ||
			    miss.shard_id == std::numeric_limits<uint32_t>::max()) {
				throw std::runtime_error("unified JPEG DCT cache miss plan is invalid");
			}
			readers.push_back(
			    miss.rowgroup->staged_read
			        ? miss.rowgroup->staged_read->reader
			        : scratch.fls_reader(*miss.fls_path, miss.rowgroup->sparse_storage_read));
			if (!readers.back()) {
				throw std::runtime_error("JPEG DCT staged rowgroup has no bound reader");
			}
		}

		std::vector<DecodedRowgroupReadResult> read_results(misses.size());
		std::vector<std::exception_ptr>         read_errors(misses.size());
		std::atomic<size_t>                     next_read {0U};
		std::vector<std::thread>                workers;
		workers.reserve(read_worker_count);
		const auto parallel_read_start = Clock::now();
		for (size_t worker_index = 0U; worker_index < read_worker_count; ++worker_index) {
			workers.emplace_back([&]() {
				while (true) {
					const size_t miss_index = next_read.fetch_add(1U, std::memory_order_relaxed);
					if (miss_index >= misses.size()) {
						return;
					}
					try {
						read_results[miss_index] = take_staged_or_read_decoded_rowgroup(
						    *readers[miss_index],
						    *misses[miss_index].rowgroup,
						    batch_unpack_n_vectors);
					} catch (...) {
						read_errors[miss_index] = std::current_exception();
					}
				}
			});
		}
		for (auto& worker : workers) {
			worker.join();
		}
		const auto parallel_read_end = Clock::now();
		execution_stats.sync_rowgroup_read_ms += elapsed_ms(parallel_read_start, parallel_read_end);
		for (const auto& error : read_errors) {
			if (error) {
				std::rethrow_exception(error);
			}
		}

		for (size_t miss_index = 0U; miss_index < misses.size(); ++miss_index) {
			auto& read_result = read_results[miss_index];
			const auto miss = misses[miss_index];
			record_decoded_rowgroup_read(read_result.io_timing, execution_stats);
			auto work = prepare_decoded_rowgroup_work_from_materialized(std::move(read_result.rowgroup),
			                                                               miss.shard_id,
			                                                               *miss.rowgroup,
			                                                               selected_coefficients,
			                                                               selection_shape,
			                                                               output_ycbcr_dct_grid,
			                                                               batch_unpack_n_vectors,
			                                                               cache,
			                                                               execution_stats);
			pending.push_back(std::move(work));
			if (pending.size() >= effective_decode_batch_rowgroups) {
				flush_pending();
			}
		}
	}
	flush_pending();
	if (fixed_transform_item_order != nullptr &&
	    fixed_transform_source_item_offset != fixed_transform_item_order->size()) {
		throw std::runtime_error("JPEG DCT deterministic fixed-transform plan was not fully consumed");
	}
}

void execute_shard_plan(const std::shared_ptr<galp::format::FlsReader>&  rdr,
                        const JpegDctDeviceShardPlan&                    shard,
	                    const std::vector<uint32_t>*                     fixed_transform_item_order,
	                    const std::vector<uint32_t>*                     fixed_transform_group_offsets,
                        int16_t*                                         output,
                        int16_t*                                         y_output,
                        int16_t*                                         cbcr_output,
                        float*                                           y_accum,
                        float*                                           cbcr_accum,
                        const uint16_t*                                  fixed_quant_tables,
                        const float*                                     fixed_resize_weight_matrices,
                        const JpegDctGridTransformSpec&                   grid_transform,
                        const bool                                       output_ycbcr_dct_grid,
                        const bool                                       output_transformed_dct_grid,
                        const std::vector<uint8_t>&                      selected_coefficients,
                        const JpegDctCoefficientSelectionShape&          selection_shape,
                        JpegDctDeviceDecodedRowgroupCache*               cache,
                        JpegDctDeviceCacheStats&                         batch_cache_stats,
                        JpegDctDeviceExecutionStats&                     execution_stats,
                        JpegDctDeviceScratch&                            scratch,
                        const size_t                                     decode_batch_rowgroups,
	                        const JpegDctDeviceRowgroupPrefetchConfig&       rowgroup_prefetch,
	                        std::vector<JpegDctDeviceCachedGatherBatchItem>& cached_pending,
	                        std::vector<JpegDctDeviceCachedFixedTransformBatchItem>& cached_fixed_pending) {
	if (!rdr) {
		throw std::runtime_error("execute_shard_plan: reader is null");
	}
	auto& pending = scratch.host_pending_works;
	pending.clear();
	const size_t effective_decode_batch_rowgroups =
	    decode_batch_rowgroups == 0 ? kDefaultJpegDctDecodeBatchRowgroups : decode_batch_rowgroups;
	pending.reserve(std::min(effective_decode_batch_rowgroups, shard.rowgroups.size()));
	unsigned batch_unpack_n_vectors = kJpegDctDeviceUnpackNVectors;
	for (const auto& rowgroup_plan : shard.rowgroups) {
		batch_unpack_n_vectors =
		    constrain_jpeg_dct_batch_unpack_n_vectors(batch_unpack_n_vectors, rowgroup_plan);
	}
	bool pending_may_insert_cache = false;
	auto prefetch_plan =
	    plan_jpeg_dct_rowgroup_prefetch(shard, cache, rowgroup_prefetch, effective_decode_batch_rowgroups);
	if (prefetch_plan.enabled) {
		size_t scheduled_position = 0;
		for (size_t rowgroup_position = 0; rowgroup_position < shard.rowgroups.size(); ++rowgroup_position) {
			if (!prefetch_plan.use_prefetch_for_position[rowgroup_position]) {
				continue;
			}
			auto& selected = prefetch_plan.selected_vectors[scheduled_position++];
			if (!selected.empty()) {
				const auto& rowgroup_plan = shard.rowgroups[rowgroup_position];
				selected = expand_selected_decode_chunks(
				    selected, rowgroup_plan.full_vector_count, batch_unpack_n_vectors);
			}
		}
	}
	std::unique_ptr<galp::runtime::RowgroupPrefetchQueue> prefetch_queue;
	if (prefetch_plan.enabled) {
		const size_t scheduled_rowgroups  = prefetch_plan.rowgroup_indices.size();
		const auto   prefetch_queue_start = Clock::now();
		// Popped rowgroups remain live until the current decode workset completes, while
		// the producer may keep `depth` more rowgroups queued. Size the shared pool for
		// both sets so workers cannot deadlock waiting for leases held by `pending`.
		const size_t pinned_pool_slots = effective_decode_batch_rowgroups + rowgroup_prefetch.depth +
		                                 std::max<size_t>(1U, rowgroup_prefetch.workers);
		if (!scratch.rowgroup_prefetch_pinned_pool ||
		    scratch.rowgroup_prefetch_pinned_pool_slots < pinned_pool_slots) {
			scratch.rowgroup_prefetch_pinned_pool =
			    galp::runtime::PinnedRowgroupBufferPool::create(pinned_pool_slots);
			scratch.rowgroup_prefetch_pinned_pool_slots = pinned_pool_slots;
		}
		prefetch_queue = std::make_unique<galp::runtime::RowgroupPrefetchQueue>(
		    rdr,
		    std::move(prefetch_plan.rowgroup_indices),
		    rowgroup_prefetch.depth,
		    rowgroup_prefetch.workers,
		    scratch.rowgroup_prefetch_pinned_pool,
		    /*max_inflight_storage_bytes=*/0,
		    std::move(prefetch_plan.selected_vectors));
		const auto prefetch_queue_end = Clock::now();
		execution_stats.prefetch_queue_start_ms += elapsed_ms(prefetch_queue_start, prefetch_queue_end);
		execution_stats.prefetched_rowgroup_count += scheduled_rowgroups;
	}
	execution_stats.prefetch_initial_cache_hit_rowgroup_count += prefetch_plan.initial_cache_hit_rowgroup_count;
	execution_stats.prefetch_candidate_rowgroup_count += prefetch_plan.candidate_rowgroup_count;
	execution_stats.prefetch_active_shard_count += prefetch_plan.enabled ? 1U : 0U;
	execution_stats.prefetch_config_disabled_shard_count += prefetch_plan.disabled_by_config ? 1U : 0U;
	execution_stats.prefetch_all_hit_shard_count += prefetch_plan.disabled_by_all_hits ? 1U : 0U;
	execution_stats.prefetch_small_batch_disabled_shard_count += prefetch_plan.disabled_by_small_batch_count ? 1U : 0U;
	execution_stats.prefetch_selected_vector_disabled_shard_count +=
	    prefetch_plan.disabled_by_selected_vector_miss ? 1U : 0U;
	execution_stats.prefetch_selected_vector_miss_rowgroup_count += prefetch_plan.selected_vector_miss_rowgroup_count;
	execution_stats.prefetch_skipped_repeated_rowgroup_count += prefetch_plan.skipped_repeated_rowgroup_count;

	const auto flush_cached = [&]() {
		execute_cached_rowgroup_hits(cached_pending, output, execution_stats, scratch);
		execute_cached_fixed_transform_hits(cached_fixed_pending,
		                                    fixed_quant_tables,
		                                    fixed_resize_weight_matrices,
		                                    grid_transform,
		                                    y_accum,
		                                    cbcr_accum,
		                                    execution_stats,
		                                    scratch);
		cached_pending.clear();
		cached_fixed_pending.clear();
	};
	size_t fixed_transform_source_item_offset = 0;
	const auto flush_pending                      = [&]() {
		// Pending dense materialization can evict cached dense buffers referenced by cached_pending.
		// Launch cached gathers first; the decoded stream will wait on their completion event.
		if (pending_may_insert_cache) {
			flush_cached();
		}
		const auto workset_item_count = fixed_transform_item_count(pending);
		const auto workset_fixed_transform_plan =
		    fixed_transform_plan_for_workset(fixed_transform_item_order,
		                                     fixed_transform_group_offsets,
		                                     fixed_transform_source_item_offset,
		                                     workset_item_count,
		                                     scratch);
        execute_decoded_rowgroup_batch(pending,
                                       workset_fixed_transform_plan.item_order,
                                       workset_fixed_transform_plan.group_offsets,
                                       output,
                                       y_output,
                                       cbcr_output,
                                       y_accum,
                                       cbcr_accum,
                                       fixed_quant_tables,
                                       fixed_resize_weight_matrices,
                                       grid_transform,
                                       selected_coefficients,
                                       selection_shape,
                                       output_ycbcr_dct_grid,
                                       output_transformed_dct_grid,
                                       cache,
                                       batch_cache_stats,
                                       execution_stats,
                                       scratch);
        fixed_transform_source_item_offset += workset_item_count;
        pending_may_insert_cache = false;
	};
	const auto pop_prefetched_rowgroup = [&](const JpegDctDeviceRowgroupPlan& rowgroup_plan,
	                                         const bool                       consumed_as_hit = false) {
		auto result = prefetch_queue->pop();
		if (result.rowgroup_index != rowgroup_plan.rowgroup_index) {
			galp::execution::free_rowgroup(result.rowgroup);
			throw std::runtime_error("JPEG DCT rowgroup prefetch queue returned an unexpected rowgroup");
		}
		execution_stats.prefetch_rowgroup_read_ms += result.timing.read_ms;
		execution_stats.rowgroup_storage_bytes_read += result.storage_bytes;
		execution_stats.compressed_payload_bytes_read += result.storage_bytes;
		execution_stats.full_compressed_payload_bytes += result.full_storage_bytes;
		execution_stats.pread_count += result.pread_count;
		if (result.used_pinned_backing) {
			++execution_stats.pinned_rowgroup_read_count;
			execution_stats.pinned_rowgroup_read_bytes += result.storage_bytes;
		}
		if (result.used_vector_bundle_read) {
			++execution_stats.vector_bundle_rowgroup_count;
			execution_stats.vector_bundle_pread_count += result.pread_count;
			execution_stats.vector_bundle_envelope_rowgroup_count +=
			    result.used_vector_bundle_envelope_read ? 1U : 0U;
		}
		execution_stats.sparse_read_supported =
		    execution_stats.sparse_read_supported || result.sparse_read_supported;
		if (!result.sparse_fallback_reason.empty()) {
			++execution_stats.sparse_read_fallback_rowgroup_count;
			execution_stats.sparse_read_fallback_reason = result.sparse_fallback_reason;
		}
		execution_stats.prefetch_depth_block_ms += result.prefetch.depth_block_ms;
		const auto ready_push = result.timing.timeline.ready_push;
		const auto wait_start = result.timing.timeline.consumer_wait_start;
		const auto wait_end   = result.timing.timeline.consumer_wait_end;
		double     wait_ms    = 0.0;
		if (wait_start != Clock::time_point {} && wait_end != Clock::time_point {} && wait_start <= wait_end) {
			wait_ms = elapsed_ms(wait_start, wait_end);
		}
		if (consumed_as_hit) {
			execution_stats.prefetch_consumed_as_hit_read_ms += result.timing.read_ms;
			execution_stats.prefetch_consumed_as_hit_wait_ms += wait_ms;
		}
		if (ready_push != Clock::time_point {} && wait_start != Clock::time_point {} && ready_push < wait_start) {
			execution_stats.prefetch_ready_ahead_ms += elapsed_ms(ready_push, wait_start);
		}
		return std::move(result.rowgroup);
	};

	for (size_t rowgroup_pos = 0; rowgroup_pos < shard.rowgroups.size(); ++rowgroup_pos) {
		const auto& rowgroup_plan = shard.rowgroups[rowgroup_pos];
		const auto  key           = JpegDctDeviceDecodedRowgroupCacheKey {shard.shard_id, rowgroup_plan.rowgroup_index};
		if (cache != nullptr && cache->capacity > 0) {
			auto it = cache->entries.find(key);
			if (it != cache->entries.end() && it->second->blocks.has_value()) {
				// Pending selected-vector decodes do not materialize dense cache entries, so they cannot
				// evict this hit. Keep accumulating them across cache hits to avoid tiny worksets.
				if (pending_may_insert_cache) {
					flush_pending();
				}
				it = cache->entries.find(key);
				if (it != cache->entries.end() && it->second->blocks.has_value()) {
					it->second->last_access = ++cache->clock;
					++batch_cache_stats.hits;
					const auto* dense = it->second->blocks->get();
					if (output_transformed_dct_grid) {
						cached_fixed_pending.reserve(cached_fixed_pending.size() +
						                             rowgroup_plan.fixed_transform_items.size());
						for (const auto& item : rowgroup_plan.fixed_transform_items) {
							cached_fixed_pending.push_back(JpegDctDeviceCachedFixedTransformBatchItem {
							    dense,
							    JpegDctDeviceFixedTransformBatchItem {
							        0U,
							        item.row_in_rowgroup,
							        item.output_block_index,
							        item.component,
							        static_cast<uint8_t>(item.zigzag_columns ? 1U : 0U),
							        item.x_factor,
							        item.y_factor,
							        item.x_subblock,
							        item.y_subblock,
							        static_cast<uint8_t>(item.x_upsample ? 1U : 0U),
							        static_cast<uint8_t>(item.y_upsample ? 1U : 0U),
							        item.x_up_factor,
							        item.y_up_factor,
							        item.x_down_factor,
							        item.y_down_factor,
							        item.quant_table_index,
							        item.x_weight_matrix_index,
							        item.y_weight_matrix_index}});
						}
					} else {
						cached_pending.reserve(cached_pending.size() + rowgroup_plan.items.size());
						for (const auto& item : rowgroup_plan.items) {
							cached_pending.push_back(JpegDctDeviceCachedGatherBatchItem {
							    dense, item.row_in_rowgroup, item.output_block_index});
						}
					}
					if (prefetch_plan.use_prefetch_for_position[rowgroup_pos]) {
						auto rowgroup = pop_prefetched_rowgroup(rowgroup_plan, /*consumed_as_hit=*/true);
						galp::execution::free_rowgroup(rowgroup);
						++execution_stats.prefetch_consumed_as_hit_count;
					}
					continue;
				}
			}
			++batch_cache_stats.misses;
			execution_stats.prefetch_initial_hit_runtime_miss_count +=
			    prefetch_plan.initial_cache_hit_for_position[rowgroup_pos] ? 1U : 0U;
			execution_stats.prefetch_skipped_repeated_runtime_miss_count +=
			    prefetch_plan.skipped_repeated_for_position[rowgroup_pos] ? 1U : 0U;
		}

		auto work                = prefetch_plan.use_prefetch_for_position[rowgroup_pos]
		                               ? prepare_decoded_rowgroup_work_from_materialized(pop_prefetched_rowgroup(rowgroup_plan),
                                                                          shard.shard_id,
                                                                          rowgroup_plan,
                                                                          selected_coefficients,
                                                                          selection_shape,
		                                                                          output_ycbcr_dct_grid,
		                                                                          batch_unpack_n_vectors,
		                                                                          cache,
                                                                          execution_stats)
			                               : prepare_decoded_rowgroup_work(*rdr,
                                                        shard.shard_id,
                                                        rowgroup_plan,
                                                        selected_coefficients,
                                                        selection_shape,
		                                                        output_ycbcr_dct_grid,
		                                                        batch_unpack_n_vectors,
		                                                        cache,
                                                        execution_stats);
		pending_may_insert_cache = pending_may_insert_cache || static_cast<bool>(work.cache_entry);
		pending.push_back(std::move(work));
		if (pending.size() >= effective_decode_batch_rowgroups) {
			flush_pending();
		}
	}
	flush_pending();
	if (fixed_transform_item_order != nullptr &&
	    fixed_transform_source_item_offset != fixed_transform_item_order->size()) {
		throw std::runtime_error("JPEG DCT deterministic fixed-transform plan was not fully consumed");
	}
	if (prefetch_queue) {
		execution_stats.prefetch_wait_ms += prefetch_queue->wait_ms();
	}
}

} // namespace

void stage_jpeg_dct_device_batch_io(JpegDctDeviceBatchPlan& plan, JpegDctHostIoContext& context) {
	stage_jpeg_dct_device_batch_io_impl(plan, context);
}

void compile_jpeg_dct_device_batch_io(JpegDctDeviceBatchPlan& plan, JpegDctHostIoContext& context) {
	compile_jpeg_dct_device_batch_io_impl(plan, context);
}

void append_jpeg_rowgroup_columns(galp::runtime::ExecutionWorkset&        workset,
                                  const galp::execution::Rowgroup&        rowgroup,
                                  const galp::execution::ExecutionConfig& cfg,
                                  const size_t                            expr_index_base,
                                  const std::vector<uint8_t>&             selected_coefficients,
                                  const std::vector<uint32_t>*            selected_vectors) {
	append_jpeg_rowgroup_columns(workset,
	                             rowgroup,
	                             cfg,
	                             expr_index_base,
	                             selected_coefficients,
	                             classify_coefficient_selection(selected_coefficients),
	                             selected_vectors);
}

void append_jpeg_rowgroup_columns(galp::runtime::ExecutionWorkset&        workset,
                                  const galp::execution::Rowgroup&        rowgroup,
                                  const galp::execution::ExecutionConfig& cfg,
                                  const size_t                            expr_index_base,
                                  const std::vector<uint8_t>&             selected_coefficients,
                                  const JpegDctCoefficientSelectionShape& selection_shape,
                                  const std::vector<uint32_t>*            selected_vectors) {
	workset.outputs.required = workset.outputs.required || cfg.write_out;
	galp::runtime::begin_workset_chunk_arena(workset, rowgroup.columns.size());
	auto* active_chunk_arena = workset.buffers.chunk_arena.get();
	galp::runtime::append_packed_rowgroup_device_scatter(workset, rowgroup, *active_chunk_arena);

	std::array<bool, kJpegDctCoefficientCount> decode_coefficients {};
	const auto mark_decode_coefficient = [&](const auto& self, const size_t coeff_idx) -> void {
		if (coeff_idx >= kJpegDctCoefficientCount || coeff_idx >= rowgroup.columns.size()) {
			throw std::out_of_range("JPEG DCT selected coefficient is outside the rowgroup column range");
		}
		if (decode_coefficients[coeff_idx]) {
			return;
		}
		decode_coefficients[coeff_idx] = true;
		const auto& column             = rowgroup.columns[coeff_idx];
		if (column.alias_of.has_value()) {
			self(self, *column.alias_of);
		}
	};
	if (selection_shape.kind == JpegDctCoefficientSelectionKind::kAll) {
		for (size_t coeff_idx = 0; coeff_idx < kJpegDctCoefficientCount && coeff_idx < rowgroup.columns.size();
		     ++coeff_idx) {
			decode_coefficients[coeff_idx] = true;
		}
	} else {
		for_each_selected_coefficient(selected_coefficients, selection_shape, [&](const size_t coeff_idx) {
			mark_decode_coefficient(mark_decode_coefficient, coeff_idx);
		});
	}

	const void* last_backing_base  = nullptr;
	size_t      last_backing_bytes = 0;
	for (size_t coeff_idx = 0; coeff_idx < kJpegDctCoefficientCount && coeff_idx < rowgroup.columns.size();
	     ++coeff_idx) {
		if (!decode_coefficients[coeff_idx]) {
			continue;
		}
		const auto& column = rowgroup.columns[coeff_idx];
		if (column.skip_decompress) {
			continue;
		}
		if (galp::runtime::has_pinned_backing(column) &&
		    (column.backing_base != last_backing_base || column.backing_bytes != last_backing_bytes)) {
			active_chunk_arena->register_backing(column.backing_base, column.backing_bytes);
			last_backing_base  = column.backing_base;
			last_backing_bytes = column.backing_bytes;
		}
		const uint32_t selected_vector_width = std::max(1U, cfg.unpack_n_vectors);
		galp::runtime::append_column_to_workset(workset,
		                                        column,
		                                        cfg,
		                                        expr_index_base + coeff_idx,
		                                        *active_chunk_arena,
		                                        nullptr,
		                                        /*emit_typed_work_items=*/true,
		                                        /*register_backing=*/false,
		                                        selected_vectors,
		                                        selected_vector_width);
	}
}

void JpegDctDeviceDecodedRowgroupCache::insert_ready_entry(
    const JpegDctDeviceDecodedRowgroupCacheKey&             key,
    std::unique_ptr<JpegDctDeviceDecodedRowgroupCacheEntry> entry,
    JpegDctDeviceCacheStats&                                batch_cache_stats) {
	if (!entry || capacity == 0) {
		return;
	}
	const auto existing = entries.find(key);
	if (existing != entries.end()) {
		resident = existing->second->bytes <= resident ? resident - existing->second->bytes : 0;
	}
	resident += entry->bytes;
	entries[key] = std::move(entry);
	++batch_cache_stats.inserts;
	while (resident > capacity && !entries.empty()) {
		auto evict_it          = entries.end();
		auto evict_last_access = std::numeric_limits<uint64_t>::max();
		for (auto it = entries.begin(); it != entries.end(); ++it) {
			if (it->first == key) {
				continue;
			}
			if (it->second->last_access < evict_last_access) {
				evict_it          = it;
				evict_last_access = it->second->last_access;
			}
		}
		if (evict_it == entries.end()) {
			break;
		}
		resident = evict_it->second->bytes <= resident ? resident - evict_it->second->bytes : 0;
		entries.erase(evict_it);
		++batch_cache_stats.evictions;
	}
}

void JpegDctDeviceDecodedRowgroupCache::set_capacity(const size_t bytes) {
	capacity = bytes;
	while (resident > capacity && !entries.empty()) {
		auto evict_it          = entries.end();
		auto evict_last_access = std::numeric_limits<uint64_t>::max();
		for (auto it = entries.begin(); it != entries.end(); ++it) {
			if (it->second->last_access < evict_last_access) {
				evict_it          = it;
				evict_last_access = it->second->last_access;
			}
		}
		if (evict_it == entries.end()) {
			break;
		}
		resident -= evict_it->second->bytes;
		entries.erase(evict_it);
	}
	if (capacity == 0) {
		clear();
	}
}

void JpegDctDeviceDecodedRowgroupCache::clear() {
	entries.clear();
	resident = 0;
	clock    = 0;
}

size_t JpegDctDeviceDecodedRowgroupCache::capacity_bytes() const noexcept {
	return capacity;
}

size_t JpegDctDeviceDecodedRowgroupCache::resident_bytes() const noexcept {
	return resident;
}

size_t JpegDctDeviceDecodedRowgroupCache::resident_rowgroups() const noexcept {
	return entries.size();
}

bool has_decoded_cache_entry(const JpegDctDeviceDecodedRowgroupCache*    cache,
                             const JpegDctDeviceDecodedRowgroupCacheKey& key) {
	if (cache == nullptr || cache->capacity == 0) {
		return false;
	}
	const auto it = cache->entries.find(key);
	return it != cache->entries.end() && it->second->blocks.has_value();
}

JpegDctDeviceRowgroupPrefetchPlan
plan_jpeg_dct_rowgroup_prefetch(const JpegDctDeviceShardPlan&              shard,
                                const JpegDctDeviceDecodedRowgroupCache*   cache,
                                const JpegDctDeviceRowgroupPrefetchConfig& config,
                                const size_t effective_decode_batch_rowgroups) {
	std::vector<bool> cache_hit_by_position(shard.rowgroups.size(), false);
	for (size_t rowgroup_pos = 0; rowgroup_pos < shard.rowgroups.size(); ++rowgroup_pos) {
		const auto& rowgroup_plan = shard.rowgroups[rowgroup_pos];
		const auto key = JpegDctDeviceDecodedRowgroupCacheKey {shard.shard_id, rowgroup_plan.rowgroup_index};
		cache_hit_by_position[rowgroup_pos] = has_decoded_cache_entry(cache, key);
	}
	return plan_jpeg_dct_rowgroup_prefetch_from_hits(shard,
	                                                 cache_hit_by_position,
	                                                 config,
	                                                 effective_decode_batch_rowgroups,
	                                                 cache == nullptr ? 0U : cache->capacity);
}

JpegDctDeviceBatch execute_jpeg_dct_device_batch_plan(JpegDctDeviceBatchPlan       plan,
                                                       JpegDctDeviceExecutionFence* reuse_fence) {
	auto impl = std::make_unique<JpegDctDeviceBatch::Impl>();
	try {
	CUDA_SAFE_CALL(cudaGetDevice(&impl->cuda_device));
	impl->layout                                             = plan.layout;
	impl->image_layouts                                      = std::move(plan.image_layouts);
	impl->block_metadata                                     = std::move(plan.block_metadata);
	impl->rowgroups                                          = std::move(plan.rowgroups);
	impl->selected_coefficients                              = std::move(plan.selected_coefficients);
	impl->coefficients_per_block                             = plan.coefficients_per_block;
	impl->execution_stats.planning_ms                        = plan.planning_ms;
	impl->execution_stats.host_io_staging_ms                 = plan.host_io_staging_ms;
	impl->execution_stats.host_io_staged_rowgroups           = plan.host_io_staged_rowgroups;
	impl->execution_stats.plan_cache_hits                    = plan.plan_cache_hits;
	impl->execution_stats.plan_cache_misses                  = plan.plan_cache_misses;
	impl->execution_stats.plan_cache_evictions               = plan.plan_cache_evictions;
	impl->execution_stats.exact_batch_plan_cache_enabled     = plan.exact_batch_plan_cache_enabled;
	impl->execution_stats.resize_weight_build_ms             = plan.resize_weight_build_ms;
	impl->execution_stats.dct_resize_weight_cache_hits       = plan.dct_resize_weight_cache_hits;
	impl->execution_stats.dct_resize_weight_cache_misses     = plan.dct_resize_weight_cache_misses;
	impl->execution_stats.dct_conversion_matrix_cache_hits   = plan.dct_conversion_matrix_cache_hits;
	impl->execution_stats.dct_conversion_matrix_cache_misses = plan.dct_conversion_matrix_cache_misses;
	impl->execution_stats.rowgroup_count                     = impl->rowgroups.size();
	impl->execution_stats.planned_selected_vector_count      = plan.planned_selected_vector_count;
	impl->execution_stats.full_vector_count                  = plan.full_vector_count;
	impl->execution_stats.planned_saved_vector_count         = plan.planned_saved_vector_count;
	impl->execution_stats.automatic_sparse_storage_candidate_rowgroup_count =
	    plan.automatic_sparse_storage_candidate_rowgroup_count;
	impl->execution_stats.automatic_sparse_storage_selected_rowgroup_count =
	    plan.automatic_sparse_storage_selected_rowgroup_count;
	impl->execution_stats.automatic_sparse_storage_rejected_rowgroup_count =
	    plan.automatic_sparse_storage_rejected_rowgroup_count;
	impl->execution_stats.automatic_sparse_storage_full_bytes = plan.automatic_sparse_storage_full_bytes;
	impl->execution_stats.automatic_sparse_storage_candidate_bytes =
	    plan.automatic_sparse_storage_candidate_bytes;
	impl->execution_stats.automatic_sparse_storage_candidate_pread_count =
	    plan.automatic_sparse_storage_candidate_pread_count;
	impl->execution_stats.automatic_sparse_storage_full_estimated_ns =
	    plan.automatic_sparse_storage_full_estimated_ns;
	impl->execution_stats.automatic_sparse_storage_candidate_estimated_ns =
	    plan.automatic_sparse_storage_candidate_estimated_ns;
	impl->execution_stats.requested_source_block_count       = plan.fixed_transform_source_block_count;
	impl->execution_stats.planned_vector_count                = plan.planned_selected_vector_count;
	impl->execution_stats.source_blocks_transformed           = plan.fixed_transform_source_block_count;
	impl->execution_stats.cache_enabled                      = plan.cache_enabled;
	impl->execution_stats.fixed_transform_image_count =
	    plan.layout == JpegDctDeviceLayout::kTransformedDctGrid ? impl->image_layouts.size() : 0;
	impl->execution_stats.fixed_transform_component_count        = plan.fixed_transform_component_count;
	impl->execution_stats.fixed_transform_source_block_count     = plan.fixed_transform_source_block_count;
	impl->execution_stats.fixed_transform_output_block_count     = plan.fixed_transform_output_block_count;
	impl->execution_stats.host_expanded_transform_items_created  = plan.host_expanded_transform_items_created;
	impl->execution_stats.host_output_block_source_lists_created = plan.host_output_block_source_lists_created;
	impl->execution_stats.host_global_transform_sort_items       = plan.host_global_transform_sort_items;
	impl->execution_stats.planless_axis_program_count            = plan.planless_axis_program_count;
	impl->execution_stats.planless_axis_phase_matrix_count       = plan.planless_axis_phase_matrix_count;
	impl->execution_stats.planless_axis_program_bytes = plan.fixed_resize_weight_matrices.size() * sizeof(float);
	impl->ycbcr_dct_grid_shape                        = plan.ycbcr_dct_grid_shape;

	const bool output_ycbcr_dct_grid =
	    plan.layout == JpegDctDeviceLayout::kYcbcrDctGrid || plan.layout == JpegDctDeviceLayout::kTransformedDctGrid;
	const bool output_weighted_grid = plan.layout == JpegDctDeviceLayout::kTransformedDctGrid;
	impl->grid_output_data_type =
	    output_weighted_grid ? plan.grid_transform.output_data_type : JpegDctGridOutputDataType::kInt16;
	impl->execution_stats.fixed_grid_output_float32 =
	    impl->grid_output_data_type == JpegDctGridOutputDataType::kFloat32;
	impl->execution_stats.fixed_grid_output_affine_applied = impl->execution_stats.fixed_grid_output_float32;
	impl->execution_stats.fixed_grid_output_add   = plan.grid_transform.output_add;
	impl->execution_stats.fixed_grid_output_scale = plan.grid_transform.output_scale;
	if (output_ycbcr_dct_grid) {
		impl->coefficient_count      = 0;
		impl->y_coefficient_count    = impl->ycbcr_dct_grid_shape.y_count();
		impl->cbcr_coefficient_count = impl->ycbcr_dct_grid_shape.cbcr_count();
		if (impl->y_coefficient_count != 0) {
			if (impl->grid_output_data_type == JpegDctGridOutputDataType::kInt16) {
				impl->y_coefficients.emplace(impl->y_coefficient_count);
				CUDA_SAFE_CALL(cudaMemset(impl->y_coefficients->get(), 0, impl->y_coefficient_count * sizeof(int16_t)));
			}
			if (output_weighted_grid) {
				impl->y_accum.emplace(impl->y_coefficient_count);
				CUDA_SAFE_CALL(cudaMemset(impl->y_accum->get(), 0, impl->y_coefficient_count * sizeof(float)));
			}
		}
		if (impl->cbcr_coefficient_count != 0) {
			if (impl->grid_output_data_type == JpegDctGridOutputDataType::kInt16) {
				impl->cbcr_coefficients.emplace(impl->cbcr_coefficient_count);
				CUDA_SAFE_CALL(
				    cudaMemset(impl->cbcr_coefficients->get(), 0, impl->cbcr_coefficient_count * sizeof(int16_t)));
			}
			if (output_weighted_grid) {
				impl->cbcr_accum.emplace(impl->cbcr_coefficient_count);
				CUDA_SAFE_CALL(cudaMemset(impl->cbcr_accum->get(), 0, impl->cbcr_coefficient_count * sizeof(float)));
			}
		}
	} else {
		impl->coefficient_count = impl->block_metadata.size() * impl->coefficients_per_block;
	}
	if (impl->coefficient_count != 0) {
		impl->coefficients.emplace(impl->coefficient_count);
	}

	int16_t*   output      = impl->coefficients.has_value() ? impl->coefficients->get() : nullptr;
	int16_t*   y_output    = impl->y_coefficients.has_value() ? impl->y_coefficients->get() : nullptr;
	int16_t*   cbcr_output = impl->cbcr_coefficients.has_value() ? impl->cbcr_coefficients->get() : nullptr;
	float*     y_accum     = impl->y_accum.has_value() ? impl->y_accum->get() : nullptr;
	float*     cbcr_accum  = impl->cbcr_accum.has_value() ? impl->cbcr_accum->get() : nullptr;
	const bool has_weighted_grid_output =
	    output_weighted_grid && (impl->y_coefficient_count != 0 || impl->cbcr_coefficient_count != 0);
	if (has_weighted_grid_output) {
		if (plan.fixed_quant_tables.empty() || plan.fixed_quant_tables.size() % 64U != 0U) {
			throw std::runtime_error("JPEG DCT fixed transform plan has invalid quantization tables");
		}
		if (!plan.uses_planless_fixed_transform &&
		    (plan.fixed_resize_weight_matrices.empty() || plan.fixed_resize_weight_matrices.size() % 64U != 0U)) {
			throw std::runtime_error("JPEG DCT fixed transform plan has invalid resize weight matrices");
		}
		impl->fixed_quant_tables.emplace(plan.fixed_quant_tables.size(), plan.fixed_quant_tables.data());
		if (!plan.fixed_resize_weight_matrices.empty()) {
			impl->fixed_resize_weight_matrices.emplace(plan.fixed_resize_weight_matrices.size(),
			                                           plan.fixed_resize_weight_matrices.data());
		}
	}
	const uint16_t* fixed_quant_tables =
	    impl->fixed_quant_tables.has_value() ? impl->fixed_quant_tables->get() : nullptr;
	const float* fixed_resize_weight_matrices =
	    impl->fixed_resize_weight_matrices.has_value() ? impl->fixed_resize_weight_matrices->get() : nullptr;
	JpegDctDeviceScratch local_scratch;
	auto&                scratch              = plan.scratch != nullptr ? *plan.scratch : local_scratch;
	scratch.configure_scheduling(
	    plan.use_low_priority_streams, plan.transform_blocks_per_launch, plan.transform_ctas_per_launch);
	impl->execution_stats.direct_dct_stream_priority      = scratch.direct_dct_stream_priority;
	impl->execution_stats.cuda_least_stream_priority      = scratch.cuda_least_stream_priority;
	impl->execution_stats.cuda_greatest_stream_priority   = scratch.cuda_greatest_stream_priority;
	impl->execution_stats.direct_dct_low_priority_streams = scratch.direct_dct_low_priority_streams;
	switch (plan.scheduling_policy) {
	case JpegDctSchedulingPolicy::kFullyOverlapped:
		impl->execution_stats.scheduling_policy = "fully-overlapped";
		break;
	case JpegDctSchedulingPolicy::kLimitedOverlap:
		impl->execution_stats.scheduling_policy = "limited-overlap";
		break;
	case JpegDctSchedulingPolicy::kSerial:
		impl->execution_stats.scheduling_policy = "serial";
		break;
	}
	auto& cached_pending       = scratch.host_cached_gather_items;
	auto& cached_fixed_pending = scratch.host_cached_fixed_transform_items;
	cached_pending.clear();
	cached_fixed_pending.clear();
	auto*      decode_cache = output_ycbcr_dct_grid && !output_weighted_grid ? nullptr : plan.cache;
	const bool use_deterministic_fixed_transform =
	    plan.unify_rowgroups_across_shards && decode_cache == nullptr && !plan.fixed_transform_item_order->empty();
	const auto* fixed_transform_item_order =
	    use_deterministic_fixed_transform ? plan.fixed_transform_item_order.get() : nullptr;
	const auto* fixed_transform_group_offsets =
	    use_deterministic_fixed_transform ? plan.fixed_transform_group_offsets.get() : nullptr;
	const bool mixed_physical_shards =
	    plan.shards->size() > 1U || (!plan.shards->empty() && plan.shards->front().mixed_physical_shards);
	if (plan.unify_rowgroups_across_shards && (mixed_physical_shards || plan.host_io_staged_rowgroups != 0U)) {
		execute_unified_image_major_plan(*plan.shards,
		                                 fixed_transform_item_order,
		                                 fixed_transform_group_offsets,
		                                 output,
		                                 y_output,
		                                 cbcr_output,
		                                 y_accum,
		                                 cbcr_accum,
		                                 fixed_quant_tables,
		                                 fixed_resize_weight_matrices,
		                                 plan.grid_transform,
		                                 output_ycbcr_dct_grid,
		                                 output_weighted_grid,
		                                 impl->selected_coefficients,
		                                 plan.coefficient_selection_shape,
		                                 decode_cache,
		                                 impl->cache_stats,
		                                 impl->execution_stats,
		                                 scratch,
		                                 plan.decode_batch_rowgroups,
		                                 plan.rowgroup_prefetch,
		                                 cached_pending,
		                                 cached_fixed_pending);
	} else {
		for (const auto& shard : *plan.shards) {
			if (shard.fls_path == nullptr) {
				throw std::runtime_error("JPEG DCT shard plan has no FLS path");
			}
			const bool enable_sparse_vector_reads =
			    std::any_of(shard.rowgroups.begin(), shard.rowgroups.end(), [](const auto& rowgroup) {
				    return rowgroup.sparse_storage_read;
			    });
			auto rdr = scratch.fls_reader(*shard.fls_path, enable_sparse_vector_reads);
			execute_shard_plan(rdr,
			                   shard,
			                   fixed_transform_item_order,
			                   fixed_transform_group_offsets,
			                   output,
			                   y_output,
			                   cbcr_output,
			                   y_accum,
			                   cbcr_accum,
			                   fixed_quant_tables,
			                   fixed_resize_weight_matrices,
			                   plan.grid_transform,
			                   output_ycbcr_dct_grid,
			                   output_weighted_grid,
			                   impl->selected_coefficients,
			                   plan.coefficient_selection_shape,
			                   decode_cache,
			                   impl->cache_stats,
			                   impl->execution_stats,
			                   scratch,
			                   use_deterministic_fixed_transform
			                       ? std::max(plan.decode_batch_rowgroups, shard.rowgroups.size())
			                       : plan.decode_batch_rowgroups,
			                   plan.rowgroup_prefetch,
			                   cached_pending,
			                   cached_fixed_pending);
		}
	}
	if (!output_ycbcr_dct_grid) {
		execute_cached_rowgroup_hits(cached_pending, output, impl->execution_stats, scratch);
	} else if (output_weighted_grid) {
		execute_cached_fixed_transform_hits(cached_fixed_pending,
		                                    fixed_quant_tables,
		                                    fixed_resize_weight_matrices,
		                                    plan.grid_transform,
		                                    y_accum,
		                                    cbcr_accum,
		                                    impl->execution_stats,
		                                    scratch);
	}
	drain_cached_gather(scratch, impl->execution_stats);
	cached_pending.clear();
	cached_fixed_pending.clear();
	if (output_weighted_grid) {
		round_fixed_ycbcr_grid_outputs(y_accum,
		                               cbcr_accum,
		                               y_output,
		                               cbcr_output,
		                               impl->y_coefficient_count,
		                               impl->cbcr_coefficient_count,
		                               scratch,
		                               impl->execution_stats,
		                               impl->fixed_grid_round_start_event,
		                               impl->completion_event,
		                               plan.grid_transform);
	}
	if (plan.cache != nullptr) {
		impl->cache_stats.capacity_bytes     = plan.cache->capacity_bytes();
		impl->cache_stats.resident_bytes     = plan.cache->resident_bytes();
		impl->cache_stats.resident_rowgroups = plan.cache->resident_rowgroups();
	}
	refresh_runtime_policy_summary(impl->execution_stats);
	impl->execution_stats.actual_saved_vector_count =
	    impl->execution_stats.full_vector_count >= impl->execution_stats.selected_vector_count
	        ? impl->execution_stats.full_vector_count - impl->execution_stats.selected_vector_count
	        : 0;
	impl->execution_stats.actual_vector_count = impl->execution_stats.selected_vector_count;
	impl->execution_stats.decode_granularity =
	    impl->execution_stats.selected_vector_count < impl->execution_stats.full_vector_count ? "selected-vector"
	                                                                                           : "rowgroup";
	impl->execution_stats.storage_read_granularity =
	    impl->execution_stats.vector_bundle_envelope_rowgroup_count != 0U
	        ? "vector-bundle-envelope"
	    : impl->execution_stats.vector_bundle_rowgroup_count != 0U
	        ? "vector-bundle-range"
	                                                 : impl->execution_stats.compressed_payload_bytes_read <
	                                                           impl->execution_stats.full_compressed_payload_bytes
	                                                     ? "selected-vector-range"
	                                                     : "rowgroup";
	impl->execution_stats.read_amplification =
	    impl->execution_stats.full_compressed_payload_bytes == 0
	        ? 0.0
	        : static_cast<double>(impl->execution_stats.compressed_payload_bytes_read) /
	              static_cast<double>(impl->execution_stats.full_compressed_payload_bytes);
	const auto native_device = galp::memory::device_pool_stats();
	const auto native_pinned = galp::memory::pinned_host_pool_stats();
	impl->execution_stats.direct_dct_h2d_stream_priority = scratch.actual_stream_priority(
	    scratch.decode_workset.transfer.h2d_stream ? scratch.decode_workset.transfer.h2d_stream.get() : nullptr);
	impl->execution_stats.direct_dct_decode_stream_priority = scratch.actual_stream_priority(
	    scratch.decode_workset.transfer.compute_stream ? scratch.decode_workset.transfer.compute_stream.get()
	                                                   : nullptr);
	impl->execution_stats.direct_dct_transform_stream_priority =
	    scratch.actual_stream_priority(scratch.transform_stream ? scratch.transform_stream.get() : nullptr);
	impl->execution_stats.direct_dct_round_stream_priority = scratch.actual_stream_priority(
	    scratch.fixed_grid_round_stream ? scratch.fixed_grid_round_stream.get() : nullptr);
	impl->execution_stats.direct_dct_stream_priority      = impl->execution_stats.direct_dct_transform_stream_priority;
	impl->execution_stats.galp_native_device_in_use_bytes = native_device.in_use_bytes;
	impl->execution_stats.galp_native_device_peak_in_use_bytes     = native_device.peak_in_use_bytes;
	impl->execution_stats.galp_native_device_cached_bytes          = native_device.cached_bytes;
	impl->execution_stats.galp_native_device_allocation_requests   = native_device.allocation_requests;
	impl->execution_stats.galp_native_device_cuda_allocation_count = native_device.cuda_allocation_count;
	impl->execution_stats.galp_native_device_cuda_allocation_bytes = native_device.cuda_allocation_bytes;
	impl->execution_stats.galp_native_pinned_in_use_bytes          = native_pinned.in_use_bytes;
	impl->execution_stats.galp_native_pinned_peak_in_use_bytes     = native_pinned.peak_in_use_bytes;
	impl->execution_stats.galp_native_pinned_cached_bytes          = native_pinned.cached_bytes;
	impl->execution_stats.galp_native_pinned_allocation_requests   = native_pinned.allocation_requests;
	impl->execution_stats.galp_native_pinned_cuda_allocation_count = native_pinned.cuda_allocation_count;
	impl->execution_stats.galp_native_pinned_cuda_allocation_bytes = native_pinned.cuda_allocation_bytes;
	if (reuse_fence != nullptr) {
		if (impl->completion_event && scratch.fixed_grid_round_stream) {
			// Record a reader-owned event after the batch-owned public event on
			// the same stream. A later submission can join this tail without
			// requiring the earlier batch object to be synchronized or destroyed.
			reuse_fence->record(scratch.fixed_grid_round_stream.get());
		} else {
			// Non-rounding paths join their workset streams internally before
			// returning from the executor.
			reuse_fence->mark_complete();
		}
	}
	return JpegDctDeviceBatch(std::move(impl));
	} catch (...) {
		if (reuse_fence != nullptr) {
			// Do not permit a possibly in-flight, partially submitted context to
			// be rebound after an execution failure.
			reuse_fence->mark_failed(std::current_exception());
		}
		throw;
	}
}

} // namespace galp::jpeg::detail
