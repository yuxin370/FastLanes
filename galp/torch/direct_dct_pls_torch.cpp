#include "direct_dct_pls_torch.hpp"
#include "direct_dct/native_batch_lifetime.hpp"
#include "galp/advanced/direct_dct_pls.hpp"
#include "galp/jpeg_dct_diagnostics.hpp"
#include "galp/profiles/registry.hpp"
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <atomic>
#include <cuda_runtime_api.h>
#include <memory>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <torch/extension.h>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using NativePool = galp::jpeg::DirectDctPlsPoolBatch;

uint64_t next_pls_lifetime_identity() noexcept {
	static std::atomic<uint64_t> next {1U};
	return next.fetch_add(1U, std::memory_order_relaxed);
}

// PLS uses the same native completion/reclaim authority as ordinary
// Direct-DCT batches. The opaque owner keeps both the DirectDctBatch and PLS
// postprocess allocations alive; no Torch allocator assumption or global
// synchronization is needed.
class PlsExternalTensorLifetime {
public:
	PlsExternalTensorLifetime(std::shared_ptr<NativePool> owner, const c10::DeviceIndex device)
	    : device_(device) {
		auto completion = std::make_shared<galp::direct_dct::NativeBatchCompletion>(
		    next_pls_lifetime_identity(),
		    reinterpret_cast<uintptr_t>(owner->cuda_completion_event()),
		    device);
		lease_ = std::make_shared<galp::direct_dct::NativeBatchLease>(
		    std::move(completion), std::static_pointer_cast<void>(std::move(owner)));
		lease_->retain_owner_reference();
	}

	~PlsExternalTensorLifetime() {
		static_cast<void>(lease_->release_owner_reference());
		static_cast<void>(galp::direct_dct::NativeBatchLease::reclaim_finished());
	}

	void retain_storage_reference() {
		lease_->retain_storage_reference();
	}

	void release_storage_reference() noexcept {
		static_cast<void>(lease_->release_storage_reference());
	}

	void register_current(const galp::direct_dct::ConsumerDependency::Source source) {
		register_stream(c10::cuda::getCurrentCUDAStream(device_), source);
	}

	void register_stream(
	    const c10::cuda::CUDAStream stream,
	    const galp::direct_dct::ConsumerDependency::Source source) {
		galp::direct_dct::ConsumerDependency dependency;
		dependency.cuda_device = stream.device_index();
		dependency.stream_identity = reinterpret_cast<uintptr_t>(stream.stream());
		dependency.source = source;
		lease_->register_consumer(std::move(dependency));
	}

private:
	std::shared_ptr<galp::direct_dct::NativeBatchLease> lease_;
	c10::DeviceIndex                                   device_;
};

galp::jpeg::DirectDctPlsCropPolicy parse_crop_policy(const std::string& value) {
	if (value == "per-sample" || value == "per_sample") {
		return galp::jpeg::DirectDctPlsCropPolicy::kPerSample;
	}
	if (value == "per-pls" || value == "per_pls") {
		return galp::jpeg::DirectDctPlsCropPolicy::kPerPls;
	}
	throw std::invalid_argument("crop_policy must be 'per-sample' or 'per-pls'");
}

galp::jpeg::DirectDctPlsOrderPolicy parse_order_policy(const std::string& value) {
	if (value == "global")
		return galp::jpeg::DirectDctPlsOrderPolicy::kGlobal;
	if (value == "closed-pool" || value == "closed_pool") {
		return galp::jpeg::DirectDctPlsOrderPolicy::kClosedPool;
	}
	if (value == "physical-order" || value == "physical_order" || value == "none") {
		return galp::jpeg::DirectDctPlsOrderPolicy::kPhysicalOrder;
	}
	throw std::invalid_argument("order_policy must be 'global', 'closed-pool', or 'physical-order'");
}

std::vector<int64_t> grid_shape(const galp::jpeg::DirectDctGridTensorDescriptor& descriptor) {
	return std::vector<int64_t>(descriptor.shape.begin(), descriptor.shape.end());
}

std::vector<int64_t> grid_strides(const galp::jpeg::DirectDctGridTensorDescriptor& descriptor) {
	return std::vector<int64_t>(descriptor.strides.begin(), descriptor.strides.end());
}

class TorchDirectDctPlsMicrobatch {
public:
	TorchDirectDctPlsMicrobatch(std::shared_ptr<NativePool>                owner,
	                            std::shared_ptr<PlsExternalTensorLifetime> storage_lifetime,
	                            const size_t                               index)
	    : owner_(std::move(owner))
	    , storage_lifetime_(std::move(storage_lifetime))
	    , index_(index) {
		if (!owner_ || !storage_lifetime_ || index_ >= owner_->microbatch_count()) {
			throw std::out_of_range("PLS microbatch index is outside the pool");
		}
	}

	torch::Tensor y() {
		return grid_tensor(view().y, y_);
	}
	torch::Tensor cbcr() {
		return grid_tensor(view().cbcr, cbcr_);
	}

	torch::Tensor targets() {
		storage_lifetime_->register_current(
		    galp::direct_dct::ConsumerDependency::Source::kGetterCompatibility);
		if (targets_.defined())
			return targets_;
		const auto  current    = view();
		const auto& descriptor = current.targets;
		if (descriptor.empty() || descriptor.data == nullptr) {
			throw std::runtime_error("native PLS pipeline did not produce model targets");
		}
		wait_for_completion(descriptor.cuda_device);
		const auto device   = static_cast<c10::DeviceIndex>(descriptor.cuda_device);
		auto       options  = torch::TensorOptions().dtype(torch::kFloat32).device(torch::Device(torch::kCUDA, device));
		auto       lifetime = storage_lifetime_;
		lifetime->retain_storage_reference();
		targets_            = torch::from_blob(
            const_cast<float*>(descriptor.data),
            {static_cast<int64_t>(descriptor.shape[0]), static_cast<int64_t>(descriptor.shape[1])},
            {static_cast<int64_t>(descriptor.strides[0]), static_cast<int64_t>(descriptor.strides[1])},
		    [lifetime = std::move(lifetime)](void*) mutable {
			    lifetime->release_storage_reference();
			    lifetime.reset();
		    },
            options);
		return targets_;
	}

	[[nodiscard]] uint32_t epoch() const noexcept {
		return owner_->epoch();
	}
	[[nodiscard]] uint32_t pool_index() const noexcept {
		return owner_->pool_index();
	}
	[[nodiscard]] size_t microbatch_index_in_pool() const noexcept {
		return index_;
	}
	[[nodiscard]] size_t pool_offset() const {
		return view().pool_offset;
	}
	[[nodiscard]] size_t image_count() const {
		return view().image_count;
	}
	[[nodiscard]] bool is_pool_end() const noexcept {
		return index_ + 1U == owner_->microbatch_count();
	}
	[[nodiscard]] std::vector<uint32_t> global_image_ids() const {
		const auto values = view().global_image_ids;
		return {values.begin(), values.end()};
	}
	[[nodiscard]] std::vector<int64_t> labels() const {
		const auto values = view().labels;
		return {values.begin(), values.end()};
	}

	void record_current_consumer_stream() const {
		wait_for_completion(owner_->batch().cuda_device());
		storage_lifetime_->register_current(
		    galp::direct_dct::ConsumerDependency::Source::kExplicitConsumer);
	}

	void record_consumer_stream(const uintptr_t stream_identity, const int cuda_device) const {
		const auto device = static_cast<c10::DeviceIndex>(owner_->batch().cuda_device());
		if (cuda_device != static_cast<int>(device)) {
			throw std::invalid_argument("PLS consumer stream device does not match the batch device");
		}
		c10::cuda::CUDAGuard guard(device);
		const auto stream = stream_identity == 0U
		                        ? c10::cuda::getDefaultCUDAStream(device)
		                        : c10::cuda::getStreamFromExternal(
		                              reinterpret_cast<cudaStream_t>(stream_identity), device);
		if (auto* event = owner_->cuda_completion_event(); event != nullptr) {
			C10_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(), static_cast<cudaEvent_t>(event), 0U));
		}
		storage_lifetime_->register_stream(
		    stream, galp::direct_dct::ConsumerDependency::Source::kExplicitConsumer);
	}

private:
	galp::jpeg::DirectDctPlsMicrobatchView view() const {
		return owner_->microbatch(index_);
	}

	torch::Tensor grid_tensor(const galp::jpeg::DirectDctGridTensorDescriptor& descriptor, torch::Tensor& cached) {
		storage_lifetime_->register_current(
		    galp::direct_dct::ConsumerDependency::Source::kGetterCompatibility);
		if (cached.defined())
			return cached;
		if (descriptor.empty() || descriptor.raw_data() == nullptr) {
			throw std::runtime_error("native PLS pipeline produced an empty model tensor");
		}
		wait_for_completion(descriptor.cuda_device);
		const auto scalar =
		    descriptor.dtype == galp::jpeg::DirectDctTensorDataType::kFloat32 ? torch::kFloat32 : torch::kInt16;
		const auto device   = static_cast<c10::DeviceIndex>(descriptor.cuda_device);
		auto       options  = torch::TensorOptions().dtype(scalar).device(torch::Device(torch::kCUDA, device));
		auto       lifetime = storage_lifetime_;
		lifetime->retain_storage_reference();
		cached              = torch::from_blob(
            const_cast<void*>(descriptor.raw_data()),
            grid_shape(descriptor),
            grid_strides(descriptor),
		    [lifetime = std::move(lifetime)](void*) mutable {
			    lifetime->release_storage_reference();
			    lifetime.reset();
		    },
            options);
		return cached;
	}

	void wait_for_completion(const int cuda_device) const {
		auto* event = owner_->cuda_completion_event();
		if (event == nullptr)
			return;
		const auto           device = static_cast<c10::DeviceIndex>(cuda_device);
		c10::cuda::CUDAGuard guard(device);
		const auto           stream = c10::cuda::getCurrentCUDAStream(device);
		C10_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(), static_cast<cudaEvent_t>(event), 0U));
	}

	std::shared_ptr<NativePool>                owner_;
	std::shared_ptr<PlsExternalTensorLifetime> storage_lifetime_;
	size_t                                     index_ = 0U;
	torch::Tensor                              y_;
	torch::Tensor                              cbcr_;
	torch::Tensor                              targets_;
};

class TorchDirectDctPlsPool : public std::enable_shared_from_this<TorchDirectDctPlsPool> {
public:
	explicit TorchDirectDctPlsPool(galp::jpeg::DirectDctPlsPoolBatch pool)
	    : owner_(std::make_shared<NativePool>(std::move(pool)))
	    , storage_lifetime_(std::make_shared<PlsExternalTensorLifetime>(
	          owner_, static_cast<c10::DeviceIndex>(owner_->batch().cuda_device()))) {
	}
	~TorchDirectDctPlsPool() {
		retire();
	}

	void retire() noexcept {
		if (owner_) {
			owner_->retire_context();
		}
	}

	std::shared_ptr<TorchDirectDctPlsMicrobatch> microbatch(const size_t index) const {
		return std::make_shared<TorchDirectDctPlsMicrobatch>(owner_, storage_lifetime_, index);
	}

	std::shared_ptr<TorchDirectDctPlsMicrobatch> next() {
		if (next_microbatch_ >= owner_->microbatch_count())
			throw py::stop_iteration();
		return microbatch(next_microbatch_++);
	}

	[[nodiscard]] uint32_t epoch() const noexcept {
		return owner_->epoch();
	}
	[[nodiscard]] uint32_t pool_index() const noexcept {
		return owner_->pool_index();
	}
	[[nodiscard]] size_t image_count() const noexcept {
		return owner_->image_count();
	}
	[[nodiscard]] size_t microbatch_count() const noexcept {
		return owner_->microbatch_count();
	}
	[[nodiscard]] const std::vector<uint32_t>& virtual_pls_ids() const noexcept {
		return owner_->virtual_pls_ids();
	}

	[[nodiscard]] py::dict execution_stats() const {
		const auto& stats = owner_->batch().execution_stats_ref();
		py::dict    out;
		out["planned_selected_vector_count"]      = stats.planned_selected_vector_count;
		out["selected_vector_count"]              = stats.selected_vector_count;
		out["full_vector_count"]                  = stats.full_vector_count;
		out["decoded_coefficient_bytes"]          = stats.decoded_coefficient_bytes;
		out["rowgroup_count"]                     = stats.rowgroup_count;
		out["fixed_transform_source_block_count"] = stats.fixed_transform_source_block_count;
		out["fixed_transform_output_block_count"] = stats.fixed_transform_output_block_count;
		out["compressed_payload_bytes_read"]      = stats.compressed_payload_bytes_read;
		out["selected_compressed_payload_bytes"]  = stats.selected_compressed_payload_bytes;
		out["full_compressed_payload_bytes"]      = stats.full_compressed_payload_bytes;
		out["physical_page_bytes_covered"]        = stats.physical_page_bytes_covered;
		out["full_physical_page_bytes"]           = stats.full_physical_page_bytes;
		out["duplicate_physical_read_count"]      = stats.duplicate_physical_read_count;
		out["physical_read_order_inversions"]     = stats.physical_read_order_inversions;
		out["bounded_physical_storage_bytes"]     = stats.bounded_physical_storage_bytes;
		out["bounded_physical_run_count"]         = stats.bounded_physical_run_count;
		out["io_uring_read_request_count"]        = stats.io_uring_read_request_count;
		out["io_uring_fallback_count"]            = stats.io_uring_fallback_count;
		out["uses_planless_fixed_transform"]      = stats.uses_planless_fixed_transform;
		out["runtime_policy_decision"]            = stats.runtime_policy_decision;
		out["runtime_policy_reason"]              = stats.runtime_policy_reason;
		return out;
	}

private:
	friend class TorchDirectDctPlsPipeline;
	std::shared_ptr<NativePool>                owner_;
	std::shared_ptr<PlsExternalTensorLifetime> storage_lifetime_;
	size_t                                     next_microbatch_ = 0U;
};

class TorchDirectDctPlsPipeline : public std::enable_shared_from_this<TorchDirectDctPlsPipeline> {
public:
	TorchDirectDctPlsPipeline(const std::string& manifest_path,
	                          const std::string& premixed_mapping_csv,
	                          const uint64_t     training_seed,
	                          const std::string& expected_mapping_sha256,
	                          const std::string& crop_policy,
	                          const std::string& order_policy,
	                          const uint32_t     segments_per_pool,
	                          const uint32_t     microbatch_images,
	                          const uint32_t     segment_images,
	                          const uint32_t     model_classes,
	                          const std::string& profile_id) {
		if (profile_id != galp::profiles::kRgbNoMoreTrainingPlsProfileId) {
			throw std::invalid_argument("DirectDctPlsPipeline requires registered profile 'rgbnomore-training-pls-v1'");
		}
		if (segments_per_pool == 0U || microbatch_images == 0U || segment_images == 0U) {
			throw std::invalid_argument("PLS, pool, and microbatch sizes must be positive");
		}
		galp::jpeg::DirectDctPlsPipelineOptions options;
		options.schedule.training_seed     = training_seed;
		options.schedule.segments_per_pool = segments_per_pool;
		options.schedule.microbatch_images = microbatch_images;
		options.schedule.crop_policy       = parse_crop_policy(crop_policy);
		options.schedule.order_policy      = parse_order_policy(order_policy);
		options.device =
		    galp::profiles::materialize_direct_dct_options(galp::profiles::resolve_direct_dct_profile(profile_id));
		options.segment_images          = segment_images;
		options.model_classes           = model_classes;
		options.expected_mapping_sha256 = expected_mapping_sha256;
		pipeline_ =
		    std::make_unique<galp::jpeg::DirectDctPlsPipeline>(manifest_path, premixed_mapping_csv, std::move(options));
	}

	void start_epoch(const uint32_t epoch) {
		static_cast<void>(galp::direct_dct::NativeBatchLease::reclaim_finished());
		ensure_open();
		current_pool_.reset();
		pipeline_->start_epoch(epoch);
	}

	std::shared_ptr<TorchDirectDctPlsPool> next_pool() {
		static_cast<void>(galp::direct_dct::NativeBatchLease::reclaim_finished());
		ensure_open();
		if (!pipeline_->has_next_pool())
			throw py::stop_iteration();
		py::gil_scoped_release release;
		auto                   pool = pipeline_->next_pool();
		return std::make_shared<TorchDirectDctPlsPool>(std::move(pool));
	}

	std::shared_ptr<TorchDirectDctPlsMicrobatch> next() {
		while (!current_pool_ || current_pool_->next_microbatch_ >= current_pool_->microbatch_count()) {
			current_pool_ = next_pool();
		}
		return current_pool_->next();
	}

	void close() noexcept {
		current_pool_.reset();
		pipeline_.reset();
		static_cast<void>(galp::direct_dct::NativeBatchLease::reclaim_finished());
	}

	[[nodiscard]] bool has_next_pool() const noexcept {
		return pipeline_ && pipeline_->has_next_pool();
	}
	[[nodiscard]] size_t sample_count() const {
		ensure_open();
		return pipeline_->layout().sample_count();
	}
	[[nodiscard]] size_t pls_count() const {
		ensure_open();
		return pipeline_->layout().pls_count();
	}
	[[nodiscard]] uint32_t segment_images() const {
		ensure_open();
		return pipeline_->layout().segment_images();
	}
	[[nodiscard]] py::dict prefetch_stats() const {
		ensure_open();
		const auto stats = pipeline_->prefetch_stats();
		py::dict out;
		out["context_capacity"]        = stats.context_capacity;
		out["live_context_count"]      = stats.live_context_count;
		out["peak_live_context_count"] = stats.peak_live_context_count;
		out["context_waiter_count"]    = stats.context_waiter_count;
		out["peak_context_waiter_count"] = stats.peak_context_waiter_count;
		out["prepare_started_count"]   = stats.prepare_started_count;
		out["prepare_completed_count"] = stats.prepare_completed_count;
		out["activation_count"]        = stats.activation_count;
		out["retired_count"]           = stats.retired_count;
		out["prepare_plan_ms"]          = stats.prepare_plan_ms;
		out["prepare_io_ms"]            = stats.prepare_io_ms;
		out["prepare_materialize_ms"]   = stats.prepare_materialize_ms;
		out["activation_wait_ms"]       = stats.activation_wait_ms;
		out["activation_ms"]            = stats.activation_ms;
		return out;
	}

private:
	void ensure_open() const {
		if (!pipeline_)
			throw std::runtime_error("DirectDctPlsPipeline is closed");
	}

	std::unique_ptr<galp::jpeg::DirectDctPlsPipeline> pipeline_;
	std::shared_ptr<TorchDirectDctPlsPool>            current_pool_;
};

} // namespace

void bind_direct_dct_pls_torch(py::module_& module) {
	py::class_<TorchDirectDctPlsMicrobatch, std::shared_ptr<TorchDirectDctPlsMicrobatch>>(module,
	                                                                                      "DirectDctPlsMicrobatch")
	    .def_property_readonly("y", &TorchDirectDctPlsMicrobatch::y)
	    .def_property_readonly("cbcr", &TorchDirectDctPlsMicrobatch::cbcr)
	    .def_property_readonly("targets", &TorchDirectDctPlsMicrobatch::targets)
	    .def("record_stream", &TorchDirectDctPlsMicrobatch::record_current_consumer_stream,
	         "Register the current CUDA stream as an actual PLS tensor consumer.")
	    .def("record_stream", &TorchDirectDctPlsMicrobatch::record_consumer_stream,
	         py::arg("stream_identity"), py::arg("cuda_device"),
	         "Register an explicit CUDA stream as an actual PLS tensor consumer.")
	    .def_property_readonly("epoch", &TorchDirectDctPlsMicrobatch::epoch)
	    .def_property_readonly("pool_index", &TorchDirectDctPlsMicrobatch::pool_index)
	    .def_property_readonly("microbatch_index_in_pool", &TorchDirectDctPlsMicrobatch::microbatch_index_in_pool)
	    .def_property_readonly("pool_offset", &TorchDirectDctPlsMicrobatch::pool_offset)
	    .def_property_readonly("image_count", &TorchDirectDctPlsMicrobatch::image_count)
	    .def_property_readonly("is_pool_end", &TorchDirectDctPlsMicrobatch::is_pool_end)
	    .def_property_readonly("global_image_ids", &TorchDirectDctPlsMicrobatch::global_image_ids)
	    .def_property_readonly("labels", &TorchDirectDctPlsMicrobatch::labels);

	py::class_<TorchDirectDctPlsPool, std::shared_ptr<TorchDirectDctPlsPool>>(module, "DirectDctPlsPool")
	    .def("__iter__", [](const std::shared_ptr<TorchDirectDctPlsPool>& pool) { return pool; })
	    .def("__next__", &TorchDirectDctPlsPool::next)
	    .def("microbatch", &TorchDirectDctPlsPool::microbatch, py::arg("index"))
	    .def("retire", &TorchDirectDctPlsPool::retire,
	         "Retire the consumed scheduling context; tensor backing remains native-lifetime protected.")
	    .def_property_readonly("epoch", &TorchDirectDctPlsPool::epoch)
	    .def_property_readonly("pool_index", &TorchDirectDctPlsPool::pool_index)
	    .def_property_readonly("image_count", &TorchDirectDctPlsPool::image_count)
	    .def_property_readonly("microbatch_count", &TorchDirectDctPlsPool::microbatch_count)
	    .def_property_readonly("virtual_pls_ids", &TorchDirectDctPlsPool::virtual_pls_ids)
	    .def_property_readonly("execution_stats", &TorchDirectDctPlsPool::execution_stats);

	py::class_<TorchDirectDctPlsPipeline, std::shared_ptr<TorchDirectDctPlsPipeline>>(module, "DirectDctPlsPipeline")
	    .def(py::init<const std::string&,
	                  const std::string&,
	                  uint64_t,
	                  const std::string&,
	                  const std::string&,
	                  const std::string&,
	                  uint32_t,
	                  uint32_t,
	                  uint32_t,
	                  uint32_t,
	                  const std::string&>(),
	         py::arg("manifest_path"),
	         py::arg("premixed_mapping_csv"),
	         py::arg("training_seed"),
	         py::arg("expected_mapping_sha256"),
	         py::arg("crop_policy")       = "per-pls",
	         py::arg("order_policy")      = "closed-pool",
	         py::arg("segments_per_pool") = 4U,
	         py::arg("microbatch_images") = 64U,
	         py::arg("segment_images")    = 1024U,
	         py::arg("model_classes")     = 1000U,
	         py::arg("profile_id")        = std::string(galp::profiles::kRgbNoMoreTrainingPlsProfileId))
	    .def("start_epoch", &TorchDirectDctPlsPipeline::start_epoch, py::arg("epoch"))
	    .def("next_pool", &TorchDirectDctPlsPipeline::next_pool)
	    .def("close", &TorchDirectDctPlsPipeline::close)
	    .def("__iter__", [](const std::shared_ptr<TorchDirectDctPlsPipeline>& pipeline) { return pipeline; })
	    .def("__next__", &TorchDirectDctPlsPipeline::next)
	    .def_property_readonly("has_next_pool", &TorchDirectDctPlsPipeline::has_next_pool)
	    .def_property_readonly("sample_count", &TorchDirectDctPlsPipeline::sample_count)
	    .def_property_readonly("pls_count", &TorchDirectDctPlsPipeline::pls_count)
	    .def_property_readonly("segment_images", &TorchDirectDctPlsPipeline::segment_images)
	    .def_property_readonly("prefetch_stats", &TorchDirectDctPlsPipeline::prefetch_stats);

	module.def("reclaim_direct_dct_pls_pools", []() {
		return galp::direct_dct::NativeBatchLease::reclaim_finished();
	});
}
