#include "jpeg/jpeg_dct_prepared_plan.hpp"
#include <utility>

namespace galp::jpeg {

JpegDctDeviceBatchPreparedPlan::JpegDctDeviceBatchPreparedPlan() noexcept = default;

JpegDctDeviceBatchPreparedPlan::JpegDctDeviceBatchPreparedPlan(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
}

JpegDctDeviceBatchPreparedPlan::~JpegDctDeviceBatchPreparedPlan()                                         = default;
JpegDctDeviceBatchPreparedPlan::JpegDctDeviceBatchPreparedPlan(JpegDctDeviceBatchPreparedPlan&&) noexcept = default;
JpegDctDeviceBatchPreparedPlan&
JpegDctDeviceBatchPreparedPlan::operator=(JpegDctDeviceBatchPreparedPlan&&) noexcept = default;

bool JpegDctDeviceBatchPreparedPlan::empty() const noexcept {
	return impl_ == nullptr;
}

JpegDctDeviceLayout JpegDctDeviceBatchPreparedPlan::layout() const noexcept {
	return impl_ ? impl_->plan.layout : JpegDctDeviceLayout::kImageMajorComponentBlockCoeff;
}

const std::vector<JpegDctDeviceImageLayout>& JpegDctDeviceBatchPreparedPlan::image_layouts() const noexcept {
	static const std::vector<JpegDctDeviceImageLayout> empty;
	return impl_ ? impl_->plan.image_layouts : empty;
}

const std::vector<JpegDctDeviceBlockMetadata>& JpegDctDeviceBatchPreparedPlan::block_metadata() const noexcept {
	static const std::vector<JpegDctDeviceBlockMetadata> empty;
	return impl_ ? impl_->plan.block_metadata : empty;
}

const std::vector<JpegDctDeviceRowgroupMetadata>& JpegDctDeviceBatchPreparedPlan::rowgroups() const noexcept {
	static const std::vector<JpegDctDeviceRowgroupMetadata> empty;
	return impl_ ? impl_->plan.rowgroups : empty;
}

size_t JpegDctDeviceBatchPreparedPlan::planned_selected_vector_count() const noexcept {
	return impl_ ? impl_->plan.planned_selected_vector_count : 0;
}

size_t JpegDctDeviceBatchPreparedPlan::estimated_selected_vector_count() const noexcept {
	return impl_ ? impl_->plan.estimated_selected_vector_count : 0;
}

size_t JpegDctDeviceBatchPreparedPlan::full_vector_count() const noexcept {
	return impl_ ? impl_->plan.full_vector_count : 0;
}

size_t JpegDctDeviceBatchPreparedPlan::planned_saved_vector_count() const noexcept {
	return impl_ ? impl_->plan.planned_saved_vector_count : 0;
}

size_t JpegDctDeviceBatchPreparedPlan::estimated_saved_vector_count() const noexcept {
	return impl_ ? impl_->plan.estimated_saved_vector_count : 0;
}

const std::vector<uint8_t>& JpegDctDeviceBatchPreparedPlan::selected_coefficients() const noexcept {
	static const std::vector<uint8_t> empty;
	return impl_ ? impl_->plan.selected_coefficients : empty;
}

size_t JpegDctDeviceBatchPreparedPlan::coefficients_per_block() const noexcept {
	return impl_ ? impl_->plan.coefficients_per_block : detail::kJpegDctCoefficientCount;
}

double JpegDctDeviceBatchPreparedPlan::planned_selected_vector_ratio() const noexcept {
	return impl_ ? impl_->plan.planned_selected_vector_ratio : 0.0;
}

double JpegDctDeviceBatchPreparedPlan::estimated_selected_vector_ratio() const noexcept {
	return impl_ ? impl_->plan.estimated_selected_vector_ratio : 0.0;
}

double JpegDctDeviceBatchPreparedPlan::planning_ms() const noexcept {
	return impl_ ? impl_->plan.planning_ms : 0.0;
}

bool JpegDctDeviceBatchPreparedPlan::uses_planless_fixed_transform() const noexcept {
	return impl_ != nullptr && impl_->plan.uses_planless_fixed_transform;
}

bool JpegDctDeviceBatchPreparedPlan::exact_batch_plan_cache_enabled() const noexcept {
	return impl_ != nullptr && impl_->plan.exact_batch_plan_cache_enabled;
}

bool JpegDctDeviceBatchPreparedPlan::decoded_rowgroup_cache_enabled() const noexcept {
	return impl_ != nullptr && impl_->plan.cache_enabled;
}

size_t JpegDctDeviceBatchPreparedPlan::automatic_sparse_storage_candidate_rowgroup_count() const noexcept {
	return impl_ ? impl_->plan.automatic_sparse_storage_candidate_rowgroup_count : 0U;
}

size_t JpegDctDeviceBatchPreparedPlan::automatic_sparse_storage_selected_rowgroup_count() const noexcept {
	return impl_ ? impl_->plan.automatic_sparse_storage_selected_rowgroup_count : 0U;
}

size_t JpegDctDeviceBatchPreparedPlan::automatic_sparse_storage_rejected_rowgroup_count() const noexcept {
	return impl_ ? impl_->plan.automatic_sparse_storage_rejected_rowgroup_count : 0U;
}

bool JpegDctDeviceBatchPreparedPlan::host_io_staged() const noexcept {
	return impl_ != nullptr && impl_->plan.host_io_staged;
}

size_t JpegDctDeviceBatchPreparedPlan::host_io_staged_rowgroups() const noexcept {
	return impl_ ? impl_->plan.host_io_staged_rowgroups : 0U;
}

double JpegDctDeviceBatchPreparedPlan::host_io_staging_ms() const noexcept {
	return impl_ ? impl_->plan.host_io_staging_ms : 0.0;
}

} // namespace galp::jpeg
