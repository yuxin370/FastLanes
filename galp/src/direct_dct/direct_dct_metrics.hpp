#ifndef GALP_DIRECT_DCT_DIRECT_DCT_METRICS_HPP
#define GALP_DIRECT_DCT_DIRECT_DCT_METRICS_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace galp::direct_dct {

inline constexpr std::string_view kDirectDctMetricsSchema = "galp-direct-dct-metrics-v2";
inline constexpr uint32_t         kDirectDctMetricsSchemaVersion = 2U;

enum class MetricValueType : uint8_t { kBoolean, kUnsignedInteger, kFloatingPoint, kString };
enum class MetricScope : uint8_t { kBatch, kPipeline };
enum class MetricUnit : uint8_t { kBoolean, kCount, kMilliseconds, kBytes, kIdentifier };
enum class MetricReducer : uint8_t { kSum, kMaximum, kInvariant };
enum class MetricCompletionRequirement : uint8_t { kHostSnapshot, kGpuCompletion };

struct MetricDescriptor final {
	std::string_view                    name;
	MetricValueType                     value_type;
	MetricScope                         scope;
	MetricUnit                          unit;
	MetricReducer                       reducer;
	MetricCompletionRequirement         completion_requirement;
	uint32_t                             schema_version;
};

[[nodiscard]] std::span<const MetricDescriptor> direct_dct_metric_descriptors() noexcept;

// One already-reduced native observation. Batch observations use counts 1/0;
// pipeline snapshots may carry larger counts and can be combined without
// teaching Python how individual fields reduce.
struct DirectDctMetricsObservation final {
	bool     host_snapshot_taken = true;
	bool     gpu_timings_finalized = false;
	double   consumer_wait_ms = 0.0;
	double   submit_to_ready_ms = 0.0;
	double   producer_ms = 0.0;
	double   planning_ms = 0.0;
	double   io_ms = 0.0;
	double   decode_ms = 0.0;
	double   transform_ms = 0.0;
	uint64_t logical_bytes = 0U;
	uint64_t physical_bytes = 0U;
	uint64_t peak_transient_bytes = 0U;
	uint64_t consumed_batches = 1U;
	uint64_t completed_batches = 0U;
};

struct DirectDctMetricsSnapshot final {
	bool     host_snapshot_taken = true;
	bool     gpu_timings_finalized = true;
	double   consumer_wait_ms = 0.0;
	double   submit_to_ready_ms = 0.0;
	double   producer_ms = 0.0;
	double   planning_ms = 0.0;
	double   io_ms = 0.0;
	double   decode_ms = 0.0;
	double   transform_ms = 0.0;
	uint64_t logical_bytes = 0U;
	uint64_t physical_bytes = 0U;
	uint64_t peak_transient_bytes = 0U;
	uint64_t consumed_batches = 0U;
	uint64_t completed_batches = 0U;
};

class DirectDctMetricsAggregator final {
public:
	void reset() noexcept;

	// Aggregate a complete or partially completed already-reduced snapshot.
	// GPU values represent exactly completed_batches; the finalized flag means
	// every consumed batch is complete, not that partial values are invalid.
	void observe(const DirectDctMetricsObservation& observation);

	// Hot-path split used by NativeLogicalBatchPipeline: host values are added
	// at delivery, then GPU values are committed exactly once after the existing
	// completion event becomes ready.
	void observe_host(const DirectDctMetricsObservation& observation);
	void observe_gpu_completion(double decode_ms, double transform_ms, uint64_t peak_transient_bytes) noexcept;

	[[nodiscard]] DirectDctMetricsSnapshot snapshot() const noexcept;

private:
	DirectDctMetricsSnapshot values_ {};
};

} // namespace galp::direct_dct

#endif // GALP_DIRECT_DCT_DIRECT_DCT_METRICS_HPP
