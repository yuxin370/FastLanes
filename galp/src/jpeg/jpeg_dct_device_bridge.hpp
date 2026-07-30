#pragma once

#include "galp/jpeg_dct_device.hpp"
#include "jpeg/jpeg_dct_device_runtime.hpp"
#include <filesystem>
#include <memory>
#include <vector>

namespace galp::jpeg::detail {

struct JpegDctDeviceBatchPlan;

// Owns the mutable CUDA resources shared by executions issued through one
// shard reader. Planning and CPU reads deliberately do not depend on this
// object and remain independently concurrent.
class JpegDctDeviceHostBridge {
public:
	JpegDctDeviceHostBridge();
	~JpegDctDeviceHostBridge();

	JpegDctDeviceHostBridge(const JpegDctDeviceHostBridge&)            = delete;
	JpegDctDeviceHostBridge& operator=(const JpegDctDeviceHostBridge&) = delete;

	void CompileIoPlan(JpegDctDeviceBatchPlan& plan);
	void StageIo(JpegDctDeviceBatchPlan& plan);
	JpegDctDeviceBatch Execute(JpegDctDeviceBatchPlan plan, const JpegDctDeviceBatchOptions& options);

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

} // namespace galp::jpeg::detail
