// DALI 2.x integration of the upstream L3 decoder and stream arrangement.
#include "dali/core/cuda_event.h"
#include "dali/core/cuda_stream.h"
#include "dali/pipeline/operator/operator.h"
#include <array>
#include <cstdint>
#include <cstring>

extern "C" void l3_decode(unsigned char*, int*, unsigned char*, cudaStream_t);

namespace dali {
class L3Decoder : public Operator<MixedBackend> {
	static constexpr int kPatch   = 32;
	static constexpr int kPatches = (512 / kPatch) * (512 / kPatch);
	static constexpr int kStreams = 32; // L3THREAD in the author's DALI patch.

public:
	explicit L3Decoder(const OpSpec& spec)
	    : Operator<MixedBackend>(spec) {
		offsets_host_.set_pinned(true);
		int least, greatest;
		CUDA_CALL(cudaDeviceGetStreamPriorityRange(&least, &greatest));
		int device = spec.GetArgument<int>("device_id");
		ready_     = CUDAEvent::CreateWithFlags(cudaEventDisableTiming, device);
		for (int i = 0; i < kStreams; i++) {
			streams_[i] = CUDAStream::CreateWithPriority(true, least, device);
			done_[i]    = CUDAEvent::CreateWithFlags(cudaEventDisableTiming, device);
		}
	}

protected:
	bool SetupImpl(std::vector<OutputDesc>& desc, const Workspace& ws) override {
		const auto& input = ws.Input<CPUBackend>(0);
		DALI_ENFORCE(input.type() == DALI_UINT8, "L3 input must be encoded bytes");
		int n = input.num_samples();
		offsets_host_.Resize(uniform_list_shape(n, TensorShape<1> {3 * (kPatches + 1)}), DALI_INT32);
		for (int i = 0; i < n; i++) {
			const auto* data  = input.tensor<uint8_t>(i);
			auto        bytes = input.tensor_shape(i).num_elements();
			DALI_ENFORCE(bytes >= 13 + 3 * kPatches * 2, "Truncated L3 header");
			int32_t width, height;
			std::memcpy(&width, data + 4, 4);
			std::memcpy(&height, data + 8, 4);
			DALI_ENFORCE(std::memcmp(data, "LLL.", 4) == 0 && width == 512 && height == 512 && data[12] == kPatch,
			             "This L3 baseline supports 512x512 RGB images with 32x32 patches");
			auto* offset = offsets_host_.mutable_tensor<int32_t>(i);
			int   total  = 13 + 3 * kPatches * 2;
			for (int c = 0; c < 3; c++) {
				offset[c * (kPatches + 1)] = 0;
				for (int p = 0; p < kPatches; p++) {
					uint16_t length;
					std::memcpy(&length, data + 13 + (c * kPatches + p) * 2, 2);
					DALI_ENFORCE(length >= kPatch * 2 && length <= kPatch * (kPatch + 2), "Invalid L3 patch length");
					offset[c * (kPatches + 1) + p + 1] = offset[c * (kPatches + 1) + p] + length;
				}
				total += offset[c * (kPatches + 1) + kPatches];
			}
			DALI_ENFORCE(total == bytes, "L3 payload length does not match its header");
		}
		desc.resize(1);
		desc[0] = {uniform_list_shape(n, TensorShape<3> {512, 512, 3}), DALI_UINT8};
		return true;
	}

	void RunImpl(Workspace& ws) override {
		const auto& input  = ws.Input<CPUBackend>(0);
		auto&       output = ws.Output<GPUBackend>(0);
		encoded_.Copy(input, ws.stream());
		offsets_gpu_.Copy(offsets_host_, ws.stream());
		CUDA_CALL(cudaEventRecord(ready_, ws.stream()));
		int active = std::min(kStreams, input.num_samples());
		for (int i = 0; i < active; i++) {
			CUDA_CALL(cudaStreamWaitEvent(streams_[i], ready_, 0));
		}
		for (int i = 0; i < input.num_samples(); i++) {
			l3_decode(encoded_.mutable_tensor<uint8_t>(i),
			          offsets_gpu_.mutable_tensor<int32_t>(i),
			          output.mutable_tensor<uint8_t>(i),
			          streams_[i % kStreams]);
		}
		CUDA_CALL(cudaGetLastError());
		// Join before downstream resize/normalize and before reusing the input buffers.
		for (int i = 0; i < active; i++) {
			CUDA_CALL(cudaEventRecord(done_[i], streams_[i]));
			CUDA_CALL(cudaStreamWaitEvent(ws.stream(), done_[i], 0));
		}
		output.SetLayout("HWC");
	}

private:
	std::array<CUDAStream, kStreams> streams_;
	std::array<CUDAEvent, kStreams>  done_;
	CUDAEvent                        ready_;
	TensorList<CPUBackend>           offsets_host_;
	TensorList<GPUBackend>           offsets_gpu_, encoded_;
};

DALI_REGISTER_OPERATOR(L3Decoder, L3Decoder, Mixed);
DALI_SCHEMA(L3Decoder)
    .DocStr("Decode repaired L3 ImageNet-512 files to RGB on the GPU.")
    .NumInput(1)
    .NumOutput(1)
    .InputDevice(0, InputDevice::CPU);
} // namespace dali
