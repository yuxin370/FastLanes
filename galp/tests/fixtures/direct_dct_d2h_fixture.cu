#include <cuda_runtime_api.h>

void rejected_direct_dct_host_transfer(void* host, const void* device, const size_t bytes, cudaStream_t stream) {
	cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost, stream);
}
