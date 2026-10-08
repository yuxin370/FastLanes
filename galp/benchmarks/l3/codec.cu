// CUDA entry points for the minimally repaired SNU-ARC/L3 prototype.
#include "upstream/l3.cuh"

extern "C" int
l3_encode_rows(unsigned char* input, unsigned char* packed, unsigned char* offsets, cudaStream_t stream) {
	for (int c = 0; c < 3; c++) {
		encoder<<<1, dim3(num_wd, num_ht), 0, stream>>>(
		    input + c * wd * ht, packed + c * (shard + 2) * num_wd * ht, offsets + c * num_wd * ht);
	}
	return cudaGetLastError();
}

extern "C" void l3_decode(unsigned char* input, int* offsets, unsigned char* output, cudaStream_t stream) {
	constexpr int count = num_wd * num_ht + 1;
	decoder<<<dim3(num_wd, num_ht), shard, 0, stream>>>(input, offsets, offsets + count, offsets + 2 * count, output);
}
