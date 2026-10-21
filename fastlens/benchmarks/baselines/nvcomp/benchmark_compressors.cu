// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// fastlens/benchmarks/nvcomp/benchmark_compressors.cu
// ────────────────────────────────────────────────────────
#include "fastlens_extensions/alp/alp_bindings.cuh"
#include "fastlens_bench/data.cuh"
#include "cuda/device_utils.cuh"
#include "cuda/launch/dispatch.cuh"
#include "codecs/decode/alp.cuh"
#include "fastlens_bench/nvcomp/benchmark_compressors.cuh"
#include "fastlens_bench/nvcomp/nvcomp_compressors.cuh"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>

template <typename T, unsigned UNPACK_N_VECTORS>
using ALPDecompressor = typename fastlens::codec::device::ALPDecompressor<
    T,
    UNPACK_N_VECTORS,
    fastlens::codec::device::
        BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, 1, fastlens::codec::device::ALPFunctor<T, UNPACK_N_VECTORS>>,
    fastlens::codec::device::StatefulALPExceptionPatcher<T, UNPACK_N_VECTORS, 1>,
    fastlens::codec::device::ALPColumn<T>>;

template <typename T, unsigned UNPACK_N_VECTORS>
using ALPExtendedDecompressor = typename fastlens::codec::device::ALPDecompressor<
    T,
    UNPACK_N_VECTORS,
    fastlens::codec::device::
        BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, 1, fastlens::codec::device::ALPFunctor<T, UNPACK_N_VECTORS>>,
    fastlens::codec::device::PrefetchAllALPExceptionPatcher<T, UNPACK_N_VECTORS, 1>,
    fastlens::codec::device::ALPExtendedColumn<T>>;

template <typename T>
BenchmarkResult benchmark_thrust(const T* input, const size_t value_count, const T value_to_search_for) {
	CudaStopwatch stopwatch = CudaStopwatch();
	double        execution_time_ms;

	thrust::device_vector<T> d_vec(input, input + value_count);
	stopwatch.start();
	bool result       = thrust::any_of(thrust::device, d_vec.begin(), d_vec.end(), is_equal_to<T>(value_to_search_for));
	execution_time_ms = stopwatch.stop();
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	return BenchmarkResult {result, execution_time_ms, 1.0};
}

template <typename T>
BenchmarkResult benchmark_alp(const fastlens::bench::nvcomp::ComparisonType  comparison_type,
                              const fastlens::bench::nvcomp::CompressionType decompressor_enum,
                              const T*                                           input,
                              const fastlens::codec::host::ALPColumn<T>&   column,
                              const T                                            value_to_search_for) {
	bool           result = false;
	GPUArray<bool> d_query_result(1, &result);
	GPUArray<T>    d_decompression_result(column.get_n_values());

	constexpr int32_t           UNPACK_N_VECTORS = 1;
	const ThreadblockMapping<T> mapping(UNPACK_N_VECTORS, column.get_n_vecs());
	CudaStopwatch               stopwatch = CudaStopwatch();
	double                      execution_time_ms;
	double                      compression_ratio;

	switch (decompressor_enum) {
	case fastlens::bench::nvcomp::ALP: {
		fastlens::codec::device::ALPColumn<T> d_column = column.copy_to_device();
		stopwatch.start();
		if (comparison_type == fastlens::bench::nvcomp::ComparisonType::DECOMPRESSION) {
			fastlens::kernels::device::decompress_column<T,
			                                   UNPACK_N_VECTORS,
			                                   1,
			                                   ALPDecompressor<T, UNPACK_N_VECTORS>,
			                                   fastlens::codec::device::ALPColumn<T>>
			    <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(d_column, d_decompression_result.get());
		} else {
			fastlens::kernels::device::
			    query_column<T, UNPACK_N_VECTORS, 1, ALPDecompressor<T, UNPACK_N_VECTORS>, fastlens::codec::device::ALPColumn<T>>
			    <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(
			        d_column, d_query_result.get(), value_to_search_for);
		}
		execution_time_ms = stopwatch.stop();
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		fastlens::codec::host::free_column(d_column);
		compression_ratio = column.get_compression_ratio();
	} break;

	case fastlens::bench::nvcomp::FASTLENS: {
		fastlens::codec::host::ALPExtendedColumn<T>   column_extended = column.create_extended_column();
		fastlens::codec::device::ALPExtendedColumn<T> d_column        = column_extended.copy_to_device();
		stopwatch.start();
		if (comparison_type == fastlens::bench::nvcomp::ComparisonType::DECOMPRESSION) {
			fastlens::kernels::device::decompress_column<T,
			                                   UNPACK_N_VECTORS,
			                                   1,
			                                   ALPExtendedDecompressor<T, UNPACK_N_VECTORS>,
			                                   fastlens::codec::device::ALPExtendedColumn<T>>
			    <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(d_column, d_decompression_result.get());
		} else {
			fastlens::kernels::device::query_column<T,
			                              UNPACK_N_VECTORS,
			                              1,
			                              ALPExtendedDecompressor<T, UNPACK_N_VECTORS>,
			                              fastlens::codec::device::ALPExtendedColumn<T>>
			    <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(
			        d_column, d_query_result.get(), value_to_search_for);
		}
		execution_time_ms = stopwatch.stop();
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		compression_ratio = column_extended.get_compression_ratio();
		fastlens::codec::host::free_column(column_extended);
		fastlens::codec::host::free_column(d_column);
	} break;
	default:
		throw std::invalid_argument("Could not parse decompressor enum for alp\n");
	}

	bool kernel_successful = false;
	if (comparison_type == fastlens::bench::nvcomp::ComparisonType::DECOMPRESSION) {
		GPUArray<T> d_input(column.get_n_values(), input);

		kernel_successful =
		    check_if_device_buffers_are_equal<T>(d_decompression_result.get(), d_input.get(), column.get_n_values());
	}

	return BenchmarkResult {kernel_successful, execution_time_ms, compression_ratio};
}

template <typename T>
BenchmarkResult benchmark_alp(const fastlens::bench::nvcomp::ComparisonType  comparison_type,
                              const fastlens::bench::nvcomp::CompressionType decompressor_enum,
                              const T*                            input,
                              const size_t                        value_count,
                              const T                             value_to_search_for) {
	fastlens::codec::host::ALPColumn<T> column = fastlens::codec::alp::encode(input, value_count, true);
	auto result = benchmark_alp(comparison_type, decompressor_enum, input, column, value_to_search_for);
	fastlens::codec::host::free_column(column);

	return result;
}

template <typename T>
BenchmarkResult benchmark_hwc(const fastlens::bench::nvcomp::ComparisonType  comparison_type,
                              const fastlens::bench::nvcomp::CompressionType compression_type,
                              const T*                            input,
                              const size_t                        value_count,
                              const T                             value_to_search_for) {
	size_t                size_in_bytes = value_count * sizeof(T);
	GPUArray<uint8_t>     d_input_buffer(size_in_bytes, reinterpret_cast<const uint8_t*>(input));
	fastlens::bench::hwc::Compressor       compressor(compression_type);
	fastlens::bench::hwc::CompressedBuffer d_compressed_buffer = compressor.compress(d_input_buffer.get(), size_in_bytes);

	GPUArray<bool> d_query_result(1);
	double         compression_ratio = d_compressed_buffer.get_compression_ratio();

	constexpr int32_t           UNPACK_N_VECTORS = 1;
	const ThreadblockMapping<T> mapping(UNPACK_N_VECTORS, fastlens::codec::utils::get_n_vecs_from_size(value_count));

	CudaStopwatch stopwatch       = CudaStopwatch();
	uint8_t*      d_output_buffer = compressor.decompress(d_compressed_buffer, stopwatch);

	if (comparison_type == fastlens::bench::nvcomp::ComparisonType::DECOMPRESSION_QUERY) {
		fastlens::bench::hwc::DummyColumn<T> d_column {reinterpret_cast<T*>(d_output_buffer), value_count};
		fastlens::kernels::device::query_column<T, UNPACK_N_VECTORS, 1, fastlens::bench::hwc::DummyDecompressor<T>, fastlens::bench::hwc::DummyColumn<T>>
		    <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(d_column, d_query_result.get(), value_to_search_for);
	}

	double execution_time_ms = stopwatch.get_result();
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	bool kernel_successful = false;
	if (comparison_type == fastlens::bench::nvcomp::ComparisonType::DECOMPRESSION) {
		kernel_successful =
		    check_if_device_buffers_are_equal<uint8_t>(d_output_buffer, d_input_buffer.get(), size_in_bytes);
	} else if (comparison_type == fastlens::bench::nvcomp::ComparisonType::DECOMPRESSION_QUERY) {
		d_query_result.copy_to_host(&kernel_successful);
	}

	CUDA_SAFE_CALL(cudaFree(d_output_buffer));
	d_compressed_buffer.free();
	compressor.free();

	return BenchmarkResult {kernel_successful, execution_time_ms, compression_ratio};
}

template BenchmarkResult
benchmark_thrust<float>(const float* input, const size_t value_count, const float value_to_search_for);
template BenchmarkResult benchmark_alp<float>(const fastlens::bench::nvcomp::ComparisonType  comparison_type,
                                              const fastlens::bench::nvcomp::CompressionType decompressor_enum,
                                              const float*                        input,
                                              const size_t                        value_count,
                                              const float                         value_to_search_for);
template BenchmarkResult benchmark_hwc<float>(const fastlens::bench::nvcomp::ComparisonType  comparison_type,
                                              const fastlens::bench::nvcomp::CompressionType compression_type,
                                              const float*                        input,
                                              const size_t                        value_count,
                                              const float                         value_to_search_for);

template BenchmarkResult
benchmark_thrust<double>(const double* input, const size_t value_count, const double value_to_search_for);
template BenchmarkResult benchmark_alp<double>(const fastlens::bench::nvcomp::ComparisonType  comparison_type,
                                               const fastlens::bench::nvcomp::CompressionType decompressor_enum,
                                               const double*                       input,
                                               const size_t                        value_count,
                                               const double                        value_to_search_for);
template BenchmarkResult benchmark_hwc<double>(const fastlens::bench::nvcomp::ComparisonType  comparison_type,
                                               const fastlens::bench::nvcomp::CompressionType compression_type,
                                               const double*                       input,
                                               const size_t                        value_count,
                                               const double                        value_to_search_for);
