// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/benchmarks/nvcomp/benchmark_compressors.cu
// ────────────────────────────────────────────────────────
#include "galp_extensions/alp/alp_bindings.cuh"
#include "galp_bench/data.cuh"
#include "cuda/device_utils.cuh"
#include "cuda/launch/dispatch.cuh"
#include "codecs/decode/alp.cuh"
#include "galp_bench/nvcomp/benchmark_compressors.cuh"
#include "galp_bench/nvcomp/nvcomp_compressors.cuh"
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
using ALPDecompressor = typename galp::codec::device::ALPDecompressor<
    T,
    UNPACK_N_VECTORS,
    galp::codec::device::
        BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, 1, galp::codec::device::ALPFunctor<T, UNPACK_N_VECTORS>>,
    galp::codec::device::StatefulALPExceptionPatcher<T, UNPACK_N_VECTORS, 1>,
    galp::codec::device::ALPColumn<T>>;

template <typename T, unsigned UNPACK_N_VECTORS>
using ALPExtendedDecompressor = typename galp::codec::device::ALPDecompressor<
    T,
    UNPACK_N_VECTORS,
    galp::codec::device::
        BitUnpackerStatefulBranchless<T, UNPACK_N_VECTORS, 1, galp::codec::device::ALPFunctor<T, UNPACK_N_VECTORS>>,
    galp::codec::device::PrefetchAllALPExceptionPatcher<T, UNPACK_N_VECTORS, 1>,
    galp::codec::device::ALPExtendedColumn<T>>;

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
BenchmarkResult benchmark_alp(const galp::bench::nvcomp::ComparisonType  comparison_type,
                              const galp::bench::nvcomp::CompressionType decompressor_enum,
                              const T*                                           input,
                              const galp::codec::host::ALPColumn<T>&   column,
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
	case galp::bench::nvcomp::ALP: {
		galp::codec::device::ALPColumn<T> d_column = column.copy_to_device();
		stopwatch.start();
		if (comparison_type == galp::bench::nvcomp::ComparisonType::DECOMPRESSION) {
			galp::kernels::device::decompress_column<T,
			                                   UNPACK_N_VECTORS,
			                                   1,
			                                   ALPDecompressor<T, UNPACK_N_VECTORS>,
			                                   galp::codec::device::ALPColumn<T>>
			    <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(d_column, d_decompression_result.get());
		} else {
			galp::kernels::device::
			    query_column<T, UNPACK_N_VECTORS, 1, ALPDecompressor<T, UNPACK_N_VECTORS>, galp::codec::device::ALPColumn<T>>
			    <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(
			        d_column, d_query_result.get(), value_to_search_for);
		}
		execution_time_ms = stopwatch.stop();
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		galp::codec::host::free_column(d_column);
		compression_ratio = column.get_compression_ratio();
	} break;

	case galp::bench::nvcomp::GALP: {
		galp::codec::host::ALPExtendedColumn<T>   column_extended = column.create_extended_column();
		galp::codec::device::ALPExtendedColumn<T> d_column        = column_extended.copy_to_device();
		stopwatch.start();
		if (comparison_type == galp::bench::nvcomp::ComparisonType::DECOMPRESSION) {
			galp::kernels::device::decompress_column<T,
			                                   UNPACK_N_VECTORS,
			                                   1,
			                                   ALPExtendedDecompressor<T, UNPACK_N_VECTORS>,
			                                   galp::codec::device::ALPExtendedColumn<T>>
			    <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(d_column, d_decompression_result.get());
		} else {
			galp::kernels::device::query_column<T,
			                              UNPACK_N_VECTORS,
			                              1,
			                              ALPExtendedDecompressor<T, UNPACK_N_VECTORS>,
			                              galp::codec::device::ALPExtendedColumn<T>>
			    <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(
			        d_column, d_query_result.get(), value_to_search_for);
		}
		execution_time_ms = stopwatch.stop();
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		compression_ratio = column_extended.get_compression_ratio();
		galp::codec::host::free_column(column_extended);
		galp::codec::host::free_column(d_column);
	} break;
	default:
		throw std::invalid_argument("Could not parse decompressor enum for alp\n");
	}

	bool kernel_successful = false;
	if (comparison_type == galp::bench::nvcomp::ComparisonType::DECOMPRESSION) {
		GPUArray<T> d_input(column.get_n_values(), input);

		kernel_successful =
		    check_if_device_buffers_are_equal<T>(d_decompression_result.get(), d_input.get(), column.get_n_values());
	}

	return BenchmarkResult {kernel_successful, execution_time_ms, compression_ratio};
}

template <typename T>
BenchmarkResult benchmark_alp(const galp::bench::nvcomp::ComparisonType  comparison_type,
                              const galp::bench::nvcomp::CompressionType decompressor_enum,
                              const T*                            input,
                              const size_t                        value_count,
                              const T                             value_to_search_for) {
	galp::codec::host::ALPColumn<T> column = galp::codec::alp::encode(input, value_count, true);
	auto result = benchmark_alp(comparison_type, decompressor_enum, input, column, value_to_search_for);
	galp::codec::host::free_column(column);

	return result;
}

template <typename T>
BenchmarkResult benchmark_hwc(const galp::bench::nvcomp::ComparisonType  comparison_type,
                              const galp::bench::nvcomp::CompressionType compression_type,
                              const T*                            input,
                              const size_t                        value_count,
                              const T                             value_to_search_for) {
	size_t                size_in_bytes = value_count * sizeof(T);
	GPUArray<uint8_t>     d_input_buffer(size_in_bytes, reinterpret_cast<const uint8_t*>(input));
	galp::bench::hwc::Compressor       compressor(compression_type);
	galp::bench::hwc::CompressedBuffer d_compressed_buffer = compressor.compress(d_input_buffer.get(), size_in_bytes);

	GPUArray<bool> d_query_result(1);
	double         compression_ratio = d_compressed_buffer.get_compression_ratio();

	constexpr int32_t           UNPACK_N_VECTORS = 1;
	const ThreadblockMapping<T> mapping(UNPACK_N_VECTORS, galp::codec::utils::get_n_vecs_from_size(value_count));

	CudaStopwatch stopwatch       = CudaStopwatch();
	uint8_t*      d_output_buffer = compressor.decompress(d_compressed_buffer, stopwatch);

	if (comparison_type == galp::bench::nvcomp::ComparisonType::DECOMPRESSION_QUERY) {
		galp::bench::hwc::DummyColumn<T> d_column {reinterpret_cast<T*>(d_output_buffer), value_count};
		galp::kernels::device::query_column<T, UNPACK_N_VECTORS, 1, galp::bench::hwc::DummyDecompressor<T>, galp::bench::hwc::DummyColumn<T>>
		    <<<mapping.n_blocks, mapping.N_THREADS_PER_BLOCK>>>(d_column, d_query_result.get(), value_to_search_for);
	}

	double execution_time_ms = stopwatch.get_result();
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	bool kernel_successful = false;
	if (comparison_type == galp::bench::nvcomp::ComparisonType::DECOMPRESSION) {
		kernel_successful =
		    check_if_device_buffers_are_equal<uint8_t>(d_output_buffer, d_input_buffer.get(), size_in_bytes);
	} else if (comparison_type == galp::bench::nvcomp::ComparisonType::DECOMPRESSION_QUERY) {
		d_query_result.copy_to_host(&kernel_successful);
	}

	CUDA_SAFE_CALL(cudaFree(d_output_buffer));
	d_compressed_buffer.free();
	compressor.free();

	return BenchmarkResult {kernel_successful, execution_time_ms, compression_ratio};
}

template BenchmarkResult
benchmark_thrust<float>(const float* input, const size_t value_count, const float value_to_search_for);
template BenchmarkResult benchmark_alp<float>(const galp::bench::nvcomp::ComparisonType  comparison_type,
                                              const galp::bench::nvcomp::CompressionType decompressor_enum,
                                              const float*                        input,
                                              const size_t                        value_count,
                                              const float                         value_to_search_for);
template BenchmarkResult benchmark_hwc<float>(const galp::bench::nvcomp::ComparisonType  comparison_type,
                                              const galp::bench::nvcomp::CompressionType compression_type,
                                              const float*                        input,
                                              const size_t                        value_count,
                                              const float                         value_to_search_for);

template BenchmarkResult
benchmark_thrust<double>(const double* input, const size_t value_count, const double value_to_search_for);
template BenchmarkResult benchmark_alp<double>(const galp::bench::nvcomp::ComparisonType  comparison_type,
                                               const galp::bench::nvcomp::CompressionType decompressor_enum,
                                               const double*                       input,
                                               const size_t                        value_count,
                                               const double                        value_to_search_for);
template BenchmarkResult benchmark_hwc<double>(const galp::bench::nvcomp::ComparisonType  comparison_type,
                                               const galp::bench::nvcomp::CompressionType compression_type,
                                               const double*                       input,
                                               const size_t                        value_count,
                                               const double                        value_to_search_for);
