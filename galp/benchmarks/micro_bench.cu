// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/benchmarks/micro_bench.cu
// ────────────────────────────────────────────────────────
#include "galp_bench/data.cuh"
#include "core/enums.cuh"
#include "cuda/kernels/dispatch.cuh"
#include "core/types.cuh"
#include "codecs/decode/alp.cuh"
#include "cuda/memory/device_pool.cuh"
#include "galp_bench/generated/kernel_bindings.cuh"
#include "galp_bench/verification.cuh"
#include <chrono>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

[[maybe_unused]] static inline void CUDA_CHECK(cudaError_t e, const char* msg) {
	if (e != cudaSuccess) {
		fprintf(stderr, "[CUDA] %s: %s\n", msg, cudaGetErrorString(e));
		throw std::runtime_error(msg);
	}
}

struct ProgramParameters {
	galp::format::DataType            data_type;
	galp::format::Encoding            encoding_type;
	galp::format::Kernel              kernel;
	uint32_t                   unpack_n_vecs;
	uint32_t                   unpack_n_vals;
	galp::format::Unpacker            unpacker;
	galp::format::Patcher             patcher;
	galp::format::Expander            expander;
	galp::bench::ValueRange<vbw_t>    bit_width_range;
	galp::bench::ValueRange<uint16_t> ec_range;
	size_t                     n_values;
	uint32_t                   n_samples;
	galp::format::Print               print_option;
};

static void print_usage(const char* program_name, const char* error = nullptr) {
	const char* const executable = program_name == nullptr ? "micro_bench" : program_name;
	if (error != nullptr) {
		std::fprintf(stderr, "micro_bench: %s\n\n", error);
	}

	std::fprintf(stderr,
	             "Usage:\n"
	             "  %s <data_type> <encoding> <kernel> <unpack_n_vecs> <unpack_n_vals> "
	             "<unpacker> <patcher> <expander> <start_vbw> <end_vbw> <start_ec> <end_ec> "
	             "<n_vecs> <n_samples> <print_debug>\n\n"
	             "Values:\n"
	             "  data_type: i8, i16, u32, u64, f32, f64\n"
	             "  encoding: alp, bit-packing, ffor, frequency, cross-rle, dictionary, slpatch, "
	             "constant, rle, dict-slpatch\n"
	             "  kernel: decompress, query, query-multi-column\n"
	             "  unpacker: none, dummy, old-fls, switch-case, stateless, stateless-branchless, "
	             "stateful-cache, stateful-local-1, stateful-local-2, stateful-local-4, "
	             "stateful-shared-1, stateful-shared-2, stateful-shared-4, stateful-register-1, "
	             "stateful-register-2, stateful-register-4, stateful-register-branchless-1, "
	             "stateful-register-branchless-2, stateful-register-branchless-4, stateful-branchless\n"
	             "  patcher: none, dummy, stateless, stateful, naive, naive-branchless, "
	             "prefetch-position, prefetch-all, prefetch-all-branchless\n"
	             "  expander: none, dummy, stateful, stateful-cache, stateful-shuffle, "
	             "prefetch-stateful, stateful-advance, stateful-extended, branchless, "
	             "prefetch-branchless\n\n"
	             "Example:\n"
	             "  %s u32 bit-packing decompress 1 1 dummy none none 1 8 0 0 1024 1 0\n",
	             executable,
	             executable);
}

struct CLIArgs {
	std::string data_type;
	std::string encoding_type;
	std::string kernel;
	uint32_t    unpack_n_vecs;
	uint32_t    unpack_n_vals;
	std::string patcher;
	std::string unpacker;
	std::string expander;
	vbw_t       start_vbw;
	vbw_t       end_vbw;
	uint16_t    start_ec;
	uint16_t    end_ec;
	size_t      n_vecs;
	uint32_t    n_samples;
	uint32_t    print_debug;

	CLIArgs(const int argc, char** argv) {
		constexpr int32_t CORRECT_ARG_COUNT = 16;
		if (argc != CORRECT_ARG_COUNT) {
			throw std::invalid_argument("Wrong arg count.");
		}

		int32_t argcounter = 0;
		data_type          = argv[++argcounter];
		encoding_type      = argv[++argcounter];
		kernel             = argv[++argcounter];
		unpack_n_vecs      = std::stoul(argv[++argcounter]);
		unpack_n_vals      = std::stoul(argv[++argcounter]);
		unpacker           = argv[++argcounter];
		patcher            = argv[++argcounter];
		expander           = argv[++argcounter];
		start_vbw          = std::stoul(argv[++argcounter]);
		end_vbw            = std::stoul(argv[++argcounter]);
		start_ec           = std::stoul(argv[++argcounter]);
		end_ec             = std::stoul(argv[++argcounter]);
		n_vecs             = std::stoul(argv[++argcounter]);
		n_samples          = std::stoul(argv[++argcounter]);
		print_debug        = std::stoul(argv[++argcounter]);
	}

	ProgramParameters parse() {
		return ProgramParameters {
		    galp::format::string_to_data_type(data_type),
		    galp::format::string_to_encoding(encoding_type),
		    galp::format::string_to_kernel(kernel),
		    unpack_n_vecs,
		    unpack_n_vals,
		    galp::format::string_to_unpacker(unpacker),
		    galp::format::string_to_patcher(patcher),
		    galp::format::string_to_expander(expander),
		    galp::bench::ValueRange<vbw_t>(start_vbw, end_vbw),
		    galp::bench::ValueRange<uint16_t>(start_ec, end_ec),
		    n_vecs * galp::codec::consts::VALUES_PER_VECTOR,
		    n_samples,
		    static_cast<galp::format::Print>(print_debug),
		};
	}

private:
};

template <typename T, typename ColumnT>
galp::bench::verification::ExecutionResult<T> decompress_column(const ColumnT& column, const ProgramParameters params) {
	auto column_device = column.copy_to_device();
	galp::memory::sync_h2d();
	T* out;

	out = galp::bench::bindings::decompress_column<T, typename ColumnT::DeviceColumnT>(column_device,
	                                                                      params.unpack_n_vecs,
	                                                                      params.unpack_n_vals,
	                                                                      params.unpacker,
	                                                                      params.patcher,
	                                                                      params.expander,
	                                                                      params.n_samples);

	galp::codec::host::free_column(column_device);

	// ---- CPU version timing (microseconds) ----
	const auto cpu_start   = std::chrono::steady_clock::now();
	const T*   correct_out = galp::bench::bindings::decompress(column);
	const auto cpu_end     = std::chrono::steady_clock::now();

	const auto cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
	std::printf("[CPU decompress] %lld us\n", static_cast<long long>(cpu_us));
	// ------------------------------------------

	auto result = galp::bench::verification::compare_data(correct_out, out, params.n_values);
	delete[] correct_out;
	delete[] out;
	return result;
}

template <typename T, typename ColumnT>
galp::bench::verification::ExecutionResult<T> decompress_column_time(const ColumnT column, const ProgramParameters params) {
	auto column_device = column.copy_to_device();
	galp::memory::sync_h2d();

	{
		const T* warm = galp::bench::bindings::decompress_column<T, typename ColumnT::DeviceColumnT>(column_device,
		                                                                                params.unpack_n_vecs,
		                                                                                params.unpack_n_vals,
		                                                                                params.unpacker,
		                                                                                params.patcher,
		                                                                                /*n_samples=*/1);
		CUDA_CHECK(cudaDeviceSynchronize(), "warmup sync");
		delete[] warm;
	}

	const T* out = galp::bench::bindings::decompress_column<T, typename ColumnT::DeviceColumnT>(column_device,
	                                                                               params.unpack_n_vecs,
	                                                                               params.unpack_n_vals,
	                                                                               params.unpacker,
	                                                                               params.patcher,
	                                                                               params.expander,
	                                                                               params.n_samples);

	printf("[KERNEL TIME] unpack_vecs=%u unpack_vals=%u patcher= %d n_samples=%u \n",
	       params.unpack_n_vecs,
	       params.unpack_n_vals,
	       (int)params.patcher,
	       params.n_samples);

	galp::codec::host::free_column(column_device);

	delete[] out;

	return galp::bench::verification::ExecutionResult<T> {};
}

template <typename T, typename ColumnT>
galp::bench::verification::ExecutionResult<T>
query_column(const ColumnT& column, const ProgramParameters params, const bool query_result, const T magic_value) {
	auto column_device = column.copy_to_device();
	galp::memory::sync_h2d();
	const bool answer = galp::bench::bindings::query_column<T, typename ColumnT::DeviceColumnT>(column_device,
	                                                                               params.unpack_n_vecs,
	                                                                               params.unpack_n_vals,
	                                                                               params.unpacker,
	                                                                               params.patcher,
	                                                                               magic_value,
	                                                                               params.n_samples);
	galp::codec::host::free_column(column_device);

	// Weird hack to avoid refactor_
	T a = query_result ? 1.0 : 0.0;
	T b = answer ? 1.0 : 0.0;

	return galp::bench::verification::compare_data(&a, &b, 1);
}

#ifndef GALP_ENABLE_MULTI_COLUMN
#define GALP_ENABLE_MULTI_COLUMN 0
#endif

#if GALP_ENABLE_MULTI_COLUMN
template <typename T, typename ColumnT>
galp::bench::verification::ExecutionResult<T>
query_multi_column(const ColumnT& column, const ProgramParameters params, const bool query_result, const T magic_value) {
	const bool answer = galp::bench::bindings::query_multi_column<T, ColumnT>(column,
	                                                             params.unpack_n_vecs,
	                                                             params.unpack_n_vals,
	                                                             params.unpacker,
	                                                             params.patcher,
	                                                             magic_value,
	                                                             params.n_samples);
	T          a      = query_result ? 1.0 : 0.0;
	T          b      = answer ? 1.0 : 0.0;
	return galp::bench::verification::compare_data(&a, &b, 1);
}
#else
template <typename T, typename ColumnT>
galp::bench::verification::ExecutionResult<T> query_multi_column(const ColumnT&, const ProgramParameters, const bool, const T) {
	throw std::invalid_argument("QueryMultiColumn is disabled at build time (GALP_ENABLE_MULTI_COLUMN=OFF).");
}
#endif

template <typename T, typename ColumnT>
galp::bench::verification::ExecutionResult<T>
execute_kernel(const ColumnT& column, const ProgramParameters params, const bool query_result, const T magic_value) {
	if (params.kernel == galp::format::Kernel::Decompress) {
		return decompress_column<T, ColumnT>(column, params);
	} else if (params.kernel == galp::format::Kernel::Query) {
		return query_column<T, ColumnT>(column, params, query_result, magic_value);
		// } else if (params.kernel == galp::format::Kernel::QueryMultiColumn) {
		// 	return query_multi_column<T, ColumnT>(column, params, query_result, magic_value);
	} else {
		throw std::invalid_argument("Kernel not implemented yet.\n");
	}
}

template <typename T>
std::vector<galp::bench::verification::ExecutionResult<T>> execute_bp(const ProgramParameters params) {
	using UINT_T = typename galp::codec::utils::same_width_uint<T>::type;
	auto results = std::vector<galp::bench::verification::ExecutionResult<T>>();

	for (vbw_t vbw {params.bit_width_range.min}; vbw <= params.bit_width_range.max; ++vbw) {
		printf("processing bitwidth = %d\n", vbw);
		auto vbw_range = galp::bench::ValueRange<vbw_t>(vbw);
		if (params.kernel == galp::format::Kernel::QueryMultiColumn || params.kernel == galp::format::Kernel::Query) {
			throw std::invalid_argument("QueryMultiColumn not supported for Bit-Packing columns.\n");
		}
		bool                      query_result = false;
		T                         magic_value  = galp::codec::consts::as<T>::MAGIC_NUMBER;
		galp::codec::host::BPColumn<T> column;

		column = galp::bench::columns::generate_random_bp_column<T>(params.n_values, vbw_range, params.unpack_n_vecs);

		results.push_back(execute_kernel<T, galp::codec::host::BPColumn<T>>(column, params, query_result, magic_value));

		galp::codec::host::free_column(column);
	}

	return results;
}

template <typename T>
std::vector<galp::bench::verification::ExecutionResult<T>> execute_ffor(const ProgramParameters params) {
	using UINT_T = typename galp::codec::utils::same_width_uint<T>::type;
	auto results = std::vector<galp::bench::verification::ExecutionResult<T>>();

	for (vbw_t vbw {params.bit_width_range.min}; vbw <= params.bit_width_range.max; ++vbw) {
		printf("processing bitwidth = %d\n", vbw);
		auto vbw_range = galp::bench::ValueRange<vbw_t>(vbw);
		if (params.kernel == galp::format::Kernel::QueryMultiColumn) {
			vbw_range = params.bit_width_range;
		}
		bool                        query_result = false;
		T                           magic_value  = galp::codec::consts::as<T>::MAGIC_NUMBER;
		galp::codec::host::FFORColumn<T> column;

		if (params.kernel == galp::format::Kernel::Query) {
			auto [_query_result, _column] =
			    galp::bench::columns::generate_binary_ffor_column<T>(params.n_values, vbw_range, params.unpack_n_vecs);
				query_result = _query_result;
				column       = std::move(_column);
		} else {
			column = galp::bench::columns::generate_random_ffor_column<T>(
			    params.n_values, vbw_range, galp::bench::ValueRange<T>(0, 100), params.unpack_n_vecs);
		}

		if (params.kernel == galp::format::Kernel::QueryMultiColumn) {
			// We do not want query multicolumn to ever find a full lane of the
			// value to query to limit write bandwidth
			magic_value = std::numeric_limits<T>::max();
		}

		results.push_back(execute_kernel<T, galp::codec::host::FFORColumn<T>>(column, params, query_result, magic_value));

		galp::codec::host::free_column(column);

		if (params.kernel == galp::format::Kernel::QueryMultiColumn) {
			break;
		}
	}

	return results;
}

template <typename T>
std::vector<galp::bench::verification::ExecutionResult<T>> execute_alp(const ProgramParameters params) {
	using UINT_T = typename galp::codec::utils::same_width_uint<T>::type;
	auto results = std::vector<galp::bench::verification::ExecutionResult<T>>();

	// for (vbw_t vbw{params.bit_width_range.min}; vbw <=
	// params.bit_width_range.max; ++vbw) {
	{
		bool query_result = false;
		T    magic_value  = galp::codec::consts::as<T>::MAGIC_NUMBER;

		auto column = galp::bench::columns::generate_alp_column<T>(
		    params.n_values, params.bit_width_range, galp::bench::ValueRange<uint16_t>(0), params.unpack_n_vecs);
		for (uint16_t ec {params.ec_range.min}; ec <= params.ec_range.max; ++ec) {
			column = galp::bench::columns::modify_alp_exception_count(std::move(column), ec);

			if (params.kernel == galp::format::Kernel::Query) {
				auto [_query_result, _magic_value] =
				    galp::bench::columns::get_value_to_query<T, galp::codec::host::ALPColumn<T>>(column);
				query_result = _query_result;
				magic_value  = _magic_value;
			}

			if (params.patcher == galp::format::Patcher::Dummy || params.patcher == galp::format::Patcher::Stateless ||
			    params.patcher == galp::format::Patcher::Stateful) {
				results.push_back(
				    execute_kernel<T, galp::codec::host::ALPColumn<T>>(column, params, query_result, magic_value));
			} else {
				auto column_extended = column.create_extended_column();

				results.push_back(execute_kernel<T, galp::codec::host::ALPExtendedColumn<T>>(
				    column_extended, params, query_result, magic_value));

				galp::codec::host::free_column(column_extended);
			}
		}

		galp::codec::host::free_column(column);
	}

	return results;
}

template <typename T>
std::vector<galp::bench::verification::ExecutionResult<T>> execute_freq(const ProgramParameters params) {
	using UINT_T = typename galp::codec::utils::same_width_uint<T>::type;
	auto results = std::vector<galp::bench::verification::ExecutionResult<T>>();

	if (params.kernel == galp::format::Kernel::QueryMultiColumn) {
		throw std::invalid_argument("QueryMultiColumn not supported for FREQ columns.\n");
	}
	// for (vbw_t vbw{params.bit_width_range.min}; vbw <= params.bit_width_range.max; ++vbw)
	{
		bool query_result = false;
		T    magic_value  = galp::codec::consts::as<T>::MAGIC_NUMBER;

		auto column = galp::bench::columns::generate_freq_column<T>(params.n_values, galp::bench::ValueRange<uint16_t>(0));
		for (uint16_t ec {params.ec_range.min}; ec <= params.ec_range.max; ++ec) {
			column = galp::bench::columns::modify_freq_exception_count(std::move(column), ec);

			if (params.kernel == galp::format::Kernel::Query) {
				throw std::invalid_argument("Query kernel not supported for FREQ columns.\n");
			}

			if (params.patcher == galp::format::Patcher::Dummy || params.patcher == galp::format::Patcher::Stateless ||
			    params.patcher == galp::format::Patcher::Stateful) {
				results.push_back(
				    execute_kernel<T, galp::codec::host::FREQColumn<T>>(column, params, query_result, magic_value));
			} else {
				auto column_extended = column.create_extended_column();

				results.push_back(execute_kernel<T, galp::codec::host::FREQExtendedColumn<T>>(
				    column_extended, params, query_result, magic_value));

				galp::codec::host::free_column(column_extended);
			}
		}

		galp::codec::host::free_column(column);
	}

	return results;
}

template <typename T>
std::vector<galp::bench::verification::ExecutionResult<T>> execute_dict(const ProgramParameters params) {
	using UINT_T = typename galp::codec::utils::same_width_uint<T>::type;
	auto results = std::vector<galp::bench::verification::ExecutionResult<T>>();

	if (params.kernel == galp::format::Kernel::QueryMultiColumn || params.kernel == galp::format::Kernel::Query) {
		throw std::invalid_argument("QueryMultiColumn/Query not supported for DICT columns.\n");
	}

	for (vbw_t vbw {params.bit_width_range.min}; vbw <= params.bit_width_range.max; ++vbw) {
		printf("processing bitwidth = %d\n", vbw);
		bool query_result = false;
		T    magic_value  = galp::codec::consts::as<T>::MAGIC_NUMBER;

		galp::codec::host::DICTFFORColumn<T> column;

		column = galp::bench::columns::generate_random_dict_column<T>(params.n_values, vbw, 20);

		results.push_back(
		    execute_kernel<T, galp::codec::host::DICTFFORColumn<T>>(column, params, query_result, magic_value));

		galp::codec::host::free_column(column);
	}

	return results;
}

template <typename T>
std::vector<galp::bench::verification::ExecutionResult<T>> execute_slpatch(const ProgramParameters params) {
	auto results = std::vector<galp::bench::verification::ExecutionResult<T>>();

	if (params.kernel == galp::format::Kernel::QueryMultiColumn || params.kernel == galp::format::Kernel::Query) {
		throw std::invalid_argument("QueryMultiColumn/Query not supported for SLPATCH columns.\n");
	}

	for (vbw_t vbw {params.bit_width_range.min}; vbw <= params.bit_width_range.max; ++vbw) {
		printf("processing bitwidth = %d\n", vbw);
		bool query_result = false;
		T    magic_value  = galp::codec::consts::as<T>::MAGIC_NUMBER;

		auto column =
		    galp::bench::columns::generate_slpatch_column<T>(params.n_values, galp::bench::ValueRange<vbw_t>(vbw), params.ec_range);

		results.push_back(execute_kernel<T, galp::codec::host::SLPATCHColumn<T>>(column, params, query_result, magic_value));
		galp::codec::host::free_column(column);
	}

	return results;
}

template <typename T>
std::vector<galp::bench::verification::ExecutionResult<T>> execute_dict_slpatch(const ProgramParameters params) {
	auto results = std::vector<galp::bench::verification::ExecutionResult<T>>();

	if (params.kernel == galp::format::Kernel::QueryMultiColumn || params.kernel == galp::format::Kernel::Query) {
		throw std::invalid_argument("QueryMultiColumn/Query not supported for DICT+SLPATCH columns.\n");
	}

	for (vbw_t vbw {params.bit_width_range.min}; vbw <= params.bit_width_range.max; ++vbw) {
		printf("processing bitwidth = %d\n", vbw);
		bool query_result = false;
		T    magic_value  = galp::codec::consts::as<T>::MAGIC_NUMBER;

		auto column = galp::bench::columns::generate_dict_slpatch_column<T>(
		    params.n_values, galp::bench::ValueRange<vbw_t>(vbw), params.ec_range, 256);

		results.push_back(
		    execute_kernel<T, galp::codec::host::DICTSLPATCHColumn<T>>(column, params, query_result, magic_value));
		galp::codec::host::free_column(column);
	}

	return results;
}

template <typename T>
std::vector<galp::bench::verification::ExecutionResult<T>> execute_rle(const ProgramParameters params) {
	auto results = std::vector<galp::bench::verification::ExecutionResult<T>>();

	if (params.kernel == galp::format::Kernel::QueryMultiColumn || params.kernel == galp::format::Kernel::Query) {
		throw std::invalid_argument("QueryMultiColumn/Query not supported for RLE columns.\n");
	}

	for (vbw_t vbw {params.bit_width_range.min}; vbw <= params.bit_width_range.max; ++vbw) {
		printf("processing bitwidth = %d\n", vbw);
		bool query_result = false;
		T    magic_value  = galp::codec::consts::as<T>::MAGIC_NUMBER;

		auto column = galp::bench::columns::generate_rle_column<T>(params.n_values);

		results.push_back(execute_kernel<T, galp::codec::host::RLEColumn<T, typename galp::codec::utils::same_width_uint<T>::type>>(
		    column, params, query_result, magic_value));
		galp::codec::host::free_column(column);
	}

	return results;
}

template <typename T>
std::vector<galp::bench::verification::ExecutionResult<T>> execute_constant(const ProgramParameters params) {
	auto results = std::vector<galp::bench::verification::ExecutionResult<T>>();

	if (params.kernel == galp::format::Kernel::QueryMultiColumn || params.kernel == galp::format::Kernel::Query) {
		throw std::invalid_argument("QueryMultiColumn/Query not supported for CONSTANT columns.\n");
	}

	bool query_result = false;
	T    magic_value  = galp::codec::consts::as<T>::MAGIC_NUMBER;

	auto column = galp::bench::columns::generate_constant_column<T>(params.n_values);
	results.push_back(execute_kernel<T, galp::codec::host::CONSTANTColumn<T>>(column, params, query_result, magic_value));
	galp::codec::host::free_column(column);

	return results;
}

// execute_cross_rle
template <typename T>
std::vector<galp::bench::verification::ExecutionResult<T>> execute_cross_rle(const ProgramParameters params) {
	using UINT_T = typename galp::codec::utils::same_width_uint<T>::type;
	auto results = std::vector<galp::bench::verification::ExecutionResult<T>>();

	if (params.kernel == galp::format::Kernel::QueryMultiColumn || params.kernel == galp::format::Kernel::Query) {
		throw std::invalid_argument("QueryMultiColumn/Query not supported for CROSS RLE columns.\n");
	}

	for (vbw_t vbw {params.bit_width_range.min}; vbw <= params.bit_width_range.max; ++vbw) {
		printf("processing bitwidth = %d\n", vbw);
		bool query_result = false;
		T    magic_value  = galp::codec::consts::as<T>::MAGIC_NUMBER;

		galp::codec::host::CROSSRLEColumn<T> column;

		column = galp::bench::columns::generate_cross_rle_column<T>(params.n_values, vbw, 20);

		if (params.expander == galp::format::Expander::StatefulExtended) {
			auto column_extended = column.create_extended_column();
			results.push_back(execute_kernel<T, galp::codec::host::CROSSRLEExtendedColumn<T>>(
			    column_extended, params, query_result, magic_value));
		} else if (params.expander == galp::format::Expander::Branchless ||
		           params.expander == galp::format::Expander::PrefetchBranchless) {
			auto column_lane_mask = column.create_lane_mask_column();
			results.push_back(execute_kernel<T, galp::codec::host::CROSSRLELaneMaskColumn<T>>(
			    column_lane_mask, params, query_result, magic_value));
		} else {
			results.push_back(
			    execute_kernel<T, galp::codec::host::CROSSRLEColumn<T>>(column, params, query_result, magic_value));
		}

		galp::codec::host::free_column(column);
	}

	return results;
}

/*
Usage:
./micro_bench \
  <data_type> <kernel> \
  <unpack_n_vecs> <unpack_n_vals> \
  <unpacker> <patcher>  <>\
  <start_vbw> <end_vbw> \
  <start_ec> <end_ec> \
  <n_vecs> <n_samples> <print_debug>
*/

template <class T>
static int32_t run_by_encoding_type(const ProgramParameters& params, bool print_debug) {
	switch (params.encoding_type) {
	case galp::format::Encoding::BIT_PACKING:
		if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>) {
			return galp::bench::verification::process_results(execute_bp<T>(params), print_debug);
		} else {
			std::cerr << "[error] bit_packing only supports u32/u64.\n";
			return 1;
		}
	case galp::format::Encoding::DICTIONARY:
		if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>) {
			return galp::bench::verification::process_results(execute_dict<T>(params), print_debug);
		} else {
			std::cerr << "[error] dictionary only supports u32/u64.\n";
			return 1;
		}
	case galp::format::Encoding::FREQUENCY:
		if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> || std::is_same_v<T, int8_t> ||
		              std::is_same_v<T, int16_t>) {
			return galp::bench::verification::process_results(execute_freq<T>(params), print_debug);
		} else {
			std::cerr << "[error] frequency only supports i8/i16/u32/u64.\n";
			return 1;
		}
	case galp::format::Encoding::CROSS_RLE:

		if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>) {
			return galp::bench::verification::process_results(execute_cross_rle<T>(params), print_debug);
		} else {
			std::cerr << "[error] cross rle only supports u32/u64.\n";
			return 1;
		}
	case galp::format::Encoding::FFOR:
		if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>) {
			return galp::bench::verification::process_results(execute_ffor<T>(params), print_debug);
		} else {
			std::cerr << "[error] ffor only supports u32/u64.\n";
			return 1;
		}
	case galp::format::Encoding::SLPATCH:
		if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> || std::is_same_v<T, int16_t>) {
			return galp::bench::verification::process_results(execute_slpatch<T>(params), print_debug);
		} else {
			std::cerr << "[error] slpatch only supports i16/u32/u64.\n";
			return 1;
		}
	case galp::format::Encoding::DICT_SLPATCH:
		if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>) {
			return galp::bench::verification::process_results(execute_dict_slpatch<T>(params), print_debug);
		} else {
			std::cerr << "[error] dict-slpatch only supports u32/u64.\n";
			return 1;
		}
	case galp::format::Encoding::RLE:
		if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>) {
			return galp::bench::verification::process_results(execute_rle<T>(params), print_debug);
		} else {
			std::cerr << "[error] rle only supports u32/u64.\n";
			return 1;
		}
	case galp::format::Encoding::CONSTANT:
		if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>) {
			return galp::bench::verification::process_results(execute_constant<T>(params), print_debug);
		} else {
			std::cerr << "[error] constant only supports u32/u64.\n";
			return 1;
		}

	case galp::format::Encoding::ALP:
		if constexpr (std::is_same_v<T, float>) {
			return galp::bench::verification::process_results(execute_alp<float>(params), print_debug);
		} else if constexpr (std::is_same_v<T, double>) {
			return galp::bench::verification::process_results(execute_alp<double>(params), print_debug);
		} else {
			std::cerr << "[error] alp only supports f32/f64.\n";
			return 1;
		}
	}

	std::cerr << "[error] unknown encoding type.\n";
	return 1;
}

int main(int argc, char** argv) {
	if (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
		print_usage(argv[0]);
		return 0;
	}

	const ProgramParameters params = [&]() {
		try {
			CLIArgs args(argc, argv);
			return args.parse();
		} catch (const std::exception& err) {
			print_usage(argv[0], err.what());
			std::exit(2);
		}
	}();

	bool print_debug = params.print_option != galp::format::Print::PrintNothing;

	int32_t exit_code = 0;
	switch (params.data_type) {
	case galp::format::DataType::I8:
		exit_code = run_by_encoding_type<int8_t>(params, print_debug);
		break;
	case galp::format::DataType::I16:
		exit_code = run_by_encoding_type<int16_t>(params, print_debug);
		break;
	case galp::format::DataType::U32:
		exit_code = run_by_encoding_type<uint32_t>(params, print_debug);
		break;
	case galp::format::DataType::U64:
		exit_code = run_by_encoding_type<uint64_t>(params, print_debug);
		break;
	case galp::format::DataType::F32:
		exit_code = run_by_encoding_type<float>(params, print_debug);
		break;
	case galp::format::DataType::F64:
		exit_code = run_by_encoding_type<double>(params, print_debug);
		break;
	default:
		std::cerr << "[error] unknown data type.\n";
		exit_code = 1;
		break;
	}

	if (params.print_option == galp::format::Print::PrintDebugExit0) {
		std::exit(0);
	}

	std::exit(exit_code);
}
