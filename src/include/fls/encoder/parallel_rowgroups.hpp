// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/include/fls/encoder/parallel_rowgroups.hpp
// ────────────────────────────────────────────────────────
#ifndef FLS_ENCODER_PARALLEL_ROWGROUPS_HPP
#define FLS_ENCODER_PARALLEL_ROWGROUPS_HPP

#include "fls/common/alias.hpp"
#include <algorithm>
#include <exception>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace fastlanes::detail {
// Keep the existing strided assignment. Admission and cancellation share a
// mutex: work admitted before an error finishes; no later rowgroup is admitted.
template <typename Function>
void parallel_rowgroups(n_t count, n_t requested_workers, const char* stage, Function&& function) {
	if (requested_workers == 0) {
		throw std::invalid_argument("rowgroup worker count must be positive");
	}
	const auto         workers = std::min(count, requested_workers);
	std::mutex         mutex;
	std::exception_ptr first_error;
	n_t                failed_index = count;
	const auto         save_error   = [&](n_t index) {
        std::lock_guard lock(mutex);
        if (!first_error) {
            first_error  = std::current_exception();
            failed_index = index;
        }
	};
	const auto run = [&](n_t worker) {
		for (n_t index = worker; index < count; index += workers) {
			{
				std::lock_guard lock(mutex);
				if (first_error) {
					return;
				}
			}
			try {
				function(index);
			} catch (...) {
				save_error(index);
				return;
			}
		}
	};
	std::vector<std::future<void>> futures;
	futures.reserve(workers);
	try {
		if (workers == 1) {
			run(0);
		} else {
			for (n_t worker = 0; worker < workers; ++worker) {
				futures.push_back(std::async(std::launch::async, run, worker));
			}
		}
	} catch (...) { save_error(count); }
	for (auto& future : futures) {
		try {
			future.get();
		} catch (...) { save_error(count); }
	}
	if (failed_index < count) {
		try {
			std::rethrow_exception(first_error);
		} catch (...) {
			std::throw_with_nested(std::runtime_error(std::string(stage) + " rowgroup " + std::to_string(failed_index) +
			                                          " failed; reload input"));
		}
	}
	if (first_error) {
		std::rethrow_exception(first_error);
	}
}
} // namespace fastlanes::detail

#endif
