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
	const auto         workers = std::min(count, requested_workers);
	std::mutex         mutex;
	std::exception_ptr first_error;
	const auto         save_error = [&] {
        std::lock_guard lock(mutex);
        if (!first_error) {
            first_error = std::current_exception();
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
				// Preserve the original exception as well as stage/index context.
				std::lock_guard lock(mutex);
				if (!first_error) {
					first_error = std::current_exception();
					try {
						std::throw_with_nested(std::runtime_error(std::string(stage) + " rowgroup " +
						                                          std::to_string(index) + " failed; reload input"));
					} catch (...) { first_error = std::current_exception(); }
				}
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
	} catch (...) { save_error(); }
	for (auto& future : futures) {
		try {
			future.get();
		} catch (...) { save_error(); }
	}
	if (first_error) {
		std::rethrow_exception(first_error);
	}
}
} // namespace fastlanes::detail

#endif
