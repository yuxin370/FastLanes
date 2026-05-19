// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// galp/src/codecs/encodings/base.cuh
// ────────────────────────────────────────────────────────
#ifndef GALP_COMPRESSION_COLUMNS_BASE_CUH
#define GALP_COMPRESSION_COLUMNS_BASE_CUH

#include "codecs/device_types.cuh"
#include "codecs/utils.cuh"
#include <cstddef>
#include <cstdint>
#include <utility>

namespace galp::codec::device {

template <typename T>
struct FunctorBase {
	using UINT_T = typename galp::codec::utils::same_width_uint<T>::type;

	__device__ __forceinline__ T operator()(const UINT_T value, [[maybe_unused]] const vi_t vector_index) {
		return static_cast<T>(value);
	}
};

template <typename T>
struct DecompressorBase {
	/* Constructor, cannot be enforced
  __device__ DecompressorBase(const ColumnT column,
  const vi_t vector_index, const lane_t lane)
  */

	void __device__ unpack_next_into([[maybe_unused]] T* __restrict out) {
	}
};

} // namespace galp::codec::device

namespace galp::codec::host {

template <typename T>
class HostArray {
public:
	HostArray() = default;
	HostArray(std::nullptr_t) noexcept {}
	HostArray(T* ptr) noexcept
	    : ptr_(ptr)
	    , owns_(ptr != nullptr) {
	}

	static HostArray borrow(T* ptr) noexcept {
		HostArray out;
		out.ptr_  = ptr;
		out.owns_ = false;
		return out;
	}

	HostArray(const HostArray&)            = delete;
	HostArray& operator=(const HostArray&) = delete;

	HostArray(HostArray&& other) noexcept
	    : ptr_(std::exchange(other.ptr_, nullptr))
	    , owns_(std::exchange(other.owns_, false)) {
	}

	HostArray& operator=(HostArray&& other) noexcept {
		if (this != &other) {
			reset();
			ptr_        = std::exchange(other.ptr_, nullptr);
			owns_       = std::exchange(other.owns_, false);
		}
		return *this;
	}

	~HostArray() {
		reset();
	}

	HostArray& operator=(T* ptr) noexcept {
		reset(ptr, ptr != nullptr);
		return *this;
	}

	void reset(T* ptr = nullptr, const bool owns = false) noexcept {
		if (owns_ && ptr_ != nullptr) {
			delete[] ptr_;
		}
		ptr_  = ptr;
		owns_ = owns;
	}

	T* release() noexcept {
		owns_ = false;
		return std::exchange(ptr_, nullptr);
	}

	T* get() const noexcept {
		return ptr_;
	}

	bool owns() const noexcept {
		return owns_;
	}

	explicit operator bool() const noexcept {
		return ptr_ != nullptr;
	}

	operator T*() const noexcept {
		return ptr_;
	}

	T& operator[](const size_t idx) const noexcept {
		return ptr_[idx];
	}

private:
	T*   ptr_  = nullptr;
	bool owns_ = false;
};

template <typename T>
HostArray<T> borrow_array(T* ptr) noexcept {
	return HostArray<T>::borrow(ptr);
}

} // namespace galp::codec::host

#endif // GALP_COMPRESSION_COLUMNS_BASE_CUH
