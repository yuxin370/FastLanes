#include "galp/advanced/direct_dct_pls.hpp"
#include "api/direct_dct_pls_postprocess.hpp"

#if GALP_WITH_JPEG_DCT

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numbers>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>

namespace galp::jpeg {
namespace {

constexpr uint64_t kTorchSeedModulus = (uint64_t {1} << 63U) - 1U;

class Sha256 {
public:
	void update(const std::string_view value) {
		for (const auto ch : value) {
			buffer_[buffer_size_++] = static_cast<uint8_t>(ch);
			if (buffer_size_ == buffer_.size()) {
				transform(buffer_.data());
				bit_count_ += 512U;
				buffer_size_ = 0U;
			}
		}
	}

	std::array<uint8_t, 32> finish() {
		bit_count_ += static_cast<uint64_t>(buffer_size_) * 8U;
		buffer_[buffer_size_++] = 0x80U;
		if (buffer_size_ > 56U) {
			std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_), buffer_.end(), 0U);
			transform(buffer_.data());
			buffer_size_ = 0U;
		}
		std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_), buffer_.begin() + 56, 0U);
		for (size_t index = 0U; index < 8U; ++index) {
			buffer_[63U - index] = static_cast<uint8_t>(bit_count_ >> (index * 8U));
		}
		transform(buffer_.data());
		std::array<uint8_t, 32> digest {};
		for (size_t word = 0U; word < state_.size(); ++word) {
			for (size_t byte = 0U; byte < 4U; ++byte) {
				digest[word * 4U + byte] = static_cast<uint8_t>(state_[word] >> (24U - byte * 8U));
			}
		}
		return digest;
	}

private:
	static constexpr std::array<uint32_t, 64> kRoundConstants {
	    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
	    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
	    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
	    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
	    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
	    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
	    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
	    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
	};

	static uint32_t choose(const uint32_t x, const uint32_t y, const uint32_t z) {
		return (x & y) ^ (~x & z);
	}
	static uint32_t majority(const uint32_t x, const uint32_t y, const uint32_t z) {
		return (x & y) ^ (x & z) ^ (y & z);
	}
	static uint32_t sigma0(const uint32_t x) {
		return std::rotr(x, 2) ^ std::rotr(x, 13) ^ std::rotr(x, 22);
	}
	static uint32_t sigma1(const uint32_t x) {
		return std::rotr(x, 6) ^ std::rotr(x, 11) ^ std::rotr(x, 25);
	}
	static uint32_t gamma0(const uint32_t x) {
		return std::rotr(x, 7) ^ std::rotr(x, 18) ^ (x >> 3U);
	}
	static uint32_t gamma1(const uint32_t x) {
		return std::rotr(x, 17) ^ std::rotr(x, 19) ^ (x >> 10U);
	}

	void transform(const uint8_t* block) {
		std::array<uint32_t, 64> schedule {};
		for (size_t index = 0U; index < 16U; ++index) {
			schedule[index] = (static_cast<uint32_t>(block[index * 4U]) << 24U) |
			                  (static_cast<uint32_t>(block[index * 4U + 1U]) << 16U) |
			                  (static_cast<uint32_t>(block[index * 4U + 2U]) << 8U) |
			                  static_cast<uint32_t>(block[index * 4U + 3U]);
		}
		for (size_t index = 16U; index < schedule.size(); ++index) {
			schedule[index] = gamma1(schedule[index - 2U]) + schedule[index - 7U] + gamma0(schedule[index - 15U]) +
			                  schedule[index - 16U];
		}
		auto a = state_[0];
		auto b = state_[1];
		auto c = state_[2];
		auto d = state_[3];
		auto e = state_[4];
		auto f = state_[5];
		auto g = state_[6];
		auto h = state_[7];
		for (size_t index = 0U; index < schedule.size(); ++index) {
			const auto t1 = h + sigma1(e) + choose(e, f, g) + kRoundConstants[index] + schedule[index];
			const auto t2 = sigma0(a) + majority(a, b, c);
			h             = g;
			g             = f;
			f             = e;
			e             = d + t1;
			d             = c;
			c             = b;
			b             = a;
			a             = t1 + t2;
		}
		state_[0] += a;
		state_[1] += b;
		state_[2] += c;
		state_[3] += d;
		state_[4] += e;
		state_[5] += f;
		state_[6] += g;
		state_[7] += h;
	}

	std::array<uint32_t, 8> state_ {
	    0x6a09e667U,
	    0xbb67ae85U,
	    0x3c6ef372U,
	    0xa54ff53aU,
	    0x510e527fU,
	    0x9b05688cU,
	    0x1f83d9abU,
	    0x5be0cd19U,
	};
	std::array<uint8_t, 64> buffer_ {};
	size_t                  buffer_size_ = 0U;
	uint64_t                bit_count_   = 0U;
};

std::array<uint8_t, 32> sha256(const std::string_view value) {
	Sha256 hash;
	hash.update(value);
	return hash.finish();
}

std::string sha256_hex(const std::string_view value) {
	const auto         digest = sha256(value);
	std::ostringstream result;
	result << std::hex << std::setfill('0');
	for (const auto byte : digest) {
		result << std::setw(2) << static_cast<unsigned>(byte);
	}
	return result.str();
}

std::string file_sha256_hex(const std::filesystem::path& path) {
	std::ifstream input(path, std::ios::binary);
	if (!input) {
		throw std::runtime_error("failed to open file for SHA-256: " + path.string());
	}
	Sha256                      hash;
	std::array<char, 1U << 20U> buffer {};
	while (input) {
		input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
		const auto count = input.gcount();
		if (count > 0)
			hash.update(std::string_view(buffer.data(), static_cast<size_t>(count)));
	}
	if (!input.eof()) {
		throw std::runtime_error("failed while hashing file: " + path.string());
	}
	const auto         digest = hash.finish();
	std::ostringstream result;
	result << std::hex << std::setfill('0');
	for (const auto byte : digest)
		result << std::setw(2) << static_cast<unsigned>(byte);
	return result.str();
}

std::string normalized_sha256(std::string_view value) {
	if (value.size() != 64U || !std::all_of(value.begin(), value.end(), [](const char ch) {
		    return std::isxdigit(static_cast<unsigned char>(ch)) != 0;
	    })) {
		throw std::invalid_argument("expected mapping SHA-256 must contain exactly 64 hexadecimal characters");
	}
	std::string result(value);
	std::transform(result.begin(), result.end(), result.begin(), [](const char ch) {
		return static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
	});
	return result;
}

uint64_t stable_seed(const std::string_view key) {
	const auto digest = sha256(key);
	uint64_t   value  = 0U;
	for (size_t index = 0U; index < 8U; ++index) {
		value |= static_cast<uint64_t>(digest[index]) << (index * 8U);
	}
	return value;
}

template <typename... Values>
std::string joined_key(const std::string_view name, const Values&... values) {
	std::ostringstream key;
	key << name;
	((key << ':' << values), ...);
	return key.str();
}

// CPython random.Random integer seeding and getrandbits behavior. Keeping this
// small implementation in the native scheduler preserves the registered
// sample/PLS order without retaining Python in the data plane.
class PythonRandom {
public:
	explicit PythonRandom(const uint64_t seed) {
		const std::array<uint32_t, 2> key {
		    static_cast<uint32_t>(seed),
		    static_cast<uint32_t>(seed >> 32U),
		};
		init_by_array(key);
	}

	uint32_t next_u32() {
		if (index_ >= state_.size()) {
			twist();
		}
		auto value = state_[index_++];
		value ^= value >> 11U;
		value ^= (value << 7U) & 0x9d2c5680U;
		value ^= (value << 15U) & 0xefc60000U;
		value ^= value >> 18U;
		return value;
	}

	uint64_t getrandbits(const unsigned bits) {
		if (bits == 0U) {
			return 0U;
		}
		if (bits <= 32U) {
			return static_cast<uint64_t>(next_u32() >> (32U - bits));
		}
		if (bits <= 64U) {
			const auto low       = static_cast<uint64_t>(next_u32());
			const auto high_bits = bits - 32U;
			const auto high      = static_cast<uint64_t>(next_u32() >> (32U - high_bits));
			return low | (high << 32U);
		}
		throw std::invalid_argument("PythonRandom supports at most 64 getrandbits bits");
	}

	uint64_t randbelow(const uint64_t upper) {
		if (upper == 0U) {
			throw std::invalid_argument("randbelow upper bound must be positive");
		}
		const auto bits  = 64U - static_cast<unsigned>(std::countl_zero(upper));
		auto       value = getrandbits(bits);
		while (value >= upper) {
			value = getrandbits(bits);
		}
		return value;
	}

	template <typename T>
	void shuffle(std::vector<T>& values) {
		for (size_t remaining = values.size(); remaining > 1U; --remaining) {
			const auto selected = static_cast<size_t>(randbelow(remaining));
			std::swap(values[remaining - 1U], values[selected]);
		}
	}

private:
	void init_genrand(const uint32_t seed) {
		state_[0] = seed;
		for (size_t index = 1U; index < state_.size(); ++index) {
			state_[index] =
			    1812433253U * (state_[index - 1U] ^ (state_[index - 1U] >> 30U)) + static_cast<uint32_t>(index);
		}
		index_ = state_.size();
	}

	void init_by_array(const std::array<uint32_t, 2>& key) {
		init_genrand(19650218U);
		size_t i = 1U;
		size_t j = 0U;
		for (size_t count = std::max(state_.size(), key.size()); count > 0U; --count) {
			state_[i] = (state_[i] ^ ((state_[i - 1U] ^ (state_[i - 1U] >> 30U)) * 1664525U)) + key[j] +
			            static_cast<uint32_t>(j);
			if (++i >= state_.size()) {
				state_[0] = state_.back();
				i         = 1U;
			}
			j = (j + 1U) % key.size();
		}
		for (size_t count = state_.size() - 1U; count > 0U; --count) {
			state_[i] =
			    (state_[i] ^ ((state_[i - 1U] ^ (state_[i - 1U] >> 30U)) * 1566083941U)) - static_cast<uint32_t>(i);
			if (++i >= state_.size()) {
				state_[0] = state_.back();
				i         = 1U;
			}
		}
		state_[0] = 0x80000000U;
		index_    = state_.size();
	}

	void twist() {
		for (size_t index = 0U; index < state_.size(); ++index) {
			const auto combined = (state_[index] & 0x80000000U) | (state_[(index + 1U) % state_.size()] & 0x7fffffffU);
			state_[index] =
			    state_[(index + 397U) % state_.size()] ^ (combined >> 1U) ^ ((combined & 1U) != 0U ? 0x9908b0dfU : 0U);
		}
		index_ = 0U;
	}

	std::array<uint32_t, 624> state_ {};
	size_t                    index_ = 624U;
};

class TorchCpuRandom {
public:
	explicit TorchCpuRandom(const uint64_t seed) {
		state_[0] = static_cast<uint32_t>(seed);
		for (size_t index = 1U; index < state_.size(); ++index) {
			state_[index] =
			    1812433253U * (state_[index - 1U] ^ (state_[index - 1U] >> 30U)) + static_cast<uint32_t>(index);
		}
	}

	uint32_t next_u32() {
		if (--left_ == 0) {
			next_state();
		}
		auto value = state_[next_++];
		value ^= value >> 11U;
		value ^= (value << 7U) & 0x9d2c5680U;
		value ^= (value << 15U) & 0xefc60000U;
		value ^= value >> 18U;
		return value;
	}

	float uniform_float() {
		return static_cast<float>(next_u32() & ((1U << 24U) - 1U)) / static_cast<float>(1U << 24U);
	}

	uint64_t next_u64() {
		return (static_cast<uint64_t>(next_u32()) << 32U) | next_u32();
	}

	double uniform_double() {
		return static_cast<double>(next_u64() & ((uint64_t {1} << 53U) - 1U)) /
		       static_cast<double>(uint64_t {1} << 53U);
	}

	float normal_float() {
		if (next_normal_.has_value()) {
			const auto value = *next_normal_;
			next_normal_.reset();
			return value;
		}
		const auto u1     = uniform_float();
		const auto u2     = uniform_float();
		const auto radius = std::sqrt(-2.0F * std::log1p(-u2));
		const auto theta  = 2.0F * std::numbers::pi_v<float> * u1;
		next_normal_      = radius * std::sin(theta);
		return radius * std::cos(theta);
	}

	double normal_double() {
		if (next_double_normal_.has_value()) {
			const auto value = *next_double_normal_;
			next_double_normal_.reset();
			return value;
		}
		const auto u1       = uniform_double();
		const auto u2       = uniform_double();
		const auto radius   = std::sqrt(-2.0 * std::log1p(-u2));
		const auto theta    = 2.0 * std::numbers::pi_v<double> * u1;
		next_double_normal_ = radius * std::sin(theta);
		return radius * std::cos(theta);
	}

	uint32_t uniform_index(const uint32_t upper) {
		if (upper == 0U) {
			throw std::invalid_argument("uniform index upper bound must be positive");
		}
		return next_u32() % upper;
	}

private:
	static uint32_t mix_bits(const uint32_t first, const uint32_t second) {
		return (first & 0x80000000U) | (second & 0x7fffffffU);
	}
	static uint32_t twist_word(const uint32_t first, const uint32_t second) {
		return (mix_bits(first, second) >> 1U) ^ ((second & 1U) != 0U ? 0x9908b0dfU : 0U);
	}
	void next_state() {
		auto* cursor = state_.data();
		left_        = static_cast<int>(state_.size());
		next_        = 0U;
		for (int count = 624 - 397 + 1; --count != 0; ++cursor) {
			*cursor = cursor[397] ^ twist_word(cursor[0], cursor[1]);
		}
		for (int count = 397; --count != 0; ++cursor) {
			*cursor = cursor[397 - 624] ^ twist_word(cursor[0], cursor[1]);
		}
		*cursor = cursor[397 - 624] ^ twist_word(cursor[0], state_[0]);
	}

	std::array<uint32_t, 624> state_ {};
	int                       left_ = 1;
	size_t                    next_ = 0U;
	std::optional<float>      next_normal_;
	std::optional<double>     next_double_normal_;
};

float sample_gamma(float alpha, TorchCpuRandom& random) {
	double scale = 1.0;
	if (alpha < 1.0F) {
		if (alpha == 0.0F)
			return 0.0F;
		scale *= std::pow(1.0 - random.uniform_double(), static_cast<double>(1.0F / alpha));
		alpha += 1.0F;
	}
	const double d = alpha - 1.0F / 3.0F;
	const auto   c = 1.0 / std::sqrt(9.0 * d);
	for (;;) {
		double x = 0.0;
		double y = 0.0;
		do {
			x = random.normal_double();
			y = 1.0 + c * x;
		} while (y <= 0.0);
		const auto v  = y * y * y;
		const auto u  = 1.0 - random.uniform_double();
		const auto xx = x * x;
		if (u < 1.0 - 0.0331F * xx * xx || std::log(u) < 0.5F * xx + d * (1.0F - v + std::log(v))) {
			return static_cast<float>(scale * d * v);
		}
	}
}

detail::DirectDctPlsMixupDecision
published_mixup_decision(const uint64_t training_seed, const uint32_t epoch, const uint64_t microbatch_index) {
	const auto     key            = sha256_hex(joined_key("dct-mixup", training_seed, epoch, microbatch_index));
	const auto     generator_seed = stable_seed(joined_key("dct-mixup-dirichlet", key));
	TorchCpuRandom random(generator_seed % kTorchSeedModulus);
	const auto     first  = sample_gamma(0.2F, random);
	const auto     second = sample_gamma(0.2F, random);
	const auto     total  = first + second;
	if (!(total > 0.0F)) {
		return {1.0F, 0.0F};
	}
	const auto left  = first / total;
	const auto right = second / total;
	return {std::max(left, right), std::min(left, right)};
}

constexpr std::array<detail::DirectDctPlsRandAugmentOp, 14> kPublishedOperations {
    detail::DirectDctPlsRandAugmentOp::kAutoContrast,
    detail::DirectDctPlsRandAugmentOp::kPosterize,
    detail::DirectDctPlsRandAugmentOp::kSolarizeAdd,
    detail::DirectDctPlsRandAugmentOp::kColor,
    detail::DirectDctPlsRandAugmentOp::kContrast,
    detail::DirectDctPlsRandAugmentOp::kBrightness,
    detail::DirectDctPlsRandAugmentOp::kMidfreqAug,
    detail::DirectDctPlsRandAugmentOp::kCutout,
    detail::DirectDctPlsRandAugmentOp::kTranslateX,
    detail::DirectDctPlsRandAugmentOp::kTranslateY,
    detail::DirectDctPlsRandAugmentOp::kRotate90,
    detail::DirectDctPlsRandAugmentOp::kAutoSaturation,
    detail::DirectDctPlsRandAugmentOp::kGrayscale,
    detail::DirectDctPlsRandAugmentOp::kChromaDrop,
};

bool is_chroma_operation(const detail::DirectDctPlsRandAugmentOp operation) {
	return operation == detail::DirectDctPlsRandAugmentOp::kColor ||
	       operation == detail::DirectDctPlsRandAugmentOp::kAutoSaturation ||
	       operation == detail::DirectDctPlsRandAugmentOp::kGrayscale ||
	       operation == detail::DirectDctPlsRandAugmentOp::kChromaDrop;
}

std::pair<float, bool> published_magnitude(const detail::DirectDctPlsRandAugmentOp operation) {
	switch (operation) {
	case detail::DirectDctPlsRandAugmentOp::kPosterize:
		return {2.0F, false};
	case detail::DirectDctPlsRandAugmentOp::kSolarizeAdd:
		return {264.9F, false};
	case detail::DirectDctPlsRandAugmentOp::kColor:
	case detail::DirectDctPlsRandAugmentOp::kContrast:
	case detail::DirectDctPlsRandAugmentOp::kBrightness:
	case detail::DirectDctPlsRandAugmentOp::kMidfreqAug:
		return {0.27F, true};
	case detail::DirectDctPlsRandAugmentOp::kCutout:
		return {1.8F, false};
	case detail::DirectDctPlsRandAugmentOp::kTranslateX:
	case detail::DirectDctPlsRandAugmentOp::kTranslateY:
		return {3.75F, true};
	case detail::DirectDctPlsRandAugmentOp::kRotate90:
		return {1.0F, true};
	default:
		return {0.0F, false};
	}
}

detail::DirectDctPlsRandAugmentDecision published_randaugment_decision(const DirectDctPlsSample&          sample,
                                                                       const DirectDctPlsScheduleOptions& options) {
	detail::DirectDctPlsRandAugmentDecision        result;
	std::vector<detail::DirectDctPlsRandAugmentOp> available(kPublishedOperations.begin(), kPublishedOperations.end());
	for (uint32_t stage = 0U; stage < 2U; ++stage) {
		const auto key = sha256_hex(
		    joined_key("dct-randaugment", options.training_seed, options.epoch, sample.logical_sample_id, stage));
		const auto choice_seed   = stable_seed(joined_key(
            "dct-randaugment-choice", options.training_seed, options.epoch, sample.logical_sample_id, stage));
		const auto operation     = available.at(static_cast<size_t>(choice_seed % available.size()));
		result.operations[stage] = operation;
		if (is_chroma_operation(operation)) {
			if (operation == detail::DirectDctPlsRandAugmentOp::kGrayscale) {
				std::erase_if(available, [](const auto candidate) { return is_chroma_operation(candidate); });
			} else {
				std::erase(available, detail::DirectDctPlsRandAugmentOp::kGrayscale);
			}
		}
		auto [magnitude, signed_magnitude] = published_magnitude(operation);
		if (signed_magnitude && stable_seed(joined_key("dct-randaugment-sign", key)) % 2U != 0U) {
			magnitude *= -1.0F;
		}
		result.magnitudes[stage] = magnitude;
		const auto internal_seed = stable_seed(joined_key("dct-randaugment-internal", key));
		if (operation == detail::DirectDctPlsRandAugmentOp::kCutout) {
			TorchCpuRandom random(internal_seed % kTorchSeedModulus);
			result.cutout_center_h[stage] = static_cast<int16_t>((random.uniform_index(28U) / 2U) * 2U);
			result.cutout_center_w[stage] = static_cast<int16_t>((random.uniform_index(28U) / 2U) * 2U);
		}
		if (operation == detail::DirectDctPlsRandAugmentOp::kChromaDrop) {
			result.chroma_drop_channel[stage] =
			    static_cast<uint8_t>(stable_seed(joined_key("chroma-drop", internal_seed)) % 2U);
		}
	}
	return result;
}

uint64_t parse_u64(const std::string_view text, const char* field) {
	uint64_t value          = 0U;
	const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
	if (error != std::errc {} || end != text.data() + text.size()) {
		throw std::runtime_error(std::string("invalid unsigned integer in ") + field);
	}
	return value;
}

int64_t parse_i64(const std::string_view text, const char* field) {
	int64_t value           = 0;
	const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
	if (error != std::errc {} || end != text.data() + text.size()) {
		throw std::runtime_error(std::string("invalid signed integer in ") + field);
	}
	return value;
}

std::vector<std::string> split_csv_line(const std::string_view line) {
	std::vector<std::string> fields;
	std::string              current;
	bool                     quoted = false;
	for (size_t index = 0U; index < line.size(); ++index) {
		const auto ch = line[index];
		if (ch == '"') {
			if (quoted && index + 1U < line.size() && line[index + 1U] == '"') {
				current.push_back('"');
				++index;
			} else {
				quoted = !quoted;
			}
		} else if (ch == ',' && !quoted) {
			fields.push_back(std::move(current));
			current.clear();
		} else if (ch != '\r') {
			current.push_back(ch);
		}
	}
	if (quoted) {
		throw std::runtime_error("unterminated quoted CSV field");
	}
	fields.push_back(std::move(current));
	return fields;
}

size_t required_column(const std::vector<std::string>& header, const std::string_view name) {
	const auto found = std::find(header.begin(), header.end(), name);
	if (found == header.end()) {
		throw std::runtime_error("premixed mapping is missing required column: " + std::string(name));
	}
	return static_cast<size_t>(std::distance(header.begin(), found));
}

uint32_t checked_u32(const uint64_t value, const char* field) {
	if (value > std::numeric_limits<uint32_t>::max()) {
		throw std::runtime_error(std::string(field) + " exceeds uint32 range");
	}
	return static_cast<uint32_t>(value);
}

uint32_t closest_crop_size(const double value, const uint32_t maximum) {
	constexpr std::array<uint32_t, 4> choices {2U, 4U, 14U, 28U};
	if (value <= choices.back()) {
		return *std::min_element(choices.begin(), choices.end(), [&](const auto left, const auto right) {
			return std::tuple {std::abs(static_cast<double>(left) - value), left} <
			       std::tuple {std::abs(static_cast<double>(right) - value), right};
		});
	}
	auto closest = static_cast<int64_t>(std::nearbyint(value / choices.back())) * choices.back();
	if (closest > maximum) {
		closest -= choices.back();
	}
	return static_cast<uint32_t>(std::max<int64_t>(closest, 0));
}

JpegDctCropBox published_crop(const uint64_t seed, const uint32_t source_width, const uint32_t source_height) {
	const auto width_blocks  = (source_width + 7U) / 8U;
	const auto height_blocks = (source_height + 7U) / 8U;
	if (width_blocks < 2U || height_blocks < 2U) {
		throw std::runtime_error("published DCT crop requires at least a 2x2 luma block grid");
	}
	TorchCpuRandom random(seed % kTorchSeedModulus);
	const auto     area = static_cast<uint64_t>(width_blocks) * height_blocks;
	for (size_t attempt = 0U; attempt < 10U; ++attempt) {
		const auto draw        = static_cast<double>(random.uniform_float());
		const auto target_area = static_cast<double>(area) * (0.05 + draw * 0.95);
		auto       width       = closest_crop_size(std::nearbyint(std::sqrt(target_area)), width_blocks);
		auto       height      = width;
		width                  = std::max<uint32_t>(width, 2U);
		height                 = std::max<uint32_t>(height, 2U);
		if (width <= width_blocks && height <= height_blocks) {
			const auto top  = (random.uniform_index(height_blocks - height + 1U) / 2U) * 2U;
			const auto left = (random.uniform_index(width_blocks - width + 1U) / 2U) * 2U;
			return {left * 8U, top * 8U, width * 8U, height * 8U};
		}
	}
	const auto width  = std::max<uint32_t>(1U, closest_crop_size(width_blocks, width_blocks));
	const auto height = std::max<uint32_t>(1U, closest_crop_size(height_blocks, height_blocks));
	const auto top    = (((height_blocks - height) / 2U) / 2U) * 2U;
	const auto left   = (((width_blocks - width) / 2U) / 2U) * 2U;
	return {left * 8U, top * 8U, width * 8U, height * 8U};
}

DirectDctGridTensorDescriptor
slice_grid(DirectDctGridTensorDescriptor source, const size_t offset, const size_t count) {
	if (offset > source.shape[0] || count > source.shape[0] - offset) {
		throw std::out_of_range("Direct-DCT PLS tensor slice exceeds pool bounds");
	}
	const auto element_offset = offset * source.strides[0];
	if (source.dtype == DirectDctTensorDataType::kFloat32) {
		source.float_data += element_offset;
	} else {
		source.data += element_offset;
	}
	source.shape[0] = count;
	return source;
}

} // namespace

detail::DirectDctPlsRandAugmentDecision
detail::derive_published_randaugment_decision(const DirectDctPlsSample&          sample,
                                              const DirectDctPlsScheduleOptions& options) {
	return published_randaugment_decision(sample, options);
}

detail::DirectDctPlsMixupDecision detail::derive_published_mixup_decision(const uint64_t training_seed,
                                                                          const uint32_t epoch,
                                                                          const uint64_t microbatch_index) {
	return published_mixup_decision(training_seed, epoch, microbatch_index);
}

DirectDctPlsLayout DirectDctPlsLayout::LoadPremixedCsv(const std::filesystem::path& mapping_csv,
                                                       const JpegDctShardManifest&  manifest,
                                                       const uint32_t               segment_images,
                                                       const std::string_view       expected_sha256,
                                                       const uint32_t               model_classes) {
	if (segment_images == 0U) {
		throw std::invalid_argument("PLS segment_images must be positive");
	}
	if (model_classes == 0U) {
		throw std::invalid_argument("PLS model_classes must be positive");
	}
	if (!expected_sha256.empty() && file_sha256_hex(mapping_csv) != normalized_sha256(expected_sha256)) {
		throw std::runtime_error("premixed mapping SHA-256 disagrees with the registered physical layout");
	}
	if (manifest.image_count > std::numeric_limits<uint32_t>::max()) {
		throw std::runtime_error("PLS manifest exceeds uint32 image index range");
	}
	std::ifstream input(mapping_csv);
	if (!input) {
		throw std::runtime_error("failed to open premixed mapping CSV: " + mapping_csv.string());
	}
	std::string line;
	if (!std::getline(input, line)) {
		throw std::runtime_error("premixed mapping CSV is empty");
	}
	const auto header         = split_csv_line(line);
	const auto planned_column = required_column(header, "planned_physical_position");
	const auto pls_column     = required_column(header, "virtual_pls_id");
	const auto in_pls_column  = required_column(header, "position_in_pls");
	const auto source_column  = required_column(header, "galp_image_id");
	const auto logical_column = required_column(header, "logical_sample_id");
	const auto label_column   = required_column(header, "label");
	const auto max_column =
	    std::max({planned_column, pls_column, in_pls_column, source_column, logical_column, label_column});

	std::vector<DirectDctPlsSample> samples;
	samples.reserve(static_cast<size_t>(manifest.image_count));
	std::vector<std::vector<uint32_t>> positions_by_pls(manifest.shards.size());
	while (std::getline(input, line)) {
		if (line.empty()) {
			continue;
		}
		const auto fields = split_csv_line(line);
		if (fields.size() <= max_column) {
			throw std::runtime_error("premixed mapping CSV row has too few fields");
		}
		const auto planned_position =
		    checked_u32(parse_u64(fields[planned_column], "planned_physical_position"), "planned_physical_position");
		if (planned_position != samples.size()) {
			throw std::runtime_error("premixed mapping positions are not contiguous physical order");
		}
		DirectDctPlsSample sample;
		sample.global_image_index = planned_position;
		sample.virtual_pls_id     = checked_u32(parse_u64(fields[pls_column], "virtual_pls_id"), "virtual_pls_id");
		sample.position_in_pls    = checked_u32(parse_u64(fields[in_pls_column], "position_in_pls"), "position_in_pls");
		sample.source_image_id    = checked_u32(parse_u64(fields[source_column], "galp_image_id"), "galp_image_id");
		sample.logical_sample_id  = fields[logical_column];
		sample.label              = parse_i64(fields[label_column], "label");
		if (sample.label < 0 || static_cast<uint64_t>(sample.label) >= model_classes) {
			throw std::runtime_error("premixed mapping label is outside the configured model class range");
		}
		if (sample.virtual_pls_id >= positions_by_pls.size()) {
			throw std::runtime_error("premixed mapping virtual_pls_id exceeds physical shard count");
		}
		if (sample.position_in_pls != positions_by_pls[sample.virtual_pls_id].size() ||
		    sample.position_in_pls >= segment_images) {
			throw std::runtime_error("premixed mapping position_in_pls is not contiguous or exceeds G");
		}
		positions_by_pls[sample.virtual_pls_id].push_back(planned_position);
		samples.push_back(std::move(sample));
	}
	if (samples.size() != manifest.image_count) {
		throw std::runtime_error("premixed mapping cardinality disagrees with the DCT manifest");
	}
	uint64_t expected_first = 0U;
	for (size_t pls = 0U; pls < manifest.shards.size(); ++pls) {
		const auto& shard = manifest.shards[pls];
		if (shard.first_global_image_index != expected_first || shard.image_count == 0U ||
		    shard.image_count > segment_images || positions_by_pls[pls].size() != shard.image_count) {
			throw std::runtime_error("physical shard boundaries do not match frozen PLS mapping");
		}
		if (pls + 1U != manifest.shards.size() && shard.image_count != segment_images) {
			throw std::runtime_error("only the final physical PLS may be shorter than G");
		}
		expected_first += shard.image_count;
	}
	if (expected_first != manifest.image_count) {
		throw std::runtime_error("physical PLS shards do not cover the manifest exactly once");
	}
	return DirectDctPlsLayout(std::move(samples), std::move(positions_by_pls), segment_images);
}

DirectDctPlsLayout::DirectDctPlsLayout(std::vector<DirectDctPlsSample>    samples,
                                       std::vector<std::vector<uint32_t>> positions_by_pls,
                                       const uint32_t                     segment_images)
    : samples_(std::move(samples))
    , positions_by_pls_(std::move(positions_by_pls))
    , segment_images_(segment_images) {
	if (segment_images_ == 0U || samples_.empty() || positions_by_pls_.empty()) {
		throw std::invalid_argument("Direct-DCT PLS layout must be non-empty with positive G");
	}
	std::vector<uint8_t> seen(samples_.size(), 0U);
	for (size_t pls = 0U; pls < positions_by_pls_.size(); ++pls) {
		if (positions_by_pls_[pls].empty() || positions_by_pls_[pls].size() > segment_images_) {
			throw std::invalid_argument("Direct-DCT PLS has invalid cardinality");
		}
		for (size_t position = 0U; position < positions_by_pls_[pls].size(); ++position) {
			const auto global = positions_by_pls_[pls][position];
			if (global >= samples_.size() || seen[global] != 0U || samples_[global].global_image_index != global ||
			    samples_[global].virtual_pls_id != pls || samples_[global].position_in_pls != position) {
				throw std::invalid_argument("Direct-DCT PLS layout is missing, duplicated, or internally inconsistent");
			}
			seen[global] = 1U;
		}
	}
	if (std::find(seen.begin(), seen.end(), 0U) != seen.end()) {
		throw std::invalid_argument("Direct-DCT PLS layout does not cover every physical image");
	}
}

size_t DirectDctPlsLayout::sample_count() const noexcept {
	return samples_.size();
}
size_t DirectDctPlsLayout::pls_count() const noexcept {
	return positions_by_pls_.size();
}
uint32_t DirectDctPlsLayout::segment_images() const noexcept {
	return segment_images_;
}

const DirectDctPlsSample& DirectDctPlsLayout::sample(const uint32_t physical_position) const {
	return samples_.at(physical_position);
}

std::span<const uint32_t> DirectDctPlsLayout::positions_in_pls(const uint32_t virtual_pls_id) const {
	return positions_by_pls_.at(virtual_pls_id);
}

DirectDctPlsEpochSchedule::DirectDctPlsEpochSchedule(const DirectDctPlsLayout&   layout,
                                                     DirectDctPlsScheduleOptions options)
    : layout_(&layout)
    , options_(options) {
	if (options_.segments_per_pool == 0U || options_.microbatch_images == 0U) {
		throw std::invalid_argument("Direct-DCT PLS M and microbatch size must be positive");
	}
	if (options_.order_policy == DirectDctPlsOrderPolicy::kGlobal) {
		global_order_.resize(layout.sample_count());
		std::iota(global_order_.begin(), global_order_.end(), 0U);
		PythonRandom(stable_seed(joined_key("global-sample-order", options_.training_seed, options_.epoch)))
		    .shuffle(global_order_);
	} else {
		pls_order_.resize(layout.pls_count());
		std::iota(pls_order_.begin(), pls_order_.end(), 0U);
		if (options_.order_policy == DirectDctPlsOrderPolicy::kClosedPool) {
			PythonRandom(stable_seed(joined_key("closed-pool-pls-order", options_.training_seed, options_.epoch)))
			    .shuffle(pls_order_);
		}
	}
}

bool DirectDctPlsEpochSchedule::has_next() const noexcept {
	return options_.order_policy == DirectDctPlsOrderPolicy::kGlobal ? next_pls_ < global_order_.size()
	                                                                 : next_pls_ < pls_order_.size();
}

size_t DirectDctPlsEpochSchedule::remaining_pool_count() const noexcept {
	if (!has_next()) {
		return 0U;
	}
	if (options_.order_policy == DirectDctPlsOrderPolicy::kGlobal) {
		const auto capacity = static_cast<size_t>(options_.segments_per_pool) * layout_->segment_images();
		return (global_order_.size() - next_pls_ + capacity - 1U) / capacity;
	}
	return (pls_order_.size() - next_pls_ + options_.segments_per_pool - 1U) / options_.segments_per_pool;
}

DirectDctPlsPoolPlan DirectDctPlsEpochSchedule::next_pool() {
	if (!has_next()) {
		throw std::out_of_range("Direct-DCT PLS epoch schedule is exhausted");
	}
	DirectDctPlsPoolPlan pool;
	pool.epoch                  = options_.epoch;
	pool.pool_index             = next_pool_index_++;
	pool.first_microbatch_index = next_microbatch_index_;
	if (options_.order_policy == DirectDctPlsOrderPolicy::kGlobal) {
		const auto capacity = static_cast<size_t>(options_.segments_per_pool) * layout_->segment_images();
		const auto end      = std::min(global_order_.size(), next_pls_ + capacity);
		pool.ordered_positions.assign(global_order_.begin() + static_cast<std::ptrdiff_t>(next_pls_),
		                              global_order_.begin() + static_cast<std::ptrdiff_t>(end));
		next_pls_ = end;
		next_microbatch_index_ +=
		    (pool.ordered_positions.size() + options_.microbatch_images - 1U) / options_.microbatch_images;
		return pool;
	}
	const auto end = std::min(pls_order_.size(), next_pls_ + options_.segments_per_pool);
	pool.virtual_pls_ids.assign(pls_order_.begin() + static_cast<std::ptrdiff_t>(next_pls_),
	                            pls_order_.begin() + static_cast<std::ptrdiff_t>(end));
	for (const auto pls : pool.virtual_pls_ids) {
		const auto positions = layout_->positions_in_pls(pls);
		pool.ordered_positions.insert(pool.ordered_positions.end(), positions.begin(), positions.end());
	}
	if (options_.order_policy == DirectDctPlsOrderPolicy::kClosedPool) {
		PythonRandom(
		    stable_seed(joined_key("closed-pool-sample-order", options_.training_seed, options_.epoch, pool.pool_index)))
		    .shuffle(pool.ordered_positions);
	}
	next_pls_ = end;
	next_microbatch_index_ +=
	    (pool.ordered_positions.size() + options_.microbatch_images - 1U) / options_.microbatch_images;
	return pool;
}

DirectDctPlsAugmentationDecision derive_direct_dct_pls_augmentation(const DirectDctPlsSample&          sample,
                                                                    const uint32_t                     source_width,
                                                                    const uint32_t                     source_height,
                                                                    const DirectDctPlsScheduleOptions& options) {
	const auto crop_key =
	    options.crop_policy == DirectDctPlsCropPolicy::kPerSample
	        ? joined_key("crop-per-sample", options.training_seed, options.epoch, sample.logical_sample_id)
	        : joined_key("crop-per-pls", options.training_seed, options.epoch, sample.virtual_pls_id);
	const auto flip_key = joined_key("horizontal-flip", options.training_seed, options.epoch, sample.logical_sample_id);
	DirectDctPlsAugmentationDecision result;
	result.crop_seed                  = stable_seed(crop_key);
	result.flip_seed                  = stable_seed(flip_key);
	result.request.global_image_index = sample.global_image_index;
	result.request.source_crop        = published_crop(result.crop_seed, source_width, source_height);
	TorchCpuRandom flip_random(result.flip_seed % kTorchSeedModulus);
	result.request.horizontal_flip   = flip_random.uniform_float() < 0.5F;
	result.request.logical_sample_id = sample.logical_sample_id;
	result.request.augmentation_key  = sha256_hex(
        joined_key("galp-training-augmentation-v1", options.training_seed, options.epoch, sample.logical_sample_id));
	return result;
}

namespace {

using PlsPoolClock = std::chrono::steady_clock;

double elapsed_ms(const PlsPoolClock::time_point begin, const PlsPoolClock::time_point end) {
	return std::chrono::duration<double, std::milli>(end - begin).count();
}

void check_pls_cuda(const cudaError_t status, const char* operation) {
	if (status != cudaSuccess) {
		throw std::runtime_error(std::string("Direct-DCT PLS ") + operation +
		                         " failed: " + cudaGetErrorString(status));
	}
}

// Bounds materialized pool contexts. The permit is part of the pool backing, so
// NativeBatchLease/NativeBatchCompletion release it only after producer,
// consumers, and external storage owners are complete.
class DirectDctPlsPoolContextSlots : public std::enable_shared_from_this<DirectDctPlsPoolContextSlots> {
public:
	static constexpr size_t kCapacity = 2U;

	class Permit {
	public:
		explicit Permit(std::shared_ptr<DirectDctPlsPoolContextSlots> owner)
		    : owner_(std::move(owner)) {
		}
		~Permit() {
			if (owner_) {
				owner_->release();
			}
		}
		Permit(const Permit&)            = delete;
		Permit& operator=(const Permit&) = delete;

	private:
		std::shared_ptr<DirectDctPlsPoolContextSlots> owner_;
	};

	std::shared_ptr<void> acquire() {
		std::unique_lock lock(mutex_);
		++stats_.context_waiter_count;
		stats_.peak_context_waiter_count =
		    std::max(stats_.peak_context_waiter_count, stats_.context_waiter_count);
		cv_.wait(lock, [this] { return stopping_ || stats_.live_context_count < kCapacity; });
		--stats_.context_waiter_count;
		if (stopping_) {
			return {};
		}
		++stats_.live_context_count;
		stats_.peak_live_context_count =
		    std::max(stats_.peak_live_context_count, stats_.live_context_count);
		return std::make_shared<Permit>(shared_from_this());
	}

	void stop() noexcept {
		{
			std::lock_guard lock(mutex_);
			stopping_ = true;
		}
		cv_.notify_all();
	}
	void reset_epoch_stats() noexcept {
		std::lock_guard lock(mutex_);
		const auto live                = stats_.live_context_count;
		stats_                         = DirectDctPlsPoolPrefetchStats {};
		stats_.live_context_count      = live;
		stats_.peak_live_context_count = live;
	}

	void prepare_started() noexcept {
		std::lock_guard lock(mutex_);
		++stats_.prepare_started_count;
	}
	void prepare_completed(const double plan_ms, const double io_ms, const double materialize_ms) noexcept {
		std::lock_guard lock(mutex_);
		++stats_.prepare_completed_count;
		stats_.prepare_plan_ms += plan_ms;
		stats_.prepare_io_ms += io_ms;
		stats_.prepare_materialize_ms += materialize_ms;
	}
	void activated(const double wait_ms, const double activation_ms) noexcept {
		std::lock_guard lock(mutex_);
		++stats_.activation_count;
		stats_.activation_wait_ms += wait_ms;
		stats_.activation_ms += activation_ms;
	}
	[[nodiscard]] DirectDctPlsPoolPrefetchStats snapshot() const noexcept {
		std::lock_guard lock(mutex_);
		return stats_;
	}

private:
	void release() noexcept {
		{
			std::lock_guard lock(mutex_);
			if (stats_.live_context_count != 0U) {
				--stats_.live_context_count;
			}
			++stats_.retired_count;
		}
		cv_.notify_one();
	}

	mutable std::mutex            mutex_;
	std::condition_variable       cv_;
	bool                          stopping_ = false;
	DirectDctPlsPoolPrefetchStats stats_;
};

struct PendingDirectDctPlsPool {
	DirectDctPlsPoolPlan        plan;
	DirectDctPlsScheduleOptions schedule;
};

struct PreparedDirectDctPlsPool {
	DirectDctPlsPoolBatch batch;
};

} // namespace

struct DirectDctPlsPoolBatch::Impl {
	// Declared first so implicit destruction releases the scheduling permit
	// only after postprocess and Direct-DCT backing owners are destroyed.
	std::shared_ptr<void>                                  pool_context_owner;
	DirectDctPlsPoolPlan                                 plan;
	DirectDctBatch                                       batch;
	std::vector<int64_t>                                 labels;
	uint32_t                                             microbatch_images = 0U;
	std::unique_ptr<detail::DirectDctPlsCudaPostprocess> postprocess;
};

DirectDctPlsPoolBatch::DirectDctPlsPoolBatch() noexcept                                   = default;
DirectDctPlsPoolBatch::~DirectDctPlsPoolBatch()                                           = default;
DirectDctPlsPoolBatch::DirectDctPlsPoolBatch(DirectDctPlsPoolBatch&&) noexcept            = default;
DirectDctPlsPoolBatch& DirectDctPlsPoolBatch::operator=(DirectDctPlsPoolBatch&&) noexcept = default;

DirectDctPlsPoolBatch::DirectDctPlsPoolBatch(DirectDctPlsPoolPlan plan,
                                             DirectDctBatch       batch,
                                             std::vector<int64_t> labels,
                                             const uint32_t       microbatch_images,
                                             std::shared_ptr<void> pool_context_owner)
    : impl_(std::make_unique<Impl>(Impl {
	      std::move(pool_context_owner),
          std::move(plan),
          std::move(batch),
          std::move(labels),
          microbatch_images,
          nullptr,
      })) {
}

void DirectDctPlsPoolBatch::retire_context() noexcept {
	// Compatibility marker only. The bounded permit follows native backing
	// lifetime and cannot be released early by a scheduling-context call.
}

uint32_t DirectDctPlsPoolBatch::epoch() const noexcept {
	return impl_ ? impl_->plan.epoch : 0U;
}
uint32_t DirectDctPlsPoolBatch::pool_index() const noexcept {
	return impl_ ? impl_->plan.pool_index : 0U;
}
size_t DirectDctPlsPoolBatch::image_count() const noexcept {
	return impl_ ? impl_->labels.size() : 0U;
}
size_t DirectDctPlsPoolBatch::microbatch_count() const noexcept {
	return impl_ ? (impl_->labels.size() + impl_->microbatch_images - 1U) / impl_->microbatch_images : 0U;
}
const std::vector<uint32_t>& DirectDctPlsPoolBatch::virtual_pls_ids() const noexcept {
	static const std::vector<uint32_t> empty;
	return impl_ ? impl_->plan.virtual_pls_ids : empty;
}
const DirectDctBatch& DirectDctPlsPoolBatch::batch() const noexcept {
	return impl_->batch;
}
void* DirectDctPlsPoolBatch::cuda_completion_event() const noexcept {
	if (!impl_)
		return nullptr;
	return impl_->postprocess ? impl_->postprocess->completion_event() : impl_->batch.cuda_completion_event();
}

DirectDctPlsMicrobatchView DirectDctPlsPoolBatch::microbatch(const size_t index) const {
	if (!impl_ || index >= microbatch_count()) {
		throw std::out_of_range("Direct-DCT PLS microbatch index is outside the pool");
	}
	const auto  offset = index * impl_->microbatch_images;
	const auto  count  = std::min<size_t>(impl_->microbatch_images, impl_->labels.size() - offset);
	const auto& ids    = impl_->batch.global_image_ids();
	if (ids.size() != impl_->labels.size()) {
		throw std::runtime_error("Direct-DCT PLS batch metadata cardinality mismatch");
	}
	const auto y       = impl_->postprocess ? impl_->postprocess->y_tensor() : impl_->batch.y_tensor_async();
	const auto c       = impl_->postprocess ? impl_->postprocess->cbcr_tensor() : impl_->batch.cbcr_tensor_async();
	auto       targets = impl_->postprocess ? impl_->postprocess->targets() : DirectDctPlsTargetTensorDescriptor {};
	if (!targets.empty()) {
		targets.data += offset * targets.strides[0];
		targets.shape[0] = count;
	}
	return DirectDctPlsMicrobatchView {
	    slice_grid(y, offset, count),
	    slice_grid(c, offset, count),
	    targets,
	    std::span<const uint32_t>(ids).subspan(offset, count),
	    std::span<const int64_t>(impl_->labels).subspan(offset, count),
	    offset,
	    count,
	};
}

struct DirectDctPlsPipeline::Impl {
	DirectDctPlsPipelineOptions              options;
	DirectDctRuntime                         runtime;
	DirectDctPlsLayout                       layout;
	std::optional<DirectDctPlsEpochSchedule> epoch;
	std::shared_ptr<DirectDctPlsPoolContextSlots>  context_slots =
	    std::make_shared<DirectDctPlsPoolContextSlots>();
	mutable std::mutex                            work_mutex;
	std::condition_variable                       work_cv;
	std::optional<PendingDirectDctPlsPool>         pending;
	std::optional<PreparedDirectDctPlsPool>        ready;
	std::exception_ptr                            worker_failure;
	bool                                          worker_busy = false;
	bool                                          stopping    = false;
	int                                           cuda_device = 0;
	std::shared_ptr<detail::DirectDctPlsCudaPostprocess::Stream> postprocess_stream;
	std::thread                                   worker;

	Impl(const std::filesystem::path& manifest_path,
	     const std::filesystem::path& mapping_csv,
	     DirectDctPlsPipelineOptions  source_options)
	    : options(std::move(source_options))
	    , runtime(manifest_path)
	    , layout(DirectDctPlsLayout::LoadPremixedCsv(mapping_csv,
	                                                 read_jpeg_dct_shard_manifest(manifest_path),
	                                                 options.segment_images,
	                                                 options.expected_mapping_sha256,
	                                                 options.model_classes)) {
		if (options.expected_mapping_sha256.empty()) {
			throw std::invalid_argument("production PLS pipeline requires the registered mapping SHA-256");
		}
		if (runtime.image_count() != layout.sample_count()) {
			throw std::runtime_error("Direct-DCT runtime and premixed PLS layout cardinalities differ");
		}
		check_pls_cuda(cudaGetDevice(&cuda_device), "capture CUDA device");
		postprocess_stream =
		    std::make_shared<detail::DirectDctPlsCudaPostprocess::Stream>(cuda_device);
		if (options.require_block_major_planless) {
			const auto stats = runtime.InitializationStats();
			if (!stats.block_major_metadata_lazy || !options.device.enable_planless_execution ||
			    options.device.layout != JpegDctDeviceLayout::kTransformedDctGrid ||
			    !options.device.grid_transform.has_value()) {
				throw std::runtime_error("production PLS pipeline requires block-major access sidecars and "
				                         "transformed-grid planless execution");
			}
		}
		worker = std::thread([this] { worker_loop(); });
	}

	~Impl() {
		{
			std::lock_guard lock(work_mutex);
			stopping = true;
		}
		context_slots->stop();
		work_cv.notify_all();
		if (worker.joinable()) {
			worker.join();
		}
	}

	void enqueue_next_locked() {
		if (!epoch.has_value() || !epoch->has_next() || pending.has_value() || worker_busy || ready.has_value()) {
			return;
		}
		pending.emplace(PendingDirectDctPlsPool {epoch->next_pool(), options.schedule});
	}

	PreparedDirectDctPlsPool prepare_pool(PendingDirectDctPlsPool source) {
		check_pls_cuda(cudaSetDevice(cuda_device), "activate worker CUDA device");
		auto context_owner = context_slots->acquire();
		if (!context_owner) {
			return {};
		}
		context_slots->prepare_started();
		const auto plan_started = PlsPoolClock::now();
		std::vector<JpegDctImageCropRequest>                    requests;
		std::vector<int64_t>                                    labels;
		std::vector<detail::DirectDctPlsRandAugmentDecision>    randaugment;
		requests.reserve(source.plan.ordered_positions.size());
		labels.reserve(source.plan.ordered_positions.size());
		randaugment.reserve(source.plan.ordered_positions.size());
		for (const auto physical_position : source.plan.ordered_positions) {
			const auto& sample       = layout.sample(physical_position);
			const auto  metadata     = runtime.ImageMetadata(sample.global_image_index);
			auto        augmentation = derive_direct_dct_pls_augmentation(
			    sample, metadata.image_width, metadata.image_height, source.schedule);
			requests.push_back(std::move(augmentation.request));
			labels.push_back(sample.label);
			randaugment.push_back(detail::derive_published_randaugment_decision(sample, source.schedule));
		}
		std::vector<detail::DirectDctPlsMixupDecision> mixup;
		const auto microbatch_count =
		    (labels.size() + source.schedule.microbatch_images - 1U) / source.schedule.microbatch_images;
		mixup.reserve(microbatch_count);
		for (size_t index = 0U; index < microbatch_count; ++index) {
			mixup.push_back(detail::derive_published_mixup_decision(
			    source.schedule.training_seed,
			    source.schedule.epoch,
			    source.plan.first_microbatch_index + index));
		}
		auto prepared          = runtime.PrepareBatch(requests, options.device);
		const auto plan_done   = PlsPoolClock::now();
		runtime.StageBatchIo(prepared);
		const auto io_done = PlsPoolClock::now();
		auto       batch   = runtime.ReadPreparedBatch(std::move(prepared));
		auto result = DirectDctPlsPoolBatch(std::move(source.plan),
		                                    std::move(batch),
		                                    std::move(labels),
		                                    options.schedule.microbatch_images,
		                                    std::move(context_owner));
		result.impl_->postprocess =
		    std::make_unique<detail::DirectDctPlsCudaPostprocess>(result.impl_->batch,
		                                                          result.impl_->labels,
		                                                          randaugment,
		                                                          mixup,
		                                                          options.schedule.microbatch_images,
		                                                          options.model_classes,
		                                                          options.enable_published_randaugment,
		                                                          options.enable_published_mixup,
		                                                          postprocess_stream);
		const auto materialize_done = PlsPoolClock::now();
		context_slots->prepare_completed(elapsed_ms(plan_started, plan_done),
		                                  elapsed_ms(plan_done, io_done),
		                                  elapsed_ms(io_done, materialize_done));
		return PreparedDirectDctPlsPool {std::move(result)};
	}

	void worker_loop() noexcept {
		for (;;) {
			std::optional<PendingDirectDctPlsPool> source;
			{
				std::unique_lock lock(work_mutex);
				work_cv.wait(lock, [this] { return stopping || pending.has_value(); });
				if (stopping) {
					return;
				}
				source.emplace(std::move(*pending));
				pending.reset();
				worker_busy = true;
			}
			try {
				auto prepared = prepare_pool(std::move(*source));
				std::lock_guard lock(work_mutex);
				worker_busy = false;
				if (stopping) {
					return;
				}
				ready.emplace(std::move(prepared));
			} catch (...) {
				std::lock_guard lock(work_mutex);
				worker_busy    = false;
				worker_failure = std::current_exception();
			}
			work_cv.notify_all();
		}
	}
};

DirectDctPlsPipeline::DirectDctPlsPipeline(const std::filesystem::path& manifest_path,
                                           const std::filesystem::path& premixed_mapping_csv,
                                           DirectDctPlsPipelineOptions  options)
    : impl_(std::make_unique<Impl>(manifest_path, premixed_mapping_csv, std::move(options))) {
}

DirectDctPlsPipeline::~DirectDctPlsPipeline()                                          = default;
DirectDctPlsPipeline::DirectDctPlsPipeline(DirectDctPlsPipeline&&) noexcept            = default;
DirectDctPlsPipeline& DirectDctPlsPipeline::operator=(DirectDctPlsPipeline&&) noexcept = default;

void DirectDctPlsPipeline::start_epoch(const uint32_t epoch) {
	std::lock_guard lock(impl_->work_mutex);
	if (impl_->pending.has_value() || impl_->ready.has_value() || impl_->worker_busy ||
	    (impl_->epoch.has_value() && impl_->epoch->has_next())) {
		throw std::logic_error("Direct-DCT PLS previous epoch still has pending pool work");
	}
	auto schedule                  = impl_->options.schedule;
	schedule.epoch                 = epoch;
	impl_->options.schedule.epoch  = epoch;
	impl_->worker_failure          = nullptr;
	impl_->epoch.emplace(impl_->layout, schedule);
	impl_->context_slots->reset_epoch_stats();
	impl_->enqueue_next_locked();
	impl_->work_cv.notify_one();
}

bool DirectDctPlsPipeline::has_next_pool() const noexcept {
	if (!impl_) {
		return false;
	}
	std::lock_guard lock(impl_->work_mutex);
	return impl_->worker_failure != nullptr || impl_->pending.has_value() || impl_->worker_busy ||
	       impl_->ready.has_value() || (impl_->epoch.has_value() && impl_->epoch->has_next());
}

DirectDctPlsPoolBatch DirectDctPlsPipeline::next_pool() {
	const auto wait_started = PlsPoolClock::now();
	std::optional<PreparedDirectDctPlsPool> prepared;
	{
		std::unique_lock lock(impl_->work_mutex);
		if (!impl_->worker_failure && !impl_->pending && !impl_->worker_busy && !impl_->ready &&
		    (!impl_->epoch.has_value() || !impl_->epoch->has_next())) {
			throw std::out_of_range("Direct-DCT PLS pipeline has no pending pool; start an epoch first");
		}
		impl_->work_cv.wait(lock, [this] { return impl_->worker_failure != nullptr || impl_->ready.has_value(); });
		if (impl_->worker_failure) {
			std::rethrow_exception(impl_->worker_failure);
		}
		prepared.emplace(std::move(*impl_->ready));
		impl_->ready.reset();
		impl_->enqueue_next_locked();
	}
	impl_->work_cv.notify_one();
	const auto activation_started = PlsPoolClock::now();
	auto       result             = std::move(prepared->batch);
	const auto activation_done = PlsPoolClock::now();
	impl_->context_slots->activated(elapsed_ms(wait_started, activation_started),
	                                elapsed_ms(activation_started, activation_done));
	return result;
}

const DirectDctPlsLayout& DirectDctPlsPipeline::layout() const noexcept {
	return impl_->layout;
}
const DirectDctPlsPipelineOptions& DirectDctPlsPipeline::options() const noexcept {
	return impl_->options;
}
DirectDctPlsPoolPrefetchStats DirectDctPlsPipeline::prefetch_stats() const noexcept {
	return impl_ ? impl_->context_slots->snapshot() : DirectDctPlsPoolPrefetchStats {};
}

} // namespace galp::jpeg

#endif // GALP_WITH_JPEG_DCT
