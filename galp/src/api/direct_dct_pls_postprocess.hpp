#ifndef GALP_DIRECT_DCT_PLS_POSTPROCESS_HPP
#define GALP_DIRECT_DCT_PLS_POSTPROCESS_HPP

#include "galp/advanced/direct_dct_pls.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace galp::jpeg::detail {

enum class DirectDctPlsRandAugmentOp : uint8_t {
	kAutoContrast,
	kPosterize,
	kSolarizeAdd,
	kColor,
	kContrast,
	kBrightness,
	kMidfreqAug,
	kCutout,
	kTranslateX,
	kTranslateY,
	kRotate90,
	kAutoSaturation,
	kGrayscale,
	kChromaDrop,
};

struct DirectDctPlsRandAugmentDecision {
	std::array<DirectDctPlsRandAugmentOp, 2> operations {};
	std::array<float, 2>                     magnitudes {};
	std::array<int16_t, 2>                   cutout_center_h {};
	std::array<int16_t, 2>                   cutout_center_w {};
	std::array<uint8_t, 2>                   chroma_drop_channel {};
};

struct DirectDctPlsMixupDecision {
	float original = 1.0F;
	float rolled   = 0.0F;
};

// Host decision derivation is exposed only to source-level parity tests and
// native pipeline composition; it is not part of the installed public API.
DirectDctPlsRandAugmentDecision derive_published_randaugment_decision(const DirectDctPlsSample&          sample,
                                                                      const DirectDctPlsScheduleOptions& options);
DirectDctPlsMixupDecision
derive_published_mixup_decision(uint64_t training_seed, uint32_t epoch, uint64_t microbatch_index);

class DirectDctPlsCudaPostprocess {
public:
	class Stream {
	public:
		explicit Stream(int cuda_device);
		~Stream();
		Stream(const Stream&)            = delete;
		Stream& operator=(const Stream&) = delete;
		Stream(Stream&&)                 = delete;
		Stream& operator=(Stream&&)      = delete;

	private:
		friend class DirectDctPlsCudaPostprocess;
		struct Impl;
		std::unique_ptr<Impl> impl_;
	};

	DirectDctPlsCudaPostprocess(const DirectDctBatch&                            source,
	                            std::span<const int64_t>                         labels,
	                            std::span<const DirectDctPlsRandAugmentDecision> randaugment,
	                            std::span<const DirectDctPlsMixupDecision>       mixup,
	                            uint32_t                                         microbatch_images,
	                            uint32_t                                         model_classes,
	                            bool                                             enable_randaugment,
	                            bool                                             enable_mixup,
	                            std::shared_ptr<Stream>                          stream);
	~DirectDctPlsCudaPostprocess();
	DirectDctPlsCudaPostprocess(const DirectDctPlsCudaPostprocess&)            = delete;
	DirectDctPlsCudaPostprocess& operator=(const DirectDctPlsCudaPostprocess&) = delete;
	DirectDctPlsCudaPostprocess(DirectDctPlsCudaPostprocess&&) noexcept;
	DirectDctPlsCudaPostprocess& operator=(DirectDctPlsCudaPostprocess&&) noexcept;

	[[nodiscard]] DirectDctGridTensorDescriptor      y_tensor() const noexcept;
	[[nodiscard]] DirectDctGridTensorDescriptor      cbcr_tensor() const noexcept;
	[[nodiscard]] DirectDctPlsTargetTensorDescriptor targets() const noexcept;
	[[nodiscard]] void*                              completion_event() const noexcept;

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

} // namespace galp::jpeg::detail

#endif // GALP_DIRECT_DCT_PLS_POSTPROCESS_HPP
