// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// data/include/data/fastlens.hpp
// ────────────────────────────────────────────────────────
#ifndef DATA_FASTLENS_HPP
#define DATA_FASTLENS_HPP

#include <array>
#include <string_view>

namespace fastlanes {

using fastlens_dataset_t = std::array<std::pair<std::string_view, std::string_view>, 12>;

class fastlens {
public:
	// FLOAT precision
	static constexpr std::string_view float_fastlens_one_vector {FLS_CMAKE_SOURCE_DIR
	                                                             "/data/generated/fastlens/float/fastlens/one_vector"};
	static constexpr std::string_view float_fastlens_single_column {
	    FLS_CMAKE_SOURCE_DIR "/data/generated/fastlens/float/fastlens/single_column"};

	static constexpr std::string_view float_constant_one_vector {FLS_CMAKE_SOURCE_DIR
	                                                             "/data/generated/fastlens/float/constant/one_vector"};
	static constexpr std::string_view float_constant_single_column {
	    FLS_CMAKE_SOURCE_DIR "/data/generated/fastlens/float/constant/single_column"};

	static constexpr std::string_view float_positive_inf_one_vector {
	    FLS_CMAKE_SOURCE_DIR "/data/generated/fastlens/float/positive_inf/one_vector"};
	static constexpr std::string_view float_positive_inf_single_column {
	    FLS_CMAKE_SOURCE_DIR "/data/generated/fastlens/float/positive_inf/single_column"};

	// DOUBLE precision
	static constexpr std::string_view double_fastlens_one_vector {
	    FLS_CMAKE_SOURCE_DIR "/data/generated/fastlens/double/fastlens/one_vector"};
	static constexpr std::string_view double_fastlens_single_column {
	    FLS_CMAKE_SOURCE_DIR "/data/generated/fastlens/double/fastlens/single_column"};

	static constexpr std::string_view double_constant_one_vector {
	    FLS_CMAKE_SOURCE_DIR "/data/generated/fastlens/double/constant/one_vector"};
	static constexpr std::string_view double_constant_single_column {
	    FLS_CMAKE_SOURCE_DIR "/data/generated/fastlens/double/constant/single_column"};

	static constexpr std::string_view double_positive_inf_one_vector {
	    FLS_CMAKE_SOURCE_DIR "/data/generated/fastlens/double/positive_inf/one_vector"};
	static constexpr std::string_view double_positive_inf_single_column {
	    FLS_CMAKE_SOURCE_DIR "/data/generated/fastlens/double/positive_inf/single_column"};

	static constexpr fastlens_dataset_t dataset {
	    {// Float entries
	     {"float_fastlens_one_vector", float_fastlens_one_vector},
	     {"float_fastlens_single_column", float_fastlens_single_column},
	     {"float_constant_one_vector", float_constant_one_vector},
	     {"float_constant_single_column", float_constant_single_column},
	     {"float_positive_inf_one_vector", float_positive_inf_one_vector},
	     {"float_positive_inf_single_column", float_positive_inf_single_column},
	     // Double entries
	     {"double_fastlens_one_vector", double_fastlens_one_vector},
	     {"double_fastlens_single_column", double_fastlens_single_column},
	     {"double_constant_one_vector", double_constant_one_vector},
	     {"double_constant_single_column", double_constant_single_column},
	     {"double_positive_inf_one_vector", double_positive_inf_one_vector},
	     {"double_positive_inf_single_column", double_positive_inf_single_column}}};
};

} // namespace fastlanes

#endif // DATA_FASTLENS_HPP
