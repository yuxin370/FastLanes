#ifndef GALP_ERRORS_HPP
#define GALP_ERRORS_HPP

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace galp {

class UnsupportedFormatError : public std::runtime_error {
public:
	UnsupportedFormatError(std::string  token,
	                       const size_t rowgroup_index,
	                       const size_t column_index,
	                       std::string  column_name)
	    : std::runtime_error(build_message(token, rowgroup_index, column_index, column_name))
	    , token_(std::move(token))
	    , column_name_(std::move(column_name))
	    , rowgroup_index_(rowgroup_index)
	    , column_index_(column_index) {
	}

	[[nodiscard]] const std::string& token() const noexcept {
		return token_;
	}

	[[nodiscard]] size_t rowgroup_index() const noexcept {
		return rowgroup_index_;
	}

	[[nodiscard]] size_t column_index() const noexcept {
		return column_index_;
	}

	[[nodiscard]] const std::string& column_name() const noexcept {
		return column_name_;
	}

private:
	static std::string build_message(const std::string& token,
	                                 const size_t       rowgroup_index,
	                                 const size_t       column_index,
	                                 const std::string& column_name) {
		std::string message = "unsupported FLS operator token: " + token +
		                      " (rowgroup=" + std::to_string(rowgroup_index) +
		                      ", column=" + std::to_string(column_index);
		if (!column_name.empty()) {
			message += ", name=" + column_name;
		}
		message += ")";
		return message;
	}

	std::string token_;
	std::string column_name_;
	size_t      rowgroup_index_ = 0;
	size_t      column_index_   = 0;
};

} // namespace galp

#endif // GALP_ERRORS_HPP
