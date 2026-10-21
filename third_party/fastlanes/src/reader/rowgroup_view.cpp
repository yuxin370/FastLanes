// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// src/reader/rowgroup_view.cpp
// ────────────────────────────────────────────────────────
#include "fls/reader/rowgroup_view.hpp"
#include "fls/common/alias.hpp"
#include "fls/common/assert.hpp"
#include "fls/footer/rowgroup_descriptor_generated.h"
#include "fls/reader/column_view.hpp"
#include "fls/std/span.hpp"
#include <cstddef> // for std::byte

namespace fastlanes {

RowgroupView::~RowgroupView()                                  = default;
RowgroupView::RowgroupView(RowgroupView&&) noexcept            = default;
RowgroupView& RowgroupView::operator=(RowgroupView&&) noexcept = default;

RowgroupView::RowgroupView(span<std::byte> ptr, const RowgroupDescriptor& footer) {
	const auto* column_descriptors = footer.m_column_descriptors();
	if (column_descriptors == nullptr) {
		return;
	}
	columns.reserve(column_descriptors->size());
	for (const auto& column_descriptor : *column_descriptors) {
		const span<std::byte> column_span = ptr;
		columns.emplace_back(column_span, *column_descriptor, footer);
	}
}

ColumnView& RowgroupView::operator[](const n_t col_idx) {
	FLS_ASSERT_NOT_EMPTY_VEC(columns)

	return columns[col_idx];
}

const ColumnView& RowgroupView::operator[](const n_t col_idx) const {
	FLS_ASSERT_NOT_EMPTY_VEC(columns)

	return columns[col_idx];
}
} // namespace fastlanes
