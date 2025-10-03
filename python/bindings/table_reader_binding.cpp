// ────────────────────────────────────────────────────────
// |                      FastLanes                       |
// ────────────────────────────────────────────────────────
// python/bindings/table_reader_binding.cpp
// ────────────────────────────────────────────────────────
#include "fls/common/alias.hpp"     
#include "fls/footer/table_descriptor.hpp"
#include "fls/reader/rowgroup_reader.hpp" 
#include "fls/reader/table_reader.hpp"
#include "fls/table/attribute.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> 
#include <pybind11/stl.h>


namespace py = pybind11;

namespace fastlanes {

std::vector<std::vector<double>> dct_to_double_list(fastlanes::TableReader& self) {
    size_t num_rowgroups = self.get_num_rowgroups();
    // n_t total_rows = 0;
    n_t total_cols = 0;
    std::vector<std::vector<double>> result;

    for (n_t rowgroup_idx = 0; rowgroup_idx < num_rowgroups; ++rowgroup_idx) {
        auto rowgroup_reader = self.get_rowgroup_reader(rowgroup_idx);
        auto rowgroup_up = rowgroup_reader->materialize();
        const auto& rowgroup = *rowgroup_up;
        const auto& desc = rowgroup_up->m_descriptor;

        const n_t n_rows = rowgroup.n_tup;
        const n_t n_cols = rowgroup.internal_rowgroup.size();

        if (total_cols == 0) {
            total_cols = n_cols;
        } else if (total_cols != n_cols) {
            throw std::runtime_error("Rowgroups have inconsistent column counts");
        }

        size_t old_size = result.size();
        result.resize(old_size + n_rows);
        for (n_t i = 0; i < n_rows; ++i) {
            result[old_size + i].resize(n_cols);
        }

        for (n_t row_idx = 0; row_idx < n_rows; row_idx++) {
            for (n_t col_idx = 0; col_idx < n_cols; col_idx++) {
                const auto& col = rowgroup.internal_rowgroup[col_idx];
                result[old_size + row_idx][col_idx] =
                    Attribute::ToDouble(col, row_idx, desc.m_column_descriptors[col_idx]->data_type);
            }
        }

        // total_rows += n_rows;
    }

    return result;
}

py::array_t<double> to_numpy_numeric(fastlanes::TableReader& self) {
    size_t num_rowgroups = self.get_num_rowgroups();
    if (num_rowgroups == 0) {
        return py::array_t<double>({0, 0});
    }

    n_t total_cols = 0;
    std::vector<double> flat_data; 

    for (n_t rg_idx = 0; rg_idx < num_rowgroups; ++rg_idx) {
        auto rowgroup_reader = self.get_rowgroup_reader(rg_idx);
        auto rowgroup_up = rowgroup_reader->materialize();
        const auto& rowgroup = *rowgroup_up;
        const auto& desc = rowgroup_up->m_descriptor;

        const n_t n_rows = rowgroup.n_tup;
        const n_t n_cols = rowgroup.internal_rowgroup.size();

        if (total_cols == 0) {
            total_cols = n_cols;
            flat_data.reserve(n_rows * n_cols * num_rowgroups); 
        } else if (total_cols != n_cols) {
            throw std::runtime_error("Rowgroups have inconsistent column counts");
        }

        size_t old_size = flat_data.size();
        flat_data.resize(old_size + n_rows * n_cols);

        for (n_t row_idx = 0; row_idx < n_rows; ++row_idx) {
            for (n_t col_idx = 0; col_idx < n_cols; ++col_idx) {
                const auto& col = rowgroup.internal_rowgroup[col_idx];
                double val = Attribute::ToDouble(col, row_idx, desc.m_column_descriptors[col_idx]->data_type);
                flat_data[old_size + row_idx * n_cols + col_idx] = val;
            }
        }
    }

    n_t total_rows = flat_data.size() / total_cols;
    auto result = py::array_t<double>({total_rows, total_cols});
    std::copy(flat_data.begin(), flat_data.end(), result.mutable_data());
    return result;
}


}

void bind_table_reader(py::module_& m) {
	py::class_<fastlanes::TableReader, fastlanes::up<fastlanes::TableReader>> cls(m, "TableReader");

	cls.def(
	       "to_csv", [](fastlanes::TableReader& self, const char* path) { self.to_csv(path); }, py::arg("file_path"))
		.def("dct_to_double_list", &fastlanes::dct_to_double_list, "Convert FLS file to double list array")
		.def("to_numpy_numeric", &fastlanes::to_numpy_numeric, "Convert numeric FLS table to NumPy float64 array")
	    .def("__repr__", [](const fastlanes::TableReader&) { return "<fastlanes.TableReader>"; })
	    .def("__dir__", []() {
		    return std::vector<std::string> {
		        "to_csv","dct_to_double_list","to_numpy_numeric", "__repr__", "__dir__"
		        // "to_csv", "__repr__", "__dir__"
		        // Add more method/field names as you expose them
		    };
	    });
}
