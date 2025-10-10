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

std::vector<std::vector<double>> to_double_list(fastlanes::TableReader& self) {
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

py::array_t<uint8_t> to_numpy_rgb(fastlanes::TableReader& self, const char* path, bool channel_first = false) {
    // self.to_rgb 现在返回 [N][3][H][W]，元素类型为 uint8_t
    auto rgb_out = self.to_rgb(path);

    if (rgb_out.empty()) {
        throw std::runtime_error("to_rgb returned empty result");
    }

    const size_t N = rgb_out.size();

    const size_t height = rgb_out[0][0].size();
    const size_t width  = (height > 0) ? rgb_out[0][0][0].size() : 0;
    if (height == 0 || width == 0) {
        throw std::runtime_error("Empty image dimensions");
    }

    // // 可选：启用严格一致性验证，防止不同图像/通道/行列尺寸不一致导致越界。
    // for (size_t img = 0; img < N; ++img) {
    //     if (rgb_out[img].size() != 3) {
    //         throw std::runtime_error("Image " + std::to_string(img) + " does not have 3 channels");
    //     }
    //     for (size_t c = 0; c < 3; ++c) {
    //         if (rgb_out[img][c].size() != height) {
    //             throw std::runtime_error("Image " + std::to_string(img) + " channel " + std::to_string(c) + " height mismatch");
    //         }
    //         for (size_t h = 0; h < height; ++h) {
    //             if (rgb_out[img][c][h].size() != width) {
    //                 throw std::runtime_error("Image " + std::to_string(img) + " channel " + std::to_string(c) + " row " + std::to_string(h) + " width mismatch");
    //             }
    //         }
    //     }
    // }

    py::array_t<uint8_t> result;
    if (!channel_first) {
        // NHWC: (N, H, W, 3)
        result = py::array_t<uint8_t>({
            static_cast<ssize_t>(N),
            static_cast<ssize_t>(height),
            static_cast<ssize_t>(width),
            static_cast<ssize_t>(3)
        });
        uint8_t* out = result.mutable_data();
        // index: ((n * height + h) * width + w) * 3 + c
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    for (size_t c = 0; c < 3; ++c) {
                        uint8_t v = rgb_out[n][c][h][w];
                        size_t idx = ((n * height + h) * width + w) * 3 + c;
                        out[idx] = v;
                    }
                }
            }
        }
    } else {
        // NCHW: (N, 3, H, W)
        result = py::array_t<uint8_t>({
            static_cast<ssize_t>(N),
            static_cast<ssize_t>(3),
            static_cast<ssize_t>(height),
            static_cast<ssize_t>(width)
        });
        uint8_t* out = result.mutable_data();
        // index: (((n * 3 + c) * height + h) * width + w)
        for (size_t n = 0; n < N; ++n) {
            for (size_t c = 0; c < 3; ++c) {
                for (size_t h = 0; h < height; ++h) {
                    for (size_t w = 0; w < width; ++w) {
                        uint8_t v = rgb_out[n][c][h][w];
                        size_t idx = ((n * 3 + c) * height + h) * width + w;
                        out[idx] = v;
                    }
                }
            }
        }
    }

    return result;
}

}

void bind_table_reader(py::module_& m) {
	py::class_<fastlanes::TableReader, fastlanes::up<fastlanes::TableReader>> cls(m, "TableReader");

	cls.def(
	       "to_csv", [](fastlanes::TableReader& self, const char* path) { self.to_csv(path); }, py::arg("file_path"))
		.def("to_double_list", &fastlanes::to_double_list, "Convert FLS file to double list array")
		.def("to_numpy_numeric", &fastlanes::to_numpy_numeric, "Convert numeric FLS table to NumPy float64 array")
		.def("to_numpy_rgb", &fastlanes::to_numpy_rgb, "Convert DCT FLS table to NumPy RGB float64 array")
	    .def("__repr__", [](const fastlanes::TableReader&) { return "<fastlanes.TableReader>"; })
	    .def("__dir__", []() {
		    return std::vector<std::string> {
		        "to_csv","to_double_list","to_numpy_numeric", "to_numpy_rgb", "__repr__", "__dir__"
		        // "to_csv", "__repr__", "__dir__"
		        // Add more method/field names as you expose them
		    };
	    });
}
