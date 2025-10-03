# ────────────────────────────────────────────────────────
# |                      FastLanes                       |
# ────────────────────────────────────────────────────────
# examples/python_jpeg_example.py
# ────────────────────────────────────────────────────────
#!/usr/bin/env python3
import sys
import os
import pyfastlanes


try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def main():
    # 1) Print module docstring & version
    print(pyfastlanes.__doc__)
    print(f"FastLanes version: {pyfastlanes.get_version()}\n")

    # 2) Paths for the demo
    jpeg_path = "/home/tangyuxin/cleanFastlanes/FastLanes/data/flower_photos/example/1.jpg"

    # 3) Clean up old output files
    fls_file = "data.fls"
    csv_file = "decoded.csv"
    if os.path.exists(fls_file):
        os.remove(fls_file)
    if os.path.exists(csv_file):
        os.remove(csv_file)

    # 4) Encode JPEG into FLS
    print("Encoding JPEG to FLS...")
    conn = pyfastlanes.connect()
    conn.inline_footer().read_jpeg(jpeg_path).to_fls(fls_file)

    # 5) Read FLS and decode to CSV (as before)
    print("Decoding FLS to CSV:", csv_file)
    reader = conn.read_fls(fls_file)
    reader.to_csv(csv_file)

    # 6) NEW: Use the new dct_to_str_list() method
    data_list_double = reader.dct_to_double_list()  # Returns List[List[double]]

    print(f"Data shape: {len(data_list_double)} rows × {len(data_list_double[0]) if data_list_double else 0} columns")

    # 7) Optional: Convert to NumPy array (dtype=object) or pandas DataFrame
    if HAS_NUMPY:
        print("\n Converting to NumPy array ...")
        np_array = np.array(data_list_double, dtype=object)

        # 6) NEW: Use the new to_numpy_numeric() method
        print("\n Calling to_numpy_numeric()...")
        data_numpy_double = reader.to_numpy_numeric()  # Returns List[List[str]]
        print(data_numpy_double)

        if HAS_TORCH:
            print("\n Converting to PyTorch tensor")
            tensor = torch.from_numpy(data_numpy_double)
            print("Tensor shape:", tensor.shape, "Tensor dtype:", tensor.dtype, "Tensor:\n", tensor)

            if torch.cuda.is_available():
                print("\n Moving tensor to GPU...")
                gpu_tensor = tensor.cuda()  # tensor.to('cuda')
                print("GPU Tensor shape:", gpu_tensor.shape, " device:", gpu_tensor.device, "  dtype:", gpu_tensor.dtype)
            else:
                print("\n CUDA not available. Skipping GPU test.")


if __name__ == "__main__":
    main()