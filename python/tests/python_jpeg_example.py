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
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


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
    print("Decoding FLS to CSV...")
    reader = conn.read_fls(fls_file)
    reader.to_csv(csv_file)
    print("✅ CSV output saved to:", csv_file)

    # 6) NEW: Use the new to_numpy_dct() method
    print("\n🚀 Calling to_numpy_dct()...")
    data_list = reader.to_numpy_dct()  # Returns List[List[str]]

    print(f"Data shape: {len(data_list)} rows × {len(data_list[0]) if data_list else 0} columns")
    print("First 3 rows (if available):")
    for i, row in enumerate(data_list[:3]):
        print(f"  Row {i}: {row}")

    # 7) Optional: Convert to NumPy array (dtype=object) or pandas DataFrame
    if HAS_NUMPY:
        print("\n📦 Converting to NumPy array (dtype=object)...")
        np_array = np.array(data_list, dtype=object)
        print("NumPy array shape:", np_array.shape)
        print("Sample element:", np_array[0, 0] if np_array.size > 0 else "N/A")

    if HAS_PANDAS:
        print("\n📊 Converting to pandas DataFrame...")
        # You may want to extract column names from footer if available
        # For now, use generic names
        num_cols = len(data_list[0]) if data_list else 0
        df = pd.DataFrame(data_list, columns=[f"col_{i}" for i in range(num_cols)])
        print(df.head())

    print("\n✅ All done!")


if __name__ == "__main__":
    main()