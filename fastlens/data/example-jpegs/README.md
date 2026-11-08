# Quickstart photographs

These four photographs are bundled for functional examples. They were taken from
the server's installed scikit-image sample data and converted to 512×512 RGB JPEGs
with 4:2:0 subsampling. No dataset download is needed.

| File | Original | Credit and reuse terms |
| --- | --- | --- |
| `image_0.jpg` | `astronaut.png`, Eileen Collins | NASA; public domain |
| `image_1.jpg` | `camera.png`, photographer | Lav Varshney; CC0 |
| `image_2.jpg` | `chelsea.png`, cat | Stefan van der Walt; CC0 |
| `image_3.jpg` | `coffee.png`, coffee cup | Rachel Michetti, courtesy of Pikolo Espresso Bar; CC0 |

Attribution and license statements are in the
[scikit-image data documentation](https://scikit-image.org/docs/stable/api/skimage.data.html)
and its [sample loaders](https://github.com/scikit-image/scikit-image/blob/main/skimage/data/_fetchers.py).
NASA's source photograph is available at <https://flic.kr/p/r9qvLn>.
CC0 terms: <https://creativecommons.org/publicdomain/zero/1.0/>.

Preparation used Pillow: convert to RGB, center-crop and resize with
`ImageOps.fit(..., (512, 512), method=Image.Resampling.LANCZOS)`, then save with
`quality=90, subsampling=2`. The camera photograph remains visually grayscale but
is encoded as three-component RGB/Y'CbCr. These images test functionality, not
representative compression ratios or throughput.

Follow the [Quickstart](../../../README.md#quick-start) from the repository root.
