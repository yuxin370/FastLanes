# DCT-domain augmentation semantics

This guide describes Direct-DCT and PLS operations. Their coefficient-domain
definitions determine the model inputs; identical names in RGB libraries do
not imply identical numerical transformations.

## Representation and processing order

Let `F[c,h,w,u,v]` be a dequantized coefficient: `c` identifies Y, Cb or Cr;
`h,w` identify an 8x8 block; `u,v` are vertical/horizontal frequencies.
`F[...,0,0]` is DC; the remaining coefficients are AC. For JPEG's orthonormal
DCT of centered samples, `DC = 8 * (mean(pixel) - 128)`.

Standard RGB-no-more training outputs are `Y=[B,1,28,28,8,8]` and
`CbCr=[B,2,14,14,8,8]`. With 4:2:0 sampling, one chroma block covers two luma
blocks along each axis.

```text
Quantized q -> source-frequency selection
    -> clamp(q * quantization_table, -1024, 1016)
    -> DCT crop, resize and horizontal flip -> int16 coefficients
    -> RandAugment input clamp and two sequential operations
    -> z = (F + 4) / 1020 -> Mixup -> model
```

Floating augmentation results use nearest rounding with ties to even and are
clamped after each operation. Raw zero maps to `4/1020`, so zeroing a normalized
tensor is not a raw-frequency mask. Zeroing all Y/Cb/Cr coefficients produces
neutral gray under the corresponding IDCT.

## Geometry

- **Block crop:** selects component block ranges. Training crops align to 16
  luma pixels. For `(x,y,width,height)=(32,48,224,224)`, luma blocks are
  `[6:34,4:32]` and chroma blocks are `[3:17,2:16]`.
- **Resize:** combines/splits DCT blocks using basis transformations, spectral
  truncation/zero padding and scale compensation. Combining an `L x M` block
  neighborhood divides the retained spectrum by `sqrt(L*M)`; expansion applies
  the inverse scaling. General ratios compose these linear transforms. This is
  spectral resampling rather than RGB bilinear interpolation.
- **Published random crop:** samples area in `[0.05,1.0]` with square candidates,
  supported block sizes and origins aligned to two luma blocks. It tries ten
  candidates before a centered fallback. Small candidate sizes include 2, 4,
  14 and 28 luma blocks; larger candidates use multiples of 28 within the source.
  Outputs are 28x28 luma and 14x14 chroma blocks.
- **Center crop:** `ResizedCenterCrop_DCT(32,28)` takes the centered 56x56 luma
  blocks of a 512x512 source, then resizes to 28x28. Its pixel crop is
  `(32,32,448,448)`.
- **Horizontal flip:** reverses block columns and negates odd horizontal
  frequencies: `F'[c,h,w,u,v] = (-1)^v F[c,h,W-1-w,u,v]`. Training probability
  is 0.5, using the same decision for every component.

Crop decisions derive from seed, epoch and sample/PLS identity. Shared PLS crops
reuse a crop decision within the partition; flips remain per sample. Resize
mixes source frequencies, so model output channels do not generally identify
all source columns required for reading.

## Published RandAugment

The recipe chooses two operations sequentially, at magnitude index 3 of 11.
`D` below denotes DC. Unmentioned coefficients remain unchanged, followed by
the common rounding and clamp.

| Operation | Transformation at the published magnitude |
| --- | --- |
| `AutoContrast` | Rescale Y DC from its per-image extrema to `[-1024,1016]` |
| `Posterize` | Each component's DC becomes `-1024+4*round((D+1024)/4)` |
| `SolarizeAdd` | Add 264 to Y DC where `D<0` |
| `Color` | Multiply Cb/Cr DC by `0.73` or `1.27` |
| `Contrast` | Multiply Y DC by `0.73` or `1.27` |
| `Brightness` | Add `mean(abs(Y_DC)) * (+/-0.27)` to every Y DC |
| `MidfreqAug` | Apply the shifted Gaussian gain below to all Y frequencies |
| `Cutout` | Zero all coefficients in a component-aligned rectangle |
| `TranslateX`, `TranslateY` | Shift whole blocks, filling exposed blocks with zero |
| `Rotate90` | Rotate blocks, transpose frequency axes and change odd-frequency signs |
| `AutoSaturation` | Rescale chroma DC using its coefficient extrema |
| `Grayscale` | Zero all Cb and Cr coefficients |
| `ChromaDrop` | Zero all coefficients of either Cb or Cr |

DC-only operations adjust block means while preserving AC texture. Adding 80
to Y DC adds 10 to that block's luma values. `Contrast` changes the mean relative
to 128. `Posterize` quantizes block means in steps of 0.5.

For `MidfreqAug`, set `m=+/-0.27`, `sigma=4-2.2*abs(m)=3.406`,
`i=(u+4) mod 8`, `j=(v+4) mod 8`, and
`G=exp(-((i-3.5)^2+(j-3.5)^2)/(2*sigma^2))`. Negative magnitude multiplies by
`G`; positive magnitude divides by it. This operation also changes DC.

Translation uses `b=2*floor(m/2)` luma blocks for `m=+/-3.75`, giving +2 or -4
luma blocks and half that shift for chroma. For square block grids:

```text
Counterclockwise: F'[h,w,u,v] = (-1)^u F[w,W-1-h,v,u]
Clockwise:        F'[h,w,u,v] = (-1)^v F[H-1-w,h,v,u]
```

Cutout aligns its sampled location to even luma-block coordinates. For grid
size `H,W`, sampled `ch,cw` and radius `p=2`, its rows are
`[max(0,H-ch-p), H-max(0,ch-p))` and columns are
`[max(0,cw-p), W-max(0,W-cw-p))`. Chroma uses half the coordinates and `p=1`.

Operation choices/signs derive from sample, seed, epoch and operation index.
Statistics are recomputed for the second operation. Choosing `Grayscale`
excludes subsequent grayscale/color/saturation/chroma-drop choices; choosing
a chroma adjustment first excludes subsequent grayscale.

## Mixup and frequency selection

Mixup samples a two-component Dirichlet weight with `alpha=0.2`, assigns the
larger weight to the original sample, and pairs samples with `roll(1)` within
each microbatch. The microbatch shares one weight:

```text
Z'_i = lambda * Z_i + (1-lambda) * Z_partner
t'_i = lambda * one_hot(label_i) + (1-lambda) * one_hot(label_partner)
```

Mixup uses normalized FP32 coefficients and soft targets. Native PLS already
applies it, so callers should not apply it again.

`coefficients` selects 1 to 64 distinct source columns in `[0,63]`; `None`
selects all. In zigzag storage, the first six columns have natural indices
`[0,1,8,16,9,2]`. Source columns, natural frequencies and model channels are
separate concepts. Omitted coefficients are zero before geometry; subsequent
resize can produce nonzero output frequencies. Frequency masks are separate
experiments, not part of default RandAugment.

## Interface and source

```python
from fastlens.profiles.rgbnomore import VALIDATION
from fastlens.torch import DirectDctReader

reader = DirectDctReader("data/example-fastlens/manifest.bin",
                         module_path="build/fastlens/torch")
batch = reader.read([0], VALIDATION, transforms=[{
    "crop": {"x": 32, "y": 32, "width": 448, "height": 448},
    "horizontal_flip": True,
    "logical_sample_id": "sample-0",
}])
y, cbcr = batch.tensors
```

Descriptors accept crop and horizontal flip; output geometry comes from the
profile. This example returns normalized validation tensors without RandAugment
or Mixup. Basic training uses crop, flip and normalization; published PLS adds
RandAugment and Mixup.

| Source | Responsibility |
| --- | --- |
| [recipe.py](benchmarks/training_pls/recipe.py) | Operation list and constants |
| [published_augmentation.py](benchmarks/training_pls/published_augmentation.py) | Python geometry and augmentation reference |
| [direct_dct_pls.cpp](src/api/direct_dct_pls.cpp) | Native crop/random decisions |
| [direct_dct_pls_postprocess.cu](src/api/direct_dct_pls_postprocess.cu) | CUDA augmentation, normalization and targets |
| [jpeg_dct_planner.cpp](src/jpeg/jpeg_dct_planner.cpp) | Component geometry and resize weights |
| [jpeg_dct_transform_kernels.cu](src/jpeg/jpeg_dct_transform_kernels.cu) | GPU transforms |
| [direct_dct.py](torch/direct_dct.py) | Reader and pipeline interfaces |

See the [PLS guide](benchmarks/training_pls/README.md) for complete training commands.
