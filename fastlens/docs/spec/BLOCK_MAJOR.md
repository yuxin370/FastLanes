# FastLens Block-major

Normative layout description. [Format index](README.md).

Blocks are ordered by semantic component slot and spatial coordinate, then by
physical image ID among images containing that coordinate. DCT frequencies form
64 signed-int16 columns in the configured natural or zigzag order. Coordinates
with no present images produce no block group. Sampling and missing components
are represented in [image metadata](IMAGE_METADATA.md).

Each positive spatial block group remains inside one FLS rowgroup. A rowgroup is
closed before adding a group that exceeds its configured capacity. Vectors contain
1024 rows; padding and partial vectors must be interpreted using real-row counts.
The [geometry descriptor](GEOMETRY.md) reconstructs group location and the rank of
an image within that group, giving its row, vector and lane. Crop and frequency
requests select the required compressed data before GPU decoding.

The public writer is:

```bash
python -m fastlens convert fastlens/data/example-jpegs \
  --output fastlens/data/example-fastlens --layout block-major --verify full
python -m fastlens inspect fastlens/data/example-fastlens
```

Its [Published Store](STORE.md) uses manifest version 1, sectioned image metadata
version 3, and geometry version 1. The native physical-layout option is
`spatial-major`; the persisted layout name is `spatial-major-image-minor`.
The converter does not resize or re-encode JPEGs.
