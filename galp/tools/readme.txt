  # decompress table
  /path/to/FastLanes/build/galp/tools/galp_cli read_table \
    /path/to/FastLanes/data/fls/galp-test/data.fls \
    /tmp/out.csv

  # only decompress one specific rowgroup
  /path/to/FastLanes/build/galp/tools/galp_cli read_table \
    /path/to/FastLanes/data/fls/galp-test/data.fls \
    /tmp/out.csv --rowgroup 0

  # benchmark：only GPU decompression, no materialize
  /path/to/FastLanes/build/galp/tools/galp_cli benchmark \
    /path/to/FastLanes/data/fls/galp-test/data.fls \
    --samples 100

  # benchmark：only GPU decompression, with kernel launch overhead
  /path/to/FastLanes/build/galp/tools/galp_cli benchmark \
    /path/to/FastLanes/data/fls/galp-test/data.fls \
    --samples 100  --estimate-launch --launch-iters 10000