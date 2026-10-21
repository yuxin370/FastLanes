# Published results for `fastlens.pdf`

These are the complete retained tables for the final PDF's evaluation, dated
2026-10-18. The tables contain the adopted measurement points, without adding
PCR results from the later manuscript. GB/MB use decimal units; throughput is
images/s. Paths inside `raw_result` and `source_csv` identify historical evidence
and use repository-relative paths. References to external historical storage use
`$HOME` or `$FASTLENS_EXPERIMENT_ROOT`; measurement values are unchanged.

| Section / experiment | Table | Rows |
| --- | --- | ---: |
| §6.1 workloads, Table 1 | [section_6_1_workloads.csv](section_6_1_workloads.csv) | 7 |
| §6.2.1 matched DCT, Fig. 9 | [section_6_2_1_matched_dct.csv](section_6_2_1_matched_dct.csv) | 56 |
| §6.2.2 shape-preserving selection, Table 2 | [selected_inference.csv](section_6_2_2/selected_inference.csv) | 45 |
| §6.2.2 compact inputs, Fig. 10 | [compact_training.csv](section_6_2_2/compact_training.csv) | 60 |
| §6.2.3 deployment, Fig. 11a | [section_6_2_3_deployment.csv](section_6_2_3_deployment.csv) | 160 |
| §6.2.4 resident attainment, Fig. 11b | [section_6_2_4_resident_training.csv](section_6_2_4_resident_training.csv) | 20 |
| §6.2.5 convergence, Fig. 12 | [convergence.csv](section_6_2_5/convergence.csv) | 1,200 |
| §6.3 storage, Table 3 | [storage_table3.csv](section_6_3/storage_table3.csv) | 8 representations × train/val |
| §6.3 H2D traffic, Fig. 14 | [h2d_traffic.csv](section_6_3/h2d_traffic.csv), [FFCV check](section_6_3/ffcv_h2d.csv) | 28 / 4 |
| §6.4 projection / staging, Fig. 13 | [projection.csv](section_6_4/projection.csv), [staging.csv](section_6_4/staging.csv) | 1 / 6 |
| §6.4 compressed transfer / direct tensor, Fig. 15 | [core_timing.csv](section_6_4/core_timing.csv), [core_h2d.csv](section_6_4/core_h2d.csv) | 4 / 8 |
| §6.5.1 block-major locality, Fig. 16 | [locality.csv](section_6_5_1/locality.csv) | 12 |
| §6.5.2 limited read bandwidth | [bandwidth.csv](section_6_5_2/bandwidth.csv) | 24 |

## Trial records and statistics

`end_to_end/` retains all **522 original trial rows**, plus the existing summaries.
These are source records: the ResNet source includes 12 additional DCT-resident
control trials that are not plotted in the final PDF. The section tables above
and the ten figures select only the adopted configurations:

| Subdirectory | Original source under repository `results/` | Trials |
| --- | --- | ---: |
| `resident/` | `resident_study_20261011_fixed/` | 132 |
| `remaining/` | `remaining_baselines_20261012/` | 282 |
| `swin_inference/` | `swin_inference_20261011_fixed_after_training/` | 48 |
| `resnet/` | `resnet_matched_20261013/` | 60 |

Matched speedups are medians of the three paired ratios; the throughput columns
are medians/minima/maxima of the three throughput measurements. The speedup is
FastLens/CPU-DCT and is repeated on both system rows for the same configuration.
Averaging the per-configuration matched speedups gives training **3.80× / 4.20×**
and inference **6.17× / 7.95×** on RTX 4090 / H100, matching Fig. 9.
Deployment excludes PCR and reuses the same RGB measurement for MobileNet24/32,
as in the PDF. Resident attainment uses per-round ratios and selects the fastest
of the six RGB paths within each round.

**Fig. 11b aggregation caveat:** the ten RGB configuration medians have an
arithmetic mean of **89.6545%** (rounds to 89.7%) and a geometric mean of
**89.5586%** (rounds to the PDF's 89.6%). The DCT arithmetic mean is
**99.0642%** (the PDF's 99.1%). The plotted points and raw measurements agree;
the PDF does not establish a uniform aggregation rule for these two prose
values. Both original values and this discrepancy are retained.

Selected-frequency inference has one measurement per point. All **45** rounded
throughput entries match Table 2. Compact training
has 53 single measurements and seven medians of three measurements, explicitly
identified by `statistic`; do not infer unmeasured error bars. Fig. 10's
**6.87×** over CPU-DCT and **13%** over the strongest ImageCalculator baseline
use geometric means over the twelve configurations (6.87214× / 1.12991×). These two tables
were previously under `fastlens/docs/paper/figures/evaluation/`.

`section_6_2_5/shared.csv` retains the original shared-crop records from
`a800_convergence_20261003/figures/convergence_observed.csv`. The two independent
JSONL files retain all metrics from `a800_swin_independent_20261012/remote/independent/`
and `a800_crop_policy_20261010/remote/independent/`. `convergence.csv` combines all
four curves; epoch is one-based. Blank columns mean that a metric was not
exported into that combined table, not zero. There is one seed per curve.

## Storage evidence and source boundaries

`storage_table3.csv` transcribes **the published rounded values** from Table 3;
it does not reconstruct exact byte counts from rounded GB. The existing
[storage_measurements.csv](section_6_3/storage_measurements.csv) is preserved
from `results/cost/storage.csv`, including its byte counts, conversion notes and
raw-result references. It covers only a subset of the Table 3 representations.
Its FFCV-JPEG validation row counts only the `.beton` file: 2,157,072,256 bytes.
The [original conversion record](section_6_3/ffcv_jpeg_val_conversion.json)
also records 22,461,127 bytes of external sample metadata, matching the
`samples.json` file beside that store. Together these are **2,179,533,383 bytes**,
which round to Table 3's **2.180 GB**. The old CSV is retained unchanged; use the
inclusive accounting when comparing with the PDF.

The H2D report includes only the 28 phase/model/system points used in Fig. 14,
plus its four FFCV verification rows. Projection reports only the RTX 4090,
K=8 point cited in §6.4; the larger development sweep stays in the raw results.
H2D tables come from `section42_resources_20261008/` and `ffcv_h2d_20261014/`.
Core/projection/locality timings come from `paper_rerun_20261007/summary.csv`;
the H2D diagnostics were separate runs under `paper_rerun_20261006/`.
Staging is the H100 table under `paper_rerun_20261007/remaining/`.
The bandwidth table retains the original 4090 rows and the H100 125/500 MB/s
rows, replacing H100 2,000 MB/s with the adopted
`bandwidth_recheck_20261012_2000/` measurements. Each row names its source.

The original large run trees, predictions, checkpoints and profiler captures
are not copied here. Some fresh experiments still need those external inputs;
see the [reproduction guide](../README.md). The ten published plots read their
small inputs from this directory:

```bash
python -m fastlens.benchmarks.paper.reproduce figures
python -m fastlens.benchmarks.paper.section_6_3.storage.run table
```

New measurements continue to go to repository `results/reproduced/`, so they
cannot overwrite these published tables.

Presentation labels use **FastLens**. Historical trial IDs such as `imagelanes`,
`imagelanes_basic`, and versioned `galp-*` schemas remain unchanged so the tables
can still be traced to the original evidence. No measurement was changed to
force agreement with a rounded prose claim.
