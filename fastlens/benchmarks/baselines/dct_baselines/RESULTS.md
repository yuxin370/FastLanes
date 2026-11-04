# DCT baseline results and evidence

## Published measurements

The paper evaluation dated 2026-10-18 has retained, machine-readable tables:

| Experiment | Available records | Scope |
| --- | --- | --- |
| Matched DCT | [section_6_2_1_matched_dct.csv](../../paper/results/section_6_2_1_matched_dct.csv) | Seven models, RTX 4090/H100, training and inference |
| Shape-preserving selection | [selected_inference.csv](../../paper/results/section_6_2_2/selected_inference.csv) | H100, MobileNet24/ResNet24/ViT-Ti, D4/D6/D8 |
| Compact training | [compact_training.csv](../../paper/results/section_6_2_2/compact_training.csv) | ResNet-50/MobileNetV2/ViT-Ti, D3/D15, RTX 4090/H100 |

See [published result notes](../../paper/results/README.md) for statistics,
trial sources and limits, and the [paper reproduction guide](../../paper/README.md)
for fresh runs. CSV source references identify the original experiments; large
raw run trees and checkpoints are external. Use the
[reference preparation guide](../../paper/REBUILDING_REFERENCES.md) to rebuild
inputs. Fresh measurements have their own hardware, code and data identities.

## Diagnostic runs

`bench` writes JSONL rows and adjacent counters; `report` validates
[result.schema.json](result.schema.json) and groups compatible contracts,
implementation hashes, store content, device and runtime settings. Keep the
raw rows and their verification JSON with any summary. Record experiment date,
code revision/native binary identity, GPU/CPU/software, source population,
checkpoint, precision, resource limits and cache/timing protocol with a run.
These details determine which comparisons are valid.

The diagnostic schema retains `headline_eligible=false`. It covers bounded
inference and training checks, reader studies and adapted IC/FS timing. A
successful check establishes its recorded scope; it does not qualify an
unmeasured full dataset, different model, code revision or resource allocation.
