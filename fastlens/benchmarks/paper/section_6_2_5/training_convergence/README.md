# §6.2.5 — Training convergence (Figure 12)

Defaults: SwinV2 and MobileNet24, both crop policies, 300 epochs, 8 processes, BF16, microbatch 64/global batch 1,024, seed 11997733, 4 segments per pool, pread. This requires the A800-scale training inputs and validation layouts. Historical shared MobileNet training changed GPU count during continuation; a fresh 8-GPU run follows the recipe, not the historical interrupted execution timeline.

## Prepare and validate

Install the [shared environment and public model dependencies](../../REBUILDING_REFERENCES.md#public-inputs).
Training requires the premixed Block-major `dct/manifest.bin` and its matching
`ordered_mapping.csv`; pass them with `--manifest` and `--mapping`. The
[existing-Store rebuild](../../REBUILDING_REFERENCES.md#rebuild-existing-paper-stores-in-the-current-directory-layout)
preserves those inputs. It does not generate a missing historical mapping. The
published mapping and original training data are not bundled, so this checkout
alone cannot reproduce that exact physical training order.

Under `--validation-root`, Swin uses `imagenet512_val_block_major` from
[native source-DCT preparation](../../REBUILDING_REFERENCES.md#prepare-native-source-dct-inference).
MobileNet uses the model-specific offline `backup/offline_model_input/imagenet512_val_mobilenet112_block_major`
representation from the [DCT model preparation tools](../../../dct_models/README.md).
These layouts are distinct; a source512 Store cannot be substituted for the
offline MobileNet input. Check all 50,000 validation samples and physical mapping
before launching distributed training. The runner binds the mapping digest;
retaining an old digest for a new mapping is invalid.

## Run and outputs

```bash
python -m fastlens.benchmarks.paper.section_6_2_5.training_convergence.run --help
python -m fastlens.benchmarks.paper.section_6_2_5.training_convergence.run run --dry-run
python -m fastlens.benchmarks.paper.section_6_2_5.training_convergence.run plot
```

Published records (repository-relative, unprefixed batch names are under `results/`):

`a800_convergence_20261003/figures/convergence_observed.csv; a800_swin_independent_20261012/remote/independent/metrics.jsonl; a800_crop_policy_20261010/remote/independent/metrics.jsonl`

Fresh runs write model/crop-policy subdirectories under `--output`, with epoch
metrics and checkpoints. `plot` redraws the retained published curves; it does
not replace them with the new run's metrics.

See [the paper reproduction guide](../../README.md) for dependencies, input preparation,
default output directories and the distinction between published plots and new runs.
