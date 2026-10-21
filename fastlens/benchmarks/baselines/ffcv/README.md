# FFCV input preparation

These scripts prepare and verify the FFCV baseline inputs. Run them from the
repository root in the environment with FFCV installed:

```bash
python -m fastlens.benchmarks.baselines.ffcv.prepare_p0_ffcv --help
python -m fastlens.benchmarks.baselines.ffcv.prepare_compressed_ffcv --help
python -m fastlens.benchmarks.baselines.ffcv.verify_p0_ffcv --help
```

The original dataset sizes, output locations and preparation defaults are
preserved. Final-paper measurements are launched through the section entries
in [`../../paper/`](../../paper/README.md).
