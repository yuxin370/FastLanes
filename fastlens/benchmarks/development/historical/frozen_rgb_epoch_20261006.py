"""Execute the byte-identical original RGB entry without changing the shared worktree."""
import hashlib
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'results/full_rgb_epochs_20261002/source_before_stream_priority/training_window_rgb.py'


def main():
    source = SOURCE.read_bytes()
    assert hashlib.sha256(source).hexdigest() == '1f8166530d77bcbfdfd94e7d6c8f7e898ad01c9d530646a4d18facf596221cdf'
    module = types.ModuleType('fastlens.benchmarks.paper.shared.training_window_rgb')
    module.__file__ = str(ROOT / 'fastlens/benchmarks/paper/shared/training_window_rgb.py')
    sys.modules[module.__name__] = module
    exec(compile(source, str(SOURCE), 'exec'), module.__dict__)
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'executed_entry.json').write_text(json.dumps(dict(source=str(SOURCE), sha256=hashlib.sha256(source).hexdigest(),
        model_stream_policy='Original default stream, identical to prior full-RGB cohort',
        shared_entry_replaced=False), indent=2) + '\n')
    if '--full-epoch' in sys.argv and sys.argv[sys.argv.index('--pipeline') + 1] == 'ffcv':
        from fastlens.benchmarks.development.historical.full_ffcv_interleaved_cache import main as run
        run()
    else:
        module.main()


if __name__ == '__main__':
    main()
