"""Separate observed implementation revisions without changing historical logs."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
REVISION = ROOT / 'results/version_refresh_20260929'
SNAPSHOT = REVISION / 'code_snapshot.json'


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def capture():
    paths = subprocess.check_output(['git', 'ls-files', 'fastlens'], cwd=ROOT, text=True).splitlines()
    paths = [p for p in paths if Path(p).suffix in ('.py', '.cpp', '.cu', '.cuh', '.hpp', '.h')]
    native = next((ROOT/'build/fastlens/torch').glob('_fastlens_direct_dct.*.so'))
    paths += [str(native.relative_to(ROOT))]
    hashes = {p: file_sha(ROOT/p) for p in sorted(paths)}
    return dict(files=hashes, fingerprint=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                native_path=str(native.relative_to(ROOT)), native_sha256=hashes[str(native.relative_to(ROOT))],
                native_mtime_ns=native.stat().st_mtime_ns,
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())


def verify_snapshot():
    expected = json.loads(Path(os.environ.get('FASTLENS_EVALUATION_SNAPSHOT', SNAPSHOT)).read_text())
    changed = [p for p, sha in expected['files'].items() if file_sha(ROOT/p) != sha]
    if changed:
        raise RuntimeError('Evaluation code changed after snapshot: ' + ', '.join(changed))
    return expected['fingerprint']


def cohort(row):
    if row.get('implementation_cohort'):
        return row['implementation_cohort']
    if row['pipeline'] != 'FastLens':
        if row['pipeline'] == 'CoorDL' and row.get('task', 'inference') == 'inference':
            return 'coordl_cpu' if 'efun' in row['raw_result'] else 'coordl_mixed'
        return 'unchanged'
    snapshot = json.loads(SNAPSHOT.read_text())
    stamp = row.get('timestamp')
    if not stamp:
        command = ROOT / Path(row['raw_result'].split('#')[0]).parent / 'run.command.json'
        if command.exists():
            stamp = json.loads(command.read_text())['timestamp']
    if stamp and datetime.datetime.fromisoformat(stamp).timestamp()*1e9 >= snapshot['native_mtime_ns']:
        # This historical classification is explicitly inferred, not a recovered hash.
        return 'native_' + snapshot['native_sha256'][:16]
    if row.get('task') == 'training_window':
        return 'historical_training_before_native_build'
    return 'historical_unverified:' + row['raw_result'].split('#')[0]


def select_current(rows):
    """Retain raw rows; choose one implementation cohort per comparison cell."""
    if not SNAPSHOT.exists():
        return rows
    current_native = 'native_' + json.loads(SNAPSHOT.read_text())['native_sha256'][:16]
    groups = {}
    for row in rows:
        row['implementation_cohort'] = cohort(row)
        if row['pipeline'] == 'FastLens' and not row.get('code_fingerprint'):
            note = ' Implementation cohort inferred from run/build timestamps; historical binary hash was not recorded.'
            if note not in row.get('notes', ''):
                row['notes'] = row.get('notes', '') + note
        groups.setdefault((row['gpu'], row['model'], row['pipeline']), []).append(row)
    selected = []
    for group in groups.values():
        preferred = [r for r in group if r['implementation_cohort'] == current_native]
        if not preferred:
            latest = max(group, key=lambda r: r.get('timestamp', ''))
            preferred = [r for r in group if r['implementation_cohort'] == latest['implementation_cohort']]
        selected.extend(preferred)
    return selected
