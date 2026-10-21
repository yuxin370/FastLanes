"""Matched prepared-input frequency reads, queued after the H100 full epochs."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.development.historical.retry_full_rgb_contention import wait_for_process
from fastlens.benchmarks.execution.revision_results import file_sha
from fastlens.benchmarks.common import gpu_uuid
from fastlens.benchmarks.common import write_csv
from fastlens.benchmarks.system_dct_major.common import parse_manifest

OLD = ROOT / 'results/ablation/prepared_frequency_20261006'
OUT = ROOT / 'results/ablation/prepared_frequency_retry_20261006'
SOURCE = ROOT / 'fastlens/data/compressed/backup/offline_model_input/imagenet512_val_mobilenet112_block_major'


def read(path):
    return json.loads(path.read_text())


def validate(directory, count, arm, reference=None):
    raw = read(directory / f'N_{count}.json')
    command = read(directory / 'run.command.json')
    assert command['returncode'] == 0 and command['code_unchanged_during_run']
    assert raw['samples'] == len(set(raw['sample_ids'])) == count
    assert len(raw['predictions']) == count and min(raw['predictions']) >= 0
    assert raw['input_geometry'] == 'precomputed' and raw['output_layout'] == 'projected'
    for shard in raw['native_shards']:
        selected, full = shard['selected_coefficient_count'], shard['full_coefficient_count']
        assert (selected == full if arm == 'off' else 0 < selected < full), 'Physical frequency control failed'
    if raw['verification_run']:
        assert raw['checks'] and all(c['input_max_abs'] == 0 for c in raw['checks'])
    if reference is not None:
        for key in ('checkpoint', 'sample_ids', 'predictions', 'top1', 'top5', 'model_profile', 'batch_size', 'precision'):
            assert raw[key] == reference[key], key
        assert raw['native_options']['grid_transform']['output_channels'] == reference['native_options']['grid_transform']['output_channels']
        if raw['pushdown'] != reference['pushdown']:
            selected, full = (raw, reference) if arm == 'on' else (reference, raw)
            selected_bytes = sum(s['rowgroup_storage_bytes_read'] for s in selected['native_shards'])
            full_bytes = sum(s['rowgroup_storage_bytes_read'] for s in full['native_shards'])
            assert 0 < selected_bytes < full_bytes, 'No physical frequency-read reduction; logical coefficient counts alone are insufficient'
    return raw


def main():
    wait_for_process(439000, 'fastlens/benchmarks/development/historical/resume_full_rgb_frozen_20261006.py')
    state = read(ROOT / 'results/full_rgb_epochs_20261002/status.json')
    if state['status'] not in ('COMPLETE', 'FINISHED_WITH_FAILURES'):
        raise RuntimeError('Primary epoch queue did not reach a completed boundary')
    wait_for_process(381608, 'fastlens.benchmarks.baselines.l3.run')
    os.environ['DCTNET_PROFILE'] = 'mobilenet24'
    from fastlens.benchmarks.dct_models import backend as B
    omitted = {'model', 'checkpoint', 'shape', 'indices', 'mean', 'std'}
    assert {k: v for k, v in read(SOURCE / 'profile.json').items() if k not in omitted} == {
        k: v for k, v in B.profile().items() if k not in omitted}
    preparation = read(SOURCE / 'validation.json')
    assert preparation['failed_shards'] == 0 and preparation['coefficients_and_qtables_exact']
    previous = OLD / 'correctness/on_0'
    previous_files = read(OLD / 'runtime_snapshot.json')['files']
    gate_dependencies = [p for p in previous_files if
        ('/fastlens/benchmarks/dct_models/' in p and not p.endswith('/evaluate_rgb.py')) or
        '_fastlens_direct_dct' in p or p in (str(B.CHECKPOINT), str(SOURCE / 'profile.json'))]
    reuse_selected = all(file_sha(p) == previous_files[p] for p in gate_dependencies)
    if reuse_selected:
        from fastlens.benchmarks.dct_models.evaluate_shards import native_options
        checked = validate(previous, 8, 'on')
        assert checked['native_options'] == native_options(True, True) and checked['model_profile'] == B.profile()
        (OUT / 'correctness_reuse.json').write_text(json.dumps(dict(raw_result=str(previous / 'N_8.json'),
            verified_dependencies=gate_dependencies, note='Only passed ON check reused; interrupted OFF check rerun.'), indent=2) + '\n')
    files = set(read(ROOT / 'results/full_rgb_epochs_20261002/runtime_snapshot_20261006.json')['files'])
    files.update(str(p.resolve()) for p in (ROOT / 'fastlens/benchmarks/dct_models').glob('*.py'))
    files.update(str(p.resolve()) for p in (ROOT / 'build/fastlens/torch').glob('_fastlens_direct_dct*.so'))
    files.update(map(str, [Path(__file__).resolve(), ROOT / 'fastlens/benchmarks/development/diagnostics/prepared_frequency_control.py',
                          ROOT / 'fastlens/benchmarks/system_dct_major/common.py', Path(B.CHECKPOINT), SOURCE / 'profile.json']))
    hashes = {p: file_sha(ROOT / p) for p in sorted(files)}
    snapshot = OUT / 'runtime_snapshot.json'
    snapshot.write_text(json.dumps(dict(files=hashes, fingerprint=hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode()).hexdigest()), indent=2) + '\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid('4090'), PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch',
               TMPDIR=str(Path.home() / 'tmp'), OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
               PYTHONDONTWRITEBYTECODE='1', FASTLENS_NSYS_FINE='0', FASTLENS_EVALUATION_SNAPSHOT=str(snapshot))
    paths = [Path(s['fls_path']) for s in parse_manifest(SOURCE / 'manifest.bin')['shards']
             if s['first_global_image_index'] < 16384]
    references, rows = {}, []
    stages = [('correctness', 8, 0)] + [('performance', 16384, t) for t in (1, 2, 3)] + [('diagnostic', 16384, 0)]
    for stage, count, trial in stages:
        for arm in (('off', 'on') if trial % 2 else ('on', 'off')):
            if stage == 'correctness' and arm == 'on' and reuse_selected:
                references[count] = validate(previous, count, arm)
                continue
            directory = OUT / stage / f'{arm}_{trial}'
            directory.mkdir(parents=True)
            command = [sys.executable, 'fastlens/benchmarks/development/diagnostics/prepared_frequency_control.py', '--new-data', str(SOURCE),
                       '--input-geometry', 'precomputed', '--native-runtime', 'legacy', '--pushdown', arm,
                       '--output-layout', 'projected', '--physical-prefix', '--count', str(count),
                       '--batch-size', '64', '--model-threads', '8', '--warmup-batches', '20',
                       '--output-dir', str(directory), '--baseline-dir', str(OUT / 'unused')]
            if stage == 'correctness':
                command += ['--verify']
            if stage == 'diagnostic':
                command += ['--diagnostic-window', str(directory / 'window.json')]
                command = ['nsys', 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
                           '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
                           '--output', str(directory / 'capture'), *command]
            run(['numactl', '--cpunodebind=0', '--membind=0', *command], directory, env,
                paths[:1] if count == 8 else paths)
            raw = validate(directory, count, arm, references.get(count))
            references.setdefault(count, raw)
            if stage == 'diagnostic':
                database = directory / 'capture.sqlite'
                subprocess.run([sys.executable, 'fastlens/benchmarks/execution/run_logged.py', '--log', str(directory / 'export.log'),
                    '--', 'nsys', 'export', '--type', 'sqlite', '--output', str(database),
                    str(directory / 'capture.nsys-rep')], check=True, env=env)
                with sqlite3.connect(database) as connection:
                    (start, end), = connection.execute("SELECT n.start,n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE COALESCE(n.text,s.value)='control.measured_pass'").fetchall()
                    copies = connection.execute('SELECT start,end,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=1 AND start<? AND end>?', (end, start)).fetchall()
                    assert copies and all(start <= a <= b <= end for a, b, _ in copies)
                    kernels = connection.execute('SELECT s.value,SUM(k.end-k.start),COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id WHERE k.start>=? AND k.end<=? GROUP BY s.value', (start, end)).fetchall()
                write_csv(directory / 'kernels.csv', [dict(kernel=n, cumulative_ns=t, launches=c) for n, t, c in kernels], ['kernel', 'cumulative_ns', 'launches'])
                for row in rows:
                    if row['arm'] == arm:
                        row.update(h2d_bytes=sum(size for _, _, size in copies), h2d_copy_count=len(copies),
                                   h2d_memcpy_s=sum(b-a for a, b, _ in copies)/1e9, diagnostic_raw=str(database))
            if stage == 'performance':
                metadata = read(directory / 'run.command.json')
                rows.append(dict(timestamp=metadata['timestamp'], git_commit=metadata['git_commit'], gpu=raw['device_name'],
                    model='MobileNetV2-DCT24', checkpoint=raw['checkpoint'], domain='DCT', pipeline='FastLens',
                    comparison_type='ablation', task='inference', precision=raw['precision'], batch_size=64,
                    trial=trial, samples=count, arm=arm, elapsed_s=raw['e2e_seconds'], images_per_s=count/raw['e2e_seconds'],
                    dependency_bytes=sum(s['selected_compressed_payload_bytes'] for s in raw['native_shards']),
                    issued_read_bytes=sum(s['rowgroup_storage_bytes_read'] for s in raw['native_shards']),
                    workset_h2d_bytes=sum(s['workset_upload_dma_bytes'] for s in raw['native_shards']),
                    input_wait_s=raw['input_wait_seconds'], model_s=raw['model_seconds'], top1=raw['top1'], correctness='PASS',
                    raw_result=str(directory / f'N_{count}.json'),
                    notes='Prepared target coefficients; fixed output channels and geometry; legacy reader. Not an online resize decomposition. Unprofiled timing; H2D from separate diagnostic.'))
                assert rows[-1]['issued_read_bytes'] > 0
                write_csv(OUT / 'performance_raw.csv', rows, list(rows[0]))
    assert max(r['dependency_bytes'] for r in rows if r['arm'] == 'on') < min(r['dependency_bytes'] for r in rows if r['arm'] == 'off')
    write_csv(OUT / 'performance_raw.csv', rows, list(rows[0]))
    summary = []
    for arm in ('off', 'on'):
        selected = [r for r in rows if r['arm'] == arm]
        summary.append(dict(arm=arm, trials=len(selected), samples=16384,
            median_images_per_s=statistics.median(r['images_per_s'] for r in selected),
            min_images_per_s=min(r['images_per_s'] for r in selected), max_images_per_s=max(r['images_per_s'] for r in selected),
            median_elapsed_s=statistics.median(r['elapsed_s'] for r in selected),
            issued_read_bytes=selected[0]['issued_read_bytes'], h2d_bytes=selected[0]['h2d_bytes'], correctness='PASS'))
    write_csv(OUT / 'summary.csv', summary, list(summary[0]))
    print('COMPLETE: prepared-input matched frequency control; all three unprofiled trials retained', flush=True)


if __name__ == '__main__':
    main()
