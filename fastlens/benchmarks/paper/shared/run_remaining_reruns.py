"""Run staging, pool-memory and simulated-read-bandwidth controls on both GPUs."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.common import gpu_uuid
from fastlens.benchmarks.common import write_csv
from fastlens.benchmarks.system_dct_major.common import parse_manifest


def read(path):
    return json.loads(path.read_text())


def boundary_sweep(output, env, library, bandwidth, trials=3, crops=(448, 160), rates=(125, 500, 2000), verify=True):
    source = ROOT / 'fastlens/data/compressed/imagenet512_val_block_major'
    paths = [Path(s['fls_path']) for s in parse_manifest(source / 'manifest.bin')['shards']
             if s['first_global_image_index'] < 16384]
    rows = []
    for crop in crops:
        for rate in (rates if bandwidth else (0,)):
            arms = ('full', 'compact') if bandwidth else ('full', 'source', 'compact')
            references = {}
            for trial in range(-1 if verify else 0, trials + 1):
                stage = 'correctness' if trial == -1 else 'warmup' if trial == 0 else f'trial_{trial}'
                for arm in (arms if trial % 2 else arms[::-1]):
                    directory = output / f'crop{crop}_MBs{rate}' / arm / stage
                    directory.mkdir(parents=True, exist_ok=False)
                    count = 8 if trial == -1 else 16384
                    command = [sys.executable, 'fastlens/benchmarks/paper/shared/boundary_control.py',
                        '--control-library', str(library), '--staging', 'source' if arm == 'source' else 'compact',
                        '--read-mbps', str(rate), '--crop-size', str(crop), '--source-coefficients', '64',
                        '--new-data', str(source), '--input-geometry', 'source512', '--pushdown', 'off',
                        '--crop-execution-mode', 'full-source-decode' if arm == 'full' else 'vector-range-read-selected-decode',
                        '--output-layout', 'projected', '--native-runtime', 'b6', '--shards-per-activation', '1',
                        '--physical-prefix', '--batch-size', '64', '--model-threads', '8', '--warmup-batches', '20',
                        '--count', str(count), '--baseline-dir', str(output / 'unused'), '--output-dir', str(directory)]
                    if trial == -1:
                        command.append('--verify')
                    run(command, directory, env, paths[:1] if trial == -1 else paths, bind_to_gpu=True)
                    raw = read(directory / f'N_{count}.json')
                    assert raw['samples'] == len(set(raw['sample_ids'])) == count
                    if trial == -1:
                        assert raw['checks'] and all(c['native_grid_and_full_source_input_exact'] for c in raw['checks'])
                    if count in references:
                        for key in ('sample_ids', 'predictions', 'checkpoint', 'precision', 'batch_size'):
                            assert raw[key] == references[count][key], (arm, crop, rate, key)
                    else:
                        references[count] = raw
                    boundary = raw['input_boundary_control']
                    if rate:
                        assert boundary['read_bytes'] / boundary['read_completion_span_s'] <= rate * 1e6 * 1.01
                    if trial > 0:
                        stats = raw['native_shards']
                        rows.append(dict(gpu=raw['device_name'], crop=crop, simulated_MB_s=rate, arm=arm,
                            trial=trial, elapsed_s=raw['e2e_seconds'], actual_fls_read_bytes=boundary['read_bytes'],
                            issued_payload_bytes=sum(s['rowgroup_storage_bytes_read'] for s in stats),
                            workset_h2d_bytes=sum(s['workset_upload_dma_bytes'] for s in stats),
                            planning_ms=sum(s['planning_ms'] for s in stats),
                            completion_span_s=boundary['read_completion_span_s'],
                            raw_result=str(directory / f'N_{count}.json')))
                        write_csv(output / 'timings.csv', rows, list(rows[0]))
            if not bandwidth:
                selected = [r for r in rows if r['crop'] == crop and r['arm'] == 'compact']
                control = [r for r in rows if r['crop'] == crop and r['arm'] == 'source']
                assert {r['issued_payload_bytes'] for r in selected} == {r['issued_payload_bytes'] for r in control}
                assert max(r['workset_h2d_bytes'] for r in selected) < min(r['workset_h2d_bytes'] for r in control)
    summary = []
    for crop, rate, arm in sorted({(r['crop'], r['simulated_MB_s'], r['arm']) for r in rows}):
        selected = [r for r in rows if (r['crop'], r['simulated_MB_s'], r['arm']) == (crop, rate, arm)]
        summary.append(dict(crop=crop, simulated_MB_s=rate, arm=arm,
            median_s=statistics.median(r['elapsed_s'] for r in selected),
            min_s=min(r['elapsed_s'] for r in selected), max_s=max(r['elapsed_s'] for r in selected),
            actual_fls_read_bytes=selected[0]['actual_fls_read_bytes'],
            issued_payload_bytes=selected[0]['issued_payload_bytes'], workset_h2d_bytes=selected[0]['workset_h2d_bytes']))
    write_csv(output / 'summary.csv', summary, list(summary[0]))


def memory_sweep(output, env, library, gpu, resume=False):
    def completed(directory, result_name):
        if not resume or not (directory / result_name).exists():
            return False
        assert read(directory / 'run.command.json')['returncode'] == 0, directory
        print('REUSE', directory, flush=True)
        return True

    input_results = []
    for arm in ('microbatch', 'pool'):
        directory = output / 'inputs' / arm
        if not completed(directory, 'inputs.json'):
            directory.mkdir(parents=True, exist_ok=False)
            run([sys.executable,
                 'fastlens/benchmarks/paper/shared/pool_memory_control.py', '--control-library', str(library), '--allocation', arm,
                 '--input-check-output', str(directory / 'inputs.json')], directory, env, None, bind_to_gpu=True)
        input_results.append(read(directory / 'inputs.json'))
    assert input_results[0] == input_results[1], 'Input tensors, targets or order differ between allocation controls'
    rows, resources = [], []
    for model in ('swinv2', 'vitti'):
        model_env = dict(env, DCTNET_PROFILE=model)
        references = {}
        for stage, trial in [('check', 0), ('performance', 0), ('performance', 1), ('performance', 2), ('memory', 0)]:
            for arm in (('microbatch', 'pool') if trial % 2 == 0 else ('pool', 'microbatch')):
                directory = output / model / arm / f'{stage}_{trial}'
                reuse = completed(directory, 'result.json')
                if not reuse:
                    directory.mkdir(parents=True, exist_ok=False)
                command = [sys.executable, 'fastlens/benchmarks/paper/shared/pool_memory_control.py',
                    '--control-library', str(library), '--allocation', arm, '--gpu', gpu, '--model', model,
                    '--pipeline', 'imagelanes', '--trial', str(trial), '--output', str(directory)]
                if stage == 'check':
                    command.append('--check')
                elif stage == 'memory':
                    command += ['--memory', '--measured-images', '16384']
                else:
                    command += ['--correctness-report', str(output / model / arm / 'check_0/result.json')]
                if stage == 'memory' and not reuse:
                    with (directory / 'gpu_process_memory.csv').open('x') as telemetry:
                        monitor = subprocess.Popen(['nvidia-smi',
                            '--query-compute-apps=timestamp,pid,gpu_uuid,used_gpu_memory',
                            '--format=csv,noheader,nounits', '-lms', '100'], stdout=telemetry)
                        try:
                            run(command, directory, model_env, None, bind_to_gpu=True)
                        finally:
                            monitor.terminate()
                            monitor.wait()
                elif not reuse:
                    run(command, directory, model_env, None, bind_to_gpu=True)
                result = read(directory / 'result.json')
                assert result['correctness_passed']
                if stage in references:
                    for key in ('sample_order_sha256', 'initial_model_hash', 'precision', 'batch',
                                'accumulation', 'augmentation_configuration', 'images', 'warmup_updates'):
                        assert result[key] == references[stage][key], (model, stage, key)
                else:
                    references[stage] = result
                if stage == 'performance':
                    assert result['images'] == 65536 and result['warmup_updates'] == 32
                    rows.append(dict(gpu=gpu, model=model, allocation=arm, trial=trial,
                        images=result['images'], seconds=result['seconds'], images_per_s=result['images']/result['seconds'],
                        precision=result['precision'], raw_result=str(directory / 'result.json')))
                    write_csv(output / 'timings.csv', rows, list(rows[0]))
                elif stage == 'memory':
                    memory = read(directory / 'memory_samples.json')
                    residency = []
                    with (directory / 'gpu_process_memory.csv').open() as stream:
                        for stamp, pid, uuid, mib in csv.reader(stream):
                            ns = int(datetime.datetime.strptime(stamp.strip(), '%Y/%m/%d %H:%M:%S.%f').timestamp()*1e9)
                            if int(pid) == memory['pid'] and memory['start_unix_ns'] <= ns <= memory['end_unix_ns']:
                                residency.append(int(mib)*1024*1024)
                    assert residency
                    resources.append(dict(gpu=gpu, model=model, allocation=arm,
                        process_peak_bytes=max(residency), torch_peak_allocated_bytes=memory['torch_peak_allocated_bytes'],
                        native_peak_in_use_bytes=max(r['native_pools']['device_pool']['peak_in_use_bytes'] for r in memory['measurements']),
                        raw_memory=str(directory / 'memory_samples.json')))
                    write_csv(output / 'memory.csv', resources, list(resources[0]))
        control, optimized = [r for r in resources if r['model'] == model and r['allocation'] == 'pool'], [
            r for r in resources if r['model'] == model and r['allocation'] == 'microbatch']
        assert control[0]['process_peak_bytes'] > optimized[0]['process_peak_bytes'], 'Pool-allocation control did not raise residency'
        assert control[0]['native_peak_in_use_bytes'] > optimized[0]['native_peak_in_use_bytes']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--libraries', type=Path, required=True)
    parser.add_argument('--experiments', choices=('staging', 'memory', 'bandwidth'), nargs='+',
                        default=('staging', 'memory', 'bandwidth'))
    parser.add_argument('--gpu', choices=('4090', 'H100'), nargs='+', default=('4090', 'H100'))
    parser.add_argument('--boundary-trials', type=int, default=3,
                        help='Measured trials per arm for staging and bandwidth, excluding correctness and warmup')
    parser.add_argument('--boundary-crops', type=int, choices=(448, 160), nargs='+', default=(448, 160))
    parser.add_argument('--read-bandwidths', type=int, choices=(125, 500, 2000), nargs='+', default=(125, 500, 2000))
    parser.add_argument('--skip-correctness', action='store_true',
                        help='Skip separate boundary correctness passes; retain warmup and paired-output checks')
    parser.add_argument('--resume-memory', action='store_true',
                        help='Reuse successful memory runs; move interrupted run directories aside before resuming')
    args = parser.parse_args()
    if args.boundary_trials < 1:
        parser.error('boundary-trials must be positive')
    for experiment in args.experiments:
        for gpu in args.gpu:
            output = args.output.resolve() / gpu / experiment
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid(gpu), DCTNET_PROFILE='mobilenet24',
                PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch', TMPDIR=str(Path.home() / 'tmp'),
                OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', PYTHONDONTWRITEBYTECODE='1')
            print('BEGIN', gpu, experiment, flush=True)
            if experiment == 'memory':
                memory_sweep(output, env, args.libraries.resolve() / 'libpool_memory_control.so', gpu,
                             resume=args.resume_memory)
            else:
                boundary_sweep(output, env, args.libraries.resolve() / 'libinput_boundary_controls.so',
                               bandwidth=experiment == 'bandwidth', trials=args.boundary_trials,
                               crops=args.boundary_crops, rates=args.read_bandwidths,
                               verify=not args.skip_correctness)
            print('COMPLETE', gpu, experiment, flush=True)


if __name__ == '__main__':
    main()
