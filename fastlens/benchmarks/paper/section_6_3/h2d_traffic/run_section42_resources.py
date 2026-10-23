"""Profile completed chapter runs without changing their measured workload."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.system_dct_major.common import parse_manifest

PIPELINES = {'pytorch': 'PyTorch', 'dali': 'DALI', 'coordl': 'CoorDL',
             'ffcv_raw': 'FFCV', 'l3': 'L3', 'cpu_dct': 'CPU-DCT',
             'fastlens': 'FastLens', 'fastlens_basic': 'FastLens'}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def jobs(run_dir):
    """Use one completed trial per published workload; never use check runs."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    seen = set()
    for record in sorted(run_dir.rglob('run.command.json')):
        if not record.parent.name.startswith('trial_'):
            continue
        metadata = read(record)
        pipeline = metadata['pipeline']
        if pipeline not in PIPELINES:
            continue
        key = (metadata['phase'], metadata['gpu'], metadata['model'], PIPELINES[pipeline])
        if key in seen:
            continue
        source = record.parent / 'result.json'
        if not source.is_file():
            raise FileNotFoundError(source)
        seen.add(key)
        yield dict(phase=key[0], gpu=key[1], model=key[2], pipeline=key[3],
                   source=str(source.resolve()), metadata=metadata)


def set_arg(command, flag, value):
    command[command.index(flag)+1] = str(value)


def training_inputs(job, command):
    if job['pipeline'] == 'FastLens':
        manifest = ROOT / 'fastlens/data/compressed/imagenet512_train_block_major_premixed/dct/manifest.bin'
        return [Path(s['fls_path']) for s in parse_manifest(manifest)['shards']]
    if job['pipeline'] == 'FFCV':
        root = (Path(command[command.index('--ffcv-root')+1]) if '--ffcv-root' in command
                else ROOT / 'results/preparation/p0_ffcv')
        return [root / 'train_window.beton']
    from fastlens.benchmarks.baselines.l3.prepare import source_samples, encoded_path
    samples = source_samples('train')
    if job['pipeline'] == 'L3':
        root = command[command.index('--l3-root')+1]
        return [encoded_path(root, s['logical_sample_id']) for s in samples]
    return [Path(s['path']) for s in samples]


def inference_inputs(job, command, destination):
    contract = None
    if '--contract' in command:
        contract = read(command[command.index('--contract') + 1])
    elif '--reference' in command:
        reference = read(command[command.index('--reference') + 1])
        if 'contract' in reference:
            contract = read(reference['contract'])
    if job['pipeline'] == 'FastLens':
        manifest = ROOT / 'fastlens/data/compressed/imagenet512_val_block_major/manifest.bin'
        return [Path(s['fls_path']) for s in parse_manifest(manifest)['shards']]
    if job['pipeline'] == 'FFCV':
        return [Path(contract['pipelines']['ffcv']['beton'])]
    if contract is not None:
        samples = read(contract['dataset']['sample_manifest'])['samples']
        if job['pipeline'] == 'L3':
            from fastlens.benchmarks.baselines.l3.prepare import encoded_path
            root = contract['pipelines']['dali']['l3']['root']
            return [encoded_path(root, s['sample_id']) for s in samples]
        return [Path(s['path']) for s in samples]
    from fastlens.benchmarks.baselines.l3.prepare import source_samples
    return [Path(s['path']) for s in source_samples('val')]


def verify(job, result, cold_start=False):
    source = read(job['source'])
    if job['phase'] == 'training':
        assert result['images'] == 65536 and result['updates'] == 64 and result['correctness_passed']
        for key in ('initial_model_hash', 'sample_order_sha256', 'precision', 'batch', 'accumulation', 'warmup_updates'):
            assert result[key] == source[key], key
        for key in ('worker_count', 'decoder_threads', 'prefetch', 'cache_mode', 'compile', 'cuda_graphs',
                    'tf32', 'pytorch_workers', 'host_threads', 'rgb_input_reference'):
            if key in source and not (cold_start and key == 'cache_mode'):
                assert result[key] == source[key], key
        return result['images']
    actual = result['repeats'][0] if 'repeats' in result else result
    expected = source['repeats'][0] if 'repeats' in source else source
    count = actual['images'] if 'images' in actual else actual['samples']
    assert count == 50000
    for key in ('sample_trace', 'sample_ids', 'image_ids', 'checkpoint', 'top1_predictions_sha256', 'predictions'):
        if key in expected:
            assert actual[key] == expected[key], key
    if 'model' in source and isinstance(source['model'], dict):
        assert result['model']['checkpoint_sha256'] == source['model']['checkpoint_sha256']
    # Accuracy is checked independently from IDs and checkpoint identity.
    key = 'accuracy_top1' if 'accuracy_top1' in actual else 'top1'
    assert abs(actual[key] - expected[key]) < 1e-5, key
    return count


def metrics(destination, mode, images):
    if mode == 'profile':
        with sqlite3.connect(destination / 'capture.sqlite') as connection:
            windows = connection.execute("SELECT n.start,n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s "
                "ON n.textId=s.id WHERE COALESCE(n.text,s.value)='fastlens.measured_pass'").fetchall()
            (start, end), = windows
            copies = connection.execute('SELECT start,end,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY '
                'WHERE copyKind=1 AND start<? AND end>?', (end, start)).fetchall()
            assert all(a >= start and b <= end for a, b, _ in copies), 'H2D crosses capture boundary'
            total = sum(size for _, _, size in copies)
            return dict(h2d_bytes=total, h2d_bytes_per_image=total/images, h2d_copy_count=len(copies))
    memory = read(destination / 'memory_samples.json')
    residency, utilization = [], []
    for row in csv.reader((destination / 'gpu_process_memory.csv').open()):
        stamp, pid, _, mib = (x.strip() for x in row)
        timestamp = int(datetime.datetime.strptime(stamp, '%Y/%m/%d %H:%M:%S.%f').timestamp()*1e9)
        if int(pid) == memory['pid'] and memory['start_unix_ns'] <= timestamp <= memory['end_unix_ns']:
            residency.append(int(mib)*1024**2)
    for row in csv.reader((destination / 'gpu_utilization.csv').open()):
        stamp, value = (x.strip() for x in row)
        timestamp = int(datetime.datetime.strptime(stamp, '%Y/%m/%d %H:%M:%S.%f').timestamp()*1e9)
        if memory['start_unix_ns'] <= timestamp <= memory['end_unix_ns']:
            utilization.append(float(value))
    assert residency and utilization
    return dict(gpu_peak_process_bytes=max(residency), gpu_utilization_mean_percent=sum(utilization)/len(utilization),
                storage_read_bytes=memory['process_tree_storage_read_bytes'],
                read_chars=memory['process_tree_read_chars'],
                torch_peak_allocated_bytes=memory['torch_peak_allocated_bytes'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'results/reproduced',
                        help='Completed chapter trials containing run.command.json and result.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/reproduced/section_6_3/h2d_traffic')
    parser.add_argument('--phases', nargs='+', choices=['training', 'inference'], default=['training', 'inference'])
    parser.add_argument('--gpus', nargs='+', choices=['4090', 'H100'], default=['4090'])
    parser.add_argument('--models', nargs='+', choices=['vitti', 'swinv2', 'mobilenet24', 'mobilenet32', 'resnet24', 'resnet64', 'efun'],
                        help='Defaults: ViT-Ti and MobileNet24 for warm diagnostics; MobileNet24 for cold IO')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--cold-start', action='store_true', help='Separate unprofiled cold-input runs, including startup IO')
    args = parser.parse_args()
    if (args.output.resolve() / 'STOP_AFTER_CURRENT').is_file() and not args.list:
        parser.exit(75, 'Paused by user; STOP_AFTER_CURRENT is present.\n')
    models = args.models or (['mobilenet24'] if args.cold_start else ['vitti', 'mobilenet24'])
    selected = [j for j in jobs(args.run_dir) if j['phase'] in args.phases and j['gpu'] in args.gpus and j['model'] in models]
    if not selected:
        parser.error(f'No completed chapter trials match the selected workloads in {args.run_dir}')
    if args.limit:
        selected = selected[:args.limit]
    if args.list:
        print(json.dumps([{k:v for k,v in j.items() if k != 'metadata'} for j in selected], indent=2))
        return
    out = args.output.resolve()
    if args.cold_start:
        out = out / 'cold_start'
    out.mkdir(parents=True, exist_ok=True)
    for job in selected:
        for mode in (('cold',) if args.cold_start else ('memory', 'profile')):
            dest = out / job['phase'] / job['gpu'] / job['model'] / job['pipeline'].lower() / mode
            if (dest / 'measurement.json').exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            assert not (dest / 'run.log').exists(), f'Incomplete run retained: {dest}'
            command = list(job['metadata']['command'])
            env = dict(os.environ)
            env.update({k:v for k,v in job['metadata']['environment'].items() if v is not None})
            for key in ('FASTLENS_EVALUATION_SNAPSHOT', 'FASTLENS_ALLOWED_RESIDENT_PIDS'):
                env.pop(key, None)
            env.pop('CUBLAS_WORKSPACE_CONFIG', None)
            env.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(Path.home() / 'tmp'),
                       PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch',
                       FASTLENS_TRAINING_NATIVE_DIR=str(ROOT / 'build/fastlens/torch'))
            raw = dest / 'result.json'
            set_arg(command, '--output', dest)
            paths = None
            # Replay the original correctness gate and exact timed loop.
            if args.cold_start:
                paths = (training_inputs(job, command) if job['phase'] == 'training'
                         else inference_inputs(job, command, dest))
            index = next(i for i, x in enumerate(command)
                         if Path(x).name in ('python', 'python3', 'coordl-python'))
            arguments = command[index+1:]
            if arguments[:1] == ['-B']:
                arguments = arguments[1:]
            save(dest / 'workload_command.json', arguments)
            command = command[:index+1] + [
                str(ROOT / 'fastlens/benchmarks/paper/section_6_3/h2d_traffic/profile_section42.py'),
                '--command', str(dest / 'workload_command.json'), '--mode', mode, '--output', str(dest)]
            if args.cold_start:
                save(dest / 'input_paths.json', [str(path) for path in paths])
                command += ['--input-paths', str(dest / 'input_paths.json')]
                paths = None
            if mode == 'profile':
                command = ['nsys', 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
                    '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--output', str(dest / 'capture'), *command]
            print('RESOURCE', job['phase'], job['gpu'], job['model'], job['pipeline'], mode, flush=True)
            with (dest / 'gpu_process_memory.csv').open('x') as memory, (dest / 'gpu_utilization.csv').open('x') as utilization:
                monitors = [subprocess.Popen(['nvidia-smi', '--query-compute-apps=timestamp,pid,gpu_uuid,used_gpu_memory',
                    '--format=csv,noheader,nounits', '-lms', '100'], stdout=memory),
                    subprocess.Popen(['nvidia-smi', '-i', env['CUDA_VISIBLE_DEVICES'],
                    '--query-gpu=timestamp,utilization.gpu', '--format=csv,noheader,nounits', '-lms', '100'], stdout=utilization)]
                try:
                    run(command, dest, env, paths)
                finally:
                    for monitor in monitors:
                        monitor.terminate()
                        monitor.wait()
            images = verify(job, read(raw), cold_start=args.cold_start)
            if mode == 'profile':
                subprocess.run(['nsys', 'export', '--type', 'sqlite', '--output', str(dest / 'capture.sqlite'),
                                str(dest / 'capture.nsys-rep')], check=True)
            record = {k:v for k,v in job.items() if k != 'metadata'}
            record.update(mode=mode, images=images, **metrics(dest, mode, images))
            if args.cold_start:
                cold = read(dest / 'cold_start_io.json')
                record.update({key:value for key,value in cold.items() if key.endswith('storage_read_bytes')})
                record['cold_start_scope'] = cold['scope']
            record['scope'] = 'Separate resource profiling of a completed chapter trial; throughput comes from the unprofiled trial'
            save(dest / 'measurement.json', record)
            records = [read(path) for path in sorted(out.glob('*/*/*/*/*/measurement.json'))]
            fields = list(dict.fromkeys(key for row in records for key in row))
            with (out / 'measurements.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(records)
            print('DONE', json.dumps(record), flush=True)
    print('RESOURCE QUEUE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
