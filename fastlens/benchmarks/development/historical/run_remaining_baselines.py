"""Queue the remaining deployment baselines after the current training and Swin runs."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.development.historical.recheck_table4_l3 import GPUS
from fastlens.benchmarks.execution.revision_results import file_sha
from fastlens.benchmarks.execution.revision_results import verify_snapshot
from fastlens.benchmarks.development.reports.report_resident_study import write_csv
from fastlens.benchmarks.development.historical.run_training_expansion import read
from fastlens.benchmarks.development.historical.run_training_expansion import save
from fastlens.benchmarks.paper.shared.remaining_baseline_inference import reference_predictions

CORE = ROOT/'results/resident_study_20261011_fixed'
SWIN = ROOT/'results/swin_inference_20261011_fixed_after_training'
RGB_PATHS = ('pytorch', 'dali', 'coordl', 'l3')
INFERENCE_PATHS = (*RGB_PATHS, 'ffcv_raw', 'ffcv_jpeg', 'cpu_dct', 'imagelanes')
MODELS = ('vitti', 'swinv2', 'mobilenet24', 'mobilenet32', 'efun')


from fastlens.benchmarks.paper.shared.references import inference_reference


def jobs():
    result = []
    for phase in ('training', 'inference'):
        for gpu in GPUS:
            models = MODELS if phase == 'training' else tuple(m for m in MODELS if m != 'swinv2')
            for model in models:
                paths = ((*RGB_PATHS, 'cpu_dct') if phase == 'training' else INFERENCE_PATHS)
                if model == 'mobilenet32':
                    paths = ('cpu_dct',) if phase == 'training' else ('cpu_dct', 'imagelanes')
                for path in paths:
                    job = dict(phase=phase, gpu=gpu, model=model, path=path)
                    if phase == 'inference':
                        ref = inference_reference(gpu, model, path)
                        reference_predictions(ref)
                        job['reference'] = str(ref)
                    result.append(job)
    assert sum(j['phase'] == 'training' for j in result)*3 == 126
    assert sum(j['phase'] == 'inference' for j in result)*3 == 156
    return result


def summarize(out, records):
    write_csv(out/'trials.csv', records)
    summary = []
    for key in dict.fromkeys((r['phase'], r['gpu'], r['model'], r['path']) for r in records):
        rates = [float(r['images_per_s']) for r in records
                 if (r['phase'], r['gpu'], r['model'], r['path']) == key]
        summary.append(dict(zip(('phase', 'gpu', 'model', 'path'), key), trials=len(rates),
            median_images_per_s=statistics.median(rates), min_images_per_s=min(rates), max_images_per_s=max(rates)))
    write_csv(out/'summary.csv', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/remaining_baselines_20261012')
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(exist_ok=True)
    if not (out/'plan.json').exists():
        plan = dict(jobs=jobs(), trials=3, training_runs=126, inference_runs=156,
            after=[str(CORE), str(SWIN)], training_protocol='restored-state-real-resident-8-v2',
            CUBLAS_WORKSPACE_CONFIG=None, model_stream_priority='greatest',
            training_images=65536, inference_images=50000, randaugment=False, mixup=False,
            order='Per GPU and model, rotate paths by 2*trial; retain every successful trial.',
            scope='Training windows and full validation; MobileNet RGB shared; no full training epoch.')
        save(out/'plan.json', plan)
        frozen = read(CORE/'runtime_snapshot.json')
        files = dict(frozen['files'])
        for folder in ('fastlens/benchmarks/system_dct_major', 'fastlens/benchmarks/baselines/system_rgbnomore/inference',
                       'fastlens/benchmarks/dct_models', 'fastlens/benchmarks/baselines/l3'):
            files.update({str(p.relative_to(ROOT)): file_sha(p) for p in (ROOT/folder).rglob('*.py')})
        for name in ('fastlens/benchmarks/development/historical/remaining_baseline_inputs.py', 'fastlens/benchmarks/paper/shared/remaining_baseline_training.py', 'fastlens/benchmarks/paper/shared/remaining_baseline_inference.py', 'fastlens/benchmarks/development/historical/run_remaining_baselines.py'):
            file = ROOT / name
            files[str(file.relative_to(ROOT))] = file_sha(file)
        native = next((SWIN.parent/'swin_inference_20261011_fixed/native_runtime').glob('*.so'))
        files[str(native.relative_to(ROOT))] = file_sha(native)
        l3 = read(ROOT/'results/l3_patch32_20261006/training/4090/vitti/l3/trial_0/result.json')['l3']
        files[l3['library']] = file_sha(l3['library'])
        save(out/'runtime_snapshot.json', dict(files=files,
            fingerprint=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()))
        save(out/'status.json', dict(status='QUEUED', pid=None, completed_runs=0, planned_runs=282, active=None))
    os.environ['FASTLENS_EVALUATION_SNAPSHOT'] = str(out/'runtime_snapshot.json')
    verify_snapshot()
    if args.prepare:
        print('Prepared 126 training and 156 inference runs; current cohorts unchanged.')
        return
    plan = read(out/'plan.json')
    state = read(out/'status.json')
    if state['status'] == 'COMPLETE':
        return
    state.update(status='WAITING_FOR_CURRENT_GROUP', pid=os.getpid())
    state.pop('error', None)
    save(out/'status.json', state)
    records = list(csv.DictReader((out/'trials.csv').open())) if (out/'trials.csv').exists() else []
    env = dict(os.environ, TMPDIR=str(Path.home()/'tmp'), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', FASTLENS_NSYS_FINE='0')
    env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    env.pop('FASTLENS_ALLOWED_RESIDENT_PIDS', None)

    def launch(job, dest, check=False, trial=0):
        phase, gpu, model, path = (job[k] for k in ('phase', 'gpu', 'model', 'path'))
        native = CORE/'native_runtime' if phase == 'training' else SWIN.parent/'swin_inference_20261011_fixed/native_runtime'
        env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu], DCTNET_PROFILE=model,
                   PYTHONPATH=f'{ROOT}:{native}', FASTLENS_TRAINING_NATIVE_DIR=str(native))
        interpreter = str(Path.home()/'tmp/coordl-build-cu128/coordl-python') if path == 'coordl' else sys.executable
        command = ['numactl', '--cpunodebind=0', '--membind=0', interpreter, '-B',
            f'fastlens/benchmarks/paper/shared/remaining_baseline_{phase}.py', '--model', model, '--path', path, '--output', str(dest)]
        if phase == 'training':
            command += ['--gpu', gpu, '--trial', str(trial)]
            if not check:
                command += ['--correctness-report', str(dest.parent/'check/result.json')]
        else:
            command += ['--reference', job['reference']]
        if check:
            command.append('--check')
        while True:
            state.update(status='RUNNING', active=str(dest.relative_to(out)))
            save(out/'status.json', state)
            if dest.exists():
                dest.rename(dest.with_name(dest.name+'_failed_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
            dest.mkdir(parents=True)
            try:
                run(command, dest, env, None)
            except RuntimeError as error:
                if not str(error).startswith('Target GPU contention:'):
                    raise
                if path == 'coordl':
                    from fastlens.benchmarks.baselines.system_rgbnomore.shared.paper_adapters import cleanup_coordl_cache
                    cleanup_coordl_cache(dest)
                print('GPU contention: wait and retry this interrupted attempt only.', flush=True)
                continue
            metadata = read(dest/'run.command.json')
            assert metadata['returncode'] == 0 and metadata['code_unchanged_during_run']
            return read(dest/'result.json')

    try:
        while True:
            current = read(CORE/'status.json')
            if current['status'] == 'FAILED' and not current.get('error', '').startswith("RuntimeError('Target GPU contention:"):
                raise RuntimeError('Current training needs inspection: '+current.get('error', ''))
            if current['status'] == 'COMPLETE' and (SWIN/'status.json').exists():
                preceding = read(SWIN/'status.json')
                if preceding['status'] == 'FAILED':
                    raise RuntimeError('Current Swin inference needs inspection: '+preceding.get('error', ''))
                if preceding['status'] == 'COMPLETE':
                    break
            time.sleep(10)
        verify_snapshot()
        for phase in ('training', 'inference'):
            for gpu in GPUS:
                for model in MODELS:
                    group = [j for j in plan['jobs'] if (j['phase'], j['gpu'], j['model']) == (phase, gpu, model)]
                    for job in group:
                        dest = out/phase/gpu/model/job['path']/'check'
                        if (dest/'result.json').exists():
                            gate = read(dest/'result.json')
                        else:
                            gate = launch(job, dest, check=True)
                        assert gate['correctness_passed'] and gate['CUBLAS_WORKSPACE_CONFIG'] is None
                        if phase == 'training':
                            refpath = 'imagelanes_basic' if job['path'] == 'cpu_dct' else 'ffcv_raw'
                            ref = read(CORE/'checks'/gpu/model/refpath/'result.json')
                            assert gate['restored_state'] == ref['restored_state']
                            assert gate['prefix_512_sha256'] == ref['prefix_512_sha256']
                            assert [t['shape'] for t in gate['tensor_contract']] == [t['shape'] for t in ref['tensor_contract']]
                            assert [t['dtype'] for t in gate['tensor_contract']] == [t['dtype'] for t in ref['tensor_contract']]
                    for trial in range(3):
                        if not group:
                            continue
                        rotate = 2*trial % len(group)
                        for job in group[rotate:]+group[:rotate]:
                            path = job['path']
                            if any((r['phase'], r['gpu'], r['model'], r['path'], int(r['trial'])) == (phase, gpu, model, path, trial) for r in records):
                                continue
                            dest = out/phase/gpu/model/path/f'trial_{trial}'
                            result = launch(job, dest, trial=trial)
                            assert result['images'] == (65536 if phase == 'training' else 50000)
                            assert result['host_threads'] == 8 and result['CUBLAS_WORKSPACE_CONFIG'] is None
                            if phase == 'training':
                                refpath = 'imagelanes_basic' if path == 'cpu_dct' else 'ffcv_raw'
                                reference = read(CORE/'formal'/gpu/model/refpath/f'trial_{trial}'/'result.json')
                                assert result['restored_state'] == reference['restored_state']
                                assert result['sample_order_sha256'] == reference['sample_order_sha256']
                            elif trial:
                                assert result['predictions_sha256'] == read(dest.parent/'trial_0/result.json')['predictions_sha256']
                            records.append(dict(phase=phase, gpu=gpu, model=model, path=path, trial=trial,
                                images=result['images'], seconds=result['seconds'], images_per_s=result['images_per_s'],
                                raw_result=str((dest/'result.json').relative_to(ROOT))))
                            summarize(out, records)
                            state['completed_runs'] = len(records)
                            save(out/'status.json', state)
                            print('COMPLETED', records[-1], flush=True)
        state.update(status='COMPLETE', active=None)
    except Exception as error:
        state.update(status='FAILED', error=repr(error))
        raise
    finally:
        save(out/'status.json', state)


if __name__ == '__main__':
    main()
