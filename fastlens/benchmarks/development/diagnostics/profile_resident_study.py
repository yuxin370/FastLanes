"""Separate 16,384-image diagnostic controls and Nsight captures after formal trials."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.development.diagnostics.analyze_resident_trace import analyze
from fastlens.benchmarks.development.diagnostics.analyze_resident_trace import render_timelines
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.development.historical.recheck_table4_l3 import GPUS
from fastlens.benchmarks.execution.revision_results import file_sha
from fastlens.benchmarks.execution.revision_results import verify_snapshot
from fastlens.benchmarks.development.reports.report_resident_study import write_csv


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def select(rows):
    candidates = [r for r in rows if r['model'] in ('vitti', 'swinv2')]
    assert len(candidates) == 4
    ordered = sorted(candidates, key=lambda r: (abs(float(r['g_DCT'])), r['gpu'], r['model']))
    result = []
    for role, row in (('largest_gap', ordered[-1]), ('closest_to_resident', ordered[0])):
        raw, jpeg = float(row['ffcv_raw']), float(row['ffcv_jpeg'])
        path = 'ffcv_jpeg' if abs(raw-jpeg)/max(raw,jpeg) <= .02 or jpeg > raw else 'ffcv_raw'
        result.append(dict(role=role, gpu=row['gpu'], model=row['model'], g_DCT=float(row['g_DCT']), ffcv=path))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    profiles = out/'profiles'
    profiles.mkdir(exist_ok=True)
    verify_snapshot()
    selection = select(list(csv.DictReader((out/'absolute_throughput.csv').open())))
    code = {str(ROOT / name): file_sha(ROOT / name)
            for name in ('fastlens/benchmarks/development/diagnostics/profile_resident_study.py', 'fastlens/benchmarks/development/diagnostics/analyze_resident_trace.py')}
    plan = dict(selection=selection, code=code, images=16384, warmup_updates=32,
        order=['imagelanes_basic', 'imagelanes_full', 'dct_resident', 'selected_ffcv', 'rgb_resident'],
        pair_order='Uninstrumented control followed by instrumented capture; both restore identical states.',
        timeline='200 ms from the first forward kernel of update 4, identical scale for every path.',
        nsight_event_caveat='CUDA event tracing can perturb execution and introduce cross-stream dependencies; only profiled-window attribution is reported.')
    if (profiles/'plan.json').exists():
        assert read(profiles/'plan.json') == plan
    else:
        save(profiles/'plan.json', plan)
    env = dict(os.environ, TMPDIR=str(Path.home()/'tmp'), PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=f'{ROOT}:{out / "native_runtime"}', FASTLENS_TRAINING_NATIVE_DIR=str(out/'native_runtime'),
        FASTLENS_NSYS_FINE='0', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    env.pop('FASTLENS_ALLOWED_RESIDENT_PIDS', None)
    base = ['numactl', '--cpunodebind=0', '--membind=0', sys.executable, '-B']
    state = dict(status='RUNNING', completed_windows=0, planned_windows=20, active=None)
    comparisons = []
    augmentation_comparisons = []

    def launch(command, dest):
        state['active'] = str(dest.relative_to(out))
        save(profiles/'status.json', state)
        if (dest/'result.json').exists():
            metadata = read(dest/'run.command.json')
            assert metadata['returncode'] == 0 and metadata['code_unchanged_during_run']
            return
        if dest.exists():
            dest.rename(dest.with_name(dest.name+'_failed_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
        dest.mkdir(parents=True)
        run(command, dest, env, None)
        metadata = read(dest/'run.command.json')
        assert metadata['returncode'] == 0 and metadata['code_unchanged_during_run']

    try:
        for choice in selection:
            gpu, model = choice['gpu'], choice['model']
            env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu], DCTNET_PROFILE=model)
            group = profiles/gpu/model
            full_gate = group/'checks'/'imagelanes_full'
            matched = group/'checks'/'matched_dct_full'
            launch([*base, 'fastlens/benchmarks/development/diagnostics/check_swin_pls_tensors.py', '--model', model,
                    '--augmentation', 'full', '--output', str(matched/'result.json')], matched)
            assert read(matched/'result.json')['passed']
            driver = ['fastlens/benchmarks/paper/section_6_2_4/resident_training/resident_training_window.py', '--gpu', gpu, '--model', model]
            launch([*base, *driver, '--path', 'imagelanes_full', '--check', '--output', str(full_gate)], full_gate)
            assert read(full_gate/'result.json')['correctness_passed']
            timeline_paths = []
            for path in ('imagelanes_basic', 'imagelanes_full', 'dct_resident', choice['ffcv'], 'rgb_resident'):
                gate = (full_gate if path == 'imagelanes_full' else out/'checks'/gpu/model/path)/'result.json'
                common = [*driver, '--path', path, '--measured-images', '16384', '--correctness-report', str(gate)]
                control, capture = group/path/'control', group/path/'profile'
                launch([*base, *common, '--output', str(control)], control)
                state['completed_windows'] += 1
                launch(['nsys', 'profile', '--trace=cuda,nvtx,osrt', '--sample=none', '--cpuctxsw=none',
                        '--cuda-event-trace=true', '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
                        '--output', str(capture/'trace'), *base, *common, '--profile', '--output', str(capture)], capture)
                state['completed_windows'] += 1
                a, b = read(control/'result.json'), read(capture/'result.json')
                assert a['images'] == b['images'] == 16384 and not a['profile'] and b['profile']
                assert a['restored_state'] == b['restored_state'] and a['sample_order_sha256'] == b['sample_order_sha256']
                sqlite = capture/'trace.sqlite'
                if not sqlite.exists():
                    with (capture/'export.log').open('w') as stream:
                        subprocess.run(['nsys', 'export', '--type', 'sqlite', '--output', str(sqlite),
                                        str(capture/'trace.nsys-rep')], stdout=stream, stderr=subprocess.STDOUT, check=True)
                summary = analyze(sqlite, capture)
                comparisons.append(dict(gpu=gpu, model=model, role=choice['role'], path=path, images=16384,
                    control_seconds=a['seconds'], profile_seconds=b['seconds'],
                    control_images_per_s=a['images_per_s'], profile_images_per_s=b['images_per_s'],
                    profiler_time_increase=b['seconds']/a['seconds']-1,
                    confirmed_input_wait_ms=summary['confirmed_input_wait_ms'], f_input_wait=summary['f_input_wait'],
                    unavailable_event_completions=summary['unavailable_event_completions'],
                    other_model_gaps_ms=summary['other_model_gaps_ms']))
                write_csv(profiles/'diagnostic_comparison.csv', comparisons)
                save(profiles/'status.json', state)
                timeline_paths.append((path, capture))
            render_timelines(group/'representative_timeline', timeline_paths)
            basic = read(group/'imagelanes_basic'/'profile'/'trace_summary.json')
            full = read(group/'imagelanes_full'/'profile'/'trace_summary.json')
            by_mode = {mode: {r['category']: r for r in data['categories']}
                       for mode, data in (('basic', basic), ('full', full))}
            for category in sorted(by_mode['basic'].keys() | by_mode['full'].keys()):
                a = by_mode['basic'].get(category, dict(calls=0, cumulative_gpu_ms=0))
                b = by_mode['full'].get(category, dict(calls=0, cumulative_gpu_ms=0))
                augmentation_comparisons.append(dict(gpu=gpu, model=model, category=category,
                    basic_calls=a['calls'], full_calls=b['calls'], calls_removed=b['calls']-a['calls'],
                    basic_cumulative_gpu_ms=a['cumulative_gpu_ms'], full_cumulative_gpu_ms=b['cumulative_gpu_ms'],
                    cumulative_gpu_ms_difference=b['cumulative_gpu_ms']-a['cumulative_gpu_ms'],
                    interpretation='Profiled kernel duration sums may overlap; fused normalization/Mixup difference is not pure Mixup cost.'))
            write_csv(profiles/'augmentation_gpu_work.csv', augmentation_comparisons)
        assert all(file_sha(Path(path)) == sha for path, sha in code.items())
        state.update(status='COMPLETE', active=None)
    except Exception as error:
        state.update(status='FAILED', error=repr(error))
        raise
    finally:
        save(profiles/'status.json', state)


if __name__ == '__main__':
    main()
