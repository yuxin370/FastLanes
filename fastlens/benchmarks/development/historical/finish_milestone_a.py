"""Continue the remaining fixed Milestone A stages after the active training runner."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.common import gpu_uuid
from fastlens.benchmarks.development.reports.summarize import FIELDS
from fastlens.benchmarks.development.reports.summarize import model_name
from fastlens.benchmarks.common import write_csv
from fastlens.benchmarks.development.reports.deployment_report import rows as csv_rows


def command(script, *arguments):
    subprocess.run([sys.executable, script, *arguments], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-pid', type=int, required=True)
    args = parser.parse_args()
    if psutil.pid_exists(args.after_pid):
        process = psutil.Process(args.after_pid)
        assert any(a.endswith('run_mobile_full_epochs.py') for a in process.cmdline())
        print('Waiting for existing full-epoch runner', args.after_pid, flush=True)
        while process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
            time.sleep(20)
    for gpu in ('4090','H100'):
        for pipeline in ('imagelanes','cpu_dct'):
            assert (ROOT/f'results/p1/mobile_b6/{gpu}/{pipeline}/accepted.json').exists(), 'Full training phase remains incomplete'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid('4090'),
        PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch', TMPDIR=str(Path.home()/'tmp'),
        OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8')
    for model, batch in [('swinv2',50),('mobilenet24',64)]:
        for domain in ('rgb','dct'):
            output = ROOT/f'results/raw/4090/{model}/model_only_{domain}.json'
            if output.exists():
                continue
            directory = ROOT/f'results/analysis/ceiling_runs/4090/{model}/{domain}'
            directory.mkdir(parents=True, exist_ok=True)
            run(['numactl','--cpunodebind=0','--membind=0',sys.executable,'fastlens/benchmarks/development/diagnostics/model_only.py',
                '--model',model,'--domain',domain,'--batch-size',str(batch),'--output',str(output)], directory,env,None)
    # Refresh only ceiling ingestion; the legacy inference eligibility table is preserved.
    rows = []
    for path in sorted((ROOT/'results/raw').glob('*/*/model_only_*.json')):
        if '.command.' in path.name:
            continue
        result = json.loads(path.read_text())
        for trial in result['trials']:
            row = dict.fromkeys(FIELDS)
            row.update(timestamp=result['timestamp'], git_commit=result['git_commit'], gpu=result['gpu'],
                model=model_name(result['model'],result['domain']), domain=result['domain'], pipeline='model_only',task='inference',
                precision=result['precision'],batch_size=result['batch_size'],trial=trial['trial'],samples=trial['samples'],
                elapsed_s=trial['elapsed_s'],torch_peak_allocated_bytes=trial['torch_peak_allocated_bytes'],
                torch_peak_reserved_bytes=trial['torch_peak_reserved_bytes'],
                raw_result=str(path.relative_to(ROOT))+f"#/trials/{trial['trial']-1}",
                correctness_status=result['correctness'],notes=result['scope'])
            rows.append(row)
    write_csv(ROOT/'results/e2e/model_only_raw.csv',rows,FIELDS)
    prior=csv_rows(ROOT/'results/analysis/model_only.csv')
    for gpu in ('4090','H100'):
        env.update(CUDA_VISIBLE_DEVICES=gpu_uuid(gpu),DCTNET_PROFILE='mobilenet24')
        for model,label in [('swinv2','SwinV2-T'),('mobilenet','MobileNetV2')]:
            for domain,pipeline in [('RGB','pytorch'),('DCT','imagelanes')]:
                gpu_label='RTX 4090' if gpu=='4090' else gpu
                if any(r['task']=='training_model_only' and (r['gpu'],r['model'],r['domain'])==(gpu_label,label,domain) for r in prior):
                    continue
                directory=ROOT/f'results/analysis/training_ceilings/{gpu}/{model}/{domain.lower()}/trial_0'
                if (directory/'accepted.json').exists(): continue
                directory.mkdir(parents=True,exist_ok=True)
                if not (directory/'result.json').exists():
                    run(['numactl','--cpunodebind=0','--membind=0',sys.executable,'fastlens/benchmarks/paper/shared/training_window_rgb.py',
                        '--gpu',gpu,'--model',model,'--pipeline',pipeline,'--trial','0','--model-only','--output',str(directory)],directory,env,None)
                result=json.loads((directory/'result.json').read_text())
                assert result['correctness_passed'] and result['images']==65536 and result['updates']==64
                assert result['mode']=='training_model_only' and not result['profile']
                (directory/'accepted.json').write_text(json.dumps(dict(correctness_passed=True,raw_result=str(directory/'result.json')))+'\n')
    command('fastlens/benchmarks/development/historical/replan.py')
    command('fastlens/benchmarks/development/reports/deployment_report.py')
    command('fastlens/benchmarks/development/historical/run_resource_windows.py','--pipelines','pytorch','dali','coordl','ffcv','imagelanes')
    command('fastlens/benchmarks/development/historical/run_minimal_ablations.py')
    command('fastlens/benchmarks/development/reports/deployment_report.py')
    command('fastlens/benchmarks/development/reports/export_milestone_a.py')
    print('Fixed Milestone A queue ended; no Phase 2 experiments started.',flush=True)


if __name__ == '__main__':
    main()
