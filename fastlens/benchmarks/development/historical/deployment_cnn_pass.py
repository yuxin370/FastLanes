#!/usr/bin/env python3
"""One missing full RGB inference pass per architecture/system, with load annotations."""
import csv
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.resource_guard import ResourceGuard
from fastlens.benchmarks.execution.resource_guard import owned_pids
from fastlens.benchmarks.common import gpu_uuid
from fastlens.benchmarks.profiling.input_motivation import cache_pages


def main():
    gpu = '4090'
    uuid = gpu_uuid(gpu)
    manifest = json.loads((ROOT/'fastlens/data/system_rgbnomore/e2e_v3/training_manifests_official_v3/val.json').read_text())
    paths = [Path(s['path']) for s in manifest['samples']]
    labels = json.loads((ROOT/'results/raw/H100/mobilenet24/inference/cpu_dct/trial_1/samples_50000.json').read_text())
    labels = {s['logical_sample_id']: s['model_label'] for s in labels}
    for profile in ('mobilenet24', 'resnet24', 'efun'):
        for route in ('pytorch', 'dali'):
            output = ROOT/f'results/p0/deployment_passes/{gpu}/{profile}/rgb_{route}'
            if (output/'accepted.json').exists():
                continue
            if output.exists():
                raise RuntimeError(f'Inspect unfinished attempt before restarting: {output}')
            output.mkdir(parents=True)
            guard = ResourceGuard(uuid)
            while guard.blockers(owned_pids()):
                print('Target GPU occupied:', guard.observations, flush=True)
                time.sleep(5)
            cache = cache_pages(paths, warm=True)
            if cache['resident_fraction'] != 1:
                raise RuntimeError(f'Page cache not fully warm: {cache}')
            (output/'warm_cache.json').write_text(json.dumps(cache, indent=2)+'\n')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, DCTNET_PROFILE=profile,
                TMPDIR=str(Path.home()/'tmp'), PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            module = ['-m','fastlens.benchmarks.dct_models.efun','evaluate_rgb'] if profile=='efun' else ['-m','fastlens.benchmarks.dct_models.evaluate_rgb']
            command = ['numactl','--cpunodebind=0','--membind=0',sys.executable,*module,
                '--route',route,'--physical-order','--batch-size','64','--workers','8',
                '--count','50000','--output-dir',str(output)]
            launched = datetime.datetime.now(datetime.timezone.utc).isoformat()
            print(launched, 'START', profile, route, flush=True)
            process = subprocess.Popen([sys.executable,'fastlens/benchmarks/execution/run_logged.py','--log',str(output/'run.log'),'--',*command],
                cwd=ROOT, env=env, start_new_session=True)
            concurrent = False
            with (output/'concurrent_load.jsonl').open('x') as stream:
                while process.poll() is None:
                    blockers = guard.blockers(owned_pids())
                    concurrent = concurrent or any('pid' in o for o in guard.observations)
                    stream.write(json.dumps(dict(timestamp=time.time(), observations=guard.observations,
                        load_average=os.getloadavg(), pressure={k:Path('/proc/pressure',k).read_text() for k in ('cpu','memory','io')}))+'\n')
                    stream.flush()
                    if blockers:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait()
                        raise RuntimeError(f'Target GPU contention; own trial stopped: {blockers}')
                    time.sleep(5)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode,command)
            raw = output/f'RGB_{route}_50000.json'
            data = json.loads(raw.read_text())
            reference = json.loads((ROOT/f'results/raw/H100/{profile}/inference/rgb_{route}/trial_1/RGB_{route}_50000.json').read_text())
            assert data['samples']==50000 and len(data['sample_ids'])==50000 and set(data['sample_ids'])==set(labels)
            assert data['sample_ids']==reference['sample_ids']
            assert Path(data['checkpoint']).resolve()==Path(reference['checkpoint']).resolve()
            assert data['preprocessing']==reference['preprocessing']
            correct = sum(p==labels[i] for i,p in zip(data['sample_ids'],data['predictions']))
            assert abs(data['top1']-100*correct/50000)<1e-9
            agreement = sum(a==b for a,b in zip(data['predictions'],reference['predictions']))/50000
            command_record = json.loads((output/'run.command.json').read_text())
            model = {'mobilenet24':'MobileNetV2 RGB','resnet24':'ResNet-50 RGB','efun':'EfficientNet-B0 reference'}[profile]
            family = {'mobilenet24':'MobileNetV2','resnet24':'ResNet-50','efun':'eFUN'}[profile]
            accepted = dict(timestamp=launched, git_commit=command_record['git_commit'],gpu='RTX 4090',
                architecture_family=family,model=model,model_variant=model,domain='RGB',checkpoint=data['checkpoint'],
                pipeline='PyTorch' if route=='pytorch' else 'DALI',comparison_type='deployment',task='inference',
                precision=data['precision'],batch_size=64,trial=1,samples=50000,elapsed_s=data['e2e_seconds'],
                images_per_s=50000/data['e2e_seconds'],top1=data['top1'],top5=data['top5'],
                correctness='PASS: full IDs, labels, preprocessing, checkpoint, recomputed accuracy',
                cross_gpu_prediction_agreement=agreement,
                raw_result=str(raw.relative_to(ROOT)), concurrent_load_observed=concurrent,
                notes='ImageNet-512; NUMA CPU/memory node 0; workers 8; model threads 8; warm page cache; concurrent host/other-GPU activity annotated in concurrent_load.jsonl, isolation not claimed.')
            (output/'accepted.json').write_text(json.dumps(accepted,indent=2)+'\n')
            subprocess.run([sys.executable,'fastlens/benchmarks/development/reports/deployment_report.py'],cwd=ROOT,check=True)
            print('ACCEPTED',profile,route,accepted['images_per_s'],flush=True)


if __name__=='__main__':
    main()
