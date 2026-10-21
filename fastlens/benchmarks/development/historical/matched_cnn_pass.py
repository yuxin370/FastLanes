#!/usr/bin/env python3
"""First missing CNN DCT passes; reuse existing H100 controls, reject failed gates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.common import gpu_uuid
from fastlens.benchmarks.system_dct_major.common import parse_manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=['deployment','matched'])
    parser.add_argument('--verify-mobilenet-fix',action='store_true',help='Validate the repaired DCT24 path on both GPUs in new attempt directories')
    parser.add_argument('--repair-attempt-name',default='fastlens_rounding_fixed')
    parser.add_argument('--checks-attempt-name',help='Reuse completed GPU-specific repair diagnostics from this attempt')
    args=parser.parse_args()
    jobs=[('4090',p,'imagelanes') for p in ('mobilenet24','mobilenet32','resnet24','resnet64','efun')] if args.stage=='deployment' else [
        *[('4090',p,'cpu_dct') for p in ('mobilenet24','mobilenet32','resnet24','resnet64','efun')],('H100','mobilenet24','cpu_dct'),('H100','mobilenet32','cpu_dct')]
    if args.verify_mobilenet_fix:
        if args.stage!='deployment': parser.error('The repair validation uses the deployment stage')
        jobs=[(gpu,'mobilenet24','imagelanes') for gpu in ('4090','H100')]
    source=ROOT/'fastlens/data/compressed/imagenet512_val_block_major'
    native_paths=[Path(s['fls_path']) for s in parse_manifest(source/'manifest.bin')['shards']]
    label_rows=json.loads((ROOT/'results/raw/H100/mobilenet24/inference/cpu_dct/trial_1/samples_50000.json').read_text())
    labels={s['logical_sample_id']:s['model_label'] for s in label_rows}
    for gpu,profile,route in jobs:
        output=ROOT/f'results/p0/deployment_passes/{gpu}/{profile}/{route}'
        if args.verify_mobilenet_fix:
            output=output.with_name(args.repair_attempt_name)
        if (output/'accepted.json').exists() or (output/'failure.json').exists(): continue
        output.mkdir(parents=True,exist_ok=False)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu_uuid(gpu),DCTNET_PROFILE=profile,TMPDIR=str(Path.home()/'tmp'),
            PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        task='evaluate_shards' if route=='imagelanes' else 'evaluate'
        driver=[sys.executable,'-m','fastlens.benchmarks.dct_models.efun',task] if profile=='efun' else [sys.executable,'-m','fastlens.benchmarks.dct_models.'+task]
        command=['numactl','--cpunodebind=0','--membind=0',*driver,'--batch-size','64']
        if route=='imagelanes':
            command+=['--new-data',str(source),'--input-geometry','source512','--pushdown','off',
                '--crop-execution-mode','vector-range-read-selected-decode','--output-layout','projected',
                '--native-runtime','b6','--shards-per-activation','1','--physical-prefix','--baseline-dir',str(output/'unused')]
            paths=native_paths
            checks_root=output.parent/args.checks_attempt_name if args.checks_attempt_name else output
            if args.verify_mobilenet_fix:
                diagnostic=checks_root/'regression_22858'
                diagnostic.mkdir(parents=True,exist_ok=True)
                if not (diagnostic/'diagnosis.json').exists():
                    run(['numactl','--cpunodebind=0','--membind=0',sys.executable,'-m',
                        'fastlens.benchmarks.dct_models.diagnose_source512','--image-id','22858',
                        '--batch-size','64','--verify','--output-dir',str(diagnostic)],diagnostic,env,paths)
                assert json.loads((diagnostic/'run.command.json').read_text())['returncode']==0
            check=checks_root/'correctness'
            check.mkdir(parents=True,exist_ok=True)
            if not (check/'N_8.json').exists():
                run([*command,'--count','8','--verify','--output-dir',str(check)],check,env,paths)
            assert json.loads((check/'run.command.json').read_text())['returncode']==0
            small=json.loads((check/'N_8.json').read_text())
            reference=json.loads((ROOT/f'results/raw/H100/{profile}/correctness/cpu_dct/J_8.json').read_text())
            assert small['sample_ids']==reference['sample_ids'] and small['predictions']==reference['predictions']
            assert small['checks'] and all(c['native_grid_and_full_source_input_exact'] for c in small['checks'])
        else:
            command+=['--route','J','--device','cuda','--physical-order','--workers','8']
            paths=[Path(s['path']) for s in label_rows]
        prefix='N' if route=='imagelanes' else 'J'
        concurrent=run([*command,'--count','50000','--output-dir',str(output)],output,env,paths)
        raw=output/f'{prefix}_50000.json'
        data=json.loads(raw.read_text())
        reference=json.loads((ROOT/f'results/raw/H100/{profile}/inference/cpu_dct/trial_1/J_50000.json').read_text())
        assert data['samples']==50000 and len(data['sample_ids'])==50000 and set(data['sample_ids'])==set(labels)
        assert data['sample_ids']==reference['sample_ids']
        assert Path(data['checkpoint']).resolve()==Path(reference['checkpoint']).resolve()
        correct=sum(pred==labels[i] for i,pred in zip(data['sample_ids'],data['predictions']))
        assert abs(data['top1']-100*correct/50000)<1e-9
        agreement=sum(a==b for a,b in zip(data['predictions'],reference['predictions']))/50000
        if agreement!=1:
            (output/'failure.json').write_text(json.dumps(dict(reason='Full matched DCT prediction gate failed',agreement=agreement,raw_result=str(raw))))
            subprocess.run([sys.executable,'fastlens/benchmarks/development/reports/deployment_report.py'],check=True)
            continue
        meta=json.loads((output/'run.command.json').read_text())
        family='MobileNetV2' if profile.startswith('mobilenet') else 'ResNet-50' if profile.startswith('resnet') else 'eFUN'
        model=family+'-DCT'+profile[-2:] if profile!='efun' else 'eFUN-DCT'
        accepted=dict(timestamp=meta['timestamp'],git_commit=meta['git_commit'],gpu='RTX 4090' if gpu=='4090' else gpu,
            architecture_family=family,model=model,model_variant=model,domain='DCT',checkpoint=data['checkpoint'],
            pipeline='FastLens' if route=='imagelanes' else 'CPU-DCT',comparison_type='deployment' if route=='imagelanes' else 'matched_dct',
            task='inference',precision=data['precision'],batch_size=64,trial=1,samples=50000,
            elapsed_s=data['e2e_seconds'],images_per_s=50000/data['e2e_seconds'],top1=data['top1'],top5=data['top5'],
            correctness='PASS: full IDs, checkpoint, exact predictions, recomputed top1; native prefix tensor control',
            raw_result=str(raw.relative_to(ROOT)),concurrent_load_observed=concurrent,
            notes='ImageNet-512; CPU/memory NUMA 0; workers 8 for CPU-DCT; model threads 8; warm pages; concurrent load annotated, isolation not claimed.')
        (output/'accepted.json').write_text(json.dumps(accepted,indent=2)+'\n')
        subprocess.run([sys.executable,'fastlens/benchmarks/development/reports/deployment_report.py'],check=True)
        print('ACCEPTED',gpu,profile,route,accepted['images_per_s'],flush=True)


if __name__=='__main__': main()
