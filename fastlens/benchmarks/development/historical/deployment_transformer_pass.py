#!/usr/bin/env python3
"""Complete only missing registered transformer inference passes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import numpy as np
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.common import gpu_uuid


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['deployment','matched'],default='deployment')
    args=parser.parse_args()
    jobs=[('4090','swinv2',p) for p in ('pytorch','dali','galp')]+[('H100','vitti','coordl')] if args.stage=='deployment' else [('4090','swinv2','rgbnomore')]
    for gpu,key,pipeline in jobs:
        output=ROOT/f'results/p0/deployment_passes/{gpu}/{key}/{pipeline}'
        if pipeline=='coordl' and (output/'failure.json').exists():
            output=output.with_name('coordl_after_cleanup')
        if (output/'accepted.json').exists(): continue
        output.mkdir(parents=True,exist_ok=False)
        source=ROOT/f'results/raw/H100/{key}/inference'
        contract=json.loads((source/'contract.json').read_text())
        contract['execution'].update(repeats=1,aggregate_exclude_first_repeat=False,warmup_batches=0)
        contract['pipelines']['enabled']=[pipeline]
        if pipeline=='coordl': contract['pipelines']['coordl']['file_list']=str(output/'coordl_file_list.txt')
        cp=output/'contract.json'
        cp.write_text(json.dumps(contract,indent=2)+'\n')
        samples=json.loads(Path(contract['dataset']['sample_manifest']).read_text())['samples']
        paths=[Path(s['path']) for s in samples]
        if pipeline=='galp':
            paths=[Path(p['path']) for p in contract['pipelines']['galp']['payload_fingerprints']]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu_uuid(gpu),TMPDIR=str(Path.home()/'tmp'),
            PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        raw=output/f'pipeline_{pipeline}.json'
        driver=[sys.executable,'fastlens/benchmarks/development/historical/swin_inference_once.py'] if key=='swinv2' else [str(Path.home()/'tmp/coordl-build-cu128/coordl-python'),'-m','fastlens.benchmarks.system_dct_major.pipeline']
        command=['numactl','--cpunodebind=0','--membind=0',*driver,'--pipeline',pipeline,'--contract',str(cp),'--output',str(raw)]
        concurrent=run(command,output,env,paths)
        data=json.loads(raw.read_text())
        measured=data['repeats'][0]
        reference=json.loads((source/f'pipeline_{"rgbnomore" if pipeline=="galp" else "pytorch" if pipeline=="coordl" else pipeline}.json').read_text())
        assert measured['images']==50000 and measured['sample_trace']==reference['repeats'][1]['sample_trace']
        domain='DCT' if pipeline in ('galp','rgbnomore') else 'RGB'
        assert data['model']['checkpoint_sha256']==contract['models'][domain.lower()]['checkpoint_sha256']
        semantic=np.load(data['semantic_artifact'])
        expected=np.load(reference['semantic_artifact'])
        for field in (('prediction_ordinals','prediction_labels') if key=='swinv2' else ('ordinals','labels')):
            assert np.array_equal(semantic[field],expected[field])
        agreement=float(np.mean(semantic['top1_predictions']==expected['top1_predictions']))
        full_labels=semantic['prediction_labels'] if key=='swinv2' else np.array([s['label'] for s in samples])
        correct=int(np.sum(semantic['top1_predictions']==full_labels))
        assert correct==measured['correct_top1']
        if domain=='DCT' and agreement!=1:
            (output/'failure.json').write_text(json.dumps(dict(reason='Matched DCT prediction gate failed',agreement=agreement)))
            continue
        if domain=='DCT':
            for field in ('input_0','input_1'):
                delta=np.abs(semantic[field]-expected[field])
                assert delta.max()<=.001 and delta.mean()<=.0001
        meta=json.loads((output/'run.command.json').read_text())
        family='SwinV2-T' if key=='swinv2' else 'ViT-Ti'
        model=family+' '+domain
        accepted=dict(timestamp=meta['timestamp'],git_commit=meta['git_commit'],gpu='RTX 4090' if gpu=='4090' else gpu,
            architecture_family=family,model=model,model_variant=model,domain=domain,checkpoint=data['model']['checkpoint'],
            pipeline={'pytorch':'PyTorch','dali':'DALI','galp':'FastLens','coordl':'CoorDL','rgbnomore':'CPU-DCT'}[pipeline],
            comparison_type='matched_dct' if pipeline=='rgbnomore' else 'deployment',task='inference',precision='fp32',batch_size=50,trial=1,samples=50000,
            elapsed_s=measured['repeat_scope_seconds'],images_per_s=50000/measured['repeat_scope_seconds'],
            top1=measured['accuracy_top1']*100,top5=measured['accuracy_top5']*100,
            correctness='PASS: full sample identities/labels, checkpoint SHA, integer accuracy; DCT tensor/prediction gate where applicable',
            raw_result=str(raw.relative_to(ROOT))+'#/repeats/0',cross_gpu_prediction_agreement=agreement,
            concurrent_load_observed=concurrent,notes='ImageNet-512; CPU/memory NUMA 0; workers 8; warm page cache; other-GPU/host activity annotated in concurrent_load.jsonl; no isolation claim.')
        (output/'accepted.json').write_text(json.dumps(accepted,indent=2)+'\n')
        subprocess.run([sys.executable,'fastlens/benchmarks/development/reports/deployment_report.py'],check=True)
        print('ACCEPTED',gpu,key,pipeline,accepted['images_per_s'],flush=True)


if __name__=='__main__': main()
