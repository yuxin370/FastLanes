#!/usr/bin/env python3
"""Replace legacy ViT RGB fixed-crop runs with checkpoint-normal evaluation."""
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
    for gpu in ('4090','H100'):
        for pipeline in ('pytorch','dali','ffcv','coordl'):
            output=ROOT/f'results/p0/deployment_passes/{gpu}/vitti/normal_{pipeline}'
            if (output/'accepted.json').exists() or (output/'failure.json').exists(): continue
            output.mkdir(parents=True,exist_ok=True)
            contract=json.loads((ROOT/'results/raw/H100/vitti/inference/contract.json').read_text())
            contract['execution'].update(repeats=1,warmup_batches=0,aggregate_exclude_first_repeat=False)
            contract['preprocess']['profile']='checkpoint-normal-rgb-resize256-center224'
            contract['preprocess']['rgb'].update(resize_shorter=256,crop_origin=[16,16],
                interpolation='bilinear_antialias',normalization='[-1,1]',normal_config='/home/tangyuxin/RGB-no-more/datasets.py:328')
            contract['pipelines']['enabled']=[pipeline]
            contract['pipelines']['coordl']['file_list']=str(output/'coordl_file_list.txt')
            contract['pipelines'][pipeline]['model_warmup_batches']=20
            if pipeline in ('ffcv','coordl'):
                contract['pipelines'][pipeline]['resize_backend']='Torch GPU bilinear antialias, uint8 rounding, center crop; full 512 RGB decode. Installed CoorDL lacks antialias and FFCV default resized crop uses a different filter.'
            cp=output/'contract.json'
            cp.write_text(json.dumps(contract,indent=2)+'\n')
            samples=json.loads(Path(contract['dataset']['sample_manifest']).read_text())['samples']
            paths=[Path(s['path']) for s in samples]
            if pipeline=='ffcv': paths=[Path(contract['pipelines']['ffcv']['beton'])]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu_uuid(gpu),PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch',
                OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR=str(Path.home()/'tmp'))
            interpreter=str(Path.home()/'tmp/coordl-build-cu128/coordl-python') if pipeline=='coordl' else sys.executable
            raw=output/f'pipeline_{pipeline}.json'
            command=['numactl','--cpunodebind=0','--membind=0',interpreter,'fastlens/benchmarks/development/historical/vit_inference_once.py',
                '--pipeline',pipeline,'--contract',str(cp),'--output',str(raw)]
            try:
                if raw.exists():
                    prior=json.loads((output/'run.command.json').read_text())
                    assert prior['returncode']==0
                    concurrent=any(any('pid' in o for o in json.loads(line)['observations']) for line in (output/'concurrent_load.jsonl').read_text().splitlines())
                else:
                    concurrent=run(command,output,env,paths)
            except subprocess.CalledProcessError as error:
                (output/'failure.json').write_text(json.dumps(dict(status='BLOCKED',returncode=error.returncode,reason='See run.log; continue other baseline systems'),indent=2)+'\n')
                continue
            data=json.loads(raw.read_text())
            measured=data['repeats'][0]
            semantic=np.load(data['semantic_artifact'])
            labels=np.array([s['label'] for s in samples])
            assert measured['images']==50000 and (len(measured['sample_trace']) if isinstance(measured['sample_trace'],list) else measured['sample_trace']['count'])==50000
            assert len(semantic['top1_predictions'])==50000
            assert measured['correct_top1']==int(np.sum(semantic['top1_predictions']==labels))
            assert data['model']['checkpoint_sha256']==contract['models']['rgb']['checkpoint_sha256']
            diagnostics={}
            if pipeline!='pytorch' or gpu!='4090':
                reference=json.loads((ROOT/'results/p0/deployment_passes/4090/vitti/normal_pytorch/pipeline_pytorch.json').read_text())
                expected=np.load(reference['semantic_artifact'])
                assert measured['sample_trace']==reference['repeats'][0]['sample_trace']
                difference=np.abs(semantic['input_0']-expected['input_0'])
                diagnostics=dict(input_max_abs=float(difference.max()),input_mean_abs=float(difference.mean()),
                    prediction_agreement=float(np.mean(semantic['top1_predictions']==expected['top1_predictions'])))
                # Reuse the artifact's existing RGB numerical bounds; do not tune after observations.
                if diagnostics['input_max_abs']>.25 or diagnostics['input_mean_abs']>.008:
                    (output/'failure.json').write_text(json.dumps(dict(status='BLOCKED',reason='RGB input equivalence gate failed',**diagnostics),indent=2)+'\n')
                    continue
            meta=json.loads((output/'run.command.json').read_text())
            accepted=dict(timestamp=meta['timestamp'],git_commit=meta['git_commit'],gpu='RTX 4090' if gpu=='4090' else gpu,
                architecture_family='ViT-Ti',model='ViT-Ti RGB',model_variant='ViT-Ti RGB',domain='RGB',
                checkpoint=data['model']['checkpoint'],pipeline={'pytorch':'PyTorch','dali':'DALI','ffcv':'FFCV','coordl':'CoorDL'}[pipeline],
                comparison_type='deployment',task='inference',precision='fp32',batch_size=50,trial=1,samples=50000,
                elapsed_s=measured['repeat_scope_seconds'],images_per_s=50000/measured['repeat_scope_seconds'],
                top1=measured['accuracy_top1']*100,top5=measured['accuracy_top5']*100,
                correctness='PASS: full identities/labels, checkpoint SHA, recomputed top1; original RGB input tolerance',
                raw_result=str(raw.relative_to(ROOT))+'#/repeats/0',concurrent_load_observed=concurrent,
                preprocessing_diagnostics=diagnostics,
                notes='ImageNet-512, checkpoint-normal Resize256/CenterCrop224. NUMA0, workers8. Twenty excluded resident model warmup batches; warm source pages. Concurrent load annotated. '+str(contract['pipelines'][pipeline].get('resize_backend','Native loader evaluation preprocessing.')))
            (output/'accepted.json').write_text(json.dumps(accepted,indent=2)+'\n')
            subprocess.run([sys.executable,'fastlens/benchmarks/development/reports/deployment_report.py'],check=True)
            print('ACCEPTED NORMAL',gpu,pipeline,accepted['top1'],accepted['images_per_s'],flush=True)


if __name__=='__main__': main()
