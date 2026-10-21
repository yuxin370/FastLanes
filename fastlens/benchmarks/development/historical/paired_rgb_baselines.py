#!/usr/bin/env python3
"""Complete CoorDL/FFCV family coverage by reusing their validated RGB readers."""
import hashlib
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.common import gpu_uuid
from fastlens.benchmarks.system_dct_major.common import sha256_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--retry-efun-cpu-decoder',action='store_true')
    args=parser.parse_args()
    source=ROOT/'results/raw/H100/vitti/inference/contract.json'
    original=json.loads(source.read_text())
    stored=json.loads(Path(original['dataset']['sample_manifest']).read_text())
    by_id={s['sample_id']:s for s in stored['samples']}
    cnn_samples=json.loads((ROOT/'results/raw/H100/mobilenet24/inference/cpu_dct/trial_1/samples_50000.json').read_text())
    for gpu in ('4090','H100'):
        for profile in ('swinv2','mobilenet24','resnet24','efun'):
            if args.retry_efun_cpu_decoder and profile!='efun': continue
            family={'swinv2':'SwinV2-T','mobilenet24':'MobileNetV2','resnet24':'ResNet-50','efun':'eFUN'}[profile]
            model=family+' RGB' if profile!='efun' else 'EfficientNet-B0 reference'
            cnn=profile!='swinv2'
            if cnn:
                reference=json.loads((ROOT/f'results/raw/H100/{profile}/inference/rgb_pytorch/trial_1/RGB_pytorch_50000.json').read_text())
                checkpoint=reference['checkpoint']
                selected=[dict(by_id[s['logical_sample_id']],label=s['model_label']) for s in cnn_samples]
            else:
                reference=json.loads((ROOT/'results/raw/H100/swinv2/inference/pipeline_pytorch.json').read_text())
                checkpoint=reference['model']['checkpoint']
                swin_contract=json.loads(Path(reference['contract']).read_text())
                selected=[dict(by_id[s['sample_id']]) for s in json.loads(Path(swin_contract['dataset']['sample_manifest']).read_text())['samples']]
            indices=[s['ordinal'] for s in selected]
            remap=[0]*50000
            for i,s in enumerate(selected):
                remap[s['ordinal']]=i
                s.update(source_fastlens_image_id=s['galp_image_id'],ordinal=i,galp_image_id=i)
            for pipeline in ('ffcv','coordl'):
                if args.retry_efun_cpu_decoder and pipeline!='coordl': continue
                output=ROOT/f'results/p0/deployment_passes/{gpu}/{profile}/paired_{pipeline}'
                if args.retry_efun_cpu_decoder: output=output.with_name('paired_coordl_cpu_decode')
                if (output/'accepted.json').exists() or (output/'failure.json').exists(): continue
                output.mkdir(parents=True,exist_ok=True)
                contract=json.loads(source.read_text())
                batch=64 if cnn else 50
                crop=224 if cnn else 256
                contract['execution'].update(batch_size=batch,repeats=1,warmup_batches=0,measurement_batches=math.ceil(50000/batch),aggregate_exclude_first_repeat=False)
                contract['models']['rgb'].update(architecture=model,checkpoint=checkpoint,
                    checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest())
                manifest=dict(stored,samples=selected,provenance={'source':str(source),'purpose':'RGB only; logical identities and checkpoint label mapping preserved; fastlens_image_id is an unused sequential adapter slot'})
                mp=output/'samples.json'
                mp.write_text(json.dumps(manifest)+'\n')
                contract['dataset'].update(sample_manifest=str(mp),sample_manifest_sha256=sha256_json(manifest),label_mapping='sorted WNID ImageFolder model labels' if cnn else 'RGB-no-more checkpoint labels')
                contract['preprocess']['profile']='checkpoint-normal-rgb'
                contract['preprocess']['rgb'].update(resize_shorter=256,crop_origin=[(256-crop)//2]*2,crop_size=[crop]*2,
                    torch_interpolation='bicubic' if profile=='efun' else 'bilinear',antialias=True,
                    model_mean=[.485,.456,.406] if cnn else [.5]*3,model_std=[.229,.224,.225] if cnn else [.5]*3)
                contract['pipelines']['enabled']=[pipeline]
                contract['pipelines']['coordl']['file_list']=str(output/'coordl_file_list.txt')
                contract['pipelines']['ffcv'].update(indices=indices,ordinal_remap=remap)
                contract['pipelines'][pipeline]['resize_backend']='Full RGB decode; Torch GPU antialiased resize, uint8 rounding, center crop, checkpoint normalization'
                if args.retry_efun_cpu_decoder:
                    contract['pipelines'][pipeline]['decode_backend']='CoorDL CPU libjpeg (use_fast_idct=False), then H2D; cache unchanged'
                cp=output/'contract.json'
                if cp.exists():
                    contract=json.loads(cp.read_text())
                    selected=json.loads(Path(contract['dataset']['sample_manifest']).read_text())['samples']
                else:
                    cp.write_text(json.dumps(contract,indent=2)+'\n')
                raw=output/f'pipeline_{pipeline}.json'
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu_uuid(gpu),DCTNET_PROFILE=profile,TMPDIR=str(Path.home()/'tmp'),
                    PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
                interpreter=str(Path.home()/'tmp/coordl-build-cu128/coordl-python') if pipeline=='coordl' else sys.executable
                command=['numactl','--cpunodebind=0','--membind=0',interpreter,'fastlens/benchmarks/development/historical/paired_rgb_once.py',
                    '--profile',profile,'--pipeline',pipeline,'--contract',str(cp),'--output',str(raw)]
                paths=[Path(contract['pipelines']['ffcv']['beton'])] if pipeline=='ffcv' else [Path(s['path']) for s in selected]
                try:
                    if not raw.exists(): run(command,output,env,paths)
                except subprocess.CalledProcessError as error:
                    (output/'failure.json').write_text(json.dumps(dict(status='BLOCKED',returncode=error.returncode,reason='See run.log'),indent=2)+'\n')
                    continue
                meta=json.loads((output/'run.command.json').read_text())
                assert meta['returncode']==0
                data=json.loads(raw.read_text()); measured=data['repeats'][0]
                semantic=np.load(data['semantic_artifact'])
                assert measured['images']==50000 and len(measured['sample_trace'])==50000
                assert data['model']['checkpoint_sha256']==contract['models']['rgb']['checkpoint_sha256']
                labels=np.array([s['label'] for s in selected])
                assert measured['correct_top1']==int(np.sum(semantic['top1_predictions']==labels))
                expected=[]
                for s in selected[:len(semantic['input_0'])]:
                    with Image.open(s['path']) as image:
                        x=image.convert('RGB').resize((256,256),Image.Resampling.BICUBIC if profile=='efun' else Image.Resampling.BILINEAR)
                        border=(256-crop)//2
                        x=np.array(x.crop((border,border,border+crop,border+crop))).transpose(2,0,1).astype(np.float32)/255
                    mean=np.array(contract['preprocess']['rgb']['model_mean'],np.float32)[:,None,None]
                    std=np.array(contract['preprocess']['rgb']['model_std'],np.float32)[:,None,None]
                    expected.append((x-mean)/std)
                difference=np.abs(semantic['input_0']-np.array(expected))*std*2
                diagnostic=dict(rgb_range_input_max_abs=float(difference.max()),rgb_range_input_mean_abs=float(difference.mean()))
                if difference.max()>.25 or difference.mean()>.008:
                    (output/'failure.json').write_text(json.dumps(dict(status='BLOCKED',reason='Original RGB input equivalence bounds failed',**diagnostic),indent=2)+'\n')
                    continue
                accepted=dict(timestamp=meta['timestamp'],git_commit=meta['git_commit'],gpu='RTX 4090' if gpu=='4090' else gpu,
                    architecture_family=family,model=model,model_variant=model,domain='RGB',checkpoint=checkpoint,
                    pipeline='FFCV' if pipeline=='ffcv' else 'CoorDL',comparison_type='deployment',task='inference',precision='fp32',
                    batch_size=batch,trial=1,samples=50000,elapsed_s=measured['repeat_scope_seconds'],images_per_s=50000/measured['repeat_scope_seconds'],
                    top1=measured['accuracy_top1']*100,top5=measured['accuracy_top5']*100,
                    correctness='PASS: full sample identity/labels; checkpoint SHA; top1 recomputation; normal PIL input reference',
                    raw_result=str(raw.relative_to(ROOT))+'#/repeats/0',preprocessing_diagnostics=diagnostic,
                    notes='Same RGB checkpoint and sample order as PyTorch RGB. NUMA0; workers8; model threads '+str(8 if cnn else 1)+'. GPU antialiased resize/crop/normalization after full RGB decode; loader-specific implementation recorded. Concurrent-load trace retained; no isolation claim.')
                if args.retry_efun_cpu_decoder:
                    accepted['notes']+=' CoorDL CPU libjpeg decoder; previous mixed decoder failed unchanged input-equivalence bounds and remains excluded.'
                (output/'accepted.json').write_text(json.dumps(accepted,indent=2)+'\n')
                subprocess.run([sys.executable,'fastlens/benchmarks/development/reports/deployment_report.py'],check=True)
                print('ACCEPTED',gpu,profile,pipeline,accepted['images_per_s'],flush=True)


if __name__=='__main__': main()
