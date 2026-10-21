"""Compare legacy-rebatched and native batch64 DALI inputs outside formal timing."""
import sys, json, copy, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import cv2
cv2.setNumThreads(1)
import torch
from fastlens.benchmarks.system_dct_major import pipeline as M
from fastlens.benchmarks.paper.shared.remaining_baseline_inference import read
from fastlens.benchmarks.paper.shared.remaining_baseline_inference import reference_predictions

def main(out):
    refpath=ROOT/'results/remaining_baselines_20261012/reference_checks/H100/vitti/dali/reference.json'
    ref=read(refpath);contract=read(ref['contract']);samples=read(contract['dataset']['sample_manifest'])['samples'];expected=reference_predictions(refpath)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda');stream=torch.cuda.Stream(priority=torch.cuda.Stream.priority_range()[1]);stream.wait_stream(torch.cuda.current_stream());torch.cuda.set_stream(stream)
    model=M.build_workload_model(rgbnomore_root=Path(contract['models']['rgbnomore_root']),checkpoint=Path(contract['models']['rgb']['checkpoint']),device=device,domain='rgb',workload='evaluation').eval()
    sources=[]
    for size in [50,64]:
        c=copy.deepcopy(contract);c['execution'].update(batch_size=size,workers=8,precision='fp32',warmup_batches=0)
        source=M.DaliAdapter(c,samples,device,'dali');source.begin_repeat();sources.append(source)
    rows=[];pred50=[];pred64=[];carry=None;pos=0
    with torch.inference_mode():
        for _ in range(20):model(torch.zeros(64,3,224,224,device=device))
        for first in range(0,50000,64):
            n=min(64,50000-first)
            while carry is None or len(carry)<n:
                chosen=samples[pos:pos+50];batch=sources[0].load(chosen);x=batch.inputs[0].clone();carry=x if carry is None else torch.cat((carry,x));pos+=len(chosen)
            x50=carry[:n];carry=carry[n:]
            batch=sources[1].load(samples[first:first+n]);x64=batch.inputs[0]
            delta=(x50-x64).abs().flatten(1);maxima=delta.amax(1).cpu().tolist();means=delta.mean(1).cpu().tolist()
            y50=model(x50);y64=model(x64)
            a=y50.topk(5,dim=1).indices.cpu().tolist();b=y64.topk(5,dim=1).indices.cpu().tolist();pred50.extend(a);pred64.extend(b)
            for i,s in enumerate(samples[first:first+n]):
                if maxima[i] or a[i][0]!=b[i][0] or b[i][0]!=expected[s['sample_id']]:
                    rows.append(dict(sample_id=s['sample_id'],input_max=maxima[i],input_mean=means[i],prediction50=a[i][0],prediction64=b[i][0],expected=expected[s['sample_id']]))
            stream.synchronize()
            if first%6400==0:print('PROGRESS',first,flush=True)
    for source in sources:source.close()
    m50=[s['sample_id'] for s,p in zip(samples,pred50) if p[0]!=expected[s['sample_id']]]
    m64=[dict(sample_id=s['sample_id'],actual=p[0],expected=expected[s['sample_id']]) for s,p in zip(samples,pred64) if p[0]!=expected[s['sample_id']]]
    failed=read(ROOT/'results/remaining_baselines_20261012/inference/H100/vitti/dali/trial_0/prediction_mismatch.json')['mismatches']
    diagnosis=dict(gpu=torch.cuda.get_device_name(),images=50000,reference=str(refpath),batch50_reference_mismatches=m50,batch64_reference_mismatches=m64,batch64_reproduces_formal_mismatch=m64==failed,input_changed_samples=sum(r['input_max']>0 for r in rows),input_max_abs=max(r['input_max'] for r in rows) if rows else 0,rows=rows)
    (out/'diagnosis.json').write_text(json.dumps(diagnosis,indent=2)+'\n')
    assert not m50,f'{len(m50)} reference mismatches for batch50 input'
    assert m64==failed,'Could not reproduce formal mismatch exactly'
    candidate=dict(ref,predictions=pred64,batch=64,decoder_batch=64,reference_input_batch=64,previous_reference=str(refpath),diagnosis=str(out/'diagnosis.json'),gpu=torch.cuda.get_device_name(),scope='Semantic-only validation using actual batch64 DALI inputs; no throughput result')
    (out/'reference.json').write_text(json.dumps(candidate)+'\n')
    print('DIAGNOSIS',json.dumps({k:v for k,v in diagnosis.items() if k not in ['rows','batch50_reference_mismatches','batch64_reference_mismatches']}),flush=True)

if __name__=='__main__':
    out=ROOT/'results/remaining_baselines_20261012/reference_checks/H100/vitti/dali_batch64'
    if '--worker' in sys.argv:
        main(out)
    else:
        from fastlens.benchmarks.execution.inference_load import run
        out.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-d6e80e78-00d8-8e4b-0c81-a403bebe0d76',PYTHONPATH=str(ROOT),TMPDIR=str(Path.home()/'tmp'),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',FASTLENS_EVALUATION_SNAPSHOT=str(ROOT/'results/remaining_baselines_20261012/runtime_snapshot.json'))
        env.pop('CUBLAS_WORKSPACE_CONFIG',None)
        run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker'],out,env,None)
