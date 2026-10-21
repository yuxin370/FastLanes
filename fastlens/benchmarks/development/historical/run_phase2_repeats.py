"""Add only missing headline deployment passes, reusing accepted configurations."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.system_dct_major.common import parse_manifest


def read(path): return json.loads(Path(path).read_text())


def predictions(data):
    if 'repeats' in data:
        return np.load(data['semantic_artifact'])['top1_predictions'].tolist()
    return data['predictions']


def main():
    assert read(ROOT/'results/paper/milestone_a.json')['status']=='COMPLETE'
    with (ROOT/'results/p0/deployment_inference_summary.csv').open() as stream:
        summaries=list(csv.DictReader(stream))
    templates=[(p,read(p)) for p in (ROOT/'results/p0/deployment_passes').glob('*/*/*/accepted.json') if 'phase2_' not in p.parent.name]
    failures=[]
    for summary in summaries:
        if int(summary['trials'])>=3: continue
        key=tuple(summary[k] for k in ('gpu','model','pipeline'))
        candidates=[(p,r) for p,r in templates if tuple(r[k] for k in ('gpu','model','pipeline'))==key and r['raw_result'] in summary['raw_results'].split(';')]
        # The repaired MobileNet path and CPU-decoded eFUN path are the accepted templates.
        candidates.sort(key=lambda item:item[0].stat().st_mtime)
        template,accepted=candidates[-1]
        metadata=read(template.parent/'run.command.json')
        original=read(ROOT/accepted['raw_result'].split('#')[0])
        profile=template.parent.parent.name
        for trial in range(int(summary['trials'])+1,4):
            prefix='normal_' if profile=='vitti' and accepted['domain']=='RGB' else ''
            directory=template.parent.parent/f'{prefix}phase2_{accepted["pipeline"].lower()}_trial_{trial}'
            if (directory/'accepted.json').exists(): continue
            if (directory/'failure.json').exists():
                failures.append(str(directory)); break
            directory.mkdir(parents=True,exist_ok=True)
            command=list(metadata['command'])
            env=dict(os.environ,**{k:v for k,v in metadata['environment'].items() if v is not None})
            if '--output' in command:
                raw=directory/Path(command[command.index('--output')+1]).name
                command[command.index('--output')+1]=str(raw)
            else:
                raw=directory/Path(accepted['raw_result'].split('#')[0]).name
                command[command.index('--output-dir')+1]=str(directory)
            if '--contract' in command:
                contract=read(command[command.index('--contract')+1])
                if 'coordl' in contract['pipelines']:
                    contract['pipelines']['coordl']['file_list']=str(directory/'coordl_file_list.txt')
                target=directory/'contract.json'
                target.write_text(json.dumps(contract,indent=2)+'\n')
                command[command.index('--contract')+1]=str(target)
                if accepted['pipeline']=='FFCV':
                    paths=[Path(contract['pipelines']['ffcv']['beton'])]
                elif accepted['pipeline']=='FastLens':
                    paths=[Path(s['fls_path']) for s in parse_manifest(Path(contract['pipelines']['galp']['manifest']))['shards']]
                else:
                    paths=[Path(s['path']) for s in read(contract['dataset']['sample_manifest'])['samples']]
            elif accepted['domain']=='DCT':
                source=Path(command[command.index('--new-data')+1])
                paths=[Path(s['fls_path']) for s in parse_manifest(source/'manifest.bin')['shards']]
            else:
                samples=read(ROOT/f'results/raw/H100/{profile}/inference/rgb_pytorch/trial_1/samples_50000.json') if (ROOT/f'results/raw/H100/{profile}/inference/rgb_pytorch/trial_1/samples_50000.json').exists() else read(ROOT/'results/raw/H100/mobilenet24/inference/cpu_dct/trial_1/samples_50000.json')
                paths=[Path(s['path']) for s in samples]
            try:
                if not raw.exists(): run(command,directory,env,paths)
            except subprocess.CalledProcessError as error:
                failure=dict(status='BLOCKED',reason='Repeat command failed; no throughput accepted',returncode=error.returncode)
                (directory/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
                failures.append(str(directory)); break
            data=read(raw)
            measured=data['repeats'][0] if 'repeats' in data else data
            old=original['repeats'][0] if 'repeats' in original else original
            ids=measured['sample_trace'] if 'repeats' in data else data['sample_ids']
            old_ids=old['sample_trace'] if 'repeats' in original else original['sample_ids']
            samples=measured['images'] if 'repeats' in data else data['samples']
            checkpoint=data['model']['checkpoint_sha256'] if 'repeats' in data else data['checkpoint']
            old_checkpoint=original['model']['checkpoint_sha256'] if 'repeats' in original else original['checkpoint']
            top1=measured['correct_top1'] if 'repeats' in data else data['top1']
            old_top1=old['correct_top1'] if 'repeats' in original else original['top1']
            if samples!=50000 or ids!=old_ids or predictions(data)!=predictions(original) or checkpoint!=old_checkpoint or top1!=old_top1:
                failure=dict(status='BLOCKED',reason='Repeated configuration differs in sample IDs, predictions, checkpoint or count; no throughput accepted',reference=accepted['raw_result'])
                (directory/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
                failures.append(str(directory)); break
            elapsed=measured['repeat_scope_seconds'] if 'repeats' in data else data['e2e_seconds']
            meta=read(directory/'run.command.json')
            result=dict(accepted,timestamp=meta['timestamp'],git_commit=meta['git_commit'],trial=trial,
                elapsed_s=elapsed,images_per_s=50000/elapsed,
                top1=measured['accuracy_top1']*100 if 'repeats' in data else data['top1'],
                top5=measured['accuracy_top5']*100 if 'repeats' in data else data['top5'],
                raw_result=str(raw.relative_to(ROOT))+('#/repeats/0' if 'repeats' in data else ''),
                correctness='PASS: 50000 samples; full IDs, predictions and checkpoint equal to accepted same-GPU configuration',
                notes=accepted['notes']+' Phase 2 repeat; full predictions matched the accepted same-GPU configuration. Concurrent-load trace retained.')
            result.pop('concurrent_load_observed',None)
            (directory/'accepted.json').write_text(json.dumps(result,indent=2)+'\n')
            subprocess.run([sys.executable,'fastlens/benchmarks/development/reports/deployment_report.py'],check=True)
            print('ACCEPTED',key,'trial',trial,flush=True)
    output=ROOT/'results/phase2'
    output.mkdir(exist_ok=True)
    (output/'repetition_status.json').write_text(json.dumps(dict(status='COMPLETE' if not failures else 'BLOCKED_CELLS',failed_directories=failures),indent=2)+'\n')


if __name__=='__main__': main()
