"""Authorized Phase 2: missing headline repeats, then two fixed matched controls."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]


def main():
    assert json.loads((ROOT/'results/paper/milestone_a.json').read_text())['status']=='COMPLETE'
    output=ROOT/'results/phase2'
    output.mkdir(exist_ok=True)
    state=dict(phase='RUNNING',stages=[])
    def stage(name,command):
        state['active']=name
        (output/'status.json').write_text(json.dumps(state,indent=2)+'\n')
        result=subprocess.run([sys.executable,'fastlens/benchmarks/execution/run_logged.py','--log',str(output/(name+'.log')),'--',*command],cwd=ROOT)
        state['stages'].append(dict(name=name,returncode=result.returncode,status='DONE' if result.returncode==0 else 'BLOCKED',raw_log=str(output/(name+'.log'))))
        (output/'status.json').write_text(json.dumps(state,indent=2)+'\n')
        return result.returncode==0
    stage('inference_repetitions',[sys.executable,'fastlens/benchmarks/development/historical/run_phase2_repeats.py'])
    stage('frequency_control',[sys.executable,'fastlens/benchmarks/development/historical/run_minimal_ablations.py','--phase2-control','frequency'])
    prepared=ROOT/'results/preparation/raw_int16_phase2_16384'
    if (prepared/'preparation.json').exists():
        prepared_ok=True
    else:
        prepared_ok=stage('raw_int16_preparation',[sys.executable,'fastlens/benchmarks/paper/section_6_4/compressed_transfer_and_tensor/prepare_compression_control.py',
            '--output-root',str(prepared),'--shard-ids',*map(str,range(16))])
    if prepared_ok:
        stage('raw_int16_control',[sys.executable,'fastlens/benchmarks/development/historical/run_minimal_ablations.py','--phase2-control','raw_int16','--prepared',str(prepared)])
    repetitions=json.loads((output/'repetition_status.json').read_text()) if (output/'repetition_status.json').exists() else {}
    state.update(phase='COMPLETE' if all(s['status']=='DONE' for s in state['stages']) and repetitions.get('status')=='COMPLETE' else 'BLOCKED_CELLS',active=None)
    (output/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    subprocess.run([sys.executable,'fastlens/benchmarks/development/reports/deployment_report.py'],cwd=ROOT,check=True)
    print(json.dumps(state),flush=True)


if __name__=='__main__': main()
