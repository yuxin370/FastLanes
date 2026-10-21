"""Wait for Milestone A, populate the paper, and run authorized Phase 2 in background."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
ROOT=Path(__file__).resolve().parents[4]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-pid',type=int,required=True)
    args=parser.parse_args()
    if psutil.pid_exists(args.after_pid):
        process=psutil.Process(args.after_pid)
        assert any(x.endswith('finish_milestone_a.py') for x in process.cmdline())
        print('Waiting without interrupting Milestone A PID',args.after_pid,flush=True)
        while process.is_running() and process.status()!=psutil.STATUS_ZOMBIE:
            time.sleep(20)
    assert json.loads((ROOT/'results/paper/milestone_a.json').read_text())['status']=='COMPLETE', 'Milestone A remains incomplete; do not start Phase 2'
    out=ROOT/'results/phase2'
    out.mkdir(exist_ok=True)
    # Pin paper generation/build to NUMA 1; the benchmark policy uses NUMA 0.
    with (ROOT/'results/paper/manuscript_population.log').open('x') as log:
        populated=subprocess.run(['taskset','-c','24',sys.executable,'fastlens/benchmarks/development/reports/populate_paper.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    with (out/'queue.log').open('x') as log:
        phase2=subprocess.Popen([sys.executable,'-u','fastlens/benchmarks/development/historical/run_phase2.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (out/'launcher.json').write_text(json.dumps(dict(pid=phase2.pid,command=[sys.executable,'-u','fastlens/benchmarks/development/historical/run_phase2.py'],milestone_a='results/paper/milestone_a.json'),indent=2)+'\n')
    build=Path.home()/'tmp/fastlens-milestone-a-paper'
    compiled=None
    if populated.returncode==0:
        with (ROOT/'results/paper/manuscript_build.log').open('x') as log:
            compiled=subprocess.run(['taskset','-c','24','bash','fastlens/docs/paper/build_paper.sh',str(build)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    state=dict(populate_returncode=populated.returncode,build_returncode=compiled.returncode if compiled else None,
        pdf=str(build/'main.pdf'),phase2_pid=phase2.pid,
        status='BUILT_REVIEW_PENDING' if compiled and compiled.returncode==0 else 'BLOCKED',
        notes='Human-readable figure/layout and evidence review still required; build success alone is not editorial completion.')
    (ROOT/'results/paper/manuscript_status.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state),flush=True)


if __name__=='__main__': main()
