"""Finish the known hardlink-metadata repair after the active serial queue exits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'results/version_refresh_20260929'


def main():
    parent = int(sys.argv[1])
    created = float(sys.argv[2])
    while psutil.pid_exists(parent):
        process = psutil.Process(parent)
        if process.create_time() != created or process.status() == psutil.STATUS_ZOMBIE: break
        time.sleep(10)
    with (OUT/'hardlink_followup.log').open('x') as stream:
        result = subprocess.run([sys.executable, '-u', 'fastlens/benchmarks/development/historical/refresh_evaluation.py'], cwd=ROOT,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode or json.loads((OUT/'status.json').read_text())['status'] != 'COMPLETE':
        (OUT/'paper_update_status.json').write_text(json.dumps(dict(status='BLOCKED', note='Refresh has unresolved cells; manuscript not regenerated automatically.')))
        return
    from fastlens.benchmarks.development.historical.refresh_evaluation import full_swin
    for gpu in ('H100','4090'):
        (OUT/'full_epoch_status.json').write_text(json.dumps(dict(status='RUNNING',gpu=gpu)))
        full_swin(gpu)
    (OUT/'full_epoch_status.json').write_text(json.dumps(dict(status='COMPLETE')))
    env = dict(os.environ, MPLCONFIGDIR=str(Path.home()/'tmp/fastlens-matplotlib'))
    with (OUT/'paper_update.log').open('x') as stream:
        for script in ('fastlens/benchmarks/development/reports/export_milestone_a.py','fastlens/benchmarks/development/reports/populate_paper.py'):
            subprocess.run([sys.executable, script], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        subprocess.run(['bash','fastlens/docs/paper/scripts/build_paper.sh',str(Path.home()/'tmp/fastlens-version-refresh-paper')],
                       cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    (OUT/'paper_update_status.json').write_text(json.dumps(dict(status='BUILT_REVIEW_PENDING',
        pdf=str(Path.home()/'tmp/fastlens-version-refresh-paper/main.pdf')),indent=2)+'\n')


if __name__ == '__main__': main()
