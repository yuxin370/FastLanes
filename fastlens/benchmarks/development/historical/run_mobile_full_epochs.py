"""Four short-warmup matched B6 MobileNet24 epochs; correctness gates precede timing."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.common import gpu_uuid


def read(p): return json.loads(p.read_text())


def main():
    root=ROOT/'fastlens/data/compressed/imagenet512_train_block_major_premixed'
    material=read(root/'materialization_contract.json')
    for gpu in ('H100','4090'):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu_uuid(gpu),DCTNET_PROFILE='mobilenet24',
            PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch',TMPDIR=str(Path.home()/'tmp'),
            OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8',FASTLENS_NSYS_FINE='0')
        gate=ROOT/f'results/p1/mobile_b6/{gpu}/correctness'
        gate.mkdir(parents=True,exist_ok=True)
        if not (gate/'gate.json').exists():
            run(['numactl','--cpunodebind=0','--membind=0',sys.executable,'fastlens/benchmarks/development/diagnostics/check_mobile_training.py',
                '--output',str(gate/'gate.json')],gate,env,None)
        assert read(gate/'gate.json')['passed']
        for pipeline in ('imagelanes','cpu_dct'):
            output=ROOT/f'results/p1/mobile_b6/{gpu}/{pipeline}'
            if (output/'accepted.json').exists(): continue
            output.mkdir(parents=True,exist_ok=True)
            command=['numactl','--cpunodebind=0','--membind=0',sys.executable,'-m',
                'fastlens.benchmarks.dct_models.training_pls','--manifest',str(root/'dct/manifest.bin'),
                '--mapping',material['ordered_mapping'],'--mapping-sha256',material['ordered_mapping_sha256'],
                '--condition','B6','--input-backend','native' if pipeline=='imagelanes' else 'jpeg',
                '--epochs','1','--short-warmup','--validation-count','0','--workers','8',
                '--transform-blocks-per-launch','32768','--transform-ctas-per-launch','512',
                '--compile-mode','default','--output-dir',str(output)]
            if not (output/'training.json').exists(): run(command,output,env,None)
            data=read(output/'training.json')
            epoch,=data['epochs']
            assert epoch['full_epoch'] and epoch['samples']==epoch['unique_samples']==1281167
            assert epoch['optimizer_updates']==1252 and math.isfinite(epoch['final_loss'])
            assert data['initial_model_hash']==data['warmup']['initial_model_hash']
            assert data['warmup']['initial_state_restored']
            if pipeline=='cpu_dct':
                native=read(output.parent/'fastlens/training.json')
                for key in ('initial_model_hash','precision','microbatch','accumulation','condition','output_channels'):
                    assert data[key]==native[key],key
                assert epoch['sample_order_digest']==native['epochs'][0]['sample_order_digest']
            (output/'accepted.json').write_text(json.dumps(dict(correctness_passed=True,
                raw_result=str((output/'training.json').relative_to(ROOT)),tensor_gate=str(gate/'gate.json'),
                comparison_type='matched_dct',initial_model_hash=data['initial_model_hash'],
                notes='One full B6 epoch; exact sample coverage and paired order digest; prefix tensor/target gate; short excluded warmup; no convergence claim.'),indent=2)+'\n')
            subprocess.run([sys.executable,'fastlens/benchmarks/development/historical/replan.py'],check=True)
            subprocess.run([sys.executable,'fastlens/benchmarks/development/reports/deployment_report.py'],check=True)


if __name__=='__main__': main()
