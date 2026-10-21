"""One matched crop pair and one matched construction pair; stop before Phase 2."""
import csv
import argparse
import datetime
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.common import gpu_uuid
from fastlens.benchmarks.common import write_csv
from fastlens.benchmarks.system_dct_major.common import parse_manifest


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase2-control',choices=['frequency','raw_int16'])
    parser.add_argument('--prepared',type=Path)
    args=parser.parse_args()
    source = ROOT/'fastlens/data/compressed/imagenet512_val_block_major'
    crop,k=448,64
    root=ROOT/'results/ablation/minimal/4090'
    cases=[('crop_on_direct','vector-range-read-selected-decode','projected',source),
           ('crop_off_direct','full-source-decode','projected',source),
           ('crop_on_grid','vector-range-read-selected-decode','grid',source)]
    pairs=[('crop_off_direct','crop_on_direct'),('crop_on_grid','crop_on_direct')]
    groups=[('crop_pushdown',pairs[0]),('direct_vs_grid',pairs[1])]
    if args.phase2_control=='frequency':
        crop,k=512,32
        root=ROOT/'results/ablation/phase2/4090/frequency'
        cases=[('frequency_selective','vector-range-read-selected-decode','projected',source),
               ('frequency_superset','full-source-decode','projected',source)]
        pairs=[('frequency_superset','frequency_selective')]
        groups=[('frequency_pushdown_matched',pairs[0])]
    elif args.phase2_control=='raw_int16':
        if not args.prepared: parser.error('--prepared is required for raw_int16')
        preparation=read(args.prepared/'preparation.json')
        assert preparation['samples']==16384
        root=ROOT/'results/ablation/phase2/4090/raw_int16'
        cases=[(name,'vector-range-read-selected-decode','projected',Path(preparation[key]))
               for name,key in [('compressed','compressed'),('raw_int16','raw_int16')]]
        pairs=[('raw_int16','compressed')]
        groups=[('fastlanes_vs_raw_int16',pairs[0])]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid('4090'), DCTNET_PROFILE='mobilenet24',
               PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch', TMPDIR=str(Path.home()/'tmp'),
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', FASTLENS_NSYS_FINE='0')
    outputs = {}
    rows = {}
    for name, mode, layout, source in cases:
        paths = [Path(s['fls_path']) for s in parse_manifest(source/'manifest.bin')['shards']]
        base = root/name
        common = [sys.executable, 'fastlens/benchmarks/paper/shared/cnn_control.py', '--crop-size', str(crop), '--source-coefficients', str(k),
                  '--new-data', str(source), '--input-geometry', 'source512', '--pushdown', 'off',
                  '--crop-execution-mode', mode, '--output-layout', layout, '--native-runtime', 'b6',
                  '--shards-per-activation', '1', '--physical-prefix', '--batch-size', '64',
                  '--baseline-dir', str(base/'unused')]
        for stage in ('correctness', 'performance', 'diagnostic'):
            directory = base/stage
            directory.mkdir(parents=True, exist_ok=True)
            count = 8 if stage == 'correctness' else 16384
            target = directory/f'N_{count}.json'
            if not target.exists():
                command = [*common, '--count', str(count), '--output-dir', str(directory)]
                if stage == 'correctness':
                    command += ['--verify']
                if stage == 'diagnostic':
                    command += ['--diagnostic-window', str(directory/'window.json')]
                    command = ['nsys', 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
                               '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
                               '--output', str(directory/'capture'), *command]
                with (directory/'gpu_process_memory.csv').open('x') as telemetry:
                    monitor = subprocess.Popen(['nvidia-smi', '--query-compute-apps=timestamp,pid,gpu_uuid,used_gpu_memory',
                                                '--format=csv,noheader,nounits', '-lms', '100'], stdout=telemetry)
                    try:
                        run(['numactl', '--cpunodebind=0', '--membind=0', *command], directory, env, paths)
                    finally:
                        monitor.terminate()
                        monitor.wait()
            result = read(target)
            assert result['samples'] == len(set(result['sample_ids'])) == count
            if stage == 'correctness':
                assert result['checks'] and all(c['native_grid_and_full_source_input_exact'] for c in result['checks'])
            if stage == 'performance':
                outputs[name] = result
            if stage == 'diagnostic':
                assert result['sample_ids'] == outputs[name]['sample_ids']
                assert result['predictions'] == outputs[name]['predictions']
                database = directory/'capture.sqlite'
                if not database.exists():
                    subprocess.run([sys.executable, 'fastlens/benchmarks/execution/run_logged.py', '--log', str(directory/'export.log'), '--',
                                    'nsys', 'export', '--type', 'sqlite', '--output', str(database),
                                    str(directory/'capture.nsys-rep')], check=True, env=env)
        raw = outputs[name]
        diagnostic = base/'diagnostic'
        with sqlite3.connect(diagnostic/'capture.sqlite') as connection:
            (start, end), = connection.execute("SELECT n.start,n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE COALESCE(n.text,s.value)='control.measured_pass'").fetchall()
            copies = connection.execute('SELECT start,end,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=1 AND start<? AND end>?', (end, start)).fetchall()
            assert all(start <= a <= b <= end for a, b, _ in copies)
            kernels = connection.execute('SELECT s.value,SUM(k.end-k.start),COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id WHERE k.start>=? AND k.end<=? GROUP BY s.value', (start, end)).fetchall()
        write_csv(diagnostic/'kernels.csv', [dict(kernel=n, cumulative_ns=t, launches=c) for n,t,c in kernels], ['kernel','cumulative_ns','launches'])
        def total(key):
            values = [s.get(key) for s in raw['native_shards']]
            return sum(values) if all(v is not None for v in values) else None
        residency = []
        window = raw['control']
        with (base/'performance/gpu_process_memory.csv').open() as stream:
            for stamp, pid, uuid, mib in csv.reader(stream):
                ns = int(datetime.datetime.strptime(stamp.strip(), '%Y/%m/%d %H:%M:%S.%f').timestamp()*1e9)
                if int(pid) == window['pid'] and window['start_unix_ns'] <= ns <= window['end_unix_ns']:
                    residency.append(int(mib)*1024*1024)
        assert residency, 'No process residency observations in the control interval'
        metadata = read(base/'performance/run.command.json')
        rows[name] = dict(timestamp=metadata['timestamp'], git_commit=metadata['git_commit'], gpu=raw['device_name'],
            model='MobileNetV2-DCT24', domain='DCT', checkpoint=raw['checkpoint'], pipeline='FastLens',
            comparison_type='ablation', task='inference', precision='fp32', batch_size=64, trial=1, samples=16384,
            case=name, retained_fraction=(crop/512)**2, source_coefficients=k, output_layout=layout,
            storage_payload_bytes=sum(p.stat().st_size for p in paths),
            requested_source_blocks=total('requested_source_block_count'), selected_vectors=total('selected_vector_count'),
            dependency_bytes=total('selected_compressed_payload_bytes'), issued_read_bytes=total('rowgroup_storage_bytes_read'),
            h2d_bytes=sum(size for _,_,size in copies), h2d_copy_count=len(copies),
            h2d_memcpy_s=sum(b-a for a,b,_ in copies)/1e9,
            input_wait_s=raw['input_wait_seconds'], gather_s=raw['organization_seconds'],
            preprocessing_kernel_ms=total('planless_transform_gpu_kernel_ms'), native_decode_ms=total('decode_ms'),
            elapsed_s=raw['e2e_seconds'], images_per_s=16384/raw['e2e_seconds'],
            gpu_peak_process_bytes=max(residency), torch_peak_allocated_bytes=window['torch_peak_allocated_bytes'],
            torch_peak_reserved_bytes=window['torch_peak_reserved_bytes'], top1=raw['top1'], correctness='PASS',
            raw_result=str((base/'performance/N_16384.json').relative_to(ROOT)),
            diagnostic_raw=str((diagnostic/'capture.sqlite').relative_to(ROOT)),
            notes=f'One unprofiled warm-cache pass; H2D from separate diagnostic. Fixed {crop}x{crop} source crop, K{k}, physical-prefix sample IDs. Whole-process residency sampled at 100ms over wrapper interval; no sum of independent peaks. Native counters may not cover all input kernels; diagnostic kernels.csv preserves per-kernel durations.')
    for left,right in pairs:
        assert outputs[left]['sample_ids'] == outputs[right]['sample_ids']
        assert outputs[left]['predictions'] == outputs[right]['predictions']
        assert outputs[left]['checkpoint'] == outputs[right]['checkpoint']
    if not args.phase2_control:
        assert rows['crop_on_direct']['issued_read_bytes'] == rows['crop_on_grid']['issued_read_bytes']
    if args.phase2_control=='frequency':
        control=outputs['frequency_superset']['native_shards']
        selected=outputs['frequency_selective']['native_shards']
        assert all(s['planless_transform_selected_coefficient_count']==32 for s in control+selected)
        assert all(s['selected_coefficient_count']==s['full_coefficient_count'] for s in control), 'Control did not physically select K64'
        assert sum(s['selected_coefficient_count'] for s in selected)<sum(s['full_coefficient_count'] for s in selected), 'No selective frequency read demonstrated; do not attribute this to frequency pushdown'
    if args.phase2_control=='raw_int16':
        assert rows['raw_int16']['selected_vectors']==rows['compressed']['selected_vectors']
        assert outputs['raw_int16']['activation_count']==outputs['compressed']['activation_count']
    for filename, names in groups:
        write_csv(ROOT/f'results/ablation/{filename}.csv', [rows[n] for n in names], list(rows[names[0]]))
    (root/'accepted.json').write_text(json.dumps(dict(correctness_passed=True,
        samples=16384, matched_predictions=True, matched_sample_ids=True, fixed_construction_read_bytes=True if not args.phase2_control else None,
        cases=list(rows)), indent=2)+'\n')
    subprocess.run([sys.executable, 'fastlens/benchmarks/development/reports/deployment_report.py'], check=True)


if __name__ == '__main__':
    main()
