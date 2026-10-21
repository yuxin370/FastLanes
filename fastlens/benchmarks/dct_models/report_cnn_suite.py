"""Summarize completed CNN artifacts without treating running jobs as results."""
import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT/'fastlens/data/system_rgbnomore/e2e_v3'
CONFIGS = [('MobileNetV2 DCT-24', 'dctnet_mobilenet24', 'rtx4090_io_pushdown_20260914'),
           ('MobileNetV2 DCT-32', 'dctnet_mobilenet32', 'rtx4090_cnn_complete_20260914'),
           ('ResNet-50 DCT-24', 'dctnet_static24', 'rtx4090_io_pushdown_20260914'),
           ('ResNet-50 DCT-64', 'dctnet_static64', 'rtx4090_cnn_complete_20260914')]
PATHS = [('RGB PyTorch', 'RGB_pytorch_50000.json'), ('RGB DALI', 'RGB_dali_50000.json'),
         ('R', 'R_50000.json'), ('O', 'O_50000.json'),
         *[(f'N {layout} {pushdown}', f'native_{layout}_{pushdown}/N_50000.json')
           for layout in ('grid', 'projected') for pushdown in ('off', 'on')]]


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def number(value, places=3):
    return 'not measured' if value is None else f'{value:.{places}f}'


def contended(run, directory, suffix):
    profile = directory.replace('dctnet_static', 'resnet').replace('dctnet_', '')
    path = DATA/'runs/dctnet_mobilenet24'/run/'queue'/f'{profile}_{suffix}.gpu_occupancy.json'
    return path.exists() and bool(json.loads(path.read_text())['competing_pids'])


def training_rows(name, route, folder):
    """The same epoch columns for archived runs and newly supplied baselines."""
    final = json.loads((folder/'training.json').read_text()) if (folder/'training.json').exists() else {}
    rows = []
    for epoch_file in sorted(folder.glob('epoch_*.json')):
        r = json.loads(epoch_file.read_text())
        v = r.get('validation', {})
        rows.append(dict(model=name if not route.startswith('rgb_') else name.split(' DCT-')[0]+' RGB',
            route=route, epoch=r['epoch']+1, samples=r['samples'], unique_samples=r['unique_samples'],
            full_epoch=r['full_epoch'], e2e_seconds=r['seconds'], images_per_second=r['images_per_second'],
            input_wait_seconds=r['input_wait_seconds'], model_stream_seconds=r['model_stream_seconds'],
            training_ce=r['ce'], validation_samples=v.get('samples'), top1=v.get('top1'), top5=v.get('top5'), ce=v.get('ce'),
            nvml_process_peak_mib=final.get('nvml_process_peak_mib'),
            torch_peak_allocated_bytes=final.get('torch_peak_allocated'), cpu_peak_rss_kib=final.get('cpu_peak_rss_kib'),
            full_run_complete=bool(final), result=str(epoch_file), input_backend=final.get('input_backend'),
            condition=final.get('condition'), baseline_store=final.get('baseline_store'),
            correctness_evidence=final.get('correctness_evidence'), headline_eligible=final.get('headline_eligible'),
            device=final.get('device'), device_uuid=final.get('device_uuid'), precision=final.get('precision'),
            initial_model_hash=final.get('initial_model_hash'), training_manifest=final.get('training_manifest')))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=DATA/'runs/dctnet_mobilenet24/rtx4090_cnn_complete_20260914')
    p.add_argument('--inference-run', default='rtx4090_cnn_memory_20260914')
    p.add_argument('--additional-training', type=Path, nargs='*', default=[], help='append completed training.json/epoch_*.json run directories to the existing tables')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inference, training, probes, memory, data_only, traces = [], [], [], [], [], []
    for name, directory, run in CONFIGS:
        for route, relative in PATHS:
            path = DATA/'runs'/directory/run/'full'/relative
            measured = DATA/'runs'/directory/args.inference_run/'full'/relative
            metric = relative.split('/')[0] if '/' in relative else relative.removesuffix('_50000.json')
            metric_path = measured.parents[1] / f'memory_{metric}.json' if '/' in relative else measured.parent / f'memory_{metric}.json'
            if measured.exists() and metric_path.exists() and not contended(args.inference_run, directory, 'inference_memory'):
                m = json.loads(metric_path.read_text())
                if m['returncode'] == 0:
                    path = measured
                    memory.append(dict(model=name, route=route, nvml_process_peak_mib=m['nvml_process_peak_mib'],
                        main_rss_bytes=m['cpu_peaks']['main_rss_bytes'], tree_rss_sum_bytes=m['cpu_peaks']['tree_rss_sum_bytes'], result=str(metric_path)))
            if not path.exists():
                continue
            r = json.loads(path.read_text())
            shards = r.get('native_shards', [])
            requested = sum(s['compressed_payload_bytes_read'] for s in shards) if shards else r.get('logical_read_bytes', r.get('source_bytes'))
            inference.append(dict(model=name, route=route, samples=r['samples'], top1=r['top1'], top5=r['top5'], ce=r['ce'],
                e2e_seconds=r['e2e_seconds'], images_per_second=r['samples']/r['e2e_seconds'],
                input_wait_seconds=r.get('input_wait_seconds'), model_seconds=r['model_seconds'],
                requested_file_bytes=requested, h2d_payload_bytes=sum(s['workset_upload_dma_bytes'] for s in shards) if shards else None,
                torch_peak_allocated_bytes=r.get('peak_cuda_allocated_bytes'),
                native_peak_in_use_bytes=max((s['galp_native_device_peak_in_use_bytes'] for s in shards), default=None),
                agreement_with_R=r.get('prediction_agreement_with_R'), source_version='imagenet_512',
                result=str(path), checkpoint=r['checkpoint']))
        folders = [('B6', DATA/'runs'/directory/'rtx4090_native_training_optimized_20260914/B6_full_m4_inplace'),
                   ('A0 JPEG', DATA/'runs'/directory/'rtx4090_cnn_training_baselines_20260914/jpeg_full')]
        if directory.endswith('24'):
            folders += [(backend, DATA/'runs'/directory/'rtx4090_cnn_training_baselines_20260914'/f'{backend}_full')
                        for backend in ('rgb_pytorch', 'rgb_d2', 'rgb_d3')]
        for route, folder in folders:
            training.extend(training_rows(name, route, folder))
        probe_path=DATA/'runs'/directory/'rtx4090_native_training_optimized_20260914/B6_probe_m4_inplace/training.json'
        if probe_path.exists():
            r=json.loads(probe_path.read_text())
            probes.append(dict(model=name, images=r['epochs'][0]['samples'], nvml_process_peak_mib=r['nvml_process_peak_mib'],
                torch_peak_allocated_bytes=r['torch_peak_allocated'], torch_peak_reserved_bytes=r['torch_peak_reserved'],
                cpu_peak_rss_kib=r['cpu_peak_rss_kib'], result=str(probe_path)))
        for path in sorted((DATA/'runs'/directory/'rtx4090_cnn_training_baselines_20260914').glob('*_data/training.json')):
            r = json.loads(path.read_text()); epoch = r['epochs'][0]
            data_only.append(dict(model=name, route=path.parent.name.removesuffix('_data'), samples=epoch['samples'],
                seconds=epoch['seconds'], images_per_second=epoch['images_per_second'], input_wait_seconds=epoch['input_wait_seconds'],
                nvml_process_peak_mib=r['nvml_process_peak_mib'], result=str(path)))
        for path in sorted((DATA/'runs'/directory/'rtx4090_cnn_training_baselines_20260914').glob('*_profile/breakdown.json')):
            r = json.loads(path.read_text())
            traces.append(dict(model=name, workload='training', route=path.parent.name.removesuffix('_profile'), samples=r['captured_images'],
                window_seconds=r['window_seconds'], gpu_input_seconds=r['groups'].get('input', {}).get('union_seconds', 0),
                gpu_model_seconds=r['groups'].get('model', {}).get('union_seconds', 0), gpu_idle_seconds=r['gpu_idle_seconds'],
                overlap_seconds=r['input_model_overlap_seconds'], result=str(path)))
        for path in sorted((DATA/'runs'/directory/args.inference_run/'profiles').glob('*/breakdown.json')):
            if contended(args.inference_run, directory, f'inference_trace_{path.parent.name}'):
                continue
            for r in json.loads(path.read_text()).values():
                traces.append(dict(model=name, workload='inference', route=path.parent.name, samples=r['captured_images'],
                    window_seconds=r['window_seconds'], gpu_input_seconds=r['gpu'].get('input', {}).get('union_seconds', 0),
                    gpu_model_seconds=r['gpu'].get('model', {}).get('union_seconds', 0), gpu_idle_seconds=r['gpu_idle_seconds'],
                    overlap_seconds=r['model_input_overlap_seconds'], result=str(path)))
    for folder in args.additional_training:
        final = json.loads((folder/'training.json').read_text())
        name = final['backbone']+' '+final['representation']
        if final['representation'] == 'DCT':
            name += '-'+str(len(final['output_channels']))
        store = final.get('baseline_store')
        route = store['method']+' '+final['condition'] if store else final['input_backend']+' '+final['condition']
        training.extend(training_rows(name, route, folder))
    write_csv(args.output_dir/'cnn_inference.csv', inference)
    write_csv(args.output_dir/'cnn_training.csv', training)
    write_csv(args.output_dir/'cnn_training_memory_probes.csv', probes)
    write_csv(args.output_dir/'cnn_inference_memory.csv', memory)
    write_csv(args.output_dir/'cnn_data_only.csv', data_only)
    write_csv(args.output_dir/'cnn_trace_breakdown.csv', traces)
    text = ['# CNN inference and training results', '',
        'Only completed measurements are included. Compare results on the same GPU and with '
        'matched input and model settings; the CSV files retain the source paths and device identities.', '',
        f'Inference measurement directory: `{args.inference_run}`. Contended runs are excluded; '
        'the CSV result column identifies the measurement used for each row.', '',
        '## Inference', '',
        '| Model | Route | Top-1 % | Top-5 % | CE | E2E s | Input wait s | Model s | Requested GB |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in inference:
        text.append('| '+ ' | '.join([r['model'], r['route'],number(r['top1']),number(r['top5']),number(r['ce']),
            number(r['e2e_seconds']),number(r['input_wait_seconds']),number(r['model_seconds']),
            number(r['requested_file_bytes']/1e9 if r['requested_file_bytes'] is not None else None)])+' |')
    text += ['', 'GB is decimal. Requested file bytes include reads served by the page cache. '
        'Timing components overlap and must not be added to reconstruct end-to-end time.', '',
        '## Training', '',
        '| Model | Route | Epoch | Images | E2E s | Images/s | Training CE | Val Top-1 % | Val Top-5 % | Val CE | Run complete |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in training:
        text.append('| '+' | '.join([r['model'],r['route'],str(r['epoch']),str(r['samples']),number(r['e2e_seconds']),
            number(r['images_per_second']),number(r['training_ce']),number(r['top1']),number(r['top5']),number(r['ce']),
            'yes' if r['full_run_complete'] else 'no'])+' |')
    text += ['', 'Epoch 1 includes initial compilation and initialization. Validation is timed separately. '
        'Early epochs during warmup do not establish final convergence.', '',
        '## Memory probes', '',
        'Probe runs use 8,192 training and 1,000 validation images; their peaks are not full-epoch peaks. '
        'NVML includes native allocations. Main-process RSS excludes worker memory.', '',
        '| Model | NVML peak MiB | Torch allocated GiB | Torch reserved GiB | Main RSS GiB |',
        '|---|---:|---:|---:|---:|']
    for r in probes:
        text.append('| '+' | '.join([r['model'],str(r['nvml_process_peak_mib']),number(r['torch_peak_allocated_bytes']/2**30),
            number(r['torch_peak_reserved_bytes']/2**30),number(r['cpu_peak_rss_kib']/2**20)])+' |')
    text += ['', '## Full-run memory', '',
        '| Model | Route | NVML MiB | Main RSS GiB | Process-tree RSS sum GiB |',
        '|---|---|---:|---:|---:|']
    for r in memory:
        text.append(f'| {r["model"]} | {r["route"]} | {number(r["nvml_process_peak_mib"])} | {number(r["main_rss_bytes"]/2**30)} | {number(r["tree_rss_sum_bytes"]/2**30)} |')
    text += ['', 'NVML peaks use 100 ms sampling. Summed RSS counts shared pages more than once and is not PSS.', '',
        '| Training model | Route | NVML MiB | Torch allocated GiB | Main RSS GiB |', '|---|---|---:|---:|---:|']
    for r in training:
        if r['epoch'] == 2 and r['full_run_complete']:
            text.append(f'| {r["model"]} | {r["route"]} | {number(r["nvml_process_peak_mib"])} | {number(r["torch_peak_allocated_bytes"]/2**30)} | {number(r["cpu_peak_rss_kib"]/2**20)} |')
    text += ['', '## Data-only and matched traces', '',
        '| Model | Input route | Samples | Data-only s | Images/s |', '|---|---|---:|---:|---:|']
    for r in data_only:
        text.append(f'| {r["model"]} | {r["route"]} | {r["samples"]} | {number(r["seconds"])} | {number(r["images_per_second"])} |')
    text += ['', 'Data-only runs cover five pools and include initial data setup. '
        'GPU active times below are interval unions, not SM utilization.', '',
        '| Model | Workload | Route | Samples | Window s | GPU input s | Model s | Idle s | Input/model overlap s |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in traces:
        text.append('| '+' | '.join([r['model'], r['workload'], r['route'], str(r['samples']),
            *[number(r[k]) for k in ('window_seconds','gpu_input_seconds','gpu_model_seconds','gpu_idle_seconds','overlap_seconds')]])+' |')
    text += ['', '## Paired early validation', '',
        '| Model | Epoch | B6 minus A0 Top-1 pp | B6 minus A0 CE |', '|---|---:|---:|---:|']
    for name, _, _ in CONFIGS:
        for epoch in (1, 2):
            paired = {r['route']:r for r in training if r['model'] == name and r['epoch'] == epoch}
            if 'B6' in paired and 'A0 JPEG' in paired:
                a, b = paired['A0 JPEG'], paired['B6']
                text.append(f'| {name} | {epoch} | {number(b["top1"]-a["top1"])} | {number(b["ce"]-a["ce"])} |')
    text += ['', 'This comparison covers a single seed and two warmup epochs.', '',
        f'Detailed inference table: `{args.output_dir/"cnn_inference.csv"}`.', '',
        f'Completed epoch table: `{args.output_dir/"cnn_training.csv"}`.', '']
    (args.output_dir/'CNN_E2E_RESULTS.md').write_text('\n'.join(text))
    print(json.dumps(dict(inference_rows=len(inference), completed_epoch_rows=len(training), output=str(args.output_dir))))


if __name__ == '__main__':
    main()
