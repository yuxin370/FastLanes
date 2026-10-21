"""Index the current Evaluation's evidence without changing measurements or paper selections."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[4]
PAPER = ROOT / 'fastlens/docs/paper/sections/07_evaluation.tex'
OUT = ROOT / 'results/paper/evaluation_records'


def read_csv(path):
    with (ROOT / path).open() as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads((ROOT / path).read_text())


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def relative(path):
    return str(Path(path).relative_to(ROOT)) if Path(path).is_relative_to(ROOT) else str(path)


def model_key(name):
    name = name.replace(' (reference)', '').replace(' DCT', '')
    return {'MobileNetV2': 'MobileNetV2-DCT24', 'EfficientNet-B0': 'eFUN-DCT',
            'EfficientNet-B0 reference': 'eFUN-DCT',
            'eFUN': 'eFUN-DCT'}.get(name, name)


def main():
    OUT.mkdir(exist_ok=True)
    text = PAPER.read_text()
    active = '\n'.join(re.split(r'(?<!\\)%', line)[0] for line in text.splitlines())
    # Match the manuscript's actual values to existing cohorts, not to the best run.
    candidates = read_csv('results/paper/training_windows.csv')
    for row in candidates:
        row['source_summary'] = 'results/paper/training_windows.csv'
    for job in read_json('results/remaining_4090_windows_20261002/plan.json')['jobs']:
        candidates.append(dict(job['historical_summary'], source_summary=
                               'results/remaining_4090_windows_20261002/plan.json'))
    reserved = 'results/h100_reserved_windows_20261002/window_summary.csv'
    for row in read_csv(reserved):
        candidates.append(dict(row, gpu='H100', domain='DCT', pipeline='FastLens',
                               source_summary=reserved))
    block = active.split(r'\label{tab:training-window}', 1)[1].split(r'\end{tabular}', 1)[0]
    table_rows = []
    gpu = None
    for line in block.splitlines():
        heading = re.search(r'\\multirow\{\d+\}\{\*\}\{(RTX 4090|H100)\}', line)
        if heading:
            gpu = heading.group(1)
        cells = line.split('&')
        if len(cells) != 7 or cells[0].strip() == 'GPU':
            continue
        if cells[0].strip() in ('RTX 4090', 'H100'):
            gpu = cells[0].strip()
        assert gpu is not None
        table_rows.append((gpu, cells[1], '&'.join(cells[2:])))
    selected, trial_rows = [], []
    systems = ['PyTorch', 'DALI', 'CoorDL', 'FFCV', 'FastLens']
    for gpu, model, values in table_rows:
        model = model.strip()
        numbers = [int(v.replace(',', '')) for v in re.findall(r'\d[\d,]*', values)]
        assert len(numbers) == 5
        for system, shown in zip(systems, numbers):
            matches = [r for r in candidates if r['gpu'] == gpu and r['pipeline'] == system
                       and model_key(r['model']) == model_key(model)
                       and round(float(r['median_images_per_s'])) == shown]
            # DCT32 shares its RGB experiments with DCT24.
            if not matches and model_key(model) == 'MobileNetV2-DCT32' and system != 'FastLens':
                matches = [r for r in candidates if r['gpu'] == gpu and r['pipeline'] == system
                           and r['model'] == 'MobileNetV2'
                           and round(float(r['median_images_per_s'])) == shown]
            unique = {r['raw_results']: r for r in matches}
            if len(unique) != 1:
                raise ValueError(f'Unresolved manuscript cell: {gpu}/{model}/{system}/{shown}: {len(unique)} cohorts')
            summary = next(iter(unique.values()))
            paths = [relative(p) for p in summary['raw_results'].split(';')]
            rates = []
            for path in paths:
                d = read_json(path)
                assert d['images'] == 65536 and d['updates'] == 64
                assert d['correctness_passed'] and math.isfinite(d['training_loss']) and not d['profile']
                assert d['batch'] == 64 and d['accumulation'] == 16 and d['warmup_updates'] == 32
                rate = d['images'] / d['seconds']
                rates.append(rate)
                trial_rows.append(dict(table='tab:training-window', gpu=gpu, paper_workload=model,
                    pipeline=system, domain=d['domain'], trial=d['trial'], samples=d['images'],
                    optimizer_updates=d['updates'], elapsed_s=d['seconds'], images_per_s=rate,
                    precision=d['precision'], batch_size=d['batch'], effective_batch_size=1024,
                    warmup_updates=d['warmup_updates'], initial_model_hash=d['initial_model_hash'],
                    sample_order_sha256=d['sample_order_sha256'], timestamp=d['timestamp'],
                    git_commit=d['git_commit'], correctness='PASS', raw_result=path,
                    source_summary=summary['source_summary']))
            median = statistics.median(rates)
            assert len(rates) == 3 and round(median) == shown
            selected.append(dict(gpu=gpu, paper_workload=model, pipeline=system,
                domain='DCT' if system == 'FastLens' else 'RGB', comparison_type='deployment',
                paper_images_per_s=shown, median_images_per_s=median,
                min_images_per_s=min(rates), max_images_per_s=max(rates), trials=3,
                samples_per_trial=65536, median_elapsed_s=statistics.median(65536/r for r in rates),
                cv_percent=100*statistics.stdev(rates)/statistics.mean(rates),
                source_summary=summary['source_summary'], raw_results=';'.join(paths)))
    assert len(selected) == 50
    write_csv('training_window_selected.csv', selected)
    write_csv('training_window_trials.csv', trial_rows)

    # Keep valid later observations visible, including those not selected by the manuscript.
    other = []
    selected_paths = {r['raw_result'] for r in trial_rows}
    for cohort in ['remaining_4090_windows_20261002', 'h100_reserved_windows_20261002',
                   'h100_swin_window_recheck_20261003_final']:
        for row in read_csv(f'results/{cohort}/window_summary.csv'):
            paths = [relative(p) for p in row['raw_results'].split(';')]
            if not set(paths) <= selected_paths:
                other.append(dict(row, raw_results=';'.join(paths),
                    source_summary=f'results/{cohort}/window_summary.csv',
                    selection_status='VALID_NOT_SELECTED_IN_CURRENT_TEX',
                    note='Not invalidated by lower throughput; retained separately.'))
    write_csv('unselected_valid_retests.csv', other)

    # Ordered like the current Evaluation. Existing files remain in place.
    catalog = [
        ('Setup', 'tab:workloads', 'results/inventory/workload_pairs.csv', 'RGB/DCT pairing and checkpoint inventory'),
        ('Setup', 'sec:eval-setup', 'results/inventory/environment.json', 'Hardware and software inventory'),
        ('Setup', 'sec:eval-setup', 'results/inventory/baseline_configs.json', 'Baseline configuration inventory'),
        ('Training', 'tab:full-training;tab:full-deployment-training', 'results/p1/full_epoch_training.csv', 'One full matched epoch per configuration; 28 rows; representative FastLens epochs reused for deployment'),
        ('Training', 'tab:full-deployment-training', 'results/p1/full_epoch_rgb_deployment.csv', 'Accepted complete RGB epochs; missing configurations remain TBD'),
        ('Training', 'tab:training-window', 'results/paper/evaluation_records/training_window_selected.csv', 'Exact current TeX selection; 50 cells, 126 distinct trials'),
        ('Inference', 'fig:deployment-inference', 'results/p0/deployment_inference_summary.csv', 'Four RGB systems and FastLens; own checkpoints; one or three trials'),
        ('Inference', 'fig:deployment-inference', 'results/p0/deployment_ratios.csv', 'Derived deployment ratios; incomplete rows are not plotted'),
        ('Inference', 'tab:matched-inference', 'results/p0/matched_dct_inference.csv', '14 complete matched pairs plus unsupported rows'),
        ('Quality', 'tab:model-quality', 'results/quality/model_accuracy.csv', '12 checkpoints; full 50000-image accuracy'),
        ('Convergence', 'fig:training-convergence', 'results/a800_convergence_20261003/figures/convergence_observed.csv', 'Swin original run and MobileNet four-rank continuation; 300 epochs each, full validation; final and best checkpoints separate'),
        ('FFCV encoding', 'tab:ffcv-compression', 'results/ffcv_jpeg_20261005/comparison.csv', 'Complete three-trial raw/JPEG pairs; same RGB model, training windows or full validation'),
        ('Resources', 'tab:storage-cost', 'results/ffcv_jpeg_20261005/storage.csv', 'Raw/JPEG Beton bytes by population; external mapping excluded in this comparison'),
        ('Resources', 'tab:storage-cost', 'results/ffcv_jpeg_20261005/preparation/train/conversion.json', 'Training-window JPEG conversion record'),
        ('Resources', 'tab:storage-cost', 'results/ffcv_jpeg_20261005/preparation/val/conversion.json', 'Full-validation JPEG conversion record'),
        ('Resources', 'tab:storage-cost', 'results/ffcv_jpeg_20261005/preparation/train_full/conversion.json', 'Full-training JPEG conversion record'),
        ('Resources', 'tab:amortization', 'results/selectivity_4090_20261001/break_even_lower_bound.csv', 'Derived from measured conversion and full matched epochs; excludes index preparation'),
        ('Resources', 'tab:storage-cost;tab:preparation-cost', 'results/resources/storage.csv', 'Persistent representations and preparation; split/view must match'),
        ('Resources', 'tab:resource-cost', 'results/resources/h2d.csv', 'Separate CUPTI profiles, 16384 images; decimal MB/image'),
        ('Resources', 'tab:resource-cost', 'results/resources/memory.csv', 'Unprofiled memory sampling, 16384 images; binary GiB/MiB'),
        ('Mechanisms', 'fig:crop-pushdown;fig:core-ablations', 'results/ablation/crop_pushdown.csv', 'Fixed 448px crop; 16384 images; timing and profiling separate'),
        ('Mechanisms', 'fig:core-ablations', 'results/ablation/direct_vs_grid.csv', 'Fixed reads; direct output versus full grids'),
        ('Mechanisms', 'fig:core-ablations', 'results/ablation/fastlanes_vs_raw_int16.csv', 'Fixed logical request; compressed versus raw int16'),
        ('Selectivity', 'fig:selectivity-spatial', 'results/selectivity_4090_20261001/spatial.csv', '30 measured passes; trial 0 excluded'),
        ('Selectivity', 'fig:selectivity-spatial', 'results/selectivity_4090_20261001/spatial_summary.csv', 'Three-trial medians and paired ranges'),
        ('Selectivity', 'Frequency and joint selectivity', 'results/selectivity_4090_20261001/channels.csv', 'Model-input-changing K sweep; not matched model semantics'),
        ('Selectivity', 'Frequency and joint selectivity', 'results/selectivity_4090_20261001/joint.csv', 'Crop/K pairs; spatial control at fixed K'),
        ('Locality', 'fig:selectivity-locality', 'results/selectivity_4090_20261001/locality_2g/locality.csv', '4096 images; common 2 GiB budget; model-free'),
        ('Locality', 'fig:selectivity-locality', 'results/selectivity_4090_20261001/locality_summary.csv', 'Use locality_2g cohort, not earlier locality directory'),
        ('Frequency', 'tab:compact-frequency', 'results/compact_projection_20261001/frequency_summary.csv', 'Block-descriptor output; 4096 images; matched coefficients'),
        ('Frequency', 'Frequency selectivity and placement cost', 'results/compact_projection_20261001/diagnostic/h2d.csv', 'Separate CUPTI capture; total versus workset H2D'),
        ('Frequency', 'Prepared-input model-inclusive frequency control', 'results/ablation/prepared_frequency_retry_20261006/performance_raw.csv', 'MobileNetV2-DCT24; prepared 112px coefficients; legacy reader; 16384 images; three unprofiled trials per arm; separate CUPTI H2D'),
        ('I/O', 'Read coalescing', 'results/selectivity_4090_20261001/coalescing.csv', 'Exact versus bounded ranges; warm payload'),
        ('I/O', 'Cold payload access', 'results/selectivity_4090_20261001/cold.csv', 'Verified cold payload; process reads are not controller traffic'),
    ]
    index, records = [], []
    ref_keys = ('raw_result', 'raw_results', 'jpeg_results', 'diagnostic_raw', 'gpu_samples', 'cpu_raw', 'fastlens_raw')
    for section, element, path, scope in catalog:
        source = ROOT / path
        rows = read_csv(path) if source.suffix == '.csv' else []
        index.append(dict(section=section, paper_element=element, source_file=path,
                          rows=len(rows) if rows else '', scope=scope))
        for number, row in enumerate(rows, 1):
            refs = [(key, ref) for key in ref_keys for ref in row.get(key, '').split(';') if ref]
            if 'locality_2g/locality.csv' in path:
                refs = [('raw_result', str(Path(path).parent /
                        f"{row['policy']}_b{row['batch_size']}_{row['layout']}_{row['trial']}.json"))]
            if 'compact_projection' in path and '/diagnostic/' in path:
                refs = [('diagnostic_raw', 'results/compact_projection_20261001/diagnostic/capture.sqlite'),
                        ('native_raw', 'results/compact_projection_20261001/diagnostic/native.json')]
            for role, ref in refs:
                ref = relative(ref)
                file, _, pointer = ref.partition('#')
                raw = ROOT / file
                exists = raw.is_file()
                pointer_status = ''
                if pointer and exists:
                    node = json.loads(raw.read_text())
                    for token in pointer.lstrip('/').split('/'):
                        token = token.replace('~1', '/').replace('~0', '~')
                        node = node[int(token)] if isinstance(node, list) else node[token]
                    pointer_status = 'RESOLVED'
                command_path = raw.parent / 'run.command.json'
                meta = json.loads(command_path.read_text()) if command_path.exists() else {}
                records.append(dict(section=section, paper_element=element, source_file=path,
                    source_row=number, raw_role=role, raw_result=ref,
                    file_status='EXISTS' if exists else 'MISSING', json_pointer=pointer,
                    pointer_status=pointer_status,
                    command_record=relative(command_path) if command_path.exists() else '',
                    command=meta.get('shell_command', row.get('command', '')),
                    git_commit=meta.get('git_commit', row.get('git_commit', '')),
                    timestamp=meta.get('timestamp', row.get('timestamp', '')),
                    source_row_json=json.dumps(row, ensure_ascii=False)))
    write_csv('source_index.csv', index)
    write_csv('raw_record_index.csv', records)

    checks = []
    claim = re.search(r'Averaged across these configurations,(.*?)respectively\.', active, re.S).group(1)
    percentages = [float(v) for v in re.findall(r'(\d+\.\d+)\\%', claim)]
    assert len(percentages) == 4
    for system, paper_percent in zip(systems[:-1], percentages):
        gains = []
        for optimized in [r for r in selected if r['pipeline'] == 'FastLens']:
            baseline = next(r for r in selected if (r['gpu'], r['paper_workload'], r['pipeline']) ==
                            (optimized['gpu'], optimized['paper_workload'], system))
            gains.append(100*(optimized['median_images_per_s']/baseline['median_images_per_s']-1))
        calculated = statistics.mean(gains)
        checks.append(dict(claim=f'Training mean deployment improvement vs {system} (%)',
            manuscript_value=paper_percent, recomputed_value=calculated,
            formula='arithmetic mean of 10 per-row (DCT/RGB-1)*100; unrounded medians',
            status='MATCH' if round(calculated, 1) == paper_percent else 'MISMATCH',
            source='training_window_selected.csv'))
    inference_claim = re.search(
        r'average end-to-end speedup of (\d+\.\d+)\$\\times\$ on the RTX 4090 and (\d+\.\d+)\$\\times\$ on the H100', active).groups()
    for gpu, stated in zip(['RTX 4090', 'H100'], map(float, inference_claim)):
        ratios = [float(r['matched_system_speedup']) for r in read_csv('results/p0/matched_dct_inference.csv')
                  if r['gpu'] == gpu and r['correctness'] == 'PASS']
        checks.append(dict(claim=f'{gpu} mean matched inference speedup', manuscript_value=stated,
            recomputed_value=statistics.mean(ratios), formula='arithmetic mean of seven unrounded matched ratios',
            status='MATCH' if round(statistics.mean(ratios), 2) == stated else 'MISMATCH',
            source='results/p0/matched_dct_inference.csv'))
    epoch_claim = re.search(
        r'arithmetic-mean speedups across the seven workloads of (\d+\.\d+)\$\\times\$ on the RTX 4090 and (\d+\.\d+)\$\\times\$ on the H100', active).groups()
    for gpu, stated in zip(['RTX 4090', 'H100'], map(float, epoch_claim)):
        pairs = {}
        for row in read_csv('results/p1/full_epoch_training.csv'):
            if row['gpu'] == gpu:
                pairs.setdefault(row['model'], {})[row['pipeline']] = int(row['samples'])/float(row['elapsed_s'])
        mean = statistics.mean(r['FastLens']/r['CPU-DCT'] for r in pairs.values())
        checks.append(dict(claim=f'{gpu} mean full-epoch matched training speedup', manuscript_value=stated,
            recomputed_value=mean, formula='arithmetic mean of seven matched full-epoch ratios',
            status='MATCH' if round(mean, 2) == stated else 'MISMATCH',
            source='results/p1/full_epoch_training.csv'))
    write_csv('claim_checks.csv', checks)
    metadata = dict(manuscript=relative(PAPER), manuscript_sha256=hashlib.sha256(PAPER.read_bytes()).hexdigest(),
        ffcv_table_sha256=hashlib.sha256((ROOT / 'fastlens/docs/paper/sections/07_ffcv_compression_table.tex').read_bytes()).hexdigest(),
        scope='Records follow current manuscript selection; no measurement or canonical selection changed.',
        training_cells=len(selected), training_trial_references=len(trial_rows),
        distinct_training_trials=len(selected_paths), raw_references=len(records),
        missing_references=sum(r['file_status'] == 'MISSING' for r in records),
        valid_unselected_retest_groups=len(other))
    (OUT / 'record_status.json').write_text(json.dumps(metadata, indent=2)+'\n')
    # A self-contained set of selected result CSVs for this manuscript, not a
    # replacement of the global canonical summaries or the original raw files.
    (OUT / 'current_results').mkdir(exist_ok=True)
    window_table = []
    for gpu, workload in dict.fromkeys((r['gpu'], r['paper_workload']) for r in selected):
        row = dict(gpu=gpu, dct_workload=workload, comparison_type='deployment', samples=65536, trials=3)
        for system in systems:
            cell = next(r for r in selected if (r['gpu'], r['paper_workload'], r['pipeline']) == (gpu, workload, system))
            row[system] = cell['median_images_per_s']
            row[system + '_paper_display'] = cell['paper_images_per_s']
        window_table.append(row)
    write_csv('current_results/training_window_table.csv', window_table)
    write_csv('current_results/training_window_summary.csv', selected)
    write_csv('current_results/training_window_raw.csv', trial_rows)
    epochs = read_csv('results/p1/full_epoch_training.csv')
    for row in epochs:
        row['images_per_s'] = int(row['samples'])/float(row['elapsed_s'])
    write_csv('current_results/full_epoch_training.csv', epochs)
    # Keep the live paper status on the same selected cohorts as the manuscript.
    def table(headers, rows):
        return '\n'.join('| ' + ' | '.join(map(str, row)) + ' |'
                         for row in [headers, ['---'] * len(headers), *rows])

    window_rows = []
    for row in window_table:
        values = [row[system] for system in systems]
        window_rows.append([row['gpu'], row['dct_workload'],
                            *[f'{v:.1f}' for v in values], f'{values[-1]/max(values[:-1]):.3f}'])
    epoch_rates = {}
    for row in epochs:
        assert row['valid'] == row['strict_matched'] == 'True'
        assert int(row['samples']) == 1281167 and int(row['optimizer_updates']) == 1252
        epoch_rates[row['gpu'], row['model'], row['pipeline']] = row['images_per_s']
    matched_rows = []
    for gpu, model in dict.fromkeys((r['gpu'], r['model']) for r in epochs):
        baseline, optimized = [epoch_rates[gpu, model, p] for p in ('CPU-DCT', 'FastLens')]
        matched_rows.append([gpu, model, f'{baseline:.3f}', f'{optimized:.3f}', f'{optimized/baseline:.3f}'])
    rgb_epochs = read_csv('results/p1/full_epoch_rgb_deployment.csv')
    deployment_rows = []
    for gpu in ('RTX 4090', 'H100'):
        for rgb_model, dct_model in [('SwinV2-T', 'SwinV2-T'), ('MobileNetV2', 'MobileNetV2-DCT24')]:
            rates = []
            for system in systems[:-1]:
                found = [r for r in rgb_epochs if gpu in r['gpu'] and r['model'] == rgb_model and r['pipeline'] == system]
                assert len(found) <= 1
                if found:
                    row = found[0]
                    assert row['correctness'] == 'PASS' and int(row['samples']) == 1281167
                    assert int(row['optimizer_updates']) == 1252
                    rates.append(int(row['samples'])/float(row['elapsed_s']))
                else:
                    rates.append(None)
            optimized = epoch_rates[gpu, dct_model, 'FastLens']
            deployment_rows.append([gpu, dct_model, *['TBD' if v is None else f'{v:.1f}' for v in rates],
                                    f'{optimized:.1f}', 'TBD' if None in rates else f'{optimized/max(rates):.3f}'])
    write_csv('current_results/full_epoch_deployment_table.csv',
        [dict(zip(['gpu', 'dct_model', *systems, 'deployment_ratio'], row)) for row in deployment_rows])
    full_table = [r'\begin{table}[t]', r'\centering',
        r'\caption{Full-epoch deployment training throughput (images/s). Each cell measures one 1,281,167-image epoch. The four baselines use RGB; \sys{} uses SwinV2-T DCT or MobileNetV2-DCT24. FFCV stores raw RGB.}',
        r'\label{tab:full-deployment-training}',
        r'\fontsize{8}{9}\selectfont\setlength{\tabcolsep}{2pt}',
        r'\begin{tabular}{@{}lrrrrr@{}}', r'\toprule',
        r'Architecture & PyTorch & DALI & CoorDL & FFCV & \sys{} \\']
    for gpu in ('RTX 4090', 'H100'):
        full_table += [r'\midrule', rf'\multicolumn{{6}}{{l}}{{{gpu}}} \\']
        for row in deployment_rows:
            if row[0] == gpu:
                label = 'MobileNetV2' if row[1] == 'MobileNetV2-DCT24' else row[1]
                values = ['TBD' if v == 'TBD' else f'{float(v):,.0f}' for v in row[2:7]]
                full_table.append(' & '.join([label, *values]) + r' \\')
    full_table += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    PAPER.with_name('07_full_epoch_deployment_table.tex').write_text('\n'.join(full_table) + '\n')
    live = ROOT/'results/PAPER_STATUS.md'
    status = live.read_text()
    prefix, remainder = status.split('## D. Training deployment window', 1)
    _, suffix = remainder.split('\n## F.', 1)
    storage_start = suffix.index('| Representation |')
    storage_end = suffix.index('RTX 4090: separate 16,384-image training diagnostics.')
    storage_rows = [r for r in read_csv('results/resources/storage.csv')
                    if r['representation'] != 'coordl_jpeg_cache']
    suffix = suffix[:storage_start] + table(
        ['Representation', 'View/split', 'Samples', 'Storage bytes', 'Conversion s', 'Storage/JPEG'],
        [[r['representation'], r['dataset_view']+'/'+r['split'], r['samples'],
          r['representation_bytes'], r['conversion_s'] or 'TBD',
          f"{float(r['storage_amplification']):.3f}"] for r in storage_rows]) + (
        '\n\nStorage scope and conversion scope follow the source notes in `resources/storage.csv`. '
        'JPEG-Beton totals include its external sample manifest; the encoding comparison separately reports Beton-only bytes. '
        'CoorDL tmpfs is a runtime cache and remains separately recorded in the source CSV.\n\n') + suffix[storage_end:]
    prefix = prefix.split('## B. Main deployment inference', 1)[0]
    deployments = [r for r in read_csv('results/p0/deployment_ratios.csv') if r['dct_top1']]
    deployment_fields = ['gpu', 'architecture_family', 'dct_variant', 'rgb_top1', 'dct_top1',
                         *systems, 'fastest_rgb_throughput', 'deployment_ratio']
    matches = [r for r in read_csv('results/p0/matched_dct_inference.csv') if r['correctness'] == 'PASS']
    match_fields = ['gpu', 'dct_model', 'cpu_dct_images_per_s', 'imagelanes_images_per_s', 'matched_system_speedup']
    prefix += '\n\n'.join([
        '## B. Main deployment inference',
        'Images/s; selected medians from results/p0/deployment_ratios.csv. Each RGB family is reused across its DCT variants. Ratios compare complete deployments, not input-system speedup.',
        table(['GPU', 'Architecture', 'DCT variant', 'RGB Top-1', 'DCT Top-1', *systems, 'Fastest RGB', 'Deployment ratio'],
              [[r[k] if i < 3 else f'{float(r[k]):.3f}' for i, k in enumerate(deployment_fields)] for r in deployments]),
        '## C. Matched DCT inference',
        'Same-model/checkpoint comparisons from results/p0/matched_dct_inference.csv; correctness-passing selected pairs only.',
        table(['GPU', 'DCT model', 'CPU-DCT', 'FastLens', 'Matched system speedup'],
              [[r[k] if i < 2 else f'{float(r[k]):.3f}' for i, k in enumerate(match_fields)] for r in matches]),
        'Additional eFUN constructors (efun-l, efun-s, efun-s-plus) remain UNSUPPORTED: no configured evaluation checkpoint.',
    ]) + '\n\n'
    status = prefix + '\n\n'.join([
        '## D. Training deployment window',
        '65,536 measured images; median of three trials. These ten rows follow the current manuscript selection. RGB/DCT ratios describe deployment throughput.',
        table(['GPU', 'DCT workload', *systems, 'Deployment ratio'], window_rows),
        '## E. Full-epoch matched DCT training',
        'One complete measured epoch: 1,281,167 samples and 1,252 updates. Current accepted pairs from results/p1/full_epoch_training.csv; no convergence claim.',
        table(['GPU', 'DCT model', 'CPU-DCT', 'FastLens', 'System speedup'], matched_rows),
        '## E2. Full-epoch deployment training',
        'One full epoch per cell; source: results/p1/full_epoch_rgb_deployment.csv and the DCT epoch records above. These independently measured deployment configurations are not an isolated GPU-scaling experiment.',
        table(['GPU', 'DCT workload', *systems, 'Deployment ratio'], deployment_rows),
    ]) + '\n\n## F.' + suffix
    live.write_text(status)
    write_csv('current_results/deployment_inference.csv', [r for r in read_csv('results/p0/deployment_ratios.csv') if r['dct_top1']])
    write_csv('current_results/matched_dct_inference.csv', [r for r in read_csv('results/p0/matched_dct_inference.csv') if r['correctness'] == 'PASS'])
    write_csv('current_results/storage.csv', [r for r in read_csv('results/resources/storage.csv')
        if r['dataset_view'] == 'imagenet512' and r['representation'] != 'coordl_jpeg_cache'])
    sources = {
        'full_epoch_rgb_deployment.csv': 'results/p1/full_epoch_rgb_deployment.csv',
        'ffcv_encoding_comparison.csv': 'results/ffcv_jpeg_20261005/comparison.csv',
        'ffcv_encoding_storage.csv': 'results/ffcv_jpeg_20261005/storage.csv',
        'inference_summary.csv': 'results/p0/deployment_inference_summary.csv',
        'model_accuracy.csv': 'results/quality/model_accuracy.csv',
        'convergence_curves.csv': 'results/a800_convergence_20261003/figures/convergence_observed.csv',
        'h2d.csv': 'results/resources/h2d.csv',
        'memory.csv': 'results/resources/memory.csv',
        'amortization.csv': 'results/selectivity_4090_20261001/break_even_lower_bound.csv',
        'crop_pushdown.csv': 'results/ablation/crop_pushdown.csv',
        'direct_vs_grid.csv': 'results/ablation/direct_vs_grid.csv',
        'fastlanes_vs_raw_int16.csv': 'results/ablation/fastlanes_vs_raw_int16.csv',
        'frequency_compact_summary.csv': 'results/compact_projection_20261001/frequency_summary.csv',
        'frequency_h2d.csv': 'results/compact_projection_20261001/diagnostic/h2d.csv',
        'frequency_prepared_model_raw.csv': 'results/ablation/prepared_frequency_retry_20261006/performance_raw.csv',
        'frequency_prepared_model_summary.csv': 'results/ablation/prepared_frequency_retry_20261006/summary.csv',
        'locality_raw.csv': 'results/selectivity_4090_20261001/locality_2g/locality.csv',
        'locality_summary.csv': 'results/selectivity_4090_20261001/locality_summary.csv',
    }
    for study in ('spatial', 'channels', 'joint', 'coalescing', 'cold'):
        for suffix in ('', '_summary'):
            sources[f'{study}{suffix}.csv'] = f'results/selectivity_4090_20261001/{study}{suffix}.csv'
    for name, source in sources.items():
        write_csv('current_results/' + name, read_csv(source))
    print(json.dumps(metadata, indent=2))
    for check in checks:
        print(check['claim'], check['recomputed_value'], check['status'])


if __name__ == '__main__':
    main()
