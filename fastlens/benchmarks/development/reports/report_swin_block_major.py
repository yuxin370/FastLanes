"""Replace Swin inference rows with the paired block-major measurements."""
import csv
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results/swin_block_major_20261001'
PAPER = ROOT / 'fastlens/docs/paper'


def read_csv(path):
    with path.open() as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write_csv(path, fields, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    summary = {}
    for gpu, label in [('4090', 'RTX 4090'), ('H100', 'H100')]:
        summary[label] = {}
        reference = None
        for arm in ['block', 'compact', 'cpu']:
            runs = [json.loads((OUT/gpu/f'trial_{i}'/arm/'result.json').read_text()) for i in range(1,4)]
            for r in runs:
                top1 = [p[0] for p in r['predictions']]
                if reference is None:
                    reference = top1
                assert top1 == reference
                assert r['image_ids'] == list(range(50000))
            rates = [r['images_per_s'] for r in runs]
            times = [r['elapsed_s'] for r in runs]
            summary[label][arm] = dict(median=statistics.median(rates), minimum=min(rates), maximum=max(rates),
                median_s=statistics.median(times), cv_pct=100*statistics.stdev(rates)/statistics.mean(rates),
                top1=runs[0]['top1'], trials=3,
                raw_results=';'.join(str((OUT/gpu/f'trial_{i}'/arm/'result.json').relative_to(ROOT)) for i in range(1,4)))
        summary[label]['layout_ratio'] = summary[label]['block']['median']/summary[label]['compact']['median']
        summary[label]['matched_ratio'] = summary[label]['block']['median']/summary[label]['cpu']['median']
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

    fields, rows = read_csv(ROOT/'results/p0/deployment_inference_summary.csv')
    for row in rows:
        if row['architecture_family'] == 'SwinV2-T' and row['pipeline'] == 'FastLens':
            s = summary[row['gpu']]['block']
            row.update(trials=3, implementation_cohort='swin_block_major_20261001', measurement_cohort='paired_physical_order',
                median_images_per_s=s['median'], min_images_per_s=s['minimum'], max_images_per_s=s['maximum'],
                cv_images_per_s_pct=s['cv_pct'], median_elapsed_s=s['median_s'], top1=s['top1'], raw_results=s['raw_results'])
    write_csv(ROOT/'results/p0/deployment_inference_summary.csv', fields, rows)
    write_csv(ROOT/'results/inference_table.csv', fields, rows)

    for name in ['p0/deployment_ratios.csv', 'paper/deployment_inference.csv']:
        fields, rows = read_csv(ROOT/'results'/name)
        for row in rows:
            if row['architecture_family'] == 'SwinV2-T':
                s = summary[row['gpu']]['block']
                row.update(FastLens=s['median'], dct_top1=s['top1'],
                    deployment_ratio=s['median']/float(row['fastest_rgb_throughput']),
                    notes='Block-major Swin, three full passes in physical sample order; RGB baselines retained. Matched CPU-DCT and compact-v3 measured separately with the same order.')
        write_csv(ROOT/'results'/name, fields, rows)
    for name in ['p0/matched_dct_inference.csv', 'paper/matched_dct_inference.csv']:
        fields, rows = read_csv(ROOT/'results'/name)
        for row in rows:
            if row['architecture_family'] == 'SwinV2-T':
                s = summary[row['gpu']]
                row.update(cpu_dct_images_per_s=s['cpu']['median'], imagelanes_images_per_s=s['block']['median'],
                    matched_system_speedup=s['matched_ratio'], correctness='PASS',
                    raw_results=s['cpu']['raw_results']+';'+s['block']['raw_results'])
        write_csv(ROOT/'results'/name, fields, rows)

    fields, rows = read_csv(ROOT/'results/p0/inference_observations_raw.csv')
    rows = [r for r in rows if 'swin_block_major_20261001' not in r['raw_result']]
    for label, s in summary.items():
        for arm, pipeline in [('block', 'FastLens'), ('cpu', 'CPU-DCT')]:
            for raw in s[arm]['raw_results'].split(';'):
                r = json.loads((ROOT/raw).read_text())
                command = json.loads((ROOT/raw).with_name('run.command.json').read_text())
                row = dict.fromkeys(fields, '')
                values = dict(timestamp=command['timestamp'], git_commit=command['git_commit'],
                    gpu=label, architecture_family='SwinV2-T', model='SwinV2-T DCT', domain='DCT',
                    pipeline=pipeline, task='inference', precision='fp32', batch_size=50, samples=50000,
                    elapsed_s=r['elapsed_s'], images_per_s=r['images_per_s'], raw_result=raw,
                    checkpoint=r['checkpoint']['checkpoint'], correctness='PASS', top1=r['top1'], top5=r['top5'],
                    comparison_type='matched_dct', trial=Path(raw).parents[1].name.removeprefix('trial_'),
                    implementation_cohort='swin_block_major_20261001', measurement_cohort='paired_physical_order')
                row.update({k:v for k,v in values.items() if k in row})
                rows.append(row)
    write_csv(ROOT/'results/p0/inference_observations_raw.csv', fields, rows)

    fields, rows = read_csv(ROOT/'results/analysis/inference_pipeline_efficiency.csv')
    for row in rows:
        if row['model']=='SwinV2-T DCT' and row['pipeline']=='FastLens':
            s = summary[row['gpu']]['block']
            row.update(end_to_end_images_per_s=s['median'], pipeline_efficiency=s['median']/float(row['model_only_images_per_s']),
                       raw_results=s['raw_results'])
    write_csv(ROOT/'results/analysis/inference_pipeline_efficiency.csv', fields, rows)

    path = PAPER/'sections/07_evaluation.tex'
    text = path.read_text()
    _, deployments = read_csv(ROOT/'results/p0/deployment_ratios.csv')
    for label, s in summary.items():
        deployment = next(r for r in deployments if r['gpu']==label and r['architecture_family']=='SwinV2-T')
        new = f"{label} & SwinV2-T DCT & " + ' & '.join(f"{float(deployment[k]):,.0f}" for k in ['PyTorch','DALI','CoorDL','FFCV'])
        new += f" & {s['cpu']['median']:,.0f} & {s['block']['median']:,.0f} & {float(deployment['deployment_ratio']):.3f} " + r'\\'
        text = re.sub(re.escape(label)+r' & SwinV2-T DCT & [^\n]+', lambda m:new if m.group().count('&')==8 else m.group(), text)
        new = f"{label} & SwinV2-T DCT & {s['cpu']['median']:,.0f} & {s['block']['median']:,.0f} & {s['matched_ratio']:.3f} " + r'\\'
        text = re.sub(re.escape(label)+r' & SwinV2-T DCT & [^\n]+', lambda m:new if m.group().count('&')==4 else m.group(), text)
    exact = [float(r['deployment_ratio']) for r in deployments
             if r['relationship'] in ('registered_architecture_pair', 'upstream_architecture_family')]
    text = re.sub(r'Across the verified architecture pairs, .*?Small observed differences do not establish a reliable performance ranking\.',
        lambda m: f'Across the verified architecture pairs, {sum(v>1 for v in exact)} of {len(exact)} model/GPU deployment observations exceed the fastest RGB median; ratios range from {min(exact):.3f} to {max(exact):.3f}'+r'$\times$. Small observed differences do not establish a reliable performance ranking.',text)
    text = text.replace('The compact representation used by SwinV2 inference is reported\nseparately from the block-major representation used by the other workloads.',
        'Headline inference results use block-major storage. The compact\nrepresentation is retained as the image-major control in the SwinV2 layout comparison.')
    text = text.replace(r'\sys{} bars show measured throughput.', r'\sys{} bars show measured throughput, with three-trial medians and ranges for SwinV2-T.')
    marker = r'\paragraph{SwinV2 layout comparison.}'
    assert marker not in text
    addition = r'''
\paragraph{SwinV2 layout comparison.}
Table~\ref{tab:swin-layout} compares block-major and image-major compact-v3
inference using the SwinV2-T checkpoint, FP32, and model batches of 50.
Both traverse all 50,000 images in ascending physical image order.
Block-major prepares complete 1,024-image segments, with an 848-image
last segment, and restores logical batches across segment boundaries.
It uses two output slots, asynchronous GPU preparation, and the bounded
selective reader with a 10\% read-amplification allowance. Swin's
full-image resize and all-frequency input select the whole source;
the planner consequently chooses full-rowgroup reads. Pushdown is
enabled but saves no source bytes for this workload.
Each path has one excluded warm-up pass and three alternating measured
passes. Timings include reader startup and input/model execution;
model loading and resident-tensor warm-up are excluded. The matched
CPU-DCT baseline is remeasured with this sample order. RGB deployment
baselines retain their original measurement protocol.
Sampled model tensors agree exactly across layouts, and all 50,000
top-1 predictions agree across both layouts and CPU-DCT.
The block-major medians replace the SwinV2 headline inference results.

\begin{table}[t]
\centering
\caption{Paired SwinV2-T inference, median images/s over three full
validation passes. Ratio is block-major / image-major compact-v3.}
\label{tab:swin-layout}
\fontsize{8}{9}\selectfont\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrr@{}}
\toprule
GPU & Compact-v3 & Block-major & Ratio & CPU-DCT \\
\midrule
'''
    for label, s in summary.items():
        addition += f"{label} & {s['compact']['median']:,.0f} & {s['block']['median']:,.0f} & {s['layout_ratio']:.3f} & {s['cpu']['median']:,.0f} " + '\\\\\n'
    addition += '\\bottomrule\n\\end{tabular}\n\\end{table}\n\n'
    text = text.replace(r'\paragraph{Training deployment windows.}', addition+r'\paragraph{Training deployment windows.}')
    path.write_text(text)
    path = PAPER/'sections/09_conclusion.tex'
    text = re.sub(r'deployment ratios range from [0-9.]+ to [0-9.]+',
        f'deployment ratios range from {min(exact):.2f} to {max(exact):.2f}',path.read_text())
    path.write_text(text)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
