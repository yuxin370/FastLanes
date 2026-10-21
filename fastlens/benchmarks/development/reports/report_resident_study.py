"""Paired throughput tables and domain-normalized figures for resident controls."""
import argparse
import csv
import json
from pathlib import Path
import statistics

MODELS = ('vitti', 'swinv2', 'mobilenet24', 'mobilenet32', 'efun')
LABELS = dict(vitti='ViT-Ti', swinv2='SwinV2-T', mobilenet24='MobileNet-DCT24',
              mobilenet32='MobileNet-DCT32', efun='eFUN / EfficientNet-B0 reference')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(out):
    out = Path(out)
    records = list(csv.DictReader((out / 'trials.csv').open()))
    lookup = {(r['gpu'], r['model'], r['path'], int(r['trial'])): r for r in records}
    rows, pairs = [], []
    for gpu in ('4090', 'H100'):
        for model in MODELS:
            rgb_model = 'mobilenet24' if model == 'mobilenet32' else model
            paths = ('ffcv_raw', 'ffcv_jpeg', 'rgb_resident', 'imagelanes_basic', 'dct_resident')
            group = {}
            for path in paths:
                source_model = rgb_model if path.startswith(('ffcv', 'rgb')) else model
                members = [lookup.get((gpu, source_model, path, i)) for i in range(3)]
                if not all(members):
                    break
                group[path] = members
            if len(group) != 5:
                continue
            row = dict(gpu=gpu, model=model, trials=3)
            for path, members in group.items():
                values = [float(r['images_per_s']) for r in members]
                row[path] = statistics.median(values)
                row[path + '_min'] = min(values)
                row[path + '_max'] = max(values)
            current_pairs = []
            for i in range(3):
                rates = {p: float(group[p][i]['images_per_s']) for p in paths}
                pair = dict(gpu=gpu, model=model, trial=i,
                    E_DCT=rates['imagelanes_basic']/rates['dct_resident'],
                    E_RGB_raw=rates['ffcv_raw']/rates['rgb_resident'],
                    E_RGB_jpeg=rates['ffcv_jpeg']/rates['rgb_resident'],
                    DCT_resident_over_RGB_resident=rates['dct_resident']/rates['rgb_resident'],
                    DCT_resident_over_FFCV_raw=rates['dct_resident']/rates['ffcv_raw'],
                    DCT_resident_over_FFCV_jpeg=rates['dct_resident']/rates['ffcv_jpeg'])
                pair.update(g_DCT=1-pair['E_DCT'], g_RGB_raw=1-pair['E_RGB_raw'], g_RGB_jpeg=1-pair['E_RGB_jpeg'])
                current_pairs.append(pair)
            for key in current_pairs[0]:
                if key in ('gpu', 'model', 'trial'):
                    continue
                values = [r[key] for r in current_pairs]
                row[key] = statistics.median(values)
                row[key + '_min'] = min(values)
                row[key + '_max'] = max(values)
            row['raw_results'] = ';'.join(m['raw_result'] for members in group.values() for m in members)
            rows.append(row)
            pairs.extend(current_pairs)
    write_csv(out / 'absolute_throughput.csv', rows)
    write_csv(out / 'paired_ratios.csv', pairs)
    if rows:
        render(out, rows)
    return rows


def render(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(11, max(3.5, len(rows)*.38)), constrained_layout=True)
    for axis, domain in zip(axes, ('RGB', 'DCT')):
        selected = [r for r in rows if domain == 'DCT' or r['model'] != 'mobilenet32']
        keys = ('E_RGB_raw', 'E_RGB_jpeg') if domain == 'RGB' else ('E_DCT',)
        colors = ('#537fa6', '#bb8247') if domain == 'RGB' else ('#488d73',)
        for j, (key, color) in enumerate(zip(keys, colors)):
            ys = [i + (j-.5)*.24 if domain == 'RGB' else i for i in range(len(selected))]
            vals = [r[key] for r in selected]
            axis.errorbar(vals, ys, xerr=[[r[key]-r[key+'_min'] for r in selected],
                                        [r[key+'_max']-r[key] for r in selected]],
                          fmt='o', capsize=3, color=color,
                          label=('FFCV raw' if j == 0 else 'FFCV JPEG') if domain == 'RGB' else 'FastLens basic')
        axis.set_yticks(range(len(selected)), [r['gpu']+' '+LABELS[r['model']] for r in selected])
        axis.axvline(1., linestyle='--', linewidth=1, color='#777777')
        axis.set_xlabel('Throughput / '+domain+' resident-input throughput')
        axis.set_xlim(left=0)
        axis.invert_yaxis()
        axis.grid(axis='x', alpha=.2)
        axis.legend(loc='best', fontsize=9)
    fig.savefig(out / 'normalized_throughput.pdf')
    fig.savefig(out / 'normalized_throughput.png', dpi=180)
    plt.close(fig)
    lines = [r'\begin{tabular}{llrrrrrr}', r'\toprule',
             r'GPU & Workload & FFCV raw & FFCV JPEG & RGB resident & FastLens basic & DCT resident & DCT efficiency \\',
             r'\midrule']
    for r in rows:
        values = ' & '.join(f"{r[p]:.0f}" for p in ('ffcv_raw','ffcv_jpeg','rgb_resident','imagelanes_basic','dct_resident'))
        lines.append(f"{r['gpu']} & {LABELS[r['model']]} & {values} & {100*r['E_DCT']:.1f}\\% \\\\")
    lines += [r'\bottomrule', r'\end{tabular}',
              '% Throughputs: medians of three uninstrumented 65,536-image windows.',
              '% Efficiency: median of trial-paired ratios; ranges are retained in paired_ratios.csv.',
              '% End-to-end windows process distinct samples; resident controls cycle eight preprocessed batches (512 samples).',
              '% MobileNet-DCT24/32 share one RGB reference; EfficientNet-B0 is only a deployment reference for eFUN.']
    (out / 'absolute_throughput.tex').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    summarize(p.parse_args().output)
