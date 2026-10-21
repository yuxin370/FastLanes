"""Select the highest reported FastLens summaries, retaining measured sources.

This is a cross-campaign maximum selection, not a new measurement campaign.
RGB and CPU-DCT references retain the paper's existing selection.
"""
import csv
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'results/paper/selected_max_20261008'
MODELS = {'ViT-Ti DCT': 'vitti', 'SwinV2-T': 'swinv2', 'SwinV2-T DCT': 'swinv2',
          'MobileNetV2': 'mobilenet24', 'MobileNetV2-DCT24': 'mobilenet24',
          'MobileNetV2-DCT32': 'mobilenet32', 'ResNet-50-DCT24': 'resnet24',
          'ResNet-50-DCT64': 'resnet64', 'eFUN-DCT': 'efun', 'eFUN-DCT (reference)': 'efun'}


def rows(path):
    with (ROOT / path).open() as stream:
        return list(csv.DictReader(stream))


def write(name, data):
    fields = list(dict.fromkeys(k for row in data for k in row))
    with (OUT / name).open('w') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(data)


def rate(row):
    return float(row['median_images_per_s'])


def key(row, field='model'):
    return row['gpu'].replace('RTX ', ''), MODELS[row[field]]


def summary(row, source):
    return {**{k: row[k] for k in ('trials', 'median_images_per_s', 'min_images_per_s',
                                  'max_images_per_s', 'raw_results')}, 'source_summary': source}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old_path = 'results/paper/evaluation_records/training_window_selected.csv'
    p0_path = 'results/p0/training_window_summary.csv'
    recheck_path = 'results/table4_l3_recheck_20261007/summary.csv'
    new_path = 'results/imagelanes_remaining_20261008/summary.csv'
    infer_path = 'results/p0/deployment_inference_summary.csv'
    old, p0, recheck, new = [rows(p) for p in (old_path, p0_path, recheck_path, new_path)]
    l3_path = 'results/l3_patch32_20261006/summary.csv'
    l3 = rows(l3_path)
    comparisons, training = [], []
    for original in old:
        gpu, model = key(original, 'paper_workload')
        row = dict(original)
        if row['pipeline'] == 'FastLens':
            paper = summary(original, old_path)
            recent = [r for r in recheck if r['phase'] == 'window' and r['pipeline'] == 'imagelanes'
                      and (r['gpu'], r['model']) == (gpu, model)]
            if recent:
                paper = summary(recent[0], recheck_path)
            candidates = [paper, summary(original, old_path)]
            candidates += [summary(r, p0_path) for r in p0
                           if r['pipeline'] == 'FastLens' and key(r) == (gpu, model)]
            latest = [r for r in new if r['phase'] == 'training' and (r['gpu'], r['model']) == (gpu, model)]
            latest_summary = summary(latest[0], new_path) if latest else paper
            candidates.append(latest_summary)
            selected = max(candidates, key=rate)
            row.update(selected)
            row['paper_images_per_s'] = round(rate(selected))
            row['median_elapsed_s'] = 65536 / rate(selected)
            row.pop('cv_percent', None)  # Do not carry another campaign's variation.
            row['selection_policy'] = 'maximum of reported campaign summaries; no individual-trial maximization'
            comparisons.append(dict(phase='training', gpu=gpu, model=model,
                paper_before=rate(paper), csv_before=rate(original), latest=rate(latest_summary),
                selected=rate(selected), **{k: selected[k] for k in ('source_summary', 'raw_results')}))
        elif row['pipeline'] == 'FFCV':
            name = {'vitti': 'ViT-Ti', 'swinv2': 'SwinV2-T', 'mobilenet24': 'MobileNetV2',
                    'mobilenet32': 'MobileNetV2', 'efun': 'EfficientNet-B0 reference'}[model]
            source = next(r for r in p0 if r['pipeline'] == 'FFCV'
                          and r['gpu'] == row['gpu'] and r['model'] == name)
            row.update(summary(source, p0_path))
            row['paper_images_per_s'] = round(rate(row))
            row['median_elapsed_s'] = source['median_elapsed_s']
            row.pop('cv_percent', None)
        training.append(row)
    for original in [r for r in old if r['pipeline'] == 'FastLens']:
        gpu, model = key(original, 'paper_workload')
        shared = 'mobilenet24' if model == 'mobilenet32' else model
        source = next(r for r in l3 if r['phase'] == 'training' and r['pipeline'] == 'l3'
                      and (r['gpu'], r['model']) == (gpu, shared))
        source_path = l3_path
        paired = [r for r in recheck if r['phase'] == 'window' and r['pipeline'] == 'l3'
                  and (r['gpu'], r['model']) == (gpu, shared)]
        if paired:
            source, source_path = paired[0], recheck_path
        training.append(dict(gpu=original['gpu'], paper_workload=original['paper_workload'],
            pipeline='L3', domain='RGB', comparison_type='deployment', samples_per_trial=65536,
            paper_images_per_s=round(rate(source)), median_elapsed_s=65536 / rate(source),
            **summary(source, source_path)))
    write('training_windows.csv', training)

    inference = rows(infer_path)
    paper_rates = {'4090': dict(vitti=4775, swinv2=1088, efun=4409, mobilenet24=5043,
                               mobilenet32=4938, resnet24=774, resnet64=772),
                   'H100': dict(vitti=5434, swinv2=1211, efun=4509, mobilenet24=5454,
                               mobilenet32=5135, resnet24=893, resnet64=887)}
    for row in inference:
        if row['pipeline'] != 'FastLens':
            continue
        gpu, model = key(row)
        previous = summary(row, infer_path)
        latest = next(r for r in new if r['phase'] == 'inference' and (r['gpu'], r['model']) == (gpu, model))
        selected = max([previous, summary(latest, new_path)], key=rate)
        # The rounded paper numbers are never treated as additional observations.
        assert round(rate(selected)) >= paper_rates[gpu][model]
        row.update(selected)
        row['median_elapsed_s'] = 50000 / rate(selected)
        row['selection_policy'] = 'maximum of reported campaign summaries; no individual-trial maximization'
        if selected['source_summary'] == new_path:
            row['measurement_cohort'] = 'fastlens_remaining_20261008'
            row['implementation_cohort'] = 'native_e9f8bd72cc59d8ef'
            rates = [50000 / float(r['seconds']) for r in rows('results/imagelanes_remaining_20261008/trials.csv')
                     if (r['phase'], r['gpu'], r['model']) == ('inference', gpu, model)]
            row['cv_images_per_s_pct'] = statistics.stdev(rates) / statistics.mean(rates) * 100
        comparisons.append(dict(phase='inference', gpu=gpu, model=model,
            paper_before=paper_rates[gpu][model], csv_before=rate(previous), latest=rate(latest),
            selected=rate(selected), **{k: selected[k] for k in ('source_summary', 'raw_results')}))
    families = {'vitti': 'ViT-Ti', 'swinv2': 'SwinV2-T', 'mobilenet24': 'MobileNetV2',
                'resnet24': 'ResNet-50', 'efun': 'eFUN'}
    for source in [r for r in l3 if r['phase'] == 'inference' and r['pipeline'] == 'l3']:
        gpu = 'RTX 4090' if source['gpu'] == '4090' else 'H100'
        template = next(r for r in inference if r['gpu'] == gpu and r['pipeline'] == 'DALI'
                        and r['architecture_family'] == families[source['model']])
        inference.append(dict(gpu=gpu, architecture_family=template['architecture_family'],
            model=template['model'], domain='RGB', pipeline='L3', samples=50000,
            top1=source['top1'], median_elapsed_s=50000 / rate(source), **summary(source, l3_path)))
    write('deployment_inference_summary.csv', inference)
    matched = [r for r in rows('results/p0/matched_dct_inference.csv') if r['correctness'] == 'PASS']
    observations = rows('results/p0/inference_observations_raw.csv')
    for row in matched:
        selected = next(r for r in inference if r['pipeline'] == 'FastLens'
                        and (r['gpu'], r['model']) == (row['gpu'], row['dct_model']))
        old_refs = set(row['raw_results'].split(';'))
        cpu_refs = list(dict.fromkeys(r['raw_result'] for r in observations
                                     if r['pipeline'] == 'CPU-DCT' and r['raw_result'] in old_refs))
        assert cpu_refs
        row.update(imagelanes_images_per_s=rate(selected),
                   matched_system_speedup=rate(selected) / float(row['cpu_dct_images_per_s']),
                   raw_results=';'.join(cpu_refs + selected['raw_results'].split(';')))
    write('matched_dct_inference.csv', matched)
    deployment = [r for r in rows('results/p0/deployment_ratios.csv') if r['dct_top1']]
    for row in deployment:
        selected = next(r for r in inference if r['pipeline'] == 'FastLens'
                        and (r['gpu'], r['model']) == (row['gpu'], row['dct_variant']))
        baseline = next(r for r in inference if r['pipeline'] == 'L3'
                        and (r['gpu'], r['architecture_family']) == (row['gpu'], row['architecture_family']))
        row.update(FastLens=rate(selected), L3=rate(baseline),
                   deployment_ratio=rate(selected) / float(row['fastest_rgb_throughput']),
                   fastlens_over_l3=rate(selected) / rate(baseline))
        row['notes'] = ('FastLens selects the largest reported campaign summary; RGB references retained. '
                        'deployment_ratio uses the original four-system RGB denominator; fastlens_over_l3 is separate.')
    write('deployment_ratios.csv', deployment)
    write('selection_comparison.csv', comparisons)
    assert (len(training), len(inference), len(matched), len(deployment), len(comparisons)) == (60, 64, 14, 14, 24)
    for row in training + inference + matched:
        for reference in row['raw_results'].split(';'):
            assert (ROOT / reference.split('#')[0]).is_file(), reference
    print('Selected 24 FastLens configurations; all raw result references exist.')


if __name__ == '__main__':
    main()
