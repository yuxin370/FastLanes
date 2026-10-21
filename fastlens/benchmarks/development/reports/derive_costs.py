#!/usr/bin/env python3
"""Derive conversion amortization solely from executed preparation/pass records."""
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.common import write_csv


def main():
    storage = list(csv.DictReader((ROOT / 'results/cost/storage.csv').open()))
    measured = list(csv.DictReader((ROOT / 'results/e2e/inference_summary.csv').open()))
    measured += list(csv.DictReader((ROOT / 'results/e2e/training_summary.csv').open()))
    costs = {r['split']: r for r in storage if r['representation'] == 'imagelanes_prepared'}
    native_pipelines = ('dct_major_pushdown', 'galp', 'imagelanes')
    rgb_counterparts = {'eFUN-DCT': 'EfficientNet-B0', 'MobileNetV2-DCT24': 'MobileNetV2',
                        'MobileNetV2-DCT32': 'MobileNetV2', 'ResNet-50-DCT24': 'ResNet-50',
                        'ResNet-50-DCT64': 'ResNet-50'}
    rows = []
    for native in measured:
        if native['pipeline'] not in native_pipelines:
            continue
        cost = costs['train' if native['task'] == 'training' else 'val']
        for baseline in measured:
            if baseline['pipeline'] in native_pipelines:
                continue
            if not all(native[key] == baseline[key] for key in ('gpu', 'task', 'precision', 'batch_size')):
                continue
            expected_model = rgb_counterparts.get(native['model'], native['model']) if baseline['domain'] == 'rgb' else native['model']
            if baseline['model'] != expected_model:
                continue
            native_s, baseline_s = float(native['median_elapsed_s']), float(baseline['median_elapsed_s'])
            conversion_complete = cost['conversion_s'] and 'duration MISSING' not in cost['notes']
            conversion = float(cost['conversion_s']) if conversion_complete else None
            saving = baseline_s - native_s
            rows.append(dict(gpu=native['gpu'], model=native['model'], task=native['task'], baseline=baseline['pipeline'],
                baseline_domain=baseline['domain'], fastlens_domain=native['domain'],
                comparison='matched DCT' if baseline['domain'] == native['domain'] and native['task'] == 'inference' else 'deployment comparison; training recipe matching must be verified',
                conversion_s=conversion, baseline_time_per_pass_s=baseline_s, fastlens_time_per_pass_s=native_s,
                saved_time_per_pass_s=saving,
                break_even_passes=conversion / saving if conversion is not None and saving > 0 else None,
                status='DEFINED' if conversion is not None and saving > 0 else 'UNDEFINED: no per-pass saving' if saving <= 0 else 'MISSING: conversion duration',
                notes='Historical measured conversion time; incomplete preparation timing stays missing; excludes separately measured verification; FFCV preparation is not deducted from baseline',
                raw_results=';'.join([cost['raw_result'], native['raw_results'], baseline['raw_results']])))
    fields = list(rows[0]) if rows else ['gpu', 'model', 'baseline', 'break_even_passes', 'status', 'raw_results']
    write_csv(ROOT / 'results/cost/break_even.csv', rows, fields)
    combined = [dict(resource='storage', **row) for row in storage]
    for name in ('h2d', 'memory'):
        path = ROOT / 'results/cost' / (name + '.csv')
        if path.exists():
            with path.open() as stream:
                combined.extend(dict(resource=name, **row) for row in csv.DictReader(stream))
    columns = list(dict.fromkeys(key for row in combined for key in row))
    write_csv(ROOT / 'results/resource_cost_table.csv', combined, columns)
    print(f'Exported {len(rows)} amortization comparisons')


if __name__ == '__main__':
    main()
