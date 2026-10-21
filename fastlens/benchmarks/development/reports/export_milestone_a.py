"""Export paper tables/figures from accepted observations, with explicit missing cells."""
import csv
import json
import statistics
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from fastlens.benchmarks.development.reports.deployment_report import rows
from fastlens.benchmarks.development.reports.deployment_report import write

R = ROOT/'results'
SYSTEMS = ['PyTorch','DALI','CoorDL','FFCV','FastLens']
COLORS = ['#777777','#E69F00','#56B4E9','#CC79A7','#009E73']


def available(path):
    return rows(path) if path.exists() else []


def export(path, records):
    if records:
        fields=list(dict.fromkeys(k for r in records for k in r))
        write(path, records, fields)


def export_training_table(training, full):
    from fastlens.benchmarks.development.reports.organize_evaluation_records import model_key
    selected = rows(R/'paper/evaluation_records/training_window_selected.csv')
    windows = []
    for source in training:
        row = dict(source, comparison_type='deployment', task='training_window')
        matching = [r for r in selected if (r['gpu'], r['pipeline'], model_key(r['paper_workload'])) ==
                    (row['gpu'], row['pipeline'], model_key(row['model']))]
        if matching:
            chosen, = {r['raw_results']: r for r in matching}.values()
            raw = json.loads((ROOT/chosen['raw_results'].split(';')[0]).read_text())
            for field in ('median_images_per_s', 'min_images_per_s', 'max_images_per_s',
                          'median_elapsed_s', 'trials', 'cv_percent', 'raw_results'):
                row[field] = chosen[field]
            row.update(precision=raw['precision'], batch_size=raw['batch'], samples=raw['images'],
                coefficient_of_variation=float(chosen['cv_percent'])/100,
                implementation_cohort='', source_summary=chosen['source_summary'],
                notes='Current manuscript cohort; all three trials retained. Other accepted cohorts remain in the raw archives.')
        row['paper_selected'] = bool(matching)
        windows.append(row)
    export(R/'training_table.csv',
        windows +
        [dict(r, comparison_type='matched_dct', images_per_s=int(r['samples'])/float(r['elapsed_s'])) for r in full] +
        rows(R/'p1/full_epoch_rgb_deployment.csv'))


def export_ablation_table(ablations):
    table = []
    for mechanism, records in ablations.items():
        if records:
            control, optimized = records
            for row in records:
                table.append(dict(row, mechanism=mechanism,
                    elapsed_ratio_vs_control=float(row['elapsed_s'])/float(control['elapsed_s']),
                    issued_ratio_vs_control=float(row['issued_read_bytes'])/float(control['issued_read_bytes']),
                    h2d_ratio_vs_control=float(row['h2d_bytes'])/float(control['h2d_bytes'])))
    prepared = rows(R/'ablation/prepared_frequency_retry_20261006/performance_raw.csv')
    controls = {r['trial']: r for r in prepared if r['arm'] == 'off'}
    for row in prepared:
        control = controls[row['trial']]
        table.append(dict(row, mechanism='frequency_prepared_input', case=row['arm'],
            elapsed_ratio_vs_control=float(row['elapsed_s'])/float(control['elapsed_s']),
            issued_ratio_vs_control=float(row['issued_read_bytes'])/float(control['issued_read_bytes']),
            h2d_ratio_vs_control=float(row['h2d_bytes'])/float(control['h2d_bytes'])))
    export(R/'ablation_table.csv', table)


def update_measurement_union():
    """Append newly accepted records while retaining historical observations."""
    from fastlens.benchmarks.development.reports.organize_evaluation_records import model_key
    def reference(value):
        parts = []
        for item in value.split(';'):
            path, separator, pointer = item.partition('#')
            path = Path(path)
            parts.append(str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path) + separator + pointer)
        return ';'.join(parts)

    observed = rows(R/'all_measurements.csv')
    seen = {reference(r['raw_result']) for r in observed if r['raw_result']}
    pairs = rows(R/'inventory/workload_pairs.csv')
    candidates = rows(R/'p0/inference_observations_raw.csv')
    for row in rows(R/'paper/evaluation_records/current_results/training_window_raw.csv'):
        pair = next(p for p in pairs if model_key(p['dct_model']) == model_key(row['paper_workload']))
        domain = row['domain'].upper()
        candidates.append(dict(row, model=pair['dct_model'] if domain == 'DCT' else pair['rgb_model'],
            domain=domain, architecture_family=pair['architecture_family'],
            task='training_window', comparison_type='deployment',
            notes='Paper-selected training window; full configuration and concurrent load remain in the raw result directory.'))
    for row in rows(R/'p1/full_epoch_training.csv'):
        assert row['valid'] == row['strict_matched'] == 'True'
        candidates.append(dict(row, comparison_type='matched_dct', correctness='PASS',
                               images_per_s=int(row['samples'])/float(row['elapsed_s'])))
    candidates += rows(R/'p1/full_epoch_rgb_deployment.csv')
    candidates += rows(R/'ablation/prepared_frequency_retry_20261006/performance_raw.csv')
    for source in rows(R/'resources/storage.csv'):
        row = dict(source, task='storage', measurement='storage', comparison_type='deployment',
                   storage_bytes=source['representation_bytes'])
        del row['storage_amplification']  # Derived ratios stay in the resource summary.
        candidates.append(row)
    added = 0
    for row in candidates:
        key = reference(row['raw_result'])
        if key not in seen:
            observed.append(row)
            seen.add(key)
            added += 1
    export(R/'all_measurements.csv', observed)
    return added


def save(figure,name):
    directory=R/'figures/milestone_a'
    directory.mkdir(parents=True,exist_ok=True)
    for suffix in ('pdf','png'):
        figure.savefig(directory/f'{name}.{suffix}',dpi=180)
    plt.close(figure)


def plot_core_ablations():
    import shutil
    sys.path.insert(0, str(ROOT/'fastlens/docs/paper/scripts'))
    from plot_core_ablations import main as draw
    draw()
    for suffix in ('pdf', 'png'):
        shutil.copyfile(ROOT/f'fastlens/docs/paper/figures/milestone_a/04_core_ablations.{suffix}',
                        R/f'figures/milestone_a/04_core_ablations.{suffix}')


def plot_figures(deployment_only=False):
    """Render from the per-figure paper scripts without changing measurements."""
    import shutil
    sys.path.insert(0, str(ROOT/'fastlens/docs/paper/scripts'))
    import plot_deployment_inference
    import plot_pipeline_efficiency
    import plot_crop_pushdown
    import plot_core_ablations
    plots = [(plot_deployment_inference, '01_deployment_inference')]
    if not deployment_only:
        plots += [(plot_pipeline_efficiency, '02_pipeline_efficiency'),
                  (plot_crop_pushdown, '03_crop_pushdown'),
                  (plot_core_ablations, '04_core_ablations')]
    (R/'figures/milestone_a').mkdir(parents=True, exist_ok=True)
    for module, name in plots:
        module.main()
        for suffix in ('pdf', 'png'):
            shutil.copyfile(ROOT/f'fastlens/docs/paper/figures/milestone_a/{name}.{suffix}',
                            R/f'figures/milestone_a/{name}.{suffix}')


def main():
    plt.rcParams.update({'font.size':8,'axes.titlesize':8,'axes.labelsize':8,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42})
    deployment=rows(R/'p0/deployment_ratios.csv')
    inference=rows(R/'p0/deployment_inference_summary.csv')
    matched=rows(R/'p0/matched_dct_inference.csv')
    training=rows(R/'paper/training_windows.csv')
    full=rows(R/'p1/full_epoch_training.csv')
    h2d=available(R/'resources/h2d.csv')
    memory=available(R/'resources/memory.csv')
    storage=rows(R/'resources/storage.csv')
    ablations={name:available(R/f'ablation/{name}.csv') for name in ('crop_pushdown','direct_vs_grid','fastlanes_vs_raw_int16')}
    export(R/'inference_table.csv',inference)
    export_training_table(training, full)
    export(R/'resource_cost_table.csv',[dict(r,measurement='storage') for r in storage]+
        [dict(r,measurement='h2d') for r in h2d]+[dict(r,measurement='memory') for r in memory])
    export_ablation_table(ablations)
    update_measurement_union()
    amortization=[]
    observed=rows(R/'p0/inference_observations_raw.csv')
    pairs=rows(R/'inventory/workload_pairs.csv')
    for native in inference:
        if native['pipeline']!='FastLens': continue
        pair=next(r for r in pairs if r['dct_model']==native['model'])
        representation='imagelanes_compact_v3' if native['architecture_family']=='SwinV2-T' else 'imagelanes_prepared'
        cost=next(r for r in storage if r['representation']==representation and r['split']=='val')
        for pipeline in ['PyTorch','DALI','CoorDL','FFCV','CPU-DCT']:
            model=native['model'] if pipeline=='CPU-DCT' else pair['rgb_model']
            selected=next(r for r in (matched if pipeline=='CPU-DCT' else inference)
                if r['gpu']==native['gpu'] and
                (r['dct_model']==model if pipeline=='CPU-DCT' else r['model']==model and r['pipeline']==pipeline))
            selected_raw=set(selected['raw_results'].split(';'))
            baseline=[r for r in observed if r['raw_result'] in selected_raw and
                (r['gpu'],r['model'],r['pipeline'])==(native['gpu'],model,pipeline)]
            if not baseline: continue
            baseline_s=statistics.median(float(r['elapsed_s']) for r in baseline)
            native_s=float(native['median_elapsed_s']); saving=baseline_s-native_s
            conversion=float(cost['conversion_s']) if cost['conversion_s'] else None
            amortization.append(dict(gpu=native['gpu'],model=native['model'],task='inference',baseline=pipeline,
                comparison_type='matched_dct' if pipeline=='CPU-DCT' else 'deployment',representation=representation,
                conversion_s=conversion,baseline_time_per_pass_s=baseline_s,fastlens_time_per_pass_s=native_s,
                saved_time_per_pass_s=saving,break_even_passes=conversion/saving if conversion is not None and saving>0 else None,
                status='UNDEFINED: no per-pass saving' if saving<=0 else 'MISSING: conversion duration' if conversion is None else 'DERIVED',
                raw_results=';'.join([cost['raw_result'],native['raw_results'],*[r['raw_result'] for r in baseline]]),
                notes='Historical measured conversion; excludes separate verification and baseline preparation. Compact-v3 and block-major preparation costs are not interchangeable.'))
    for native in full:
        if native['pipeline']!='FastLens': continue
        baseline=next((r for r in full if (r['gpu'],r['model'],r['pipeline'])==(native['gpu'],native['model'],'CPU-DCT')),None)
        if baseline:
            cost=next(r for r in storage if r['representation']=='imagelanes_prepared' and r['split']=='train')
            saving=float(baseline['elapsed_s'])-float(native['elapsed_s'])
            amortization.append(dict(gpu=native['gpu'],model=native['model'],task='training_epoch',baseline='CPU-DCT',
                comparison_type='matched_dct',representation='imagelanes_prepared',conversion_s=None,
                baseline_time_per_pass_s=baseline['elapsed_s'],fastlens_time_per_pass_s=native['elapsed_s'],
                saved_time_per_pass_s=saving,break_even_passes=None,status='MISSING: complete conversion duration',
                raw_results=';'.join([cost['raw_result'],native['raw_result'],baseline['raw_result']]),
                notes='Recorded training conversion covers compression only; missing access-index time prevents complete amortization.'))
    export(R/'resources/break_even.csv',amortization)
    efficiencies=rows(R/'analysis/inference_pipeline_efficiency.csv')
    scaling=[]
    for low in efficiencies:
        if low['gpu']!='RTX 4090': continue
        high=next((r for r in efficiencies if r['gpu']=='H100' and all(r[k]==low[k] for k in ('model','domain','pipeline','batch_size','precision'))),None)
        if high:
            scaling.append(dict(model=low['model'],domain=low['domain'],pipeline=low['pipeline'],
                batch_size=low['batch_size'],precision=low['precision'],
                gpu_model_scaling=float(high['model_only_images_per_s'])/float(low['model_only_images_per_s']),
                gpu_e2e_scaling=float(high['end_to_end_images_per_s'])/float(low['end_to_end_images_per_s']),
                efficiency_4090=low['pipeline_efficiency'],efficiency_h100=high['pipeline_efficiency'],
                raw_results=low['raw_results']+';'+high['raw_results'],
                ceiling_raw_results=low['ceiling_raw_results']+';'+high['ceiling_raw_results'],
                notes='DERIVED; same host and common batch/precision; own domain/model ceiling. Concurrent-load annotations apply.'))
    export(R/'analysis/gpu_scaling.csv',scaling)
    paper=R/'paper'
    for name,data in [('model_accuracy',rows(R/'quality/model_accuracy.csv')),('deployment_inference',deployment),
                      ('matched_dct_inference',matched),('training_windows',training),('full_epoch_training',full)]:
        export(paper/f'{name}.csv',data)
    # A numeric measurement is never manufactured for an unfinished stage.
    requirements=dict(checkpoint_accuracies=len(rows(R/'quality/model_accuracy.csv'))==12,
        deployment_inference=len(inference)==54,matched_dct_inference=sum(r['correctness']=='PASS' for r in matched)==14,
        training_windows=all(any(r['gpu']==gpu and r['model']==model and r['pipeline']==system
            and int(r['trials'])>=3 and int(r['samples'])==65536 for r in training)
            for gpu in ('RTX 4090','H100') for model in ('SwinV2-T','MobileNetV2') for system in SYSTEMS),
        full_training_epochs=len(full)==8,
        storage=len(storage)>=10,h2d=len(h2d)==10,memory=len(memory)==10,
        crop_pushdown=len(ablations['crop_pushdown'])==2,direct_vs_grid=len(ablations['direct_vs_grid'])==2)
    ceilings=rows(R/'analysis/model_only.csv')
    requirements['representative_model_ceilings']=all(any(r['gpu']==g and r['model']==m and r['domain']==d and r['task']=='inference_model_only'
        for r in ceilings) for g in ('RTX 4090','H100') for m,d in [('SwinV2-T-256-window8','RGB'),('SwinV2-T-256-window8','DCT'),('MobileNetV2','RGB'),('MobileNetV2-DCT24','DCT')])
    requirements['training_model_ceilings']=all(any((r['gpu'],r['model'],r['domain'],r['task'])==(g,m,d,'training_model_only')
        for r in ceilings) for g in ('RTX 4090','H100') for m in ('SwinV2-T','MobileNetV2') for d in ('RGB','DCT'))
    complete=all(requirements.values())
    repeated=[r for r in inference if int(r['trials'])>1]
    unstable=[r for r in repeated if float(r['cv_images_per_s_pct'])>2]
    milestone=dict(status='COMPLETE' if complete else 'INCOMPLETE',requirements=requirements,
        timing_repeatability=dict(repeated_inference_groups=len(repeated),cv_above_2_percent=len(unstable),
            interpretation='Matrix coverage is complete; shared-host timing does not establish stable causal performance.'),
        phase_2=['repeat optimized inference to three trials','blocked/unconfigured checkpoints if supplied','frequency matched control','optional H100 ablations'],
        limitations=['Historical conversion/verification times absent from raw logs remain MISSING.',
            'Optimized FastLens inference uses one selected pass per workload/GPU, versus repeated RGB and CPU-DCT baselines; H100 VLLM remains resident. The anomalous first MobileNet24 observation is retained separately.',
            'eFUN RGB reference is not a verified exact counterpart.',
            'Shared-host concurrent activity is recorded; historical affinity gaps remain disclosed.',
            f'{len(unstable)} of {len(repeated)} repeated inference groups exceed 2% throughput CV; timing rankings remain provisional.',
            'Full training epochs measure throughput, not convergence.',
            'Crop ablation uses one retained area; no sweep or interpolation.'])
    (paper/'milestone_a.json').write_text(json.dumps(milestone,indent=2)+'\n')
    if not complete:
        print(json.dumps(milestone)); return
    plot_figures()
    with (R/'PAPER_STATUS.md').open('a') as stream:
        stream.write('\n## Milestone A\n\nCOMPLETE: the first-pass matrix is populated. Implementation-version refresh and Phase 2 observations are tracked separately; see the refresh status and performance-change report.\n\n')
        stream.write('\n'.join('- '+s for s in milestone['limitations'])+'\n')
    print(json.dumps(milestone))


if __name__=='__main__':
    main()
