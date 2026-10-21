#!/usr/bin/env python3
"""Audit existing observations and export the reduced paper plan without launching jobs."""
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.execution.revision_results import select_current
R = ROOT / 'results'
GPUS = ('RTX 4090', 'H100')
MODELS = ('SwinV2-T', 'MobileNetV2')
SYSTEMS = ('PyTorch', 'DALI', 'CoorDL', 'FFCV', 'FastLens')
MANIFEST_FIELDS = 'experiment_id task gpu model domain pipeline sample_count status result_file valid notes'.split()
EXTRA = 'worker_count cpu_affinity numa_binding source_log command strict_matched valid implementation_cohort code_fingerprint measurement_cohort'.split()
RAW_FIELDS = next(csv.reader((R / 'unified_raw.csv').open())) + EXTRA


def read(path):
    return json.loads(path.read_text())


def csvread(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def gpu_name(name):
    return 'H100' if 'H100' in name else 'RTX 4090' if '4090' in name else name


def system(name):
    return {'pytorch': 'PyTorch', 'pytorch_rgb': 'PyTorch', 'rgb_pytorch': 'PyTorch',
            'dali': 'DALI', 'rgb_dali': 'DALI', 'rgb_d2': 'DALI', 'coordl': 'CoorDL', 'ffcv': 'FFCV',
            'imagelanes': 'FastLens', 'galp': 'FastLens', 'dct_major_pushdown': 'FastLens',
            'cpu_dct': 'CPU-DCT', 'rgbnomore': 'CPU-DCT', 'model-only': 'model_only'}.get(name, name)


def strict_prediction_pass(gate):
    if 'prediction_agreement' in gate:
        return gate['sample_order_equal'] and gate['prediction_agreement'] == 1.0 and gate['numerical_control_pass']
    if 'semantic_comparisons' in gate:
        comparisons = [c for c in gate['semantic_comparisons'] if c['enforcement'] == 'strict']
        return bool(comparisons) and all(c['ok'] and c['sample_identity_match']
                                        and c['full_top1_prediction_agreement'] == 1.0 for c in comparisons)
    comparisons = [c for c in gate['semantic_validation']['comparisons'] if c['enforcement'] == 'strict']
    return bool(comparisons) and all(c['ok'] and c['full_prediction']['count'] == 50000
                                    and c['full_prediction']['top1_agreement'] == 1.0 for c in comparisons)


def summaries(rows):
    keys = 'gpu model domain pipeline task precision batch_size effective_batch_size'.split()
    groups = {}
    for row in rows:
        if not row['valid']:
            continue
        groups.setdefault(tuple(row.get(k, '') for k in keys), []).append(row)
    out = []
    for key, group in groups.items():
        rates = [float(x['samples']) / float(x['elapsed_s']) for x in group]
        out.append(dict(zip(keys, key), trials=len(group), samples=group[0]['samples'],
                        median_images_per_s=statistics.median(rates), min_images_per_s=min(rates),
                        max_images_per_s=max(rates), median_elapsed_s=statistics.median(float(x['elapsed_s']) for x in group),
                        raw_results=';'.join(x['raw_result'] for x in group), notes=group[0]['notes']))
    return out


SUMMARY_FIELDS = 'gpu model domain pipeline task precision batch_size effective_batch_size trials samples median_images_per_s min_images_per_s max_images_per_s median_elapsed_s raw_results notes'.split()


def main():
    manifest, p0, ceilings, epochs = [], [], [], []
    queue_path = R/'p0/queue_state.json'
    queue = read(queue_path) if queue_path.exists() else None
    def entry(identifier, task, gpu, model, domain, pipeline, samples, status, result='', valid=False, notes=''):
        manifest.append(dict(zip(MANIFEST_FIELDS, (identifier, task, gpu, model, domain, pipeline, samples, status, result, valid, notes))))

    # Keep every historical motivation result discoverable, including results outside the new matrix.
    for path in sorted((ROOT / 'benchmark_results').glob('input_motivation*/*/result.json')):
        d = read(path)
        rel = str(path.relative_to(ROOT))
        log = path.parent / 'run.log'
        good = (not d.get('profile', False) and math.isfinite(d.get('seconds', float('nan')))
                and d.get('seconds', 0) > 0 and log.exists())
        entry('existing:' + rel, d.get('mode', 'training'), gpu_name(d['gpu']), d['model'],
              d.get('domain', 'rgb').upper(), system(d['arm']), d['images'], 'DONE', rel, good,
              'Historical observation; profile runs are diagnostic only; no historical CPU/NUMA affinity or git commit recorded.')
        if (path.parent.parent.name != 'input_motivation_20260925_nocudagraph'
                or d['model'] not in ('swinv2', 'mobilenet') or d.get('mode', 'training') != 'training'
                or d['images'] != 65536 or d.get('profile', False)):
            continue
        good = good and d['updates'] == 64 and d['batch'] * d['accumulation'] == 1024
        good = good and d['cache_before']['resident_fraction'] == 1.0
        # A result is only written after the existing runner's loss/gradient finite check succeeds.
        good = good and 'measuring 64 updates' in log.read_text()
        row = dict(gpu=gpu_name(d['gpu']), model='SwinV2-T' if d['model'] == 'swinv2' else 'MobileNetV2',
                   domain='RGB', pipeline=system(d['arm']), task='training_window', precision=d['precision'],
                   batch_size=d['batch'], effective_batch_size=d['batch'] * d['accumulation'],
                   trial=d['trial'], samples=d['images'], elapsed_s=d['seconds'], optimizer_updates=d['updates'],
                   worker_count=d['pytorch_workers'], cpu_affinity='NOT_RECORDED', numa_binding='NOT_RECORDED',
                   source_log=str(log.relative_to(ROOT)), command=json.dumps(d['command']), raw_result=rel,
                   valid=good, strict_matched=False,
                   correctness_status='PASS: completed 64 updates of 16 full 64-image batches; deferred finite-loss/gradient check',
                   notes='Reused ImageNet-512 training window; 32 excluded warmup updates; no profiler; BF16, compile, no CUDA graphs; '
                         'historical affinity and git commit unknown; DALI D3 uses native crop/order; deployment comparison only')
        if d['arm'] == 'model-only':
            row['task'] = 'training_model_only'
            ceilings.append(row)
        else:
            p0.append(row)

    for path in sorted([*(R/'p0/raw').glob('*/*/*/trial_*/result.json'), *(R/'analysis/training_ceilings').glob('*/*/*/trial_*/result.json')]):
        if not (path.parent/'accepted.json').exists():
            continue
        d = read(path)
        accepted = read(path.parent/'accepted.json')
        row = dict(timestamp=d['timestamp'], git_commit=d['git_commit'], gpu=gpu_name(d['gpu']),
                   model='SwinV2-T' if d['model']=='swinv2' else 'MobileNetV2', domain=d['domain'].upper(),
                   pipeline=system(d['arm']), task='training_window', precision=d['precision'],
                   batch_size=d['batch'], effective_batch_size=d['batch']*d['accumulation'], trial=d['trial'],
                   samples=d['images'], elapsed_s=d['seconds'], optimizer_updates=d['updates'],
                   training_loss=d['training_loss'], initial_model_hash=d['initial_model_hash'],
                   worker_count=d['pytorch_workers'] if d['pytorch_workers'] is not None else d.get('native_pls_workers'), cpu_affinity=json.dumps(d['cpu_affinity']),
                   numa_binding=d['numa_policy'], source_log=str((path.parent/'run.log').relative_to(ROOT)),
                   command=json.dumps(d['command']), raw_result=str(path.relative_to(ROOT)),
                   strict_matched=False, valid=d['correctness_passed'],
                   correctness_status='PASS: actual sample IDs/count/uniqueness, 64 updates, finite loss/gradient norm; sampled host GPU guard',
                   notes=d['scope']+'; '+d['notes'])
        row.update({k: accepted[k] for k in ('implementation_cohort', 'code_fingerprint') if k in accepted})
        if d['mode']=='training_model_only':
            row.update(task='training_model_only',correctness_status='PASS: resident tensor contract, 65536 image executions, 64 updates, finite loss/gradient norm; sampled GPU guard')
            ceilings.append(row)
        else:
            p0.append(row)

    all_p0 = p0[:]
    p0 = select_current(p0)
    write(R/'p0/training_window_all_versions_raw.csv', all_p0, RAW_FIELDS)
    for gpu in GPUS:
        for model in MODELS:
            for pipeline in SYSTEMS:
                domain = 'DCT' if pipeline == 'FastLens' else 'RGB'
                for trial in range(3):
                    prior = next((r for r in p0 if (r['gpu'], r['model'], r['pipeline'], r['trial']) == (gpu, model, pipeline, trial)), None)
                    trial_path = Path('results/p0/raw') / ('4090' if gpu == 'RTX 4090' else gpu) / ('swinv2' if model == 'SwinV2-T' else 'mobilenet') / pipeline.lower() / f'trial_{trial}'
                    running = bool(queue and queue['phase'] == 'RUNNING' and queue.get('active') == str(trial_path))
                    failure_path = ROOT/trial_path/'failure.json'
                    if not failure_path.exists():
                        failures = sorted((ROOT/trial_path.parent).glob('trial_*/failure.json'))
                        if failures:
                            failure_path = failures[0]
                    entry(f'p0:{gpu}:{model}:{pipeline}:{trial}', 'training_window', gpu, model, domain, pipeline, 65536,
                          'DONE' if prior and prior['valid'] else 'RUNNING' if running else 'BLOCKED' if failure_path.exists() else 'TODO',
                          prior['raw_result'] if prior else str(failure_path.relative_to(ROOT)) if failure_path.exists() else '',
                          bool(prior and prior['valid']), prior['notes'] if prior else read(failure_path)['reason'] if failure_path.exists()
                          else 'P0; 3 unprofiled trials; wait for competing host resources before measurement')

    for path in sorted((R/'interrupted').glob('*/*/*/*/trial_*/failure.json')):
        failure = read(path)
        _, gpu, model, pipeline, *_ = path.relative_to(R/'interrupted').parts
        entry('interrupted:'+str(path.relative_to(ROOT)), 'training_window', gpu_name(gpu),
              'SwinV2-T' if model=='swinv2' else 'MobileNetV2', 'DCT' if pipeline=='imagelanes' else 'RGB',
              system(path.parents[1].name), '', 'FAILED', str(path.relative_to(ROOT)), False, failure['reason'])

    # Completed epoch records remain usable observations even when the enclosing adaptive controller stopped.
    for path in sorted((R / 'raw').glob('*/*/training/*/trial_*/runs/B6/seed_11997733/metrics.jsonl')):
        base = path.parents[3]
        pause = path.parent / 'pause_result.json'
        if not pause.exists():
            entry('interrupted:' + str(base.relative_to(ROOT)), 'training_epoch', gpu_name(path.relative_to(R/'raw').parts[0]),
                  'SwinV2-T', 'DCT', system(base.parent.name), '', 'FAILED', str(path.relative_to(ROOT)), False,
                  'Incomplete legacy run; deferred by new plan; all files retained')
            continue
        d = read(pause)
        gate = d['integration_checks']
        good = all(gate[k] for k in ('coverage_counters_valid', 'gradients_finite', 'loss_finite', 'parameters_finite'))
        records = [json.loads(line) for line in path.read_text().splitlines()]
        warm = next(x for x in records if x.get('scope') == 'epoch' and x['epoch'] == 2)
        good = good and warm['epoch_samples'] == 1281167 and warm['epoch_optimizer_updates'] == 1252 and math.isfinite(warm['train_loss'])
        meta = read(base.with_suffix('.command.json'))
        initial = read(path.parent / 'model_only_dct.json')['initial_model_hash']
        row = dict(gpu=gpu_name(path.relative_to(R/'raw').parts[0]), model='SwinV2-T', domain='DCT', pipeline=system(base.parent.name),
                   task='training_epoch', timestamp=meta['timestamp'], git_commit=meta['git_commit'],
                   precision=d['precision'], batch_size=64, effective_batch_size=1024, trial=int(base.name.split('_')[1]),
                   samples=warm['epoch_samples'], elapsed_s=warm['epoch_seconds'], optimizer_updates=warm['epoch_optimizer_updates'],
                   training_loss=warm['train_loss'], initial_model_hash=initial, worker_count=8, cpu_affinity='NOT_RECORDED', numa_binding='NOT_RECORDED',
                   raw_result=str(path.relative_to(ROOT)), command=meta['shell_command'], source_log=str(base.with_suffix('.log').relative_to(ROOT)),
                   valid=good, strict_matched=False, correctness_status='PASS: coverage, updates, finite loss; B6 tensor prefix gate passed (128 samples); host contention unresolved',
                   notes='Reused completed warm epoch 2; old protocol used epoch 1 as warmup; no new full warmup epoch requested. '
                         'Matching initialization/sample-order digests; native crop/flip/augmentation digests are ownership strings, not comparable hashes. '
                         'Independent B6 tensor gate: results/checks/swin_b6_native_cpu_tensors.json (128 samples; no convergence claim). '
                         'Shared-host competing jobs were present; do not attribute a causal speedup.')
        epochs.append(row)
        entry('existing:' + row['raw_result'], 'training_epoch', row['gpu'], row['model'], 'DCT', row['pipeline'], row['samples'],
              'DONE', row['raw_result'], good, row['notes'])
    epoch_paths=[*(R/'p1/raw').glob('*/*/accepted.json'),*(R/'p1/short_warmup_v2').glob('*/*/accepted.json'),*(R/'p1/short_warmup_retry').glob('*/*/accepted.json'),*(R/'p1/short_warmup_refresh_20260929').glob('*/*/accepted.json')]
    for batch in ('recheck_20260929','low_load_recheck_20260929'):
        epoch_paths.extend((R/batch/'full_epoch').glob('*/accepted.json'))
    for path in sorted(epoch_paths):
        accepted = read(path)
        base = path.parent
        epoch = accepted['epoch']
        meta = read(base/'run.command.json')
        warm = read(ROOT/accepted['warmup'])
        cohort_name=path.relative_to(R).parts[0]
        recheck=cohort_name in ('recheck_20260929','low_load_recheck_20260929')
        gpu=base.name if recheck else base.parent.name
        pair_complete = all(any((R/'p1'/attempt/gpu/p/'accepted.json').exists()
            for attempt in ('short_warmup_v2','short_warmup_retry')) for p in ('imagelanes','cpu_dct'))
        good = accepted['correctness_passed'] and accepted['resource_guard_passed']
        epochs.append(dict(gpu=gpu_name(gpu), model='SwinV2-T', domain='DCT',
            pipeline='FastLens' if recheck else system(base.name), task='training_epoch', timestamp=meta['timestamp'], git_commit=meta['git_commit'],
            precision=accepted['precision'], batch_size=64, effective_batch_size=1024, trial=0,
            samples=epoch['epoch_samples'], elapsed_s=epoch['epoch_seconds'], optimizer_updates=epoch['epoch_optimizer_updates'],
            training_loss=epoch['train_loss'], final_loss=epoch.get('final_loss'), initial_model_hash=accepted['initial_model_hash'],
            worker_count=warm['worker_count'], cpu_affinity=json.dumps(warm['cpu_affinity']), numa_binding='CPU/memory node 0',
            raw_result=accepted['raw_result'], command=meta['shell_command'], source_log=str((base/'run.log').relative_to(ROOT)),
            code_fingerprint=accepted.get('code_fingerprint'), implementation_cohort=accepted.get('implementation_cohort'),
            measurement_cohort=cohort_name if recheck else 'historical',
            valid=good, strict_matched=good and pair_complete,
            correctness_status='PASS: coverage, updates, finite loss; initialization restored after excluded warmup; sampled resource guard',
            notes='One complete epoch after short page-cache/dataloader/CUDA warmup. Training loss is final epoch mean, not last-batch loss. Strict pair requires equal initialization, recipe, sample-order and pool-membership digests; independent tensor-prefix gate. No convergence claim.'+
                (' Separate recheck; CPU control reused from another time. Shared-host conditions are not matched.' if recheck else '')))
    for path in sorted((R/'p1/mobile_b6').glob('*/*/accepted.json')):
        accepted=read(path)
        base=path.parent
        data=read(base/'training.json')
        epoch,=data['epochs']
        meta=read(base/'run.command.json')
        matched=all((base.parent/p/'accepted.json').exists() for p in ('imagelanes','cpu_dct'))
        epochs.append(dict(gpu=gpu_name(base.parent.name),model='MobileNetV2-DCT24',domain='DCT',
            pipeline=system(base.name),task='training_epoch',timestamp=meta['timestamp'],git_commit=meta['git_commit'],
            precision=data['precision'],batch_size=data['microbatch'],effective_batch_size=data['microbatch']*data['accumulation'],
            trial=0,samples=epoch['samples'],elapsed_s=epoch['seconds'],optimizer_updates=epoch['optimizer_updates'],
            training_loss=epoch['ce'],final_loss=epoch['final_loss'],initial_model_hash=data['initial_model_hash'],
            worker_count=8,cpu_affinity='0-23,48-71',numa_binding='CPU/memory node 0',
            raw_result=accepted['raw_result'],command=meta['shell_command'],source_log=str((base/'run.log').relative_to(ROOT)),
            valid=True,strict_matched=matched,correctness_status='PASS: coverage, updates, finite loss, restored initialization; B6 tensor/target gate',
            notes='One complete epoch after short page-cache/dataloader/CUDA warmup. '+accepted['notes']))
    for gpu in GPUS:
        for model, phase in [('SwinV2-T', 'p1'), ('MobileNetV2-DCT24', 'p2')]:
            for pipeline in ('CPU-DCT', 'FastLens'):
                have = [r for r in epochs if r['gpu'] == gpu and r['model'] == model and r['pipeline'] == pipeline and r['valid']]
                matched = [r for r in have if r['strict_matched']]
                entry(f'{phase}:{gpu}:{model}:{pipeline}', 'strict_training_epoch', gpu, model, 'DCT', pipeline, 1281167,
                      'DONE' if matched else 'BLOCKED' if have else 'TODO', ';'.join(r['raw_result'] for r in matched or have), bool(matched),
                      'Matched short-warmup full epoch accepted' if matched else 'Existing full epochs retained; shared-host interference or strict pair remains unresolved' if have
                      else 'One full epoch after short warmup; P2 deferred until preceding phases complete')

    inference = []
    unstable = {(gpu_name(r['gpu']), r['model'], system(r['pipeline']))
                for r in csvread(R/'e2e/inference_summary.csv') if r['measurement_status'] == 'PROVISIONAL'}
    for row in csvread(R/'e2e/inference_raw.csv'):
        row.update(gpu=gpu_name(row['gpu']), domain=row['domain'].upper(), pipeline=system(row['pipeline']), valid=True, strict_matched=False)
        raw = ROOT / row['raw_result'].split('#')[0]
        gatepath = raw.parents[2] / 'correctness.json'
        if gatepath.exists() and row['domain'] == 'DCT':
            gate = read(gatepath)
            row['strict_matched'] = strict_prediction_pass(gate)
            if not row['strict_matched']:
                row['valid'] = False
                row['correctness_status'] = 'REJECTED for strict table: full prediction equality failed'
                row['notes'] += '; retained raw observation; prediction agreement=' + str(gate['prediction_agreement'])
        elif row['domain'] == 'DCT':
            row['strict_matched'] = strict_prediction_pass(read(raw.parent/'results.json'))
            row['valid'] = row['strict_matched']
        # Stability failures are retained, but cannot complete a paper-ready cell.
        if 'PROVISIONAL' in row['notes'] or (row['gpu'], row['model'], row['pipeline']) in unstable:
            row['valid'] = False
            row['notes'] += '; PROVISIONAL: existing trial group failed stability threshold; raw observations retained'
        entry('existing:' + row['raw_result'], 'inference', row['gpu'], row['model'], row['domain'], row['pipeline'], row['samples'],
              'DONE', row['raw_result'], row['valid'], row['notes'])
        inference.append(row)
    for gpu in GPUS:
        for model in ('ViT-Ti', 'SwinV2-T-256-window8', 'MobileNetV2-DCT24', 'MobileNetV2-DCT32', 'ResNet-50-DCT24', 'ResNet-50-DCT64', 'eFUN-DCT', 'eFUN-L', 'eFUN-S', 'eFUN-S-Plus'):
            for pipeline in ('CPU-DCT', 'FastLens'):
                have = [r for r in inference if r['gpu'] == gpu and r['model'] == model and r['pipeline'] == pipeline and r['valid']]
                extra = model in ('eFUN-L', 'eFUN-S', 'eFUN-S-Plus')
                entry(f'inference:{gpu}:{model}:{pipeline}', 'inference', gpu, model, 'DCT', pipeline, 50000,
                      'DONE' if have else 'BLOCKED' if extra else 'TODO', ';'.join(r['raw_result'] for r in have), bool(have),
                      'Training constructor found, but no configured checkpoint/full-validation native adapter for this variant' if extra
                      else 'Reuse all accepted full-validation trials; one initial run for missing cells; repeat only suspicious results')
        for model in MODELS:
            for pipeline in SYSTEMS:
                names = ('SwinV2-T-256-window8',) if model == 'SwinV2-T' else ('MobileNetV2', 'MobileNetV2-DCT24')
                have = [r for r in inference if r['gpu'] == gpu and r['model'] in names and r['pipeline'] == pipeline and r['valid']]
                entry(f'system-inference:{gpu}:{model}:{pipeline}', 'system_inference', gpu, model,
                      'DCT' if pipeline == 'FastLens' else 'RGB', pipeline, 50000,
                      'DONE' if have else 'TODO', ';'.join(r['raw_result'] for r in have), bool(have),
                      'Deployment comparison; never compute a causal RGB/DCT ratio')
        for model in MODELS:
            for domain in ('RGB', 'DCT'):
                have = [r for r in ceilings if r['gpu'] == gpu and r['model'] == model and r['domain'] == domain]
                entry(f'ceiling:{gpu}:{model}:{domain}', 'training_model_only', gpu, model, domain, 'model_only', '',
                      'DONE' if have else 'TODO', ';'.join(r['raw_result'] for r in have), bool(have), 'Ceiling must match task, tensor shape, precision, optimizer and compilation')
    for task in ('h2d', 'memory'):
        for model in MODELS:
            for pipeline in SYSTEMS:
                entry(f'resources:{task}:{model}:{pipeline}', task, 'RTX 4090', model, 'DCT' if pipeline == 'FastLens' else 'RGB', pipeline,
                      16384, 'TODO', notes='Separate diagnostic run; no profiler in primary throughput trials')
    for ablation in ('crop_pushdown', 'direct_vs_grid', 'frequency_pushdown', 'raw_int16'):
        entry('ablation:' + ablation, ablation, 'RTX 4090', 'MobileNetV2-DCT24' if ablation == 'direct_vs_grid' else 'SwinV2-T',
              'DCT', 'FastLens', 16384, 'TODO', notes='Matched correctness control required; frequency/raw-int16 deferred if Phase 1 already complete')
    external_finished = bool(queue and queue['external_finished'])
    entry('foreign:580333', 'external_motivation_suite', 'RTX 4090', 'multiple', 'DCT', 'CPU-DCT/model_only', '', 'DONE' if external_finished else 'RUNNING',
          'benchmark_results/input_motivation_dct_training_20260926_nocudagraph/continuation.log', False,
          'External suite left untouched; completion checked by PID and creation time when wait controller is active')
    if queue:
        entry('p0:wait-controller', 'queue_controller', 'H100/RTX4090', 'SwinV2-T/MobileNetV2', 'RGB/DCT', 'five systems', '',
              'DONE' if queue['phase'] in ('RGB_SUBSET_FINISHED','P0_QUEUE_FINISHED') else 'RUNNING', 'results/p0/queue_state.json', False,
              'Current phase: '+queue['phase']+'; '+queue['scope'])
    ffcv_preparation = R/'preparation/p0_ffcv'
    if (ffcv_preparation/'verification.json').exists():
        verification = read(ffcv_preparation/'verification.json')
        entry('preparation:p0_ffcv', 'dataset_preparation', 'CPU', 'SwinV2-T/MobileNetV2', 'RGB', 'FFCV',
              verification['samples'], 'DONE' if verification['passed'] else 'FAILED',
              'results/preparation/p0_ffcv/verification.json', verification['passed'],
              'All labels/ordinals verified; three exact pixel checks. Writer completed; interpreter hung during exit and was terminated after independent verification. Conversion time is writer-only.')
    preflight = R/'checks/h100_shared_host_control/preflight.json'
    if preflight.exists():
        entry('diagnostic:shared_host_preflight', 'resource_isolation', 'H100', 'MobileNetV2', 'RGB', 'PyTorch', '',
              'BLOCKED', str(preflight.relative_to(ROOT)), False,
              'Workers migrated onto NUMA 0 and another bulk CPU job appeared; no performance run launched, no concurrency allowance established.')
    write(R/'experiment_manifest.csv', manifest, MANIFEST_FIELDS)
    write(R/'p0/training_window_raw.csv', p0, RAW_FIELDS)
    p0summary = summaries(p0)
    write(R/'p0/training_window_summary.csv', p0summary, SUMMARY_FIELDS)
    write(R/'training_table.csv', p0summary, SUMMARY_FIELDS)
    write(R/'p1/full_epoch_swin.csv', epochs, [*RAW_FIELDS,'final_loss'])
    full_epochs = [r for r in epochs if r.get('valid') and r.get('notes','').startswith('One complete epoch')]
    write(R/'p1/full_epoch_training_all_versions.csv',full_epochs,[*RAW_FIELDS,'final_loss'])
    selected_epochs=select_current(full_epochs)
    for gpu,batch in [('H100','low_load_recheck_20260929'),('RTX 4090','recheck_20260929')]:
        replacements=[r for r in selected_epochs if r['gpu']==gpu and r.get('measurement_cohort')==batch]
        if replacements:
            assert len(replacements)==1
            selected_epochs=[r for r in selected_epochs if not (r['gpu']==gpu and r['model']=='SwinV2-T' and r['pipeline']=='FastLens')]+replacements
    write(R/'p1/full_epoch_training.csv',selected_epochs,[*RAW_FIELDS,'final_loss'])
    write(R/'inference/dct_all_models.csv', [r for r in inference if r['domain'] == 'DCT'], RAW_FIELDS)
    direct = [r for r in inference if r['model'].startswith(('Swin', 'MobileNetV2')) and r['pipeline'] in SYSTEMS]
    write(R/'inference/system_baselines.csv', direct, RAW_FIELDS)
    inference_ceilings = csvread(R/'e2e/model_only_raw.csv')
    for row in inference_ceilings:
        row.update(gpu=gpu_name(row['gpu']), domain=row['domain'].upper(), task='inference_model_only', valid=True)
    write(R/'analysis/model_only.csv', ceilings + inference_ceilings, RAW_FIELDS)
    efficiencies = []
    for row in p0summary:
        matches = [r for r in ceilings if all(r[k] == row[k] for k in ('gpu', 'model', 'domain', 'precision', 'batch_size', 'effective_batch_size')) and r['valid']]
        if matches:
            ceiling = statistics.median(r['samples']/r['elapsed_s'] for r in matches)
            efficiencies.append(dict(row, model_only_images_per_s=ceiling, pipeline_efficiency=row['median_images_per_s']/ceiling,
                                     ceiling_raw_results=';'.join(r['raw_result'] for r in matches)))
    write(R/'analysis/pipeline_efficiency.csv', efficiencies, SUMMARY_FIELDS + ['model_only_images_per_s', 'pipeline_efficiency', 'ceiling_raw_results'])
    storage = csvread(R/'cost/storage.csv')
    for record in storage:
        if record['representation']=='cpu_stored_dct_optional':
            record['representation']='imagelanes_compact_v3'
            record['notes']+='; used by configured SwinV2 FastLens validation pipeline (contract pipelines.fastlens.manifest); CPU-DCT baseline reads JPEGs. Historical inventory label corrected; bytes unchanged.'
    if (ffcv_preparation/'verification.json').exists():
        conversion = read(ffcv_preparation/'conversion.json')
        verified = read(ffcv_preparation/'verification.json')
        command = read(R/'preparation/p0_ffcv_conversion.command.json')
        if verified['passed']:
            storage.append(dict(timestamp=command['timestamp'], git_commit=command['git_commit'],
                representation='ffcv_beton', dataset_view='imagenet512', split='train_p0_selection',
                samples=verified['samples'], payload_bytes=verified['payload_bytes'],
                metadata_bytes=verified['embedded_metadata_bytes']+verified['external_metadata_bytes'],
                index_bytes=verified['index_bytes'], padding_bytes=verified['padding_bytes'],
                representation_bytes=verified['representation_bytes'], allocated_bytes=verified['allocated_bytes'],
                jpeg_bytes=conversion['jpeg_bytes'],
                storage_amplification=verified['representation_bytes']/conversion['jpeg_bytes'],
                conversion_s=conversion['conversion_s'], verification_s=verified['verification_s'],
                notes='98,560-image training selection only; amplification against exactly those JPEGs. Raw RGB. Conversion is writer-only, one worker, CPU 25, nice 19; interpreter hung after output and was terminated. Full process completion time unavailable. Verification covers all labels/ordinals and three pixel samples.',
                raw_result='results/preparation/p0_ffcv/conversion.json;results/preparation/p0_ffcv/verification.json;results/preparation/p0_ffcv_conversion.command.json'))
    coordl_runs=sorted((R/'resources/raw/4090').glob('*/coordl/memory/accepted.json'))
    if coordl_runs:
        base=coordl_runs[0].parent
        result=read(base/'result.json')
        cache=result['coordl_cache']
        metadata=(base/'coordl_files.txt').stat().st_size
        storage.append(dict(timestamp=result['timestamp'],git_commit=result['git_commit'],
            representation='coordl_jpeg_cache',dataset_view='imagenet512',split='train_p0_selection',
            samples=cache['samples'],payload_bytes=cache['representation_bytes'],metadata_bytes=metadata,index_bytes=0,
            representation_bytes=cache['representation_bytes']+metadata,allocated_bytes=cache['allocated_bytes']+(base/'coordl_files.txt').stat().st_blocks*512,
            jpeg_bytes=cache['source_jpeg_bytes'],storage_amplification=(cache['representation_bytes']+metadata)/cache['source_jpeg_bytes'],
            conversion_s=cache['cache_population_s'],verification_s=None,temporary_peak_bytes=None,
            notes='Executed bounded CoorDL cache-fill pass; JPEG cache measured in tmpfs before own-cache cleanup; metadata file separate. Cache population time includes excluded decoding, not persistent offline conversion. '+cache['scope'],
            raw_result=str((base/'result.json').relative_to(ROOT))+'#/coordl_cache'))
    write(R/'resources/storage.csv', storage, list(storage[0]))
    write(R/'resource_cost_table.csv', storage, list(storage[0]))
    queue_note = ('P0 controller: '+queue['phase']+'. See p0/queue_state.json. '+queue['scope']+'.'
                  if queue else 'External RTX 4090 suite was running at audit; no new performance job launched.')
    status = ['# Live paper status', '', 'ImageNet-512 view. Replanned 2026-09-26; historical raw results retained. No missing values imputed.', '',
              'P0 is incomplete. Old adaptive/full-epoch expansion stopped. '+queue_note, '',
              '## A. Direct system training window', '', '65,536 measured images, 64 updates, microbatch 64, effective batch 1,024, BF16. Values: median [min, max] images/s from three unprofiled trials. RGB/DCT comparisons are deployment comparisons.', '',
              'Reused September 25 RGB results exclude 32 warmup updates. Historical CPU affinity/NUMA policy and git commit were not recorded. DALI D3 uses native sample order/crops. These limitations must accompany the table.', '',
              '| GPU | Model | Domain | PyTorch | DALI | CoorDL | FFCV | FastLens |', '|---|---|---|---|---|---|---|---|']
    for gpu in GPUS:
        for model in MODELS:
            for domain in ('RGB', 'DCT'):
                vals = []
                for pipeline in SYSTEMS:
                    group = next((r for r in p0summary if (r['gpu'],r['model'],r['domain'],r['pipeline']) == (gpu,model,domain,pipeline)), None)
                    applicable = (domain == 'DCT') == (pipeline == 'FastLens')
                    blocked = any(r['task']=='training_window' and r['gpu']==gpu and r['model']==model
                                  and r['pipeline']==pipeline and r['status']=='BLOCKED' for r in manifest)
                    vals.append(f"{group['median_images_per_s']:.2f} [{group['min_images_per_s']:.2f}, {group['max_images_per_s']:.2f}]" if group else ('BLOCKED' if blocked else 'TBD') if applicable else '—')
                status.append('| ' + ' | '.join([gpu, model, domain, *vals]) + ' |')
    status += ['', '## B. Strict full-epoch DCT training', '', 'H100 completed epochs pass sample/update/finite-loss checks. A separate 128-sample B6 native/CPU-DCT tensor check passes the existing one-coefficient-unit tolerance, with Mixup targets within the predefined tolerance (checks/swin_b6_native_cpu_tensors.json). This is prefix validation, not convergence evidence. Existing epochs overlapped other host jobs; their raw times are retained in p1/full_epoch_swin.csv and no causal ratio is published.', '',
               '| GPU | Model | DCT baseline | FastLens | Speedup |', '|---|---|---|---|---|']
    for gpu in GPUS:
        values = []
        for pipeline in ('CPU-DCT', 'FastLens'):
            matches = [r for r in epochs if r['gpu']==gpu and r['pipeline']==pipeline and r['strict_matched'] and r['valid']]
            values.append(statistics.median(r['samples']/r['elapsed_s'] for r in matches) if matches else None)
        ratio = f'{values[1]/values[0]:.3f}×' if all(values) else 'TBD'
        status.append('| ' + ' | '.join([gpu, 'SwinV2-T', *[f'{v:.2f}' if v else 'TBD' for v in values], ratio]) + ' |')
    status += ['', '## C. DCT inference', '', 'Full 50,000-image validation observations are retained in inference/dct_all_models.csv. Failed strict prediction equality is explicitly rejected. Stability-qualified cells remain pending.', '',
               '| GPU | Model | DCT baseline | FastLens | Speedup |', '|---|---|---|---|---|']
    infsummary = summaries(inference)
    write(R/'inference_table.csv', infsummary, SUMMARY_FIELDS)
    for gpu in GPUS:
        for model in sorted({r['model'] for r in inference if r['domain'] == 'DCT'}):
            vals = []
            for pipeline in ('CPU-DCT','FastLens'):
                group = next((r for r in infsummary if (r['gpu'],r['model'],r['pipeline']) == (gpu,model,pipeline)), None)
                vals.append(group['median_images_per_s'] if group else None)
            # Only a same-model, accepted DCT pair can produce a ratio.
            ratio = f'{vals[1]/vals[0]:.3f}×' if all(vals) else 'TBD'
            status.append('| ' + ' | '.join([gpu,model,*[f'{v:.2f}' if v else 'TBD' for v in vals],ratio]) + ' |')
    status += ['', '## D. Resources', '', 'Storage observations reused in resources/storage.csv; representations/splits and conversion-time gaps remain explicit. Required 16,384-image H2D/memory diagnostics pending.', '',
               '| System | Storage | H2D MB/img | GPU peak | Host peak |', '|---|---|---|---|---|']
    for pipeline in SYSTEMS:
        representation={'PyTorch':'jpeg','DALI':'jpeg','CoorDL':'jpeg','FFCV':'ffcv_beton','FastLens':'imagelanes_prepared'}[pipeline]
        stored=next(r for r in storage if r['representation']==representation and r['dataset_view']=='imagenet512' and r['split']=='val')
        size=f"{int(stored['representation_bytes'])/1e9:.3f} GB"
        if pipeline=='CoorDL':
            size+=' JPEG + cache TBD'
        status.append(f'| {pipeline} | {size} | TBD | TBD | TBD |')
    status += ['', 'Storage cells above cover the complete 50,000-image validation view; decimal GB. Prepared representations include measured indexes/metadata. These are not extrapolations from the P0 training subset.']
    if (ffcv_preparation/'verification.json').exists():
        status += ['', 'The separate 98,560-image FFCV training selection is prepared and independently verified. Its measured size and writer-only conversion time are recorded in resources/storage.csv, with shutdown failure and verification scope disclosed; it is not a full-training-dataset storage comparison.']
    status += ['', '## E. Ablations', '', '| Ablation | Baseline | Optimized | Delta |', '|---|---|---|---|']
    for name in ('Crop pushdown', 'Direct vs full grid', 'Frequency pushdown', 'Raw int16'):
        status.append(f'| {name} | TBD | TBD | TBD |')
    status += ['', '## Execution and stop condition', '',
               'Order: audit/reuse → P0 → Swin full epochs → DCT inference → matched model ceilings → RTX 4090 resources → crop → direct/grid. Stop expanding when the six Phase-1 requirements are satisfied. Frequency/raw-int16 and MobileNet full epochs follow only if Phase 1 has not already met its stop condition, otherwise Phase 2.', '',
               'Both RTX 4090 and H100 attach to NUMA node 0, CPUs 0–23,48–71. New runs bind CPU/memory to node 0. The evaluation jobs run sequentially. Other GPU jobs may coexist only with verified remote CPU/NUMA placement and no observed input I/O; unverified competing work is waited out before timing. External tasks are not terminated.', '',
               'No whole-epoch DALI/FFCV/CoorDL, extra model epochs, layout/batch/prefetch sweeps, cold I/O, convergence or RTX PRO 6000 work is scheduled.', '',
               'Audit: inventory/replan_host_audit.txt. Machine-readable queue and reuse decisions: experiment_manifest.csv. Regenerate using `python fastlens/benchmarks/development/historical/replan.py` (read-only with respect to raw results).', '']
    (R/'PAPER_STATUS.md').write_text('\n'.join(status))
    print(json.dumps(dict(manifest_entries=len(manifest), reusable_p0_trials=len(p0), training_ceiling_trials=len(ceilings), completed_epoch_observations=len(epochs))))


if __name__ == '__main__':
    main()
