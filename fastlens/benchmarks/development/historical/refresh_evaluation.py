"""Serial, version-separated refresh; never overwrite an existing measurement."""
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
from fastlens.benchmarks.execution.inference_load import run
from fastlens.benchmarks.execution.revision_results import REVISION
from fastlens.benchmarks.execution.revision_results import SNAPSHOT
from fastlens.benchmarks.execution.revision_results import verify_snapshot
from fastlens.benchmarks.execution.revision_results import cohort
from fastlens.benchmarks.execution.revision_results import file_sha
from fastlens.benchmarks.common import gpu_uuid
from fastlens.benchmarks.development.historical.run_phase2_repeats import predictions
from fastlens.benchmarks.system_dct_major.common import parse_manifest


def read(path): return json.loads(Path(path).read_text())
def rows(path): return list(csv.DictReader(Path(path).open()))
def save(path, value): Path(path).write_text(json.dumps(value, indent=2)+'\n')


def write_csv(path, records, fields):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def report():
    for script in ('fastlens/benchmarks/development/historical/replan.py', 'fastlens/benchmarks/development/reports/deployment_report.py'):
        subprocess.run([sys.executable, script], cwd=ROOT, check=True)
    regressions = []
    for task, name, field in [('inference', 'deployment_inference_summary.csv', 'model'),
                              ('training_window', 'training_window_summary.csv', 'model')]:
        old = rows(REVISION/'before/results/p0'/name)
        new = rows(ROOT/'results/p0'/name)
        for row in new:
            if row['pipeline'] not in ('FastLens', 'CoorDL'): continue
            if 'refresh_20260929' not in row['raw_results']: continue
            prior = next(r for r in old if all(r[k] == row[k] for k in ('gpu', field, 'pipeline')))
            if row['raw_results'] == prior['raw_results']: continue
            before, after = float(prior['median_images_per_s']), float(row['median_images_per_s'])
            regressions.append(dict(task=task, gpu=row['gpu'], model=row[field], pipeline=row['pipeline'],
                old_images_per_s=before, current_images_per_s=after, change_percent=100*(after/before-1),
                status='OBSERVED_DECREASE' if after < before else 'OBSERVED_INCREASE',
                old_raw=prior['raw_results'], current_raw=row['raw_results'],
                notes='Observed version/cohort comparison, not isolated code causality. Historical host load and incomplete version fingerprints retained.'))
    historical_epochs = rows(REVISION/'before/results/p1/full_epoch_training.csv')
    for row in rows(ROOT/'results/p1/full_epoch_training.csv'):
        if 'short_warmup_refresh_20260929' not in row['raw_result']: continue
        prior = next(r for r in historical_epochs if all(r[k]==row[k] for k in ('gpu','model','pipeline')))
        before = int(prior['samples'])/float(prior['elapsed_s'])
        after = int(row['samples'])/float(row['elapsed_s'])
        regressions.append(dict(task='full_epoch',gpu=row['gpu'],model=row['model'],pipeline=row['pipeline'],
            old_images_per_s=before,current_images_per_s=after,change_percent=100*(after/before-1),
            status='OBSERVED_DECREASE' if after<before else 'OBSERVED_INCREASE',
            old_raw=prior['raw_result'],current_raw=row['raw_result'],
            notes='One complete epoch per implementation; shared-host load differs. No isolated code-causality claim.'))
    fields='task gpu model pipeline old_images_per_s current_images_per_s change_percent status old_raw current_raw notes'.split()
    write_csv(REVISION/'performance_changes.csv', regressions, fields)
    text = ['# Version refresh and observed performance changes', '',
            'Historical raw results are retained. Changes below compare observed cohorts; they do not isolate code from shared-host load.', '',
            '| Task | GPU | Model | System | Before img/s | Current img/s | Change |',
            '|---|---|---|---|---:|---:|---:|']
    for r in regressions:
        text.append(f"| {r['task']} | {r['gpu']} | {r['model']} | {r['pipeline']} | {r['old_images_per_s']:.2f} | {r['current_images_per_s']:.2f} | {r['change_percent']:+.2f}% |")
    (REVISION/'PERFORMANCE_CHANGES.md').write_text('\n'.join(text)+'\n')
    with (ROOT/'results/PAPER_STATUS.md').open('a') as stream:
        stream.write('\n## Current-code refresh\n\nSee `results/version_refresh_20260929/status.json` and `PERFORMANCE_CHANGES.md`. '
                     'Historical completion does not establish current-version coverage; old and new native implementations are not pooled.\n')


def environment(gpu, profile):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid(gpu), DCTNET_PROFILE=profile,
                TMPDIR=str(Path.home()/'tmp'), PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                FASTLENS_EVALUATION_SNAPSHOT=str(SNAPSHOT))


def original_data(row):
    path, *pointer = row['raw_result'].split('#')
    data = read(ROOT/path)
    measured = data['repeats'][int(pointer[0].rsplit('/', 1)[1])] if pointer else data
    return data, measured


def inference(row, profile, trial, native_cohort):
    gpu = '4090' if row['gpu'] == 'RTX 4090' else 'H100'
    native = row['pipeline'] == 'FastLens'
    prefix = 'normal_' if profile == 'vitti' and not native else ''
    output = ROOT/f'results/p0/deployment_passes/{gpu}/{profile}/{prefix}refresh_20260929_{row["pipeline"].lower()}_trial_{trial}'
    if (output/'run.log').exists() and 'payload identity changed after contract creation' in (output/'run.log').read_text():
        output = output.with_name(output.name+'_hardlink_identity')
    if (output/'accepted.json').exists(): return 'REUSED'
    if (output/'run.log').exists() or (output/'failure.json').exists(): return 'BLOCKED: existing incomplete attempt retained'
    output.mkdir(parents=True, exist_ok=True)
    old, old_measured = original_data(row)
    env = environment(gpu, profile)
    source = ROOT/row['raw_result'].split('#')[0]
    if 'repeats' in old:
        contract = read(source.parent/'contract.json')
        pipeline = ('dct_major_pushdown' if profile == 'vitti' else 'galp') if native else 'coordl'
        contract['execution'].update(repeats=1, aggregate_exclude_first_repeat=False, warmup_batches=0)
        contract['pipelines']['enabled'] = [pipeline]
        if profile == 'vitti' and native:
            from fastlens.benchmarks.system_dct_major.common import file_identity
            # The raw-int16 control added hardlinks to the first 16 source shards.
            # A link-count change modifies ctime without modifying payload contents.
            links = ROOT/'results/preparation/raw_int16_phase2_16384/compressed'
            linked = [Path(s['fls_path']) for s in parse_manifest(links/'manifest.bin')['shards']]
            changes = []
            for payload in contract['dataset']['dct_major_storage']['payloads']:
                path = Path(payload['path'])
                actual = file_identity(path)
                if actual == payload['file_identity']: continue
                delta = {k for k,v in actual.items() if payload['file_identity'][k] != v}
                assert delta == {'ctime_ns'} and any(path.samefile(p) for p in linked), 'Unexplained source payload change'
                changes.append(dict(path=str(path), previous=payload['file_identity'], observed=actual,
                                    reason='Same inode/size/mtime; verified link to the compression-control subset'))
                payload['file_identity'] = actual
                payload['sha256'] = file_sha(path)
            save(output/'payload_identity_refresh.json', changes)
        if not native:
            contract['pipelines']['coordl'].update(decoder_device='mixed', file_list=str(output/'coordl_file_list.txt'))
        cp = output/'contract.json'
        save(cp, contract)
        raw = output/f'pipeline_{pipeline}.json'
        if native:
            driver = [sys.executable, ('fastlens/benchmarks/development/historical/vit_inference_once.py' if profile == 'vitti' else 'fastlens/benchmarks/development/historical/swin_inference_once.py'), '--pipeline', pipeline]
            config = contract['pipelines']['galp' if profile == 'swinv2' else 'dct_major_pushdown']
            # ViT stores the source manifest in the common FASTLENS configuration.
            manifest = config.get('manifest', contract['pipelines'].get('galp', {}).get('manifest'))
            paths = [Path(s['fls_path']) for s in parse_manifest(Path(manifest))['shards']]
        else:
            python = str(Path.home()/'tmp/coordl-build-cu128/coordl-python')
            driver = ([python, 'fastlens/benchmarks/development/historical/vit_inference_once.py', '--pipeline', 'coordl'] if profile == 'vitti' else
                      [python, 'fastlens/benchmarks/development/historical/paired_rgb_once.py', '--profile', profile, '--pipeline', 'coordl'])
            paths = [Path(s['path']) for s in read(contract['dataset']['sample_manifest'])['samples']]
        command = [*driver, '--contract', str(cp), '--output', str(raw)]
    else:
        base = ROOT/'fastlens/data/compressed/imagenet512_val_block_major'
        paths = [Path(s['fls_path']) for s in parse_manifest(base/'manifest.bin')['shards']]
        module = ['-m', 'fastlens.benchmarks.dct_models.efun', 'evaluate_shards'] if profile == 'efun' else ['-m', 'fastlens.benchmarks.dct_models.evaluate_shards']
        command = [sys.executable, *module, '--batch-size', str(row['batch_size']), '--new-data', str(base),
            '--input-geometry', 'source512', '--pushdown', 'off', '--crop-execution-mode', 'vector-range-read-selected-decode',
            '--output-layout', 'projected', '--native-runtime', 'b6', '--shards-per-activation', '1', '--physical-prefix',
            '--baseline-dir', str(output/'unused'), '--count', '50000', '--output-dir', str(output)]
        raw = output/'N_50000.json'
        # Tensor validation runs separately; the 50k timing has no validation instrumentation.
        if trial == 1 or not any(output.parent.glob('refresh_20260929_fastlens_trial_*/accepted.json')):
            check = output/'tensor_check'
            check.mkdir()
            validation = list(command)
            validation[validation.index('--count')+1] = '8'
            validation[validation.index('--output-dir')+1] = str(check)
            run(['numactl', '--cpunodebind=0', '--membind=0', *validation, '--verify'], check, env, paths)
            assert read(check/'N_8.json')['checks'], 'Missing tensor correctness observations'
    run(['numactl', '--cpunodebind=0', '--membind=0', *command], output, env, paths)
    data = read(raw)
    measured = data['repeats'][0] if 'repeats' in data else data
    keys = ('images', 'sample_trace', 'correct_top1') if 'repeats' in data else ('samples', 'sample_ids', 'top1')
    assert measured[keys[0]] == 50000
    assert all(measured[k] == old_measured[k] for k in keys), 'Sample/count/accuracy mismatch'
    assert predictions(data) == predictions(old), 'Full prediction mismatch'
    checkpoint = data['model']['checkpoint_sha256'] if 'repeats' in data else data['checkpoint']
    reference = old['model']['checkpoint_sha256'] if 'repeats' in old else old['checkpoint']
    assert checkpoint == reference, 'Checkpoint mismatch'
    if 'repeats' in data and native:
        actual, expected = np.load(data['semantic_artifact']), np.load(old['semantic_artifact'])
        for name in ('input_0', 'input_1'):
            delta = np.abs(actual[name]-expected[name])
            assert delta.max() <= .001 and delta.mean() <= .0001, 'DCT tensor mismatch'
    meta = read(output/'run.command.json')
    assert meta['code_unchanged_during_run']
    elapsed = measured['repeat_scope_seconds'] if 'repeats' in data else data['e2e_seconds']
    accepted = dict(row, timestamp=meta['timestamp'], git_commit=meta['git_commit'], trial=trial,
        elapsed_s=elapsed, images_per_s=50000/elapsed, code_fingerprint=meta['code_fingerprint'],
        top1=measured['accuracy_top1']*100 if 'repeats' in data else data['top1'],
        top5=measured['accuracy_top5']*100 if 'repeats' in data else data['top5'],
        implementation_cohort=native_cohort if native else 'coordl_mixed',
        raw_result=str(raw.relative_to(ROOT))+('#/repeats/0' if 'repeats' in data else ''),
        correctness='PASS: all 50000 sample IDs/predictions, count, top1 and checkpoint equal to accepted reference; native tensor gate; stable source/binary fingerprint',
        notes=row['notes']+' Version refresh; previous results retained. Decoder=mixed for CoorDL. Shared-host load recorded; no isolation inferred.')
    save(output/'accepted.json', accepted)
    return 'DONE'


def training(gpu, model, trial, native_cohort):
    output = ROOT/f'results/p0/raw/{gpu}/{model}/fastlens/trial_{trial}_refresh_20260929'
    if (output/'accepted.json').exists(): return 'REUSED'
    if (output/'run.log').exists(): return 'BLOCKED: incomplete attempt retained'
    output.mkdir(parents=True, exist_ok=True)
    env = environment(gpu, 'mobilenet24' if model == 'mobilenet' else 'swinv2')
    env.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', FASTLENS_NSYS_FINE='0')
    command = ['numactl', '--cpunodebind=0', '--membind=0', sys.executable, 'fastlens/benchmarks/paper/shared/training_window_rgb.py',
               '--gpu', gpu, '--model', model, '--pipeline', 'imagelanes', '--trial', str(trial), '--output', str(output)]
    run(command, output, env, None)
    data = read(output/'result.json')
    previous = read(output.parent/f'trial_{trial}_tail_fixed/result.json')
    assert data['correctness_passed'] and data['images'] == 65536 and data['updates'] == 64
    assert math.isfinite(data['training_loss'])
    for k in ('sample_order_sha256', 'initial_model_hash', 'batch', 'accumulation', 'precision'):
        assert data[k] == previous[k], f'Training contract changed: {k}'
    metadata = read(output/'run.command.json')
    assert metadata['code_unchanged_during_run']
    save(output/'accepted.json', dict(raw_result=str((output/'result.json').relative_to(ROOT)), correctness_passed=True,
        implementation_cohort=native_cohort, code_fingerprint=metadata['code_fingerprint']))
    return 'DONE'


def full_swin(gpu, base=None):
    """Replace the two pre-optimization native epochs; reuse their CPU controls."""
    refresh_tables = base is None
    base = base or ROOT/f'results/p1/short_warmup_refresh_20260929/{gpu}/fastlens'
    if (base/'accepted.json').exists(): return
    if (base/'run.log').exists(): raise RuntimeError('Incomplete full epoch retained: '+str(base))
    prior = ROOT/f'results/p1/short_warmup_v2/{gpu}/fastlens'
    accepted = read(prior/'accepted.json')
    metadata = read(prior/'run.command.json')
    command = [v.replace(str(prior),str(base)) for v in metadata['command']]
    output = Path(command[command.index('--output-dir')+1])
    output.mkdir(parents=True)
    manifest = Path(command[command.index('--run-manifest')+1])
    original_manifest = prior/'runs/B6/seed_11997733/run_manifest.json'
    from fastlens.benchmarks.training_pls.contracts import code_version, condition_identity_hash
    from fastlens.benchmarks.training_pls.recipe import sha256_json
    contract = json.loads(original_manifest.read_text().replace(str(prior),str(base)))
    contract['code_version'] = code_version(ROOT)
    contract['condition_hash'] = condition_identity_hash(contract)
    contract['run_manifest_hash'] = sha256_json({k:v for k,v in contract.items() if k!='run_manifest_hash'})
    save(manifest, contract)
    env = environment(gpu,'swinv2')
    env.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    driver_sha = file_sha(ROOT/'fastlens/benchmarks/development/historical/full_epoch_short_warmup.py')
    run(['numactl','--cpunodebind=0','--membind=0',*command],base,env,None)
    assert file_sha(ROOT/'fastlens/benchmarks/development/historical/full_epoch_short_warmup.py') == driver_sha
    pause = read(output/'pause_result.json')
    assert pause['processed_images']==1281167 and pause['optimizer_update']==1252
    assert all(pause['integration_checks'][k] for k in ('coverage_counters_valid','gradients_finite','loss_finite','parameters_finite','contract_hash_valid'))
    metrics = [json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
    epoch, = [r for r in metrics if r.get('record_type')=='train' and r.get('scope')=='epoch']
    for key in ('sample_order_digest','pool_membership_digest','epoch_samples','epoch_optimizer_updates'):
        assert epoch[key] == accepted['epoch'][key], 'Full epoch semantics changed: '+key
    assert math.isfinite(epoch['train_loss'])
    warm_path = output.parent/'short_warmup.json'
    warm = read(warm_path)
    assert warm['initial_state_restored'] and warm['initial_model_hash']==accepted['initial_model_hash']
    meta = read(base/'run.command.json')
    snapshot = read(SNAPSHOT)
    save(base/'accepted.json',dict(accepted,epoch=epoch,warmup=str(warm_path.relative_to(ROOT)),
        raw_result=str((output/'metrics.jsonl').relative_to(ROOT)),
        implementation_cohort='native_'+snapshot['native_sha256'][:16],
        code_fingerprint=meta['code_fingerprint'],full_epoch_driver_sha256=driver_sha,
        notes='Replacement native epoch after optimization; original epoch retained; CPU matched control reused. Shared-host load recorded.'))
    if refresh_tables:
        report()


def main():
    verify_snapshot()
    snapshot = read(SNAPSHOT)
    native_cohort = 'native_'+snapshot['native_sha256'][:16]
    before = rows(REVISION/'before/results/p0/deployment_inference_raw.csv')
    profiles = {'ViT-Ti DCT':'vitti', 'SwinV2-T DCT':'swinv2', 'MobileNetV2-DCT24':'mobilenet24',
        'MobileNetV2-DCT32':'mobilenet32', 'ResNet-50-DCT24':'resnet24', 'ResNet-50-DCT64':'resnet64', 'eFUN-DCT':'efun',
        'ViT-Ti RGB':'vitti', 'SwinV2-T RGB':'swinv2', 'MobileNetV2 RGB':'mobilenet24', 'ResNet-50 RGB':'resnet24'}
    groups = {}
    for r in before:
        if r['pipeline']=='FastLens' or (r['pipeline']=='CoorDL' and r['model'] in profiles):
            groups.setdefault((r['gpu'],r['model'],r['pipeline']), []).append(r)
    jobs = []
    extras = []
    for key, group in groups.items():
        native = key[2]=='FastLens'
        current = [r for r in group if cohort(r)==native_cohort] if native else group
        reference = max(current or group, key=lambda r:r['timestamp'])
        for trial in range(len(current)+1, 4):
            item = ('inference', reference, profiles[reference['model']], trial)
            (jobs if not native or not current and trial==1 else extras).append(item)
    # Coverage first, then the 12 requested bounded training trials, then remaining repeats.
    jobs += [('training', gpu, model, trial) for gpu in ('H100','4090') for model in ('swinv2','mobilenet') for trial in range(3)]
    jobs += extras
    state = dict(status='RUNNING', pid=os.getpid(), total=len(jobs), completed=[], active=None,
                 policy='Sequential GPUs; other-GPU/host activity annotated; no extra full epochs automatically.')
    report()
    for kind, first, second, trial in jobs:
        label = [kind, first['gpu'], first['model'], first['pipeline'], trial] if kind=='inference' else [kind,first,second,trial]
        state['active'] = label
        save(REVISION/'status.json', state)
        print('START', label, flush=True)
        verify_snapshot()
        try:
            status = inference(first, second, trial, native_cohort) if kind=='inference' else training(first,second,trial,native_cohort)
        except (AssertionError, subprocess.CalledProcessError) as error:
            status = 'FAILED: '+str(error)
        state['completed'].append(dict(experiment=label, status=status))
        save(REVISION/'status.json', state)
        if status != 'REUSED': report()
        print(status, label, flush=True)
    state.update(active=None, status='COMPLETE' if all(x['status'] in ('DONE','REUSED') for x in state['completed']) else 'BLOCKED_CELLS')
    save(REVISION/'status.json', state)


if __name__ == '__main__': main()
