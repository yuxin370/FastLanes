#!/usr/bin/env python3
"""Reuse executed observations under the accuracy/deployment/causal evaluation plan."""
import csv
import hashlib
import json
from functools import cache
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
R = ROOT/'results'
from fastlens.benchmarks.execution.revision_results import select_current
SYSTEMS = ['PyTorch', 'DALI', 'CoorDL', 'FFCV']
GPUS = ['RTX 4090', 'H100']
RECHECKS = ('recheck_20260929', 'low_load_recheck_20260929', 'remaining_inference_recheck_20260929')
LATEST = 'latest_inference_20260930'
MANIFEST = 'experiment_id phase task gpu architecture rgb_model dct_model model_variant domain pipeline checkpoint sample_count precision batch_size status result_file correctness notes'.split()
RAW = 'timestamp git_commit gpu architecture_family model model_variant domain checkpoint pipeline comparison_type task precision batch_size trial samples optimizer_updates elapsed_s images_per_s model_only_images_per_s top1 top5 storage_bytes metadata_bytes issued_read_bytes h2d_bytes h2d_copy_count gpu_peak_process_bytes torch_peak_allocated_bytes torch_peak_reserved_bytes native_peak_bytes host_rss_peak_bytes pinned_peak_bytes correctness notes raw_result accuracy_reference implementation_cohort code_fingerprint measurement_cohort'.split()


@cache
def read(path):
    return json.loads(Path(path).read_text())


@cache
def checkpoint_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def write(path, data, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data)


def discover_pairs():
    inventory = read(R/'inventory/models.json')['models']
    pairs = []
    for model in inventory:
        c = model.get('configuration', {})
        identifier = model['id']
        if 'rgb_checkpoint_name' in c:
            key = 'swinv2' if 'swinv2' in identifier else 'vitti'
            family = 'SwinV2-T' if key == 'swinv2' else 'ViT-Ti'
            actual = R/'raw/H100'/key/'inference'
            rgb = read(actual/'pipeline_pytorch.json')['model']
            dct = read(actual/'pipeline_rgbnomore.json')['model']
            pairs.append(dict(architecture_family=family, rgb_model=family+' RGB', dct_model=family+' DCT',
                rgb_checkpoint=rgb['checkpoint'], dct_checkpoint=dct['checkpoint'], dct_profile=c['dct_profile_id'],
                inference_supported=True, training_supported=True, relationship='registered_architecture_pair',
                source=model['source'], key=key, legacy_model='SwinV2-T-256-window8' if key=='swinv2' else 'ViT-Ti',
                notes='Registered RGB/DCT pair. '+('Inference 256/window8; training 224/window7, separate workloads.' if key=='swinv2' else '')))
        elif identifier in ('mobilenet24', 'mobilenet32', 'resnet24', 'resnet64', 'efun'):
            mobile = identifier.startswith('mobilenet')
            residual = identifier.startswith('resnet')
            family = 'MobileNetV2' if mobile else 'ResNet-50' if residual else 'eFUN'
            rgb_key = 'mobilenet24' if mobile else 'resnet24' if residual else 'efun'
            rgb = read(R/f'raw/H100/{rgb_key}/inference/rgb_pytorch/trial_1/RGB_pytorch_50000.json')
            dct_name = family+'-DCT'+identifier[-2:] if mobile or residual else 'eFUN-DCT'
            pairs.append(dict(architecture_family=family, rgb_model=family+' RGB' if mobile or residual else 'EfficientNet-B0 reference',
                dct_model=dct_name, rgb_checkpoint=rgb['checkpoint'], dct_checkpoint=c['checkpoint'], dct_profile=identifier,
                inference_supported=True, training_supported=True,
                relationship='upstream_architecture_family' if mobile or residual else 'reference_only',
                source='fastlens/benchmarks/dct_models/DCTNet/classification/models/imagenet/'+('mobilenetv2.py' if mobile else 'resnet.py') if mobile or residual else 'fastlens/benchmarks/dct_models/FUN/timm/models/eFUN.py;fastlens/benchmarks/dct_models/rgb.py',
                key=identifier, legacy_model=dct_name,
                notes='Upstream RGB initialization/architecture; DCT stem/channel/stride changes remain deployment differences.' if mobile or residual else 'EfficientNet-B0 is the artifact RGB reference. No exact RGB eFUN counterpart/checkpoint is configured; do not present this as an exact paired architecture.'))
        elif model['task'] == 'training,model-only':
            pairs.append(dict(architecture_family='eFUN', rgb_model='', dct_model=identifier,
                rgb_checkpoint='', dct_checkpoint='', dct_profile=identifier, inference_supported=False,
                training_supported=True, relationship='unpaired_training_constructor', source=model.get('source', ''),
                key=identifier, legacy_model=identifier,
                notes='Constructor discovered automatically; no configured evaluation checkpoint or exact RGB counterpart.'))
    return pairs


def collect_quality(pairs):
    observations, checks = {}, []
    samples_path = R/'raw/H100/mobilenet24/inference/cpu_dct/trial_1/samples_50000.json'
    labels = {r['logical_sample_id']: r['model_label'] for r in read(samples_path)}
    for pair in pairs:
        if not pair['inference_supported']:
            continue
        for domain in ('RGB', 'DCT'):
            model = pair['rgb_model'] if domain=='RGB' else pair['dct_model']
            if model in observations:
                continue
            if pair['key']=='vitti' and domain=='RGB' and not (R/'p0/deployment_passes/4090/vitti/normal_pytorch/accepted.json').exists():
                continue
            key = pair['key']
            if key in ('vitti', 'swinv2'):
                pipeline = 'pytorch' if domain=='RGB' else 'rgbnomore'
                path = R/f'raw/H100/{key}/inference/pipeline_{pipeline}.json'
                repeat=1
                if key=='vitti' and domain=='RGB':
                    quality_gpu='H100' if (R/'p0/deployment_passes/H100/vitti/normal_pytorch/accepted.json').exists() else '4090'
                    path=R/f'p0/deployment_passes/{quality_gpu}/vitti/normal_pytorch/pipeline_pytorch.json'
                    repeat=0
                data = read(path)
                run = data['repeats'][repeat]
                count, correct = run['images'], run['correct_top1']
                top1, top5 = run['accuracy_top1']*100, run['accuracy_top5']*100
                checkpoint, batch, precision = data['model']['checkpoint'], data['execution']['batch_size'], data['execution']['precision']
                trace = run['sample_trace']
                trace_count = len(trace) if isinstance(trace, list) else trace['count']
                if count != 50000 or trace_count != 50000 or abs(top1-100*correct/count)>1e-9:
                    raise ValueError(f'Quality count/trace failed: {path}')
                raw = str(path.relative_to(ROOT))+f'#/repeats/{repeat}'
                check = 'Logged integer correct count and complete sample trace'
            else:
                if domain=='RGB':
                    key = 'mobilenet24' if key.startswith('mobilenet') else 'resnet24' if key.startswith('resnet') else key
                path = R/f'raw/H100/{key}/inference'/('rgb_pytorch/trial_1/RGB_pytorch_50000.json' if domain=='RGB' else 'cpu_dct/trial_1/J_50000.json')
                data = read(path)
                ids, predictions = data['sample_ids'], data['predictions']
                if len(ids)!=50000 or len(predictions)!=50000 or set(ids)!=set(labels):
                    raise ValueError(f'Quality sample coverage failed: {path}')
                count = len(ids)
                correct = sum(p==labels[i] for i,p in zip(ids,predictions))
                top1, top5 = data['top1'], data['top5']
                if abs(top1-100*correct/count)>1e-9:
                    raise ValueError(f'Prediction-derived correct count differs: {path}')
                checkpoint, batch, precision = data['checkpoint'], data['batch_size'], data['precision']
                raw = str(path.relative_to(ROOT))
                check = 'Correct count recomputed from all predictions and model_label mapping: '+str(samples_path.relative_to(ROOT))
            if not Path(checkpoint).is_file():
                raise FileNotFoundError(checkpoint)
            observations[model] = dict(model=model, architecture_family=pair['architecture_family'], domain=domain,
                checkpoint=checkpoint, gpu=('H100' if key!='vitti' or domain!='RGB' or quality_gpu=='H100' else 'RTX 4090'), top1_accuracy=top1, top5_accuracy=top5, correct_predictions=correct,
                total_samples=count, precision=precision, batch_size=batch, evaluation_pipeline='PyTorch RGB' if domain=='RGB' else 'CPU-DCT',
                dataset_view='ImageNet-512', correctness='PASS', raw_result=raw,
                notes=check+'; checkpoint accuracy evidence independent of throughput stability; not training convergence evidence.')
            checks.append(dict(model=model, raw_result=raw, correct_predictions=correct, total_samples=count, passed=True, method=check))
    (R/'quality').mkdir(exist_ok=True)
    (R/'quality/accuracy_reuse_checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    write(R/'quality/model_accuracy.csv', observations.values(), list(next(iter(observations.values()))))
    return observations


def family_for(model):
    if model.startswith('Swin'): return 'SwinV2-T'
    if model.startswith('ViT'): return 'ViT-Ti'
    if model.startswith('Mobile'): return 'MobileNetV2'
    if model.startswith('ResNet'): return 'ResNet-50'
    return 'eFUN'


def main():
    pairs = discover_pairs()
    quality = collect_quality(pairs)
    for p in pairs:
        p.update(rgb_accuracy_available=p['rgb_model'] in quality, dct_accuracy_available=p['dct_model'] in quality)
    write(R/'inventory/workload_pairs.csv', pairs, 'architecture_family rgb_model rgb_checkpoint dct_model dct_checkpoint dct_profile rgb_accuracy_available dct_accuracy_available inference_supported training_supported relationship source notes'.split())
    deltas = [dict(architecture_family=p['architecture_family'], rgb_model=p['rgb_model'], dct_model=p['dct_model'],
        rgb_top1=quality.get(p['rgb_model'],{}).get('top1_accuracy'), dct_top1=quality[p['dct_model']]['top1_accuracy'],
        accuracy_delta_pp=quality[p['dct_model']]['top1_accuracy']-quality[p['rgb_model']]['top1_accuracy'] if p['rgb_model'] in quality else None,
        relationship=p['relationship'], rgb_raw=quality.get(p['rgb_model'],{}).get('raw_result'), dct_raw=quality[p['dct_model']]['raw_result'])
        for p in pairs if p['inference_supported']]
    write(R/'quality/accuracy_deltas.csv', deltas, list(deltas[0]))
    accepted = {source for summary in rows(R/'inventory/inference_before_deployment_plan.csv') for source in summary['raw_results'].split(';')}
    observations, configs = [], []
    from fastlens.benchmarks.development.historical.replan import system
    from fastlens.benchmarks.development.historical.replan import gpu_name
    for old in rows(R/'e2e/inference_raw.csv'):
        if old['raw_result'] not in accepted:
            continue
        if old['model']=='ViT-Ti' and old['domain'].upper()=='RGB':
            continue  # Historical fixed-224 crop is not the checkpoint's normal RGB evaluation.
        rawpath = ROOT/old['raw_result'].split('#')[0]
        data = read(rawpath)
        domain, family = old['domain'].upper(), family_for(old['model'])
        pair = next(p for p in pairs if p['inference_supported'] and p['architecture_family']==family
                    and (domain=='RGB' or p['legacy_model']==old['model']))
        model = pair['rgb_model'] if domain=='RGB' else pair['dct_model']
        checkpoint = data['model']['checkpoint'] if 'repeats' in data else data['checkpoint']
        if Path(checkpoint).resolve() != Path(quality[model]['checkpoint']).resolve():
            raise ValueError(f'Unexpected checkpoint: {rawpath}')
        pipeline = system(old['pipeline'])
        measured = data['repeats'][int(old['raw_result'].rsplit('/',1)[1])] if 'repeats' in data else data
        actual_top1 = measured['accuracy_top1']*100 if 'repeats' in data else measured['top1']
        actual_top5 = measured['accuracy_top5']*100 if 'repeats' in data else measured['top5']
        row = dict(old, gpu=gpu_name(old['gpu']), architecture_family=family, model=model,
            model_variant=model, domain=domain, checkpoint=checkpoint, pipeline=pipeline,
            comparison_type='matched_dct' if pipeline=='CPU-DCT' else 'deployment',
            correctness='PASS (reused accepted legacy gate)', accuracy_reference=quality[model]['raw_result'],
            top1=actual_top1, top5=actual_top5)
        observations.append(row)
        if pipeline in SYSTEMS and not any(c['model']==model and c['pipeline']==pipeline and c['gpu']==row['gpu'] for c in configs):
            configs.append(dict(model=model, pipeline=pipeline, gpu=row['gpu'], checkpoint=checkpoint,
                precision=old['precision'], batch_size=old['batch_size'], raw_result=old['raw_result'],
                preprocessing=data.get('preprocess',data.get('preprocessing')),
                execution=data.get('execution', {k:data.get(k) for k in ('workers','model_threads','dali_prefetch_queue_depth')}),
                pipeline_config=data.get('pipeline_config'),
                notes='Exact recorded configuration; historical CPU affinity/NUMA policy absent unless present in raw metadata. Model quality reference does not replace per-loader correctness gates.'))
    accepted_paths=sorted((R/'p0/deployment_passes').glob('*/*/*/accepted.json'))
    for batch in (*RECHECKS, LATEST):
        accepted_paths.extend(sorted((R/batch/'inference').glob('*/trial_*/accepted.json')))
    for accepted_path in accepted_paths:
        row = dict(read(accepted_path))
        recheck=accepted_path.relative_to(R).parts[0] in (*RECHECKS, LATEST)
        row['measurement_cohort']=accepted_path.relative_to(R).parts[0] if recheck else 'historical'
        measured_path, *pointer = row['raw_result'].split('#')
        measured_data = read(ROOT/measured_path)
        if pointer:
            measured = measured_data['repeats'][int(pointer[0].rsplit('/', 1)[1])]
            row.update(top1=100*measured['accuracy_top1'], top5=100*measured['accuracy_top5'])
        else:
            row.update(top1=measured_data['top1'], top5=measured_data['top5'])
        if row['measurement_cohort'] in (RECHECKS[-1], LATEST):
            # Replayed configurations inherit identity, never a prior pass's counters.
            for field in ('model_only_images_per_s','storage_bytes','metadata_bytes','issued_read_bytes',
                          'h2d_bytes','h2d_copy_count','gpu_peak_process_bytes','torch_peak_allocated_bytes',
                          'torch_peak_reserved_bytes','native_peak_bytes','host_rss_peak_bytes','pinned_peak_bytes'):
                row[field]=''
            if pointer:
                row.update(torch_peak_allocated_bytes=measured.get('peak_torch_gpu_allocated_bytes',''),
                    torch_peak_reserved_bytes=measured.get('peak_torch_gpu_reserved_bytes',''),
                    host_rss_peak_bytes=measured.get('host_peak_rss_bytes',''))
        if row['model']=='ViT-Ti RGB' and not recheck and not accepted_path.parent.name.startswith('normal_'):
            continue
        row['accuracy_reference'] = quality[row['model']]['raw_result']
        observations.append(row)
        if row['domain']=='RGB':
            contract_path=accepted_path.parent/'contract.json'
            contract=read(contract_path) if contract_path.exists() else {}
            configs.append(dict(model=row['model'], pipeline=row['pipeline'], gpu=row['gpu'], checkpoint=row['checkpoint'], raw_result=row['raw_result'], cpu_affinity='0-23,48-71', numa_binding=0, workers=8, model_threads=8 if row['architecture_family'] not in ('SwinV2-T','ViT-Ti') else 1, preprocessing=contract.get('preprocess'), execution=contract.get('execution'), pipeline_config=contract.get('pipelines'), notes=row['notes']))
    (R/'inventory/baseline_configs.json').write_text(json.dumps(configs,indent=2)+'\n')
    campaign=R/RECHECKS[-1]
    campaign_state=read(campaign/'status.json') if (campaign/'status.json').exists() else {}
    completed_keys=set()
    if campaign_state:
        completed_keys={tuple(j['reference'][k] for k in ('gpu','model','pipeline'))
            for j in read(campaign/'plan.json')['jobs'] if j['id'] in campaign_state['finished_groups']}
    # Publish a completed trial group as a whole; never blend partial repeats with history.
    current_observations = select_current([r for r in observations if r.get('measurement_cohort')!=RECHECKS[-1]
        or (r['gpu'],r['model'],r['pipeline']) in completed_keys])
    # Use the entire explicitly requested lower-load campaign, never its best trial.
    lower_load=[r for r in current_observations if r.get('measurement_cohort')=='low_load_recheck_20260929']
    replaced={(r['gpu'],r['model'],r['pipeline']) for r in lower_load}
    current_observations=[r for r in current_observations if
        (r['gpu'],r['model'],r['pipeline']) not in replaced or r in lower_load]
    current_observations=[r for r in current_observations if
        (r['gpu'],r['model'],r['pipeline']) not in completed_keys or r.get('measurement_cohort')==RECHECKS[-1]]
    # Explicit single-pass refresh selection; keep the anomalous first MobileNet24
    # observation in raw data, separately from its requested diagnostic follow-up.
    latest_summary=rows(R/LATEST/'inference_summary.csv')
    assert len(latest_summary)==14
    selected_raw={r['selected_raw'] for r in latest_summary}
    latest_rows=[r for r in observations if r['raw_result'] in selected_raw]
    assert len(latest_rows)==14 and all(r['measurement_cohort']==LATEST for r in latest_rows)
    current_observations=[r for r in current_observations if r['pipeline']!='FastLens']+latest_rows
    write(R/'p0/deployment_inference_raw.csv', [r for r in observations if r['pipeline']!='CPU-DCT'], RAW)
    write(R/'p0/inference_observations_raw.csv',observations,RAW)
    campaign_rows=[r for r in observations if r.get('measurement_cohort')==RECHECKS[-1]]
    if campaign_rows:
        write(campaign/'inference_raw.csv',campaign_rows,RAW)
    groups = {}
    for r in current_observations:
        groups.setdefault((r['gpu'],r['model'],r['pipeline']), []).append(r)
    summaries = []
    for (gpu,model,pipeline), group in groups.items():
        rates = [int(r['samples'])/float(r['elapsed_s']) for r in group]
        summaries.append(dict(gpu=gpu, architecture_family=group[0]['architecture_family'], model=model,
            domain=group[0]['domain'], checkpoint=group[0]['checkpoint'], pipeline=pipeline, trials=len(group),
            implementation_cohort=group[0].get('implementation_cohort', 'historical'),
            measurement_cohort=group[0].get('measurement_cohort','historical'),
            samples=50000, precision=group[0]['precision'], batch_size=group[0]['batch_size'],
            median_images_per_s=statistics.median(rates), min_images_per_s=min(rates), max_images_per_s=max(rates),
            cv_images_per_s_pct=100*statistics.stdev(rates)/statistics.mean(rates) if len(rates)>1 else '',
            median_elapsed_s=statistics.median(float(r['elapsed_s']) for r in group),
            top1=group[0]['top1'], raw_results=';'.join(r['raw_result'] for r in group)))
    write(R/'p0/deployment_inference_summary.csv', [r for r in summaries if r['pipeline']!='CPU-DCT'], list(summaries[0]))
    # Reuse measured resident-input ceilings without mixing domains or tensor shapes.
    ceiling_groups={}
    for old in rows(R/'e2e/model_only_raw.csv'):
        data=read(ROOT/old['raw_result'].split('#')[0])
        domain=old['domain'].upper()
        pair=next(p for p in pairs if p['key']==data['model'])
        model=pair['rgb_model'] if domain=='RGB' else pair['dct_model']
        if model not in quality: continue
        if Path(data['checkpoint']).resolve()!=Path(quality[model]['checkpoint']).resolve():
            if checkpoint_sha(data['checkpoint'])!=checkpoint_sha(quality[model]['checkpoint']):
                raise ValueError('Model-only checkpoint differs from deployment: '+model)
        key=(gpu_name(old['gpu']),model,domain,str(old['batch_size']))
        ceiling_groups.setdefault(key,[]).append(dict(old,model=model,domain=domain,gpu=key[0],checkpoint=data['checkpoint']))
    efficiencies=[]
    for summary in summaries:
        group=ceiling_groups.get((summary['gpu'],summary['model'],summary['domain'],str(summary['batch_size'])))
        if not group: continue
        precision_alias={'fp32':'float32','float32':'float32'}
        if precision_alias[summary['precision']]!=precision_alias[group[0]['precision']]: continue
        ceiling=statistics.median(int(r['samples'])/float(r['elapsed_s']) for r in group)
        efficiencies.append(dict(gpu=summary['gpu'],model=summary['model'],domain=summary['domain'],task='inference',
            pipeline=summary['pipeline'],batch_size=summary['batch_size'],precision=summary['precision'],
            end_to_end_images_per_s=summary['median_images_per_s'],model_only_images_per_s=ceiling,
            pipeline_efficiency=summary['median_images_per_s']/ceiling,
            raw_results=summary['raw_results'],ceiling_raw_results=';'.join(r['raw_result'] for r in group),
            notes='Each deployment normalized to its own registered model/domain/shape. Historical model-only host affinity is not recorded.'))
    write(R/'analysis/inference_pipeline_efficiency.csv',efficiencies,list(efficiencies[0]))
    prior_eff=rows(R/'analysis/pipeline_efficiency.csv')
    train_eff=[r for r in prior_eff if r.get('task')!='inference']
    fields=list(dict.fromkeys([key for r in [*train_eff,*efficiencies] for key in r]))
    write(R/'analysis/pipeline_efficiency.csv',[*train_eff,*efficiencies],fields)
    def find(gpu,model,pipeline):
        return next((r for r in summaries if (r['gpu'],r['model'],r['pipeline'])==(gpu,model,pipeline)), None)
    def rate(gpu,model,pipeline):
        row = find(gpu,model,pipeline)
        return row['median_images_per_s'] if row else None
    deployment, matched, manifest = [], [], []
    for p in pairs:
        for gpu in GPUS:
            rgb = [rate(gpu,p['rgb_model'],s) for s in SYSTEMS]
            native, cpu = rate(gpu,p['dct_model'],'FastLens'), rate(gpu,p['dct_model'],'CPU-DCT')
            fastest = max(rgb) if all(v is not None for v in rgb) else None
            # A partial matrix cannot establish the fastest of all four RGB systems.
            deployment.append(dict(gpu=gpu, architecture_family=p['architecture_family'], dct_variant=p['dct_model'],
                relationship=p['relationship'], rgb_top1=quality.get(p['rgb_model'],{}).get('top1_accuracy'),
                dct_top1=quality.get(p['dct_model'],{}).get('top1_accuracy'), **dict(zip(SYSTEMS,rgb)),
                FastLens=native, fastest_rgb_throughput=fastest,
                deployment_ratio=native/fastest if native and fastest else None,
                notes='Four-system RGB denominator requires complete accepted coverage; ViT legacy fixed-crop RGB runs excluded pending normal Resize256/CenterCrop224. Reference-only eFUN row is not an exact paired architecture.'))
            matched.append(dict(gpu=gpu, architecture_family=p['architecture_family'], dct_model=p['dct_model'],
                checkpoint=p['dct_checkpoint'], cpu_dct_images_per_s=cpu, imagelanes_images_per_s=native,
                matched_system_speedup=native/cpu if native and cpu else None,
                correctness='PASS' if native and cpu else 'MISSING/BLOCKED',
                raw_results=';'.join(r['raw_results'] for r in summaries if r['gpu']==gpu and r['model']==p['dct_model'])))
            for pipeline in [*SYSTEMS,'FastLens','CPU-DCT']:
                is_rgb = pipeline in SYSTEMS
                model = p['rgb_model'] if is_rgb else p['dct_model']
                identifier=f'inference:{gpu}:{model or p["dct_model"]+":unpaired"}:{pipeline}'
                if any(m['experiment_id']==identifier for m in manifest): continue
                have=find(gpu,model,pipeline)
                unsupported=not p['inference_supported']
                manifest.append(dict(experiment_id=identifier, phase='P0-C' if pipeline=='CPU-DCT' else 'P0-B', task='inference',
                    gpu=gpu, architecture=p['architecture_family'], rgb_model=p['rgb_model'], dct_model=p['dct_model'],
                    model_variant=model, domain='RGB' if is_rgb else 'DCT', pipeline=pipeline,
                    checkpoint=p['rgb_checkpoint'] if is_rgb else p['dct_checkpoint'], sample_count=50000,
                    precision=have['precision'] if have else 'fp32', batch_size=have['batch_size'] if have else 50 if p['key'] in ('vitti','swinv2') else 64,
                    status='DONE' if have else 'UNSUPPORTED' if unsupported else 'BLOCKED' if p['key']=='vitti' and is_rgb else 'TODO',
                    result_file=have['raw_results'] if have else '', correctness='PASS' if have else 'TBD', notes=p['notes']+(' No evaluation adapter registered for this architecture/system; training adapters do not establish evaluation support.' if unsupported and p['inference_supported'] else '')))
    for q in quality.values():
        manifest.append(dict(experiment_id='quality:'+q['model'], phase='P0-A', task='checkpoint_accuracy',
            gpu=q['gpu'], architecture=q['architecture_family'], model_variant=q['model'], domain=q['domain'],
            checkpoint=q['checkpoint'], pipeline=q['evaluation_pipeline'], sample_count=50000, precision=q['precision'],
            batch_size=q['batch_size'], status='DONE', result_file=q['raw_result'], correctness=q['correctness'], notes=q['notes']))
    for old in rows(R/'inventory/manifest_before_deployment_plan.csv'):
        manifest.append(dict(experiment_id='legacy:'+old['experiment_id'], phase='legacy retained', task=old['task'],
            gpu=old['gpu'], architecture=family_for(old['model']), model_variant=old['model'], domain=old['domain'],
            pipeline=old['pipeline'], sample_count=old['sample_count'], status=old['status'], result_file=old['result_file'],
            correctness=old['valid'], notes='Historical audit snapshot; current matrix uses new entries. '+old['notes']))
    for row in rows(R/'p0/training_window_raw.csv'):
        manifest.append(dict(experiment_id=f'training:{row["gpu"]}:{row["model"]}:{row["pipeline"]}:{row["trial"]}',
            phase='P0-D', task='training_window', gpu=row['gpu'], architecture=family_for(row['model']), model_variant=row['model'],
            domain=row['domain'], pipeline=row['pipeline'], checkpoint='recorded initialization: '+row['initial_model_hash'],
            sample_count=row['samples'], precision=row['precision'], batch_size=row['batch_size'], status='DONE',
            result_file=row['raw_result'], correctness=row['correctness_status'], notes=row['notes']))
    training_done={m['experiment_id'] for m in manifest if m['task']=='training_window' and m['phase']=='P0-D'}
    queue=read(R/'p0/queue_state.json') if (R/'p0/queue_state.json').exists() else {}
    for gpu in GPUS:
        for architecture in ('SwinV2-T','MobileNetV2'):
            for pipeline in [*SYSTEMS,'FastLens']:
                for trial in range(3):
                    identifier=f'training:{gpu}:{architecture}:{pipeline}:{trial}'
                    if identifier in training_done: continue
                    active=queue.get('active') or ''
                    prefix=f'results/p0/raw/{"4090" if gpu=="RTX 4090" else gpu}/{"swinv2" if architecture=="SwinV2-T" else "mobilenet"}/{pipeline.lower()}/trial_{trial}'
                    running=queue.get('phase')=='RUNNING' and active.startswith(prefix)
                    manifest.append(dict(experiment_id=identifier,phase='P0-D',task='training_window',gpu=gpu,
                        architecture=architecture,model_variant=architecture+(' DCT' if pipeline=='FastLens' else ' RGB'),
                        domain='DCT' if pipeline=='FastLens' else 'RGB',pipeline=pipeline,sample_count=65536,
                        precision='bf16-autocast',batch_size=64,status='RUNNING' if running else 'TODO',
                        result_file=active if running else '',correctness='TBD',notes='Effective batch 1024; 64 updates; exactly 65536 model-consumed images. Audited native tail fix skips incomplete microbatches and records their IDs; full-epoch readers unchanged.'))
    for m in manifest:
        if not m['experiment_id'].startswith('inference:') or m['status'] in ('DONE','UNSUPPORTED'):
            continue
        pair=next(p for p in pairs if p['rgb_model']==m['model_variant'] or p['dct_model']==m['model_variant'])
        pipeline={'PyTorch':'pytorch','DALI':'dali','CoorDL':'coordl','FFCV':'ffcv','FastLens':'galp' if pair['key']=='swinv2' else 'imagelanes','CPU-DCT':'rgbnomore' if pair['key'] in ('vitti','swinv2') else 'cpu_dct'}[m['pipeline']]
        if pair['key'] not in ('vitti','swinv2') and m['domain']=='RGB': pipeline='rgb_'+pipeline
        path=R/'p0/deployment_passes'/('4090' if m['gpu']=='RTX 4090' else m['gpu'])/pair['key']/pipeline
        if m['domain']=='RGB' and pair['key']=='vitti':
            path=path.parent/('normal_'+m['pipeline'].lower())
        elif m['domain']=='RGB' and m['pipeline'] in ('CoorDL','FFCV'):
            path=path.parent/('paired_'+m['pipeline'].lower())
        if pair['key']=='mobilenet24' and m['pipeline']=='FastLens' and (path.parent/'fastlens_rounding_fixed').exists():
            path=path.parent/'fastlens_rounding_fixed'
            if (path.parent/'fastlens_rounding_fixed_after_contention').exists():
                path=path.parent/'fastlens_rounding_fixed_after_contention'
            if (path.parent/'fastlens_rounding_fixed_full_pass').exists():
                path=path.parent/'fastlens_rounding_fixed_full_pass'
        if pair['key']=='efun' and m['pipeline']=='CoorDL' and (path.parent/'paired_coordl_cpu_decode').exists():
            path=path.parent/'paired_coordl_cpu_decode'
        if (path/'failure.json').exists():
            m.update(status='FAILED',result_file=str((path/'failure.json').relative_to(ROOT)),correctness='FAIL',notes=m['notes']+' '+str(read(path/'failure.json')))
        elif (path/'run.command.json').exists():
            command=read(path/'run.command.json')
            status='RUNNING' if 'returncode' not in command else 'FAILED' if command['returncode'] else 'TODO'
            m.update(status=status,result_file=str((path/'run.log').relative_to(ROOT)),notes=m['notes']+' Raw execution exists; acceptance pending unless DONE.')
    full_epochs=rows(R/'p1/full_epoch_training.csv')
    if campaign_state:
        for job in read(campaign/'plan.json')['jobs']:
            reference=job['reference']
            directory=campaign/'inference'/job['id']
            trials=set(range(1,4)) | {int(p.name.split('_')[1]) for p in directory.glob('trial_*')}
            for trial in sorted(trials):
                output=directory/f'trial_{trial}'
                if (output/'accepted.json').exists():
                    continue  # Recorded below with its raw measurement.
                status='FAILED' if (output/'failure.json').exists() else 'RUNNING' if campaign_state['active']==[job['id'],trial] else 'TODO'
                manifest.append(dict(experiment_id=f'{campaign.name}:{job["id"]}:{trial}',phase='P8',
                    task='inference',gpu=reference['gpu'],architecture=reference['architecture_family'],
                    model_variant=reference['model'],domain=reference['domain'],pipeline=reference['pipeline'],
                    checkpoint=reference['checkpoint'],sample_count=50000,precision=reference['precision'],
                    batch_size=reference['batch_size'],status=status,
                    result_file=str(output.relative_to(ROOT)),correctness='FAIL' if status=='FAILED' else 'PENDING',
                    notes='Requested serial interleaved remeasurement; historical results retained.'))
    for row in [r for r in observations if r.get('measurement_cohort') in (*RECHECKS, LATEST)]:
        manifest.append(dict(experiment_id='recheck:'+row['raw_result'],phase='P8',task='inference',
            gpu=row['gpu'],architecture=row['architecture_family'],model_variant=row['model'],
            domain=row['domain'],pipeline=row['pipeline'],checkpoint=row['checkpoint'],
            sample_count=row['samples'],precision=row['precision'],batch_size=row['batch_size'],
            status='DONE',result_file=row['raw_result'],correctness=row['correctness'],
            notes=row['notes']+' Measurement cohort: '+row['measurement_cohort']))
    for failure in sorted((R/LATEST/'inference').glob('*/trial_*/failure.json')):
        job=next(j for j in read(R/LATEST/'plan.json')['jobs'] if j['id']==failure.parent.parent.name)
        ref=job['reference']
        manifest.append(dict(experiment_id='failed:'+str(failure.relative_to(ROOT)),phase='P8',task='inference',
            gpu=ref['gpu'],architecture=ref['architecture_family'],model_variant=ref['model'],domain='DCT',
            pipeline='FastLens',sample_count=50000,status='FAILED',result_file=str(failure.relative_to(ROOT)),
            correctness='NOT_COMPLETED',notes=read(failure)['error']))
    for row in rows(R/'p1/full_epoch_training_all_versions.csv'):
        if row.get('measurement_cohort') not in ('recheck_20260929','low_load_recheck_20260929'):
            continue
        manifest.append(dict(experiment_id='recheck:'+row['raw_result'],phase='P8',task='training_epoch',
            gpu=row['gpu'],architecture=row['model'],model_variant=row['model'],domain=row['domain'],
            pipeline=row['pipeline'],checkpoint='initialization: '+row['initial_model_hash'],
            sample_count=row['samples'],precision=row['precision'],batch_size=row['batch_size'],
            status='DONE',result_file=row['raw_result'],correctness=row['correctness_status'],notes=row['notes']))
    for row in full_epochs:
        manifest.append(dict(experiment_id=f'full_epoch:{row["gpu"]}:{row["model"]}:{row["pipeline"]}',
            phase='P1',task='training_epoch',gpu=row['gpu'],architecture=row['model'],model_variant=row['model'],
            domain='DCT',pipeline=row['pipeline'],checkpoint='initialization: '+row['initial_model_hash'],
            sample_count=row['samples'],precision=row['precision'],batch_size=row['batch_size'],status='DONE',
            result_file=row['raw_result'],correctness=row['correctness_status'],notes=row['notes']))
    for gpu in GPUS:
        for model in ('SwinV2-T','MobileNetV2-DCT24'):
            for pipeline in ('CPU-DCT','FastLens'):
                identifier=f'full_epoch:{gpu}:{model}:{pipeline}'
                if any(m['experiment_id']==identifier for m in manifest): continue
                interrupted=gpu=='RTX 4090' and model=='SwinV2-T' and pipeline=='CPU-DCT'
                attempt=R/'p1/short_warmup_retry/4090/cpu_dct/run.command.json'
                retry_running=interrupted and attempt.exists() and 'returncode' not in read(attempt)
                if model=='MobileNetV2-DCT24':
                    attempt=R/f'p1/mobile_b6/{"4090" if gpu=="RTX 4090" else gpu}/{"cpu_dct" if pipeline=="CPU-DCT" else "imagelanes"}/run.command.json'
                    retry_running=attempt.exists() and 'returncode' not in read(attempt)
                manifest.append(dict(experiment_id=identifier,phase='P1',task='training_epoch',gpu=gpu,
                    architecture=model,model_variant=model,domain='DCT',pipeline=pipeline,sample_count=1281167,
                    precision='bf16-autocast',batch_size=64,status='RUNNING' if retry_running else 'BLOCKED' if interrupted else 'TODO',
                    result_file=str(attempt.parent/'run.log') if retry_running else '',correctness='TBD',
                    notes='Measured epoch attempt running; acceptance pending.' if retry_running else 'Interrupted by same-GPU task; queued retry after competing sequence.' if interrupted else 'Full measured epoch remains required.'))
    resource_rows={name:rows(R/f'resources/{name}.csv') if (R/f'resources/{name}.csv').exists() else [] for name in ('h2d','memory')}
    ablation_rows={name:rows(R/f'ablation/{name}.csv') if (R/f'ablation/{name}.csv').exists() else [] for name in ('crop_pushdown','direct_vs_grid','frequency_pushdown_matched','fastlanes_vs_raw_int16')}
    for model in ('swinv2','mobilenet'):
        for pipeline in ('pytorch','dali','coordl','ffcv','imagelanes'):
            for mode in ('memory','profile'):
                directory=R/f'resources/raw/4090/{model}/{pipeline}/{mode}'
                accepted=(directory/'accepted.json').exists()
                failed=(directory/'failure.json').exists()
                command=read(directory/'run.command.json') if (directory/'run.command.json').exists() else {}
                state='DONE' if accepted else 'FAILED' if failed else 'RUNNING' if command and 'returncode' not in command else 'TODO'
                manifest.append(dict(experiment_id=f'resource:4090:{model}:{pipeline}:{mode}',phase='P6',task='resource_'+mode,
                    gpu='RTX 4090',architecture=model,domain='DCT' if pipeline=='imagelanes' else 'RGB',pipeline=pipeline,
                    sample_count=16384,precision='bf16-autocast',batch_size=64,status=state,
                    result_file=str((directory/'result.json').relative_to(ROOT)) if accepted else str((directory/'run.log').relative_to(ROOT)) if command else '',
                    correctness='PASS' if accepted else 'TBD',notes='Separate fixed resource diagnostic; never primary throughput.'))
    for name, records in ablation_rows.items():
        manifest.append(dict(experiment_id='ablation:'+name,phase='P7',task='ablation',gpu='RTX 4090',
            architecture='MobileNetV2',model_variant='MobileNetV2-DCT24',domain='DCT',pipeline='FastLens',
            sample_count=16384,precision='fp32',batch_size=64,status='DONE' if records else 'TODO',
            result_file=f'results/ablation/{name}.csv' if records else '',correctness='PASS' if records else 'TBD',
            notes='Fixed crop, K, checkpoint and sample IDs; one warm performance pass and separate H2D capture.'))
    write(R/'experiment_manifest.csv', manifest, MANIFEST)
    write(R/'p0/deployment_ratios.csv', deployment, list(deployment[0]))
    write(R/'p0/matched_dct_inference.csv', matched, list(matched[0]))
    def value(v): return 'TBD' if v is None or v=='' else f'{v:.3f}' if isinstance(v,float) else str(v)
    def table(headers,data):
        return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(value(v) for v in row)+' |' for row in data]
    status=['# FastLens deployment evaluation', '', 'ImageNet-512, 50,000-image validation. Observed measurements are reused; missing values remain TBD. Accuracy is in percent, delta in percentage points. No accuracy-preservation or convergence claim is inferred.', '',
        '## A. Model pairs and quality', '', 'Quality reference uses H100 consistently once its normal ViT RGB run is available; raw deployment rows retain each GPU’s actual accuracy. ViT RGB uses checkpoint-normal Resize(256)/CenterCrop(224); unmeasured normal-path cells remain TBD. Legacy fixed-crop accuracy is preserved in quality/legacy_fixed_crop_vitti_accuracy.csv and excluded from the primary comparison.', '']+table(['Architecture','RGB model','DCT model','RGB Top-1','DCT Top-1','Delta (pp)'],
        [[r[k] for k in ['architecture_family','rgb_model','dct_model','rgb_top1','dct_top1','accuracy_delta_pp']] for r in deltas])
    status+=['','eFUN / EfficientNet-B0 is a **reference-only comparison**, not a verified exact counterpart. Extra eFUN constructors have no configured evaluation checkpoint. Quality raw references and integer-count checks: quality/model_accuracy.csv and quality/accuracy_reuse_checks.json.','',
        '## B. Main deployment inference','', 'Images/s; existing accepted repeats summarized by median. RGB runs are reused across DCT variants. Top-1 columns reference the normal checkpoint evaluation; each loader’s observed accuracy is retained separately in deployment_inference_raw.csv. Fastest RGB and deployment ratio remain TBD until all four RGB systems are accepted.','']+table(
        ['GPU','Architecture','DCT variant','RGB Top-1','DCT Top-1',*SYSTEMS,'FastLens DCT','Fastest RGB','Deployment ratio'],
        [[r[k] for k in ['gpu','architecture_family','dct_variant','rgb_top1','dct_top1',*SYSTEMS,'FastLens','fastest_rgb_throughput','deployment_ratio']] for r in deployment if r['dct_top1'] is not None])
    status+=['','## C. Matched DCT inference','','Same-model/checkpoint comparisons only; failed prediction equality and unstable timing remain excluded.','']+table(
        ['GPU','DCT model','CPU-DCT','FastLens','Matched system speedup'], [[r[k] for k in ['gpu','dct_model','cpu_dct_images_per_s','imagelanes_images_per_s','matched_system_speedup']] for r in matched])
    training=rows(R/'p0/training_window_summary.csv')
    status+=['','## D. Training deployment window','','65,536 measured images, three trials; BF16, microbatch 64, effective batch 1,024. RGB/DCT deployment comparison; checkpoint accuracy is separate from this from-scratch training workload.','']+table(
        ['GPU','Architecture',*SYSTEMS,'FastLens DCT'], [[gpu,model,*[next((float(r['median_images_per_s']) for r in training if (r['gpu'],r['model'],r['pipeline'])==(gpu,model,p)),None) for p in [*SYSTEMS,'FastLens']]] for gpu in GPUS for model in ['SwinV2-T','MobileNetV2']])
    epoch_table=[]
    for gpu in GPUS:
        for model in ('SwinV2-T','MobileNetV2-DCT24'):
            pair=[next((r for r in full_epochs if (r['gpu'],r['model'],r['pipeline'])==(gpu,model,p)),None) for p in ('CPU-DCT','FastLens')]
            rates=[int(r['samples'])/float(r['elapsed_s']) if r else None for r in pair]
            ratio=rates[1]/rates[0] if all(r and r['strict_matched']=='True' for r in pair) else None
            epoch_table.append([gpu,model+' DCT' if model=='SwinV2-T' else model,*rates,ratio])
    status+=['','## E. Full-epoch matched DCT training','','Images/s. Complete epochs after short excluded warmup; ratios require the accepted strict pair. Historical contended epochs remain archived and excluded. Failed/interrupted attempts are retained; no convergence claim.','']+table(
        ['GPU','DCT model','CPU-DCT','FastLens','System speedup'],epoch_table)
    storage=rows(R/'resources/storage.csv')
    status+=['','## F. Resource cost','','Distinct representations/splits are separate. H2D and memory require the requested 16,384-image diagnostics.','']+table(
        ['Representation','View/split','Storage bytes','Conversion s','H2D MB/image','GPU peak','Host peak'], [[r['representation'],r['dataset_view']+'/'+r['split'],r['representation_bytes'],r['conversion_s'],None,None,None] for r in storage])
    measured_resources=[]
    for model in ('swinv2','mobilenet'):
        for pipeline in ('pytorch','dali','coordl','ffcv','imagelanes'):
            h=next((r for r in resource_rows['h2d'] if (r['model'],r['pipeline'])==(model,pipeline)),{})
            m=next((r for r in resource_rows['memory'] if (r['model'],r['pipeline'])==(model,pipeline)),{})
            measured_resources.append([model,pipeline,float(h['h2d_bytes_per_image'])/1e6 if h else None,
                m.get('gpu_peak_process_bytes'),m.get('host_rss_peak_bytes'),m.get('process_tree_rss_peak_bytes')])
    status+=['','RTX 4090: separate 16,384-image training diagnostics. MB is decimal; GPU/RSS columns are bytes. Tree RSS includes loader workers and may double-count shared pages.','']+table(
        ['Model','System','H2D MB/image','GPU process peak','Main RSS peak','Process-tree RSS peak'],measured_resources)
    ablation_table=[]
    for name,title in [('crop_pushdown','Crop pushdown'),('direct_vs_grid','Direct vs grids'),('frequency_pushdown_matched','Frequency pushdown'),('fastlanes_vs_raw_int16','FastLanes vs raw int16')]:
        records=ablation_rows[name]
        for record in records:
            ablation_table.append([title,record['case'],record['issued_read_bytes'],record['h2d_bytes'],record['elapsed_s'],record['images_per_s']])
        if not records: ablation_table.append([title,None,None,None,None,None])
    status+=['','## G. Ablations','','Matched 16,384-image controls. Crop/construction/raw-int16 use fixed 448×448 crop and K64; the Phase 2 frequency control uses full spatial extent and fixed K32. One unprofiled performance pass; H2D from a separate capture. No crop-area sweep is implied.','']+table(
        ['Mechanism','Case','Issued bytes','H2D bytes','Elapsed s','Images/s'],ablation_table)
    status+=['','Frequency and raw-int16 controls remain TBD until their matched-input and physical-control checks succeed. Compression-specific attribution requires the raw-int16 result.']
    status+=['','## Execution priority','','Accuracy/reuse → deployment inference (one clean pass) → matched DCT inference → representative training windows → eight matched full epochs → own-model ceilings → resources → crop and direct/grid. Stop new experiments at Milestone A; frequency/raw-int16 and repeated headline inference follow only in Milestone B.','',
        'After Milestone A, the authorized continuation populates fastlens/docs/paper/main.tex and starts Phase 2 in the background: missing headline repetitions, then matched frequency and raw-int16 controls. No additional full training epochs are queued.',
        'Our performance runs remain sequential. Other users’ GPU jobs are permitted and annotated unless concrete contention prevents measurement; this does not establish resource isolation. New runs use CPU/memory NUMA 0; historical affinity gaps remain disclosed. RGB baseline configurations: inventory/baseline_configs.json. All missing/blocked work remains explicit in experiment_manifest.csv.','']
    if (R/'phase2/status.json').exists():
        phase2=read(R/'phase2/status.json')
        status+=['## Phase 2 execution','',phase2['phase']+'; active: '+str(phase2.get('active')),'']
        status+=table(['Stage','Status','Raw log'],[[s['name'],s['status'],s['raw_log']] for s in phase2['stages']])
    repeated=[r for r in summaries if r['pipeline']!='CPU-DCT' and r['trials']>1]
    unstable=[r for r in repeated if r['cv_images_per_s_pct']>2]
    timing_review=(f'DERIVED: {len(unstable)} of {len(repeated)} repeated deployment-inference configurations '
        'exceed 2% throughput CV (sample standard deviation / mean). Correctness acceptance is not timing stability. '
        'Medians and min/max retain all selected-cohort observations; rankings are provisional under shared-host load. '
        'See results/p0/deployment_inference_summary.csv for CV and raw-result paths. '
        'MISSING: resource-isolated paired timing evidence sufficient to attribute the observed version changes solely to code.')
    latest_note=('FastLens inference uses one selected full-validation pass per model/GPU from '+LATEST+
        '; RGB and CPU-DCT retain their complete three-trial groups. No old/new implementation averaging. '
        'H100 VLLM residency is annotated, and missing process counters do not prove inactivity. '
        'The first H100 MobileNet24 pass (981.12 images/s, 43.07s input wait) remains in raw data; '
        'the requested follow-up (5427.00 images/s) is selected, without claiming a diagnosed cause for the anomaly. '
        'Training, resource diagnostics and mechanism controls retain their separately measured implementation versions.')
    status+=['## Inference timing validity','',timing_review,'',latest_note,'']
    if campaign_state:
        status+=['## Remaining inference remeasurement','',
            f"Status: {campaign_state['status']}; active: {campaign_state['active']}; accepted passes: {len(campaign_state['completed'])}; completed configurations: {len(completed_keys)}/63; failures: {len(campaign_state['failures'])}.", '',
            'Serial interleaved deployment and CPU-DCT passes; 3 trials, 2 additional if CV exceeds 2%. All trials retained; no throughput-based rejection. Completed groups replace historical summaries as a whole. New versus previous medians, CV, and raw paths: results/remaining_inference_recheck_20260929/inference_summary.csv. Host activity is recorded, not assumed isolated.', '']
        for recovery in sorted(campaign.glob('recovery_*/recovery.json')):
            record=read(recovery)
            status+=['Execution interruption: '+record['reason']+' Recovery record: `'+str(recovery.relative_to(ROOT))+'`.','']
    (R/'PAPER_STATUS.md').write_text('\n'.join(status))
    report_path=ROOT/'benchmark_results/archive/experiments/REPORT.md'
    report=report_path.read_text().split('\n<!-- deployment-generated -->')[0]
    appendix=['','<!-- deployment-generated -->','## New deployment-pass records','',
        'These are executed observations; no missing value is estimated. Single-pass refresh observations are distinguished from repeated baseline medians.','',timing_review,'',latest_note,'']
    for recovery in sorted(campaign.glob('recovery_*/recovery.json')):
        record=read(recovery)
        appendix+=['Execution interruption and retry: '+record['reason'],
            'Recovery record: `'+str(recovery.relative_to(ROOT))+'`; rejected raw attempt: `'+record['failed_attempt']+'`.','']
    for command_path in sorted({p.parent/'run.command.json' for p in accepted_paths} | set((R/'p0/deployment_passes').glob('*/*/*/run.command.json'))):
        command=read(command_path)
        accepted_path=command_path.parent/'accepted.json'
        observation=read(accepted_path) if accepted_path.exists() else None
        appendix+=['### '+str(command_path.parent.relative_to(ROOT)),'',
            '- Machine: ruc-NF5468M6; GPU UUID: '+str(command['environment']['CUDA_VISIBLE_DEVICES']),
            '- Git commit: '+command['git_commit']+'; dirty working tree, exact driver and contract retained.',
            '- Status: '+('OBSERVED / accepted' if observation else 'MISSING acceptance; raw exit status '+str(command.get('returncode','RUNNING'))),
            '- Exact command: `'+command['shell_command']+'`',
            '- Concurrent-load evidence: `'+str((command_path.parent/'concurrent_load.jsonl').relative_to(ROOT))+'`']
        if observation:
            appendix+=['- Model/checkpoint: '+observation['model']+' / `'+observation['checkpoint']+'`',
                '- Domain / system / comparison: '+observation['domain']+' / '+observation['pipeline']+' / '+observation['comparison_type'],
                '- Samples: '+str(observation['samples'])+'; elapsed seconds: '+str(observation['elapsed_s'])+'; images/s: '+str(observation['images_per_s']),
                '- Raw result: `'+observation['raw_result']+'`; correctness: '+observation['correctness'],
                '- Notes: '+observation['notes']]
        appendix+=['']
    appendix+=['## Accepted short-warmup full epochs','']
    for row in rows(R/'p1/full_epoch_training_all_versions.csv'):
        appendix+=['### '+row['gpu']+' / '+row['model']+' / '+row['pipeline'],'',
            '- OBSERVED: '+row['samples']+' samples; '+row['elapsed_s']+' seconds; '+row['optimizer_updates']+' updates.',
            '- Machine: ruc-NF5468M6; git commit: '+row['git_commit'],
            '- DCT initialization: `'+row['initial_model_hash']+'`; precision: '+row['precision'],
            '- Exact command: `'+row['command']+'`',
            '- Raw result: `'+row['raw_result']+'`; correctness: '+row['correctness_status'],
            '- Notes: '+row['notes'],'']
    appendix+=['## Resource and ablation records','']
    for pattern in ('resources/raw/4090/*/*/*/run.command.json','ablation/minimal/4090/*/*/run.command.json','ablation/phase2/4090/*/*/*/run.command.json','analysis/ceiling_runs/4090/*/*/run.command.json','analysis/training_ceilings/*/*/*/trial_*/run.command.json'):
        for path in sorted(R.glob(pattern)):
            command=read(path)
            appendix+=['### '+str(path.parent.relative_to(ROOT)),'',
                '- Machine: ruc-NF5468M6; GPU UUID: '+str(command['environment']['CUDA_VISIBLE_DEVICES']),
                '- Git commit: '+command['git_commit']+'; working tree may be dirty.',
                '- Raw command/log directory: `'+str(path.parent.relative_to(ROOT))+'`',
                '- Exact command: `'+command['shell_command']+'`',
                '- Exit status: '+str(command.get('returncode','RUNNING'))+'; accepted results require the correctness gates recorded in the CSV/JSON outputs.',
                '- Concurrent load: `'+str((path.parent/'concurrent_load.jsonl').relative_to(ROOT))+'`','']
    report_path.write_text(report+'\n'.join(appendix))
    revision = R/'version_refresh_20260929'
    if (revision/'code_snapshot.json').exists():
        with report_path.open('a') as stream:
            stream.write('\n## Implementation-version refresh\n\n'
                '- OBSERVED: original raw files are retained; pre-refresh tables and manuscript are in `results/version_refresh_20260929/before/`.\n'
                '- OBSERVED: new command logs include source/binary fingerprints and before/after code checks. See `code_snapshot.json` in the refresh directory.\n'
                '- DERIVED: summaries select one implementation cohort per GPU/model/system. Historical post-build membership is timestamp-inferred where an original binary hash is absent.\n'
                '- DERIVED: `performance_changes.csv` compares refreshed medians with the saved table, retaining every decrease. This does not isolate code from shared-host contention.\n'
                '- BLOCKED: the previous frequency control did not physically select K64; its timings remain excluded.\n'
                '- Current queue and retries: `results/version_refresh_20260929/status.json`, `runner.log`, and `hardlink_followup.log`.\n')
    print(json.dumps(dict(pairs=len(pairs), checkpoint_accuracies=len(quality), accepted_inference_groups=len(summaries), deployment_rows=len(deployment), manifest_entries=len(manifest))))


if __name__=='__main__':
    main()
