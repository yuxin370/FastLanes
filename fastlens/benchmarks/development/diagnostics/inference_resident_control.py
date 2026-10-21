"""H100 MobileNet resident-input inference controls for the completed deployment cohort."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
COHORT = ROOT/'results/remaining_baselines_20261012'
OUT = ROOT/'results/inference_resident_h100_20261012'
NATIVE = ROOT/'results/swin_inference_20261011_fixed/native_runtime'
ARMS = ('rgb', 'dct24', 'dct32')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def worker(arm, dest):
    os.environ['DCTNET_PROFILE'] = 'mobilenet32' if arm == 'dct32' else 'mobilenet24'
    import cv2
    cv2.setNumThreads(1)
    import torch
    from fastlens.benchmarks.dct_models import backend as B, rgb
    from fastlens.benchmarks.dct_models.evaluate import samples
    from fastlens.benchmarks.execution.revision_results import file_sha
    from fastlens.benchmarks.execution.revision_results import verify_snapshot
    fingerprint = verify_snapshot()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    assert torch.cuda.get_device_name() == 'NVIDIA H100 80GB HBM3'
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') is None
    stream = torch.cuda.Stream(priority=torch.cuda.Stream.priority_range()[1])
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    entries = samples(50000, physical_order=True)
    dct = arm != 'rgb'
    model_name = 'mobilenet32' if arm == 'dct32' else 'mobilenet24'
    refpath = COHORT/'inference/H100'/model_name/('imagelanes' if dct else 'l3')/'trial_0/result.json'
    reference = read(refpath)
    checkpoint = B.CHECKPOINT if dct else rgb.CHECKPOINT
    assert str(checkpoint) == reference['checkpoint']
    assert file_sha(checkpoint) == reference['checkpoint_sha256']
    model = (B.model() if dct else rgb.model()).to(device).eval()
    expected = dict(zip(reference['sample_ids'], reference['predictions']))
    batches = []
    print('Preparing eight real preprocessed batches outside timing', flush=True)
    with torch.inference_mode():
        if dct:
            import _fastlens_direct_dct as native
            from fastlens.benchmarks.dct_models.evaluate_shards import native_options, apply_b6_runtime
            from fastlens.benchmarks.dct_models.online_crop import source_options, CROP
            manifest = ROOT/'fastlens/data/compressed/imagenet512_val_block_major/manifest.bin'
            os.environ['FASTLENS_BLOCK_MAJOR_ACCESS_DIR'] = str(manifest.parent/'access')
            reader = native.DirectDctReader(str(manifest))
            options = apply_b6_runtime(source_options(native_options(False, True), 'vector-range-read-selected-decode'))
            source = reader.pipeline_batch_options(**options, output_batch_images=64)
            ids = [s['galp_image_id'] for s in entries]
            groups = [ids[i:i+1024] for i in range(0, len(ids), 1024)]
            source.reset(groups, transforms_by_batch=[[dict(crop=CROP, horizontal_flip=False) for _ in group] for group in groups])
            batch = next(source)
            assert list(batch.global_image_ids) == groups[0]
            for start in range(0, 512, 64):
                x = batch.projected_range(start, 64)
                resident = x.clone()
                assert resident.stride() == x.stride()
                selected = entries[start:start+64]
                labels = torch.tensor([s['model_label'] for s in selected], device=device)
                batches.append((resident, labels, [s['logical_sample_id'] for s in selected]))
                del x
            del batch
            source.close()
            del source, reader
        else:
            from fastlens.benchmarks.baselines.l3.prepare import encoded_path
            l3 = read(ROOT/'results/l3_patch32_20261006/training/4090/vitti/l3/trial_0/result.json')['l3']
            chosen = [dict(e, path=str(encoded_path(l3['root'], e['logical_sample_id']))) for e in entries]
            source = rgb.dali_loader(chosen, 64, 8, l3_library=Path(l3['library']))
            for start in range(0, 512, 64):
                values = next(source)[0]
                ordinals = values['ordinal'].reshape(-1).cpu().tolist()
                assert ordinals == list(range(start, start+64))
                x = values['image']
                resident = x.clone()
                assert resident.stride() == x.stride()
                labels = torch.tensor([entries[i]['model_label'] for i in ordinals], device=device)
                batches.append((resident, labels, [entries[i]['logical_sample_id'] for i in ordinals]))
                del values, x
            del source
        torch.cuda.synchronize()
        # Verify the actual stored-input prefix against the accepted full run.
        for x, labels, ids in batches:
            predictions = model(x).topk(5, dim=1).indices.cpu().tolist()
            assert [p[0] for p in predictions] == [expected[s][0] for s in ids]
        contract = dict(shape=list(batches[0][0].shape), stride=list(batches[0][0].stride()),
                        dtype=str(batches[0][0].dtype), labels_dtype=str(batches[0][1].dtype))
        batch_hashes = [hashlib.sha256(x.cpu().numpy().tobytes()+y.cpu().numpy().tobytes()).hexdigest()
                       for x, y, _ in batches]
        dummy = torch.zeros(tuple(batches[0][0].shape), device=device)
        for _ in range(20):
            model(dummy)
        del dummy
        torch.cuda.synchronize()
        predictions = []
        sample_ids = []
        ce, correct1, correct5 = 0., 0, 0
        started = time.perf_counter()
        for index, first in enumerate(range(0, 50000, 64)):
            size = min(64, 50000-first)
            x, labels, ids = batches[index % 8]
            x, labels = x[:size], labels[:size]
            logits = model(x)
            ce += float(torch.nn.functional.cross_entropy(logits, labels, reduction='sum'))
            top5 = logits.topk(5, dim=1).indices
            matches = top5.eq(labels[:, None])
            correct1 += int(matches[:, :1].sum())
            correct5 += int(matches.sum())
            predictions.extend(top5.cpu().tolist())
            sample_ids.extend(ids[:size])
            stream.synchronize()
            del x, logits, labels, top5, matches
        seconds = time.perf_counter()-started
    assert len(predictions) == 50000
    assert len(sample_ids) == 50000
    assert verify_snapshot() == fingerprint
    result = dict(arm=arm, images=50000, unique_samples=512, resident_batches=8,
        cycle='microbatch index modulo 8; final batch contains 16 images',
        seconds=seconds, images_per_s=50000/seconds, batch=64, precision='fp32', tf32=False,
        compile=False, cuda_graphs=False, host_threads=8, model_stream_priority=stream.priority,
        CUBLAS_WORKSPACE_CONFIG=None, warmup_batches=20, correctness_passed=True,
        reference=str(refpath), checkpoint=str(checkpoint), checkpoint_sha256=file_sha(checkpoint),
        tensor_contract=contract, batch_hashes=batch_hashes, input_ids=[s for _, _, ids in batches for s in ids],
        predictions_sha256=hashlib.sha256(json.dumps(predictions).encode()).hexdigest(),
        gpu=torch.cuda.get_device_name(), driver_sha256=file_sha(__file__), code_fingerprint=fingerprint,
        timing_scope='Resident inputs -> logits, CE, top1/top5, prediction copies and per-batch synchronization; preparation excluded.',
        scope='Execution reference only; repeats 512 samples, not a full-validation accuracy or performance upper bound.')
    save(dest/'result.json', result)
    print(json.dumps(dict(arm=arm, images_per_s=result['images_per_s'], correctness_passed=True)), flush=True)


def run_all():
    from fastlens.benchmarks.execution.inference_load import run
    from fastlens.benchmarks.development.reports.report_resident_study import write_csv
    OUT.mkdir(exist_ok=True)
    state = dict(status='RUNNING', pid=os.getpid(), completed_runs=0, planned_runs=9, active=None)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-d6e80e78-00d8-8e4b-0c81-a403bebe0d76',
        PYTHONPATH=f'{ROOT}:{NATIVE}', TMPDIR=str(Path.home()/'tmp'), OMP_NUM_THREADS='8',
        MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
        FASTLENS_EVALUATION_SNAPSHOT=str(COHORT/'runtime_snapshot.json'))
    env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    env.pop('FASTLENS_ALLOWED_RESIDENT_PIDS', None)
    records = []
    try:
        for trial in range(3):
            for arm in ARMS[trial:]+ARMS[:trial]:
                dest = OUT/arm/f'trial_{trial}'
                state['active'] = str(dest.relative_to(OUT))
                save(OUT/'status.json', state)
                if not (dest/'result.json').exists():
                    while True:
                        if dest.exists():
                            dest.rename(dest.with_name(dest.name+'_interrupted_'+str(time.time_ns())))
                        dest.mkdir(parents=True)
                        try:
                            run(['numactl', '--cpunodebind=0', '--membind=0', sys.executable, '-B',
                                 str(Path(__file__).resolve()), '--arm', arm, '--output', str(dest)], dest, env, None)
                        except RuntimeError as error:
                            if not str(error).startswith('Target GPU contention:'):
                                raise
                            print('GPU contention: retaining interrupted attempt and waiting.', flush=True)
                            continue
                        break
                r = read(dest/'result.json')
                assert r['correctness_passed'] and r['images'] == 50000 and r['unique_samples'] == 512
                if trial:
                    first = read(OUT/arm/'trial_0/result.json')
                    assert r['batch_hashes'] == first['batch_hashes']
                    assert r['predictions_sha256'] == first['predictions_sha256']
                records.append(dict(arm=arm, trial=trial, seconds=r['seconds'], images_per_s=r['images_per_s']))
                state['completed_runs'] = len(records)
                save(OUT/'status.json', state)
                write_csv(OUT/'trials.csv', records)
                print('COMPLETED', records[-1], flush=True)
        summary = []
        for arm in ARMS:
            values = [r['images_per_s'] for r in records if r['arm'] == arm]
            summary.append(dict(arm=arm, trials=len(values), median_images_per_s=statistics.median(values),
                min_images_per_s=min(values), max_images_per_s=max(values)))
        write_csv(OUT/'summary.csv', summary)
        state.update(status='COMPLETE', active=None)
    except Exception as error:
        state.update(status='FAILED', error=repr(error))
        raise
    finally:
        save(OUT/'status.json', state)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=ARMS)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.arm:
        worker(args.arm, args.output)
    else:
        run_all()
