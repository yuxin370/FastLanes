"""Crop A/B using the native read/timing loop from locality_microbench.py."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import resource
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / 'build/fastlens/torch')]
CASES = ('full', 'inside128', 'center128', 'random128')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--manifest-a', type=Path, required=True, help='one-image-per-rowgroup manifest')
parser.add_argument('--manifest-b', type=Path, required=True, help='tiled vector-rowgroup manifest')
parser.add_argument('--sample-manifest', type=Path, required=True, help='shared ordered pipeline sample_manifest.json')
parser.add_argument('--output-dir', type=Path, required=True, help='new output directory')
parser.add_argument('--validate', action='store_true', help=argparse.SUPPRESS)
parser.add_argument('--variant', choices=('v2', 'v3'), help=argparse.SUPPRESS)
parser.add_argument('--case', choices=CASES, help=argparse.SUPPRESS)
parser.add_argument('--trial', type=int, help=argparse.SUPPRESS)
ARGS = parser.parse_args()
OUT = ARGS.output_dir.resolve()
MANIFESTS = {'v2': ARGS.manifest_a.resolve(), 'v3': ARGS.manifest_b.resolve()}
SAMPLE_MANIFEST = ARGS.sample_manifest.resolve()
SAMPLES = json.loads(SAMPLE_MANIFEST.read_text())['samples']
IDS = [s['galp_image_id'] for s in SAMPLES]
RNG = random.Random(11997733)
CROPS = {
    'full': [[0, 0, 512, 512] for _ in IDS],
    'inside128': [[64, 64, 128, 128] for _ in IDS],
    'center128': [[192, 192, 128, 128] for _ in IDS],
    'random128': [[RNG.randrange(25)*16, RNG.randrange(25)*16, 128, 128] for _ in IDS],
}
OPTIONS = dict(layout='compact', dct_coeffs='all', cache_capacity_mib=0,
               plan_cache_capacity=0, decode_batch_rowgroups=64,
               enable_rowgroup_prefetch=True, rowgroup_prefetch_depth=16,
               rowgroup_prefetch_workers=8, rowgroup_prefetch_min_decode_batches=1,
               crop_execution_mode='auto')
COUNTS = ('planned_selected_vector_count', 'selected_vector_count', 'full_vector_count',
          'compressed_payload_bytes_read', 'full_compressed_payload_bytes',
          'rowgroup_storage_bytes_read', 'pread_count', 'rowgroup_count',
          'workset_upload_dma_bytes')
PEAKS = ('galp_native_device_peak_in_use_bytes', 'galp_native_pinned_peak_in_use_bytes')


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def paths(variant):
    from fastlens.benchmarks.system_dct_major.common import parse_manifest
    return [Path(s['fls_path']) for s in parse_manifest(MANIFESTS[variant])['shards']]


def read(reader, ids, crops):
    return reader.read_batch(ids, transforms=[dict(crop=c, horizontal_flip=False) for c in crops], **OPTIONS)


def validate():
    import torch
    import _fastlens_direct_dct as native
    torch.set_num_threads(8)
    indices = list(range(0, len(IDS), len(IDS)//64))
    ids = [IDS[i] for i in indices]
    readers = {v: native.DirectDctReader(str(p.resolve())) for v,p in MANIFESTS.items()}
    full = read(readers['v2'], ids, [CROPS['full'][i] for i in indices])
    reference = full.coefficients.cpu()
    keys = ('global_image_index', 'semantic_slot_id', 'block_x', 'block_y')
    metadata = list(full.block_metadata)
    locations = {tuple(m[k] for k in keys): i for i,m in enumerate(metadata)}
    del full
    for case in CASES:
        crops = [CROPS[case][i] for i in indices]
        by_id = dict(zip(ids, crops))
        expected_keys = []
        for m in metadata:
            x,y,w,h = by_id[m['global_image_index']]
            scale = 8 if m['semantic_slot_id'] == 0 else 16
            if x//scale <= m['block_x'] < (x+w)//scale and y//scale <= m['block_y'] < (y+h)//scale:
                expected_keys.append(tuple(m[k] for k in keys))
        expected = reference[[locations[k] for k in expected_keys]]
        for variant,reader in readers.items():
            batch = read(reader, ids, crops)
            observed_keys = [tuple(m[k] for k in keys) for m in batch.block_metadata]
            assert observed_keys == expected_keys, (case,variant,'sample/component/block order')
            torch.testing.assert_close(batch.coefficients.cpu(), expected, rtol=0, atol=0)
            print(json.dumps(dict(validation='pass',case=case,variant=variant,images=len(ids))),flush=True)
            del batch
    write(OUT/'validation.json', dict(ok=True,images=len(ids),cases=CASES,
          reference='full-image v2 decode, select source blocks by CPU coordinates; int16 exact equality',
          sample_ids=ids))


def leg(variant, case, trial):
    import torch
    import _fastlens_direct_dct as native
    from fastlens.benchmarks.profiling.input_motivation import cache_pages
    torch.set_num_threads(8)
    cache = cache_pages(paths(variant))
    assert cache['resident_fraction'] == 1.0, cache
    reader = native.DirectDctReader(str(MANIFESTS[variant].resolve()))
    requests = [(IDS[i:i+64],CROPS[case][i:i+64]) for i in range(0,len(IDS),64)]
    # One complete unmeasured traversal warms reader metadata and allocations.
    for ids,crops in requests:
        batch=read(reader,ids,crops)
        tensor=batch.coefficients
        torch.cuda.synchronize()
        del tensor,batch
    counters={k:0 for k in COUNTS}
    peaks={k:0 for k in PEAKS}
    elapsed=0.0
    io_before={k:int(v) for k,v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}
    for ids,crops in requests:
        torch.cuda.synchronize()
        start=time.perf_counter()
        batch=read(reader,ids,crops)
        tensor=batch.coefficients
        torch.cuda.synchronize()
        elapsed+=time.perf_counter()-start
        stats=dict(batch.execution_stats)
        for k in COUNTS: counters[k]+=stats[k]
        for k in PEAKS: peaks[k]=max(peaks[k],stats[k])
        del tensor,batch
    io_after={k:int(v) for k,v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}
    result=dict(variant=variant,case=case,trial=trial,images=len(IDS),batch_size=64,
        elapsed_s=elapsed,images_per_s=len(IDS)/elapsed,counters=counters,peaks=peaks,
        actual_block_device_read_bytes=io_after['read_bytes']-io_before['read_bytes'],
        host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        cache=cache,gpu=torch.cuda.get_device_name(),options=OPTIONS,
        timing_scope='synchronous native raw int16 DCT read/decode/gather to GPU; no model or resize; reader startup and warmup excluded',
        warmup_images=len(IDS),sample_manifest=str(SAMPLE_MANIFEST))
    write(OUT/f'{case}_{trial}_{variant}.json',result)
    print(json.dumps({k:result[k] for k in ('variant','case','trial','elapsed_s','images_per_s')}),flush=True)


def summarize():
    rows=[]
    for case in CASES:
        by_variant={v:[json.loads((OUT/f'{case}_{trial}_{v}.json').read_text()) for trial in range(5)] for v in MANIFESTS}
        row=dict(case=case)
        for v,legs in by_variant.items():
            speeds=[r['images_per_s'] for r in legs]
            full=json.loads((OUT/f'full_0_{v}.json').read_text())['counters']
            counts=legs[0]['counters']
            assert all(r['counters']==counts for r in legs), (case,v,'counter drift')
            row[v]=dict(median_images_per_s=statistics.median(speeds),
                cv=statistics.pstdev(speeds)/statistics.mean(speeds),
                min_images_per_s=min(speeds),max_images_per_s=max(speeds),
                decoded_vectors=counts['selected_vector_count'],
                vector_ratio_to_full=counts['selected_vector_count']/full['selected_vector_count'],
                payload_bytes=counts['compressed_payload_bytes_read'],
                payload_ratio_to_full=counts['compressed_payload_bytes_read']/full['compressed_payload_bytes_read'])
        row['v3_over_v2_throughput']=row['v3']['median_images_per_s']/row['v2']['median_images_per_s']
        rows.append(row)
    write(OUT/'summary.json',dict(rows=rows,validation=json.loads((OUT/'validation.json').read_text())))
    print(json.dumps(rows,indent=2),flush=True)


def run():
    from fastlens.benchmarks.profiling.input_motivation import cache_pages
    command=[sys.executable,'-B',str(Path(__file__).resolve()),
             '--manifest-a',str(MANIFESTS['v2']),'--manifest-b',str(MANIFESTS['v3']),
             '--sample-manifest',str(SAMPLE_MANIFEST),'--output-dir',str(OUT)]
    subprocess.run(command+['--validate'],check=True,cwd=ROOT)
    with ThreadPoolExecutor(max_workers=2) as pool:
        warmed=dict(zip(MANIFESTS,pool.map(lambda v:cache_pages(paths(v),warm=True),MANIFESTS)))
    print('Cache residency:',json.dumps(warmed),flush=True)
    assert all(v['resident_fraction']==1 for v in warmed.values())
    # Each measurement is a fresh process; reverse A/B order on alternate trials.
    for trial in range(5):
        order=('v2','v3') if trial%2==0 else ('v3','v2')
        for case in CASES[trial%4:]+CASES[:trial%4]:
            for variant in order:
                subprocess.run(command+['--variant',variant,'--case',case,'--trial',str(trial)],check=True,cwd=ROOT)
    summarize()


if __name__ == '__main__':
    if ARGS.validate:
        validate()
    elif ARGS.variant:
        leg(ARGS.variant, ARGS.case, ARGS.trial)
    else:
        OUT.mkdir(parents=True, exist_ok=False)
        run()
