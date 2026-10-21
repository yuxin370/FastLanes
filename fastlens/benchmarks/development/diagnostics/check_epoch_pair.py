"""Offline evidence for native-owned augmentation markers in saved B6 epochs.

Replay logical decisions only: this does not decode images or rerun training.
Native tensor equivalence is supported by the independent sampled input/update
gate, not by pretending that a native ownership marker is an observed hash.
"""
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from fastlens.benchmarks.development.historical.run_training_expansion import read

NATIVE_MARKER = "native-owned-by-rgbnomore-training-pls-v1"
AUGMENTATION_DIGESTS = ("crop_key_digest", "flip_key_digest", "randaugment_digest", "mixup_digest")
OBSERVED_DIGESTS = ("sample_order_digest", "pool_membership_digest")
CONTRACT_FIELDS = (
    "training_seed", "condition_id", "recipe_hash", "layout_hash", "train_manifest_hash",
    "crop_policy", "crop_key_scope", "flip", "randaugment", "mixup", "order_policy",
    "segments_per_pool", "microbatch_size", "gradient_accumulation", "effective_update_batch",
    "model_configuration", "model_input_contract", "model_execution", "initial_model_hash",
    "precision", "optimizer_configuration", "scheduler", "weight_decay_configuration",
    "sample_order_digest", "pool_membership_digest", "crop_key_digest", "code_version",
)


def replay_digests(mapping, contract):
    from fastlens.benchmarks.training_pls.core_schedule import epoch_position_pools, crop_key, flip_key
    from fastlens.benchmarks.training_pls.published_augmentation import (
        published_randaugment_records, published_mixup_record,
    )

    hashes = {k: hashlib.sha256() for k in (*OBSERVED_DIGESTS, *AUGMENTATION_DIGESTS)}
    seed = contract['training_seed']
    microbatch = 0
    first_ids = []
    for _, pls_ids, positions in epoch_position_pools(
        mapping, condition_id=contract['condition_id'], seed=seed, epoch=0
    ):
        for pls_id in (*pls_ids, 2**63 - 1):
            hashes['pool_membership_digest'].update(int(pls_id).to_bytes(8, 'little'))
        for begin in range(0, len(positions), contract['microbatch_size']):
            selected = positions[begin:begin + contract['microbatch_size']]
            ids = [mapping.logical_sample_ids[i] for i in selected]
            if microbatch < 2:
                first_ids.append(ids)
            for position, logical_id in zip(selected, ids):
                hashes['sample_order_digest'].update(int(position).to_bytes(8, 'little'))
                hashes['crop_key_digest'].update(crop_key(training_seed=seed, epoch=0,
                    logical_sample_id=logical_id, virtual_pls_id=position // mapping.segment_images,
                    crop_policy=contract['crop_policy']).encode('ascii'))
                hashes['flip_key_digest'].update(flip_key(training_seed=seed, epoch=0,
                    logical_sample_id=logical_id).encode('ascii'))
            records = published_randaugment_records(training_seed=seed, epoch=0, logical_sample_ids=ids)
            mixup = published_mixup_record(training_seed=seed, epoch=0,
                microbatch_index=microbatch, alpha=contract['mixup']['alpha'])
            for field, value in (('randaugment_digest', records), ('mixup_digest', mixup)):
                hashes[field].update(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())
            microbatch += 1
    return {k: h.hexdigest() for k, h in hashes.items()}, first_ids


def verify_transformer_pair(jobs, values, *, diagnostic_precision=None):
    paired = {j['pipeline']: (j, v) for j, v in zip(jobs, values)}
    if len(jobs) != 2 or set(paired) != {'imagelanes', 'cpu_dct'}:
        raise ValueError('Expected one native/CPU-DCT epoch pair')
    native_job, native = paired['imagelanes']
    cpu_job, cpu = paired['cpu_dct']
    directories = [Path(j['output'])/'runs/B6/seed_11997733' for j in (native_job, cpu_job)]
    contracts = [read(d/'run_manifest.json') for d in directories]
    for field in CONTRACT_FIELDS:
        if contracts[0][field] != contracts[1][field]:
            raise ValueError(f'Matched epoch contract mismatch: {field}')
    contract = contracts[0]
    if contract['condition_id'] != 'B6' or contract['microbatch_size'] != 64:
        raise ValueError('Offline reconstruction requires the B6 64-image microbatch contract')
    for directory in directories:
        epochs = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()
                  if json.loads(line).get('scope') == 'epoch']
        if len(epochs) != 1 or epochs[0]['epoch'] != 1:
            raise ValueError('Offline reconstruction requires the first complete epoch')
    if set(native['matching_digests']) != set(cpu['matching_digests']) or set(cpu['matching_digests']) != set(
        (*OBSERVED_DIGESTS, *AUGMENTATION_DIGESTS)
    ):
        raise ValueError('Unexpected epoch digest fields')
    for field in OBSERVED_DIGESTS:
        if native['matching_digests'][field] != cpu['matching_digests'][field]:
            raise ValueError(f'Matched epoch mismatch: {field}')
    for field in AUGMENTATION_DIGESTS:
        if native['matching_digests'][field] != NATIVE_MARKER:
            raise ValueError(f'Expected native ownership marker: {field}')

    physical = contract['physical_execution']
    mapping_path = Path(physical['premixed_mapping_csv'])
    with mapping_path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != physical['premixed_mapping_sha256']:
            raise ValueError('Saved physical mapping changed')
    cpu_command = read(Path(cpu_job['output'])/'run.command.json')['command']
    if (cpu_command[cpu_command.index('--premixed-mapping-csv') + 1] != str(mapping_path)
        or cpu_command[cpu_command.index('--expected-mapping-sha256') + 1] != physical['premixed_mapping_sha256']):
        raise ValueError('CPU-DCT used a different physical mapping')
    with mapping_path.open() as stream:
        ids = [row['logical_sample_id'] for row in csv.DictReader(stream)]
    count, segment = len(ids), physical['segment_images']
    if count != native['samples']:
        raise ValueError('Mapping population differs from measured epoch')
    mapping = SimpleNamespace(sample_count=count, logical_sample_ids=ids, segment_images=segment,
        positions_by_pls=[range(i, min(i + segment, count)) for i in range(0, count, segment)])
    replay, first_ids = replay_digests(mapping, contract)
    if replay != cpu['matching_digests']:
        raise ValueError('Logical augmentation replay differs from observed CPU-DCT digests')

    reports = []
    for job, value in zip(jobs, values):
        report = read(job['correctness_report'])
        update = report['update_check']
        if (not report['passed'] or report['samples'] != 128
            or 'B6 epoch index 0 first two microbatches' not in report['scope']
            or report['cuda_visible_devices'] != read(Path(job['output'])/'run.command.json')['environment']['CUDA_VISIBLE_DEVICES']
            or update['initial_model_hash'] != value['initial_model_hash']
            or update['precision'] != (diagnostic_precision or value['precision'])):
            raise ValueError('Independent input/update evidence does not match this epoch')
        observations = report['observations']
        if [row['sample_ids'] for row in observations] != first_ids:
            raise ValueError('Independent input gate used different logical samples')
        if not all(row['targets']['allclose'] and all(t['allclose'] for t in row['tensors']) for row in observations):
            raise ValueError('Independent tensor/target comparison failed')
        if len(update['observations']) != 2 or not all(
            row['gradients_finite'] and row['parameters_changed'] and row['optimizer_updates'] == 1
            for row in update['observations']
        ):
            raise ValueError('Independent update comparison failed')
        reports.append(job['correctness_report'])
    return dict(passed=True, training_precision=native['precision'],
        diagnostic_precision=diagnostic_precision or native['precision'],
        observed_schedule_digests={k: replay[k] for k in OBSERVED_DIGESTS},
        cpu_observed_augmentation_digests={k: replay[k] for k in AUGMENTATION_DIGESTS},
        native_augmentation='ownership markers, not observed hashes',
        logical_decisions='complete epoch-0 keyed replay matches all six CPU-DCT digests',
        matched_contract_fields=list(CONTRACT_FIELDS), independent_reports=reports,
        scope='matched logical recipe and sampled numerical input/update equivalence; no full-epoch bitwise input claim')
