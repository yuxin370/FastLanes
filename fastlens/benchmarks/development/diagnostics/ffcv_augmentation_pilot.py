"""Isolated FFCV RGB augmentation pilot; does not change the paper's runners.

Aligns RandAugment counts/magnitude and Mixup parameters with FastLens.
RGB and DCT operator sets and crop/order remain different. This is a recipe
comparison, not an exact augmentation-equivalence or input-system comparison.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
SEED = 11997733
POLICY = 'rgbnomore-rgb-ra2-m3-bins11-mixup0.2-pilot-v1'


def mixup(inputs, labels, microbatch):
    import torch
    from fastlens.benchmarks.training_pls.core_schedule import mixup_key, stable_seed
    key = mixup_key(training_seed=SEED, epoch=2, microbatch_index=microbatch)
    seed = stable_seed('dct-mixup-dirichlet', key) % (2**63 - 1)
    # Use the published keyed distribution without reseeding model dropout.
    generator = torch.Generator().manual_seed(seed)
    weights = torch._sample_dirichlet(torch.tensor([0.2, 0.2]), generator=generator).sort(descending=True).values
    original, rolled = map(float, weights)
    mixed = tuple(value * original + value.roll(1, 0) * rolled for value in inputs)
    targets = torch.nn.functional.one_hot(labels, 1000).to(inputs[0].dtype)
    return mixed, targets * original + targets.roll(1, 0) * rolled


class RgbAugmentation:
    def __init__(self):
        import torch
        sys.path.insert(0, str(Path(os.environ.get('RGBNOMORE_ROOT', Path.home() / 'RGB-no-more'))))
        from utils.custom_transforms import RandAugment_bv, _apply_op
        self.torch = torch
        self.ra = RandAugment_bv(num_ops=2, magnitude=3, num_magnitude_bins=11, fill=128)
        self.apply_op = _apply_op
        self.space = self.ra._augmentation_space(11, (224, 224))

    def plan(self, sample_id):
        seed = int.from_bytes(hashlib.sha256(f'{SEED}:2:{sample_id}:rgb-ra'.encode()).digest()[:8], 'little')
        rng = random.Random(seed)
        result = []
        for _ in range(2):
            op = rng.choice(self.ra.ops_list)
            magnitudes, signed = self.space[op]
            magnitude = float(magnitudes[3]) if magnitudes.ndim else 0.0
            if signed and rng.randrange(2):
                magnitude = -magnitude
            result.append((op, magnitude, rng.randrange(2**63)))
        return result

    def operation(self, images, op, magnitude, seed):
        # Cutout chooses a location per image; never share that location across
        # a batch. Other operators are deterministic given op and magnitude.
        if op == 'Cutout':
            with self.torch.random.fork_rng(devices=[]):
                self.torch.random.default_generator.manual_seed(seed)
                return self.apply_op(images, op, magnitude, self.ra.interpolation, 128)
        return self.apply_op(images, op, magnitude, self.ra.interpolation, 128)

    def apply(self, images, plans):
        result = images.clone()
        for stage in range(2):
            groups = defaultdict(list)
            for i, plan in enumerate(plans):
                op, magnitude, seed = plan[stage]
                if op == 'Cutout':
                    result[i] = self.operation(result[i], op, magnitude, seed)
                else:
                    groups[op, magnitude].append(i)
            for (op, magnitude), positions in groups.items():
                indices = self.torch.tensor(positions, device=images.device)
                result[indices] = self.operation(result[indices], op, magnitude, 0)
        return result


def verify_augmentation(device='cpu'):
    import torch
    from fastlens.benchmarks.training_pls.published_augmentation import apply_published_mixup
    aug = RgbAugmentation()
    # Exercise every RGB operator, both signs, and independent Cutout locations.
    plans = []
    for i, op in enumerate(aug.ra.ops_list):
        magnitudes, signed = aug.space[op]
        value = float(magnitudes[3]) if magnitudes.ndim else 0.0
        for sign in (-1, 1) if signed else (1,):
            plans.append([(op, sign * value, i), ('Cutout', 4.0, 100 + len(plans))])
    images = torch.randint(0, 256, (len(plans), 3, 32, 32), dtype=torch.uint8).to(device)
    expected = images.clone()
    for i, plan in enumerate(plans):
        for op, magnitude, seed in plan:
            expected[i] = aug.operation(expected[i], op, magnitude, seed)
    actual = aug.apply(images, plans)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1)
    labels = torch.arange(len(plans), device=device)
    normalized = actual.float().div(127.5).sub(1)
    reference, reference_targets, record = apply_published_mixup((normalized,), labels,
        training_seed=SEED, epoch=2, microbatch_index=0, alpha=0.2)
    rng_before = torch.get_rng_state().clone()
    mixed, targets = mixup((normalized,), labels, 0)
    assert torch.equal(rng_before, torch.get_rng_state())
    torch.testing.assert_close(mixed[0], reference[0])
    torch.testing.assert_close(targets, reference_targets)
    weight = record['lambda_original']
    torch.testing.assert_close(mixed[0], normalized * weight + normalized.roll(1, 0) * record['lambda_rolled'])
    torch.testing.assert_close(targets.sum(1), torch.ones(len(plans), device=device))
    assert torch.isfinite(mixed[0]).all() and targets.shape == (len(plans), 1000)
    assert aug.plan('sample-1') == aug.plan('sample-1')
    print(f'PASS: per-image RandAugment reference and Mixup on {device}', flush=True)


def worker(mode, argv):
    # Preserve the existing FFCV/OpenCV JPEG library import order.
    import cv2
    cv2.setNumThreads(1)
    from ffcv.loader import Loader  # noqa: F401
    import torch
    from fastlens.benchmarks.development.historical import p0_adapters
    from fastlens.benchmarks.paper.shared import training_window_rgb
    from fastlens.benchmarks.baselines.system_rgbnomore.training.pipeline import TrainingBatch
    from fastlens.benchmarks.baselines.system_rgbnomore.training.sample_order import SampleIdentity
    check = '--check' in argv
    output = Path(argv[argv.index('--output') + 1])
    if mode == 'augmented':
        if not check:
            gate = json.loads(Path(argv[argv.index('--correctness-report') + 1]).read_text())
            if gate.get('augmentation_policy') != POLICY:
                raise ValueError('Correctness check belongs to another augmentation policy')
        if check:
            verify_augmentation('cuda')
        counts = {'augmented_batches': 0}

        class AugmentedFfcvWindow(p0_adapters.FfcvWindow):
            def __init__(self, samples, device, decisions, root=None):
                self.augmentation = RgbAugmentation()
                self.plans = [self.augmentation.plan(s.logical_sample_id) for s in samples]
                self.microbatch = 0
                super().__init__(samples, device, decisions, root=root)

            def next_batch(self):
                images, labels, ordinals = next(self.iterator)
                ids = ordinals.reshape(-1).tolist()
                labels = labels.reshape(-1).long()
                # The existing check validates the unaugmented crop/flip prefix
                # on its first warmup microbatch. All remaining check batches
                # and every performance batch execute RA and Mixup.
                prefix_check = check and self.microbatch == 0
                if not prefix_check:
                    images = self.augmentation.apply(images, [self.plans[i] for i in ids])
                inputs = (images.float().div_(127.5).sub_(1),)
                if not prefix_check:
                    inputs, labels = mixup(inputs, labels, self.microbatch)
                    counts['augmented_batches'] += 1
                self.microbatch += 1
                return TrainingBatch(inputs=inputs, labels=labels,
                    identities=[SampleIdentity(2, i, self.samples[i].logical_sample_id) for i in ids],
                    augmentations=[], on_device=True, stage_seconds={})

        p0_adapters.FfcvWindow = AugmentedFfcvWindow
    sys.argv = [__file__, *argv]
    training_window_rgb.main()
    path = output / 'result.json'
    result = json.loads(path.read_text())
    result['augmentation_policy'] = POLICY if mode == 'augmented' else 'existing-crop-flip-only'
    if mode == 'augmented':
        assert counts['augmented_batches'] > 0
        result.update(counts)
        result['augmentation_configuration'] += '; RGB-no-more RGB RandAugment 2 ops, magnitude 3/11; Mixup alpha=.2, roll-by-one, soft labels'
        result['notes'] += '; RGB/DCT RandAugment operators are not identical. RGB per-sample RA decisions are preplanned before timing; all image transforms and Mixup execute inside the timed input path.'
    path.write_text(json.dumps(result, indent=2) + '\n')


def run_pilot(args):
    from fastlens.benchmarks.execution.inference_load import run
    from fastlens.benchmarks.development.historical.run_compressed_ffcv import GPUS
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS[args.gpu],
        PYTHONPATH=f'{ROOT}:{ROOT}/build/fastlens/torch', TMPDIR=str(Path.home() / 'tmp'),
        OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
        PYTHONDONTWRITEBYTECODE='1', FASTLENS_NSYS_FINE='0')
    for key in ('CUBLAS_WORKSPACE_CONFIG', 'FASTLENS_EVALUATION_SNAPSHOT', 'FASTLENS_ALLOWED_RESIDENT_PIDS'):
        env.pop(key, None)
    beton_root = (ROOT / 'results/preparation/p0_ffcv' if args.encoding == 'raw'
                  else ROOT / 'results/ffcv_jpeg_20261005/preparation/train')
    records = []

    def execute(mode, trial, check=False):
        dest = output / mode / ('check' if check else f'trial_{trial}')
        dest.mkdir(parents=True)
        command = ['numactl', '--cpunodebind=0', '--membind=0', sys.executable,
            str(Path(__file__).resolve()), '--worker', mode,
            '--gpu', args.gpu, '--model', args.model, '--pipeline', 'ffcv',
            '--trial', str(trial), '--ffcv-root', str(beton_root), '--output', str(dest)]
        command += ['--check'] if check else ['--correctness-report', str(output / mode / 'check/result.json')]
        run(command, dest, env, None)
        result = json.loads((dest / 'result.json').read_text())
        assert result['correctness_passed'] and result['loss_finite'] and result['parameter_update_verified']
        if not check:
            assert result['images'] == 65536 and result['cache_before']['resident_fraction'] == 1
            records.append(dict(mode=mode, trial=trial, images_per_s=result['images'] / result['seconds'],
                                raw_result=str(dest / 'result.json')))
            print('COMPLETED', records[-1], flush=True)
            with (output / 'trials.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)

    for mode in ('control', 'augmented'):
        execute(mode, 0, check=True)
    for trial in range(3):
        for mode in (('control', 'augmented') if trial % 2 == 0 else ('augmented', 'control')):
            execute(mode, trial)
    summaries = []
    for mode in ('control', 'augmented'):
        rates = [r['images_per_s'] for r in records if r['mode'] == mode]
        summaries.append(dict(gpu=args.gpu, model=args.model, encoding=args.encoding, mode=mode,
            mean_images_per_s=statistics.mean(rates), median_images_per_s=statistics.median(rates),
            stdev_images_per_s=statistics.stdev(rates), min_images_per_s=min(rates), max_images_per_s=max(rates)))
    with (output / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print('COMPLETE', json.dumps(summaries), flush=True)


if __name__ == '__main__':
    if '--worker' in sys.argv:
        position = sys.argv.index('--worker')
        worker(sys.argv[position + 1], sys.argv[1:position] + sys.argv[position + 2:])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--self-test', action='store_true')
        parser.add_argument('--gpu', choices=['4090', 'H100'], default='4090')
        parser.add_argument('--model', choices=['vitti', 'swinv2'], default='vitti')
        parser.add_argument('--encoding', choices=['raw', 'jpeg'], default='raw')
        parser.add_argument('--output', type=Path, default=ROOT / 'results/ffcv_augmentation_pilot_20261008_4090_vitti_raw')
        args = parser.parse_args()
        if args.self_test:
            verify_augmentation()
        else:
            run_pilot(args)
