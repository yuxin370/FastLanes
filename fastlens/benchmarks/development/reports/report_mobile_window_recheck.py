"""Publish only complete three-trial MobileNetV2-DCT24 recheck cohorts."""
import csv
import json
from pathlib import Path
import shlex
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'results/mobilenet24_window_recheck_20261002'


def read(path):
    return json.loads(path.read_text())


def rows(name):
    return list(csv.DictReader((ROOT / name).open()))


def write(name, records):
    path = ROOT / name
    before = OUT / 'before' / name
    if path.exists() and not before.exists():
        before.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, before)
    fields = list(dict.fromkeys(k for r in records for k in r))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    for device, gpu in [('4090', 'RTX 4090'), ('H100', 'H100')]:
        directories = [OUT / device / f'trial_{trial}' for trial in range(3)]
        if not all((d / 'accepted.json').exists() for d in directories):
            continue
        observations = []
        for directory in directories:
            d = read(directory / 'result.json')
            a = read(directory / 'accepted.json')
            m = read(directory / 'run.command.json')
            assert a['correctness_passed'] and m['returncode'] == 0 and m['code_unchanged_during_run']
            observations.append(dict(d, gpu=gpu, model='MobileNetV2', domain='DCT', pipeline='FastLens',
                task='training_window', samples=d['images'], elapsed_s=d['seconds'],
                images_per_s=d['images'] / d['seconds'], batch_size=d['batch'],
                raw_result=str((directory / 'result.json').relative_to(ROOT)),
                source_log=str((directory / 'run.log').relative_to(ROOT)), command=shlex.join(m['command']),
                concurrent_load_notes=str((directory / 'concurrent_load.jsonl').relative_to(ROOT)),
                cpu_affinity=json.dumps(d['cpu_affinity']), numa_binding=d['numa_policy'],
                correctness='PASS', correctness_status='PASS', correctness_report=a['correctness_report'],
                valid=True, strict_matched=False, code_fingerprint=a['code_fingerprint'],
                native_sha256=read(OUT / 'handoff.json')['native_sha256'],
                implementation_cohort=OUT.name, measurement_cohort=OUT.name))
        # Detailed native counters remain in source JSON rather than stringified CSV dictionaries.
        observations = [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in observations]
        for name in ['results/p0/training_window_raw.csv', 'results/all_measurements.csv']:
            previous = rows(name)
            known = {r['raw_result'] for r in previous}
            write(name, previous + [r for r in observations if r['raw_result'] not in known])
        rates = [r['images_per_s'] for r in observations]
        median = statistics.median(rates)
        cv = statistics.stdev(rates) / statistics.mean(rates)
        def matches(r):
            return (r['gpu'] == gpu and r['model'] == 'MobileNetV2' and r['domain'] == 'DCT'
                    and r['pipeline'] == 'FastLens' and r['task'] == 'training_window')
        current = rows('results/p0/training_window_summary.csv')
        summary = dict(next(r for r in current if matches(r)), median_images_per_s=median,
            min_images_per_s=min(rates), max_images_per_s=max(rates),
            median_elapsed_s=statistics.median(r['elapsed_s'] for r in observations),
            cv_percent=100 * cv, coefficient_of_variation=cv,
            raw_results=';'.join(r['raw_result'] for r in observations), implementation_cohort=OUT.name,
            notes='All three predeclared recheck trials retained; historical observations retained. Shared-host load recorded per run.')
        for name in ['results/p0/training_window_summary.csv', 'results/training_table.csv', 'results/paper/training_windows.csv']:
            write(name, [dict(r, **summary) if matches(r) else r for r in rows(name)])
        ratios = rows('results/paper/training_deployment_ratios.csv')
        for r in ratios:
            if r['gpu'] == gpu and r['dct_model'] == 'MobileNetV2-DCT24':
                r.update(imagelanes_images_per_s=median, deployment_ratio=median / float(r['rgb_images_per_s']),
                         dct_raw=summary['raw_results'])
        write('results/paper/training_deployment_ratios.csv', ratios)
        ratio = next(r for r in ratios if r['gpu'] == gpu and r['dct_model'] == 'MobileNetV2-DCT24')
        status_path = ROOT / 'results/PAPER_STATUS.md'
        lines = status_path.read_text().splitlines()
        for index, line in enumerate(lines):
            if line.startswith(f'| {gpu} | MobileNetV2-DCT24 |') and len(line.split('|')) == 10:
                cells = line.split('|')
                cells[7] = f' {median:.1f} '
                cells[8] = f" {float(ratio['deployment_ratio']):.3f} "
                lines[index] = '|'.join(cells)
        status_path.write_text('\n'.join(lines) + '\n')
        efficiencies = rows('results/analysis/pipeline_efficiency.csv')
        for r in efficiencies:
            if matches(r):
                reference = read(ROOT / r['ceiling_raw_results'])
                for key in ('initial_model_hash', 'precision', 'batch', 'accumulation', 'compile', 'tf32'):
                    assert reference[key] == observations[0][key], key
                r.update(summary, end_to_end_images_per_s=median,
                         pipeline_efficiency=median / float(r['model_only_images_per_s']))
        write('results/analysis/pipeline_efficiency.csv', efficiencies)
        manifest = rows('results/experiment_manifest.csv')
        for r in manifest:
            if r['experiment_id'].startswith(f'{OUT.name}:{device}:'):
                r.update(status='DONE', correctness='PASS')
        write('results/experiment_manifest.csv', manifest)
        original = next(r for r in read(OUT / 'previous_window_summary.json') if matches(r))
        print(json.dumps(dict(gpu=gpu, median=median, min=min(rates), max=max(rates), cv_percent=100 * cv,
                             change_percent=100 * (median / float(original['median_images_per_s']) - 1))))


if __name__ == '__main__':
    main()
