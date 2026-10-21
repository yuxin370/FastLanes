"""Review the completed old-entry MobileNet epoch without altering its raw records."""
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fastlens.benchmarks.development.historical.run_full_rgb_epochs import OUT
from fastlens.benchmarks.development.historical.run_full_rgb_epochs import read
from fastlens.benchmarks.development.historical.run_full_rgb_epochs import save
from fastlens.benchmarks.development.historical.run_full_rgb_epochs import report
from fastlens.benchmarks.development.historical.resume_full_rgb_epochs import accepted
from fastlens.benchmarks.execution.revision_results import file_sha


def main():
    state = read(OUT / 'status.json')
    job, = [j for j in state['jobs'] if (j['gpu'], j['model'], j['pipeline']) == ('H100', 'mobilenet24', 'pytorch')]
    path = Path(job['output'])
    raw, command = read(path / 'result.json'), read(path / 'run.command.json')
    archived = OUT / 'source_before_stream_priority/training_window_rgb.py'
    snapshot = read(OUT / 'runtime_snapshot_20261006.json')
    entry = str(ROOT / 'fastlens/benchmarks/paper/shared/training_window_rgb.py')
    assert file_sha(archived) == snapshot['files'][entry]
    assert command['returncode'] == 86 and not command['code_unchanged_during_run']
    assert command['code_fingerprint'] == snapshot['fingerprint']
    assert command['code_change_error'] == ('Evaluation code changed after snapshot: ' + entry + ', ' +
        str(ROOT / 'fastlens/benchmarks/dct_models/evaluate_rgb.py'))
    assert 'model_stream_priority' not in raw
    log_lines = (path / 'run.log').read_text().splitlines()
    assert json.loads(log_lines[-1]) == raw
    assert not any('Traceback (most recent call last)' in line for line in log_lines)
    assert raw['correctness_passed'] and raw['full_epoch'] and raw['images'] == 1281167 and raw['updates'] == 1252
    assert raw['loss_finite'] and raw['warmup_initial_state_restored'] and raw['final_update_samples'] == 143
    assert raw['parameter_update_verified']
    ids = read(Path(raw['sample_ids_file']))
    assert len(ids) == len(set(ids)) == 1281167
    assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == raw['sample_order_sha256']
    reference = accepted(Path(next(j['output'] for j in state['jobs'] if j['gpu'] == '4090' and j['model'] == 'mobilenet24' and j['pipeline'] == 'pytorch')))
    for key in ('initial_model_hash', 'precision', 'batch', 'accumulation', 'seed', 'sample_order_sha256',
                'augmentation_configuration', 'compile', 'tf32', 'effective_batch_size'):
        assert raw[key] == reference[key], key
    review = dict(accepted=True, observed_at=datetime.datetime.now().astimezone().isoformat(),
        raw_result_sha256=file_sha(path / 'result.json'), command_record_sha256=file_sha(path / 'run.command.json'),
        archived_executed_entry=str(archived), archived_entry_sha256=file_sha(archived),
        snapshot_start_entry_sha256=snapshot['files'][entry], original_wrapper_returncode=86,
        raw_records_modified=False, child_exit_status='not separately retained by the original wrapper',
        evidence='Start snapshot verified exact archived entry. The already running __main__ function was loaded before edits and is not reloaded. Final raw schema lacks the newly added stream field, and the final log line equals the complete result. RGB evaluation CLI is not imported by this training path. Full coverage, update, finite-state, initialization and order checks passed.',
        limitation='Disk-wide code check failed; accepted only after this specific in-memory execution review. No throughput from an incomplete epoch is used.')
    save(path / 'code_change_review.json', review)
    save(path / 'accepted.json', dict(correctness=True, comparison_type='deployment', raw_result=str(path / 'result.json'),
         code_change_review=str(path / 'code_change_review.json'), notes=review['evidence']))
    job.update(status='DONE', acceptance_notes=' Loaded old entry verified against start SHA; disk-change wrapper status 86 retained. See raw directory/code_change_review.json.')
    job.pop('error', None)
    save(OUT / 'status_before_frozen_resume_20261006.json', state)
    report(state)
    print(json.dumps(dict(accepted=True, images_per_s=raw['images']/raw['seconds'], review=str(path / 'code_change_review.json'))))


if __name__ == '__main__':
    main()
