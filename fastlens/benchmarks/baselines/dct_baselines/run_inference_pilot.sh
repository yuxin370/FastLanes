#!/usr/bin/env bash
# Matched all64 inference: the 8194-image pilot and full ImageNet-1K validation.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
: "${PYTHON:?Set PYTHON to the fastlanes-cuda Python executable}"
: "${CUDA_VISIBLE_DEVICES:?Select one idle GPU by UUID}"
prepared="${1:?Usage: run_inference_pilot.sh PREPARED OUTPUT [options]}"
output="${2:?Usage: run_inference_pilot.sh PREPARED OUTPUT [options]}"
shift 2
protocol=pilot
resume=false
resume_measurements=false
reuse_verification=
verify_only=false
wait_for_gpu=false
run_kind=inference
expected=8194
settings=
declare -A measurement_cache_mib=()
profile_stages=false
request_order=
seed=11997733
models=(resnet24 vitti)
names=(cpu_dct fastlens ic_shared ic_batched fs)
while (( $# )); do
  case "$1" in
    --ic-protocol|--fs-protocol)
      if [[ "$protocol" != pilot ]]; then echo 'Choose one measurement protocol.' >&2; exit 1; fi
      protocol="$1"; shift ;;
    --resume-validation) resume=true; shift ;;
    --resume-measurements) resume=true; resume_measurements=true; shift ;;
    --reuse-verification) reuse_verification="$2"; shift 2 ;;
    --verify-only) verify_only=true; shift ;;
    --wait-for-gpu) wait_for_gpu=true; shift ;;
    --input-only) run_kind=input_only; shift ;;
    --expected-images) expected="$2"; shift 2 ;;
    --settings) settings="$2"; shift 2 ;;
    --measurement-cache-mib)
      if [[ ! "$2" =~ ^(ic_shared|ic_batched|fs)=(0|[1-9][0-9]*)$ ]]; then
        echo 'Expected --measurement-cache-mib ic_shared|ic_batched|fs=NONNEGATIVE_MIB' >&2; exit 1
      fi
      measurement_cache_mib["${BASH_REMATCH[1]}"]="${BASH_REMATCH[2]}"; shift 2 ;;
    --profile-stages) profile_stages=true; shift ;;
    --request-order) request_order="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --models) read -ra models <<< "$2"; shift 2 ;;
    --systems) read -ra names <<< "$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done
if [[ -n "$reuse_verification" ]]; then
  if "$resume"; then
    echo 'Use --reuse-verification with a new output directory; use --resume-measurements for an existing run.' >&2
    exit 1
  fi
  reuse_verification="$(cd "$reuse_verification" && pwd)"
  # Require all requested gates before starting any work. Missing evidence
  # must not silently fall back to another full numerical verification.
  for model in "${models[@]}"; do
    for name in "${names[@]}"; do
      for file in "$name.contract/matched.json" "$name.verified.json"; do
        if [[ ! -f "$reuse_verification/$model/$file" ]]; then
          echo "Missing passed-verification artifact: $reuse_verification/$model/$file" >&2
          exit 1
        fi
      done
    done
  done
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$HOME/tmp"
export FASTLENS_BLOCK_MAJOR_ACCESS_DIR="$PWD/fastlens/data/compressed/imagenet512_val_block_major/access"
cpus="${CPU_LIST:-$("$PYTHON" -B -c 'import os; cpus=sorted(os.sched_getaffinity(0)); assert len(cpus)>=8; print(",".join(str(cpu) for cpu in cpus[:8]))')}"
python=(taskset -c "$cpus" "$PYTHON" -B)
"${python[@]}" - "$prepared/contract/source.json" "$expected" <<'PY'
import json, sys
from fastlens.benchmarks.baselines.dct_baselines.source import validate_population
source = json.load(open(sys.argv[1]))
count = int(sys.argv[2])
print(validate_population(source, source['images'], expected_images=count,
                          expected_classes=1000 if count == 50000 else None))
PY
if "$resume"; then
  test -d "$output"
  if ! "$resume_measurements" && compgen -G "$output/*/*.repeat*.jsonl" >/dev/null; then
    echo "Validation resume requires a run with no performance measurements yet." >&2
    exit 1
  fi
else
  mkdir "$output"
fi
if [[ -n "$reuse_verification" ]]; then
  for model in "${models[@]}"; do
    mkdir -p "$output/$model"
    for name in "${names[@]}"; do
      ln -s "$reuse_verification/$model/$name.contract" "$output/$model/$name.contract"
      ln -s "$reuse_verification/$model/$name.verified.json" "$output/$model/$name.verified.json"
    done
  done
  resume=true
fi
bench_options=()
if [[ "$protocol" == --ic-protocol ]]; then
  bench_options=(--warmup-seconds 60 --min-seconds 180 --observe-cache)
elif [[ "$protocol" == --fs-protocol ]]; then
  bench_options=(--min-seconds 300 --cold-cache)
fi
if "$profile_stages"; then bench_options+=(--profile-stages); fi
verification_progress() {
  # Limit terminal and log volume without changing the verified implementation.
  awk '
    /^(Source q\/Q checked:|Inference samples checked:) [0-9]+\/[0-9]+$/ {
      split($NF, count, "/")
      if (count[1] < count[2] && count[1] - last[$1] < 1024) next
      last[$1] = count[1]
    }
    { print; fflush() }
  '
}
check_gpu() {
  # Detect the observed GPU-contention failure without terminating anyone.
  "${python[@]}" - "$CUDA_VISIBLE_DEVICES" "$wait_for_gpu" <<'PY'
import subprocess, sys, time
from fastlens.benchmarks.dct_models.gpu_occupancy import gpu_occupancy
uuid, wait = sys.argv[1:]
if not uuid.startswith('GPU-') or ',' in uuid:
    raise ValueError('select one GPU by its full UUID')
subprocess.run(['nvidia-smi', '--id='+uuid, '--query-gpu=name', '--format=csv,noheader'], check=True)
was_busy, idle_since, previous = False, None, None
while True:
    occupied = gpu_occupancy(uuid)['blocking_pids']
    if occupied:
        if wait != 'true':
            raise RuntimeError('selected GPU has active compute processes: '+str(occupied))
        was_busy, idle_since = True, None
        if occupied != previous:
            print('Waiting for selected GPU: '+str(occupied), flush=True)
    elif not was_busy:
        break
    elif idle_since is None:
        idle_since = time.monotonic()
        print('GPU became free; waiting for 60 continuous idle seconds.', flush=True)
    elif time.monotonic()-idle_since >= 60:
        print('GPU remained free; continuing the workload.', flush=True)
        break
    previous = occupied
    time.sleep(10)
PY
}
select_backend() {
  case "$1" in
    cpu_dct) backend=cpu_dct; root= ;;
    fastlens) backend=fastlens; root="$PWD/fastlens/data/compressed/imagenet512_val_block_major/manifest.bin" ;;
    ic_shared) backend=ic; root="$prepared/ic_matched_shared" ;;
    ic_batched) backend=ic; root="$prepared/ic_matched_batched" ;;
    fs) backend=fs; root="$prepared/fs_matched_b8" ;;
    *) echo "Unknown system: $1" >&2; exit 1 ;;
  esac
  # Settings change physical execution only. Logical order, model batch,
  # precision and the common host input budget stay fixed across systems.
  resolved="$("${python[@]}" - "$settings" "$1" "$protocol" "$root" <<'PY'
import json, sys
path, name, protocol, root = sys.argv[1:]
opts = dict(pool_size=1024, workers=4, decode_budget_mib=512, reader_budget_mib=2048,
            decoded_cache_mib=0, prefetch_factor=2)
if protocol != 'pilot':
    opts.update(pool_size=256, workers=8, decode_budget_mib=1024, reader_budget_mib=1024)
if path:
    settings = json.load(open(path))
    override = settings.get(name, {})
    unknown = set(override)-set(opts)-{'root'}
    if unknown:
        raise ValueError('unknown physical settings: '+str(unknown))
    root = override.get('root', root)
    opts.update({key: value for key, value in override.items() if key != 'root'})
if name in ('cpu_dct', 'fastlens') and opts['decoded_cache_mib']:
    raise ValueError('decoded NPZ cache applies only to IC/FS')
print(root)
print('--cpu-loader')
print('threadpool' if protocol == 'pilot' else 'dataloader')
for key, value in opts.items():
    print('--'+key.replace('_', '-'))
    print(value)
PY
  )"
  mapfile -t resolved_args <<< "$resolved"
  root="${resolved_args[0]}"
  prepare_options=("${resolved_args[@]:1}" --host-input-budget-mib 32768)
  [[ "$protocol" == pilot ]] || prepare_options+=(--request-order source)
  [[ -z "$request_order" ]] || prepare_options+=(--request-order "$request_order")
  extra=()
  [[ -z "$root" ]] || extra+=(--root "$root")
  if [[ "$backend" == fastlens ]]; then
    extra+=(--module-path "$PWD/build/fastlens/torch" --source-workers 4)
  fi
  # The benchmark already supports changing cache capacity without changing
  # numerical evidence. Record the actual capacity in each measurement row.
  measurement_options=()
  measurement_cache_bytes=
  if [[ -n "${measurement_cache_mib[$1]+set}" ]]; then
    measurement_cache_bytes=$((measurement_cache_mib[$1] * 1024 * 1024))
    measurement_options=(--cache-bytes "$measurement_cache_bytes")
  fi
}
echo "$expected-image matched $run_kind: CPU affinity $cpus; GPU $CUDA_VISIBLE_DEVICES; batch64; protocol $protocol"
for model in "${models[@]}"; do
  directory="$output/$model"
  mkdir -p "$directory"
  for name in "${names[@]}"; do
    select_backend "$name"
    contract="$directory/$name.contract/matched.json"
    if [[ ! -f "$contract" ]]; then
      "${python[@]}" -m fastlens.benchmarks.baselines.dct_baselines prepare \
        --source-manifest "$prepared/contract/source.json" --model "$model" --upstream "$HOME/RGB-no-more" \
        --batch-size 64 --torch-threads 2 --seed "$seed" "${prepare_options[@]}" \
        --atol 1e-5 --rtol 1e-5 --output "$directory/$name.contract" 2>&1 | tee "$directory/$name.prepare.log"
    fi
    "${python[@]}" - "$contract" "${prepare_options[@]}" --seed "$seed" <<'PY'
import json, sys
config = json.load(open(sys.argv[1]))
args = dict(zip(sys.argv[2::2], sys.argv[3::2]))
fields = dict(pool_size='preparation_pool_size', workers='workers', prefetch_factor='prefetch_factor',
              seed='trace_generation_seed', decode_budget_mib='decode_budget_bytes',
              reader_budget_mib='reader_memory_budget_bytes', decoded_cache_mib='decoded_cache_bytes',
              host_input_budget_mib='host_input_budget_bytes')
for option, field in fields.items():
    key = '--'+option.replace('_', '-')
    if key in args:
        expected = int(args[key])*(1024**2 if option.endswith('_mib') else 1)
        assert config[field] == expected, 'settings changed; use a fresh output directory: '+field
assert config['cpu_loader'] == args['--cpu-loader']
assert config['request_order'] == args.get('--request-order', 'shuffle')
PY
    if "$resume" && [[ -f "$directory/$name.verified.json" ]]; then
      "${python[@]}" - "$contract" "$directory/$name.verified.json" "$backend" "$expected" "$prepared/contract/source.json" <<'PY'
import json, sys
from fastlens.benchmarks.baselines.dct_baselines.contracts import artifact
from fastlens.benchmarks.baselines.dct_baselines.gates import numerical_runtime
from fastlens.benchmarks.baselines.dct_baselines.runner import implementation_artifacts, inputs
contract, path, backend, expected, source_path = sys.argv[1:]
config, source, profile, records, trace = inputs(contract)
evidence = json.load(open(path))
assert evidence['status'] == 'passed' and evidence['phase'] == 'pretrained_inference'
assert config['source_manifest'] == artifact(source_path), 'verification belongs to a different prepared source'
assert evidence['resolved_contract'] == artifact(contract)
assert evidence['implementation'] == implementation_artifacts()
assert evidence['device'] == 'cuda:0' and evidence['runtime'] == numerical_runtime('cuda:0')
assert evidence['backend']['kind'] == backend and evidence['logits_match']
assert evidence['occurrence_count'] == len(trace) == int(expected)
refs = evidence['model_implementation'] + evidence.get('native_sources', [])
refs += [evidence['backend'][key] for key in ('binding', 'coefficient_bridge', 'decoder', 'manifest')
         if key in evidence['backend']]
assert all(artifact(ref['path']) == ref for ref in refs), 'evidence dependency changed'
print('Reusing passed verification: '+path)
PY
      continue
    fi
    check_gpu
    log="$directory/$name.verify.$(date +%Y%m%d-%H%M%S).log"
    source_gate=()
    for passed in "$output"/*/"$name.verified.json"; do
      if [[ -f "$passed" ]]; then source_gate=(--source-evidence "$passed"); break; fi
    done
    echo "Verifying $model/$name"
    verifier=(-m fastlens.benchmarks.baselines.dct_baselines.verification --cache-dir "$output/verification_cache")
    if [[ "$backend" == fastlens ]]; then
      verifier=(-m fastlens.benchmarks.baselines.dct_baselines verify-inference)
    else
      for passed in "$directory"/*.verified.json; do
        if [[ -f "$passed" ]]; then verifier+=(--model-evidence "$passed"); break; fi
      done
    fi
    "${python[@]}" "${verifier[@]}" \
      --config "$contract" --backend "$backend" "${extra[@]}" \
      "${source_gate[@]}" \
      --device cuda:0 --output "$directory/$name.verified.json" 2>&1 | verification_progress | tee "$log"
  done
  if "$verify_only"; then continue; fi
  for repeat in 0 1 2; do
    for ((position=0; position<${#names[@]}; position++)); do
      name="${names[$(( (position + repeat) % ${#names[@]} ))]}"
      select_backend "$name"
      prefix="$directory/$name.repeat$repeat"
      if "$resume_measurements" && { [[ -e "$prefix.jsonl" ]] || [[ -e "$prefix.process.json" ]]; }; then
        "${python[@]}" - "$prefix" "$directory/$name.contract/matched.json" "$directory/$name.verified.json" \
          "$protocol" "$CUDA_VISIBLE_DEVICES" "$cpus" "$run_kind" "$profile_stages" "$backend" "$root" "$measurement_cache_bytes" <<'PY'
import json, sys
from pathlib import Path
from fastlens.benchmarks.baselines.dct_baselines.contracts import artifact, load_artifact
prefix, contract, evidence, protocol, gpu, cpus, kind, profile, backend, root, cache_override = sys.argv[1:]
process = json.loads(Path(prefix+'.process.json').read_text())
assert process['valid'] and process['returncode'] == 0, 'invalid prior measurement: '+prefix
assert process['gpu_uuid'] == gpu and process['rss_limit_mib'] == 32768
rows = [json.loads(line) for line in Path(prefix+'.jsonl').read_text().splitlines()]
assert len(rows) == 1, 'expected one measurement per repeat'
row = rows[0]
assert row['status'] == 'completed' and row['measurement_mode'] == 'actual'
assert row['artifacts']['resolved_contract'] == artifact(contract)
assert row['artifacts']['correctness_evidence'] == artifact(evidence)
if backend in ('ic', 'fs'):
    assert row['artifacts']['backend']['store']['manifest'] == artifact(Path(root)/'store.json')
elif backend == 'fastlens':
    assert row['artifacts']['backend']['manifest'] == artifact(root)
assert row['run_kind'] == kind
assert row['environment']['hardware']['gpu_uuid'] == gpu.removeprefix('GPU-')
assert row['environment']['hardware']['cpu_affinity'] == cpus
seconds, warmup, policy = {'pilot': (0., 0., 'continuous_post_warmup'),
    '--ic-protocol': (180., 60., 'continuous_post_warmup'),
    '--fs-protocol': (300., 0., 'cold_each_pass')}[protocol]
params = row['resolved_parameters']
cache_bytes = int(cache_override) if cache_override else json.loads(Path(contract).read_text()).get('decoded_cache_bytes', 0)
assert params['decoded_cache_bytes_per_reader'] == cache_bytes, 'measurement cache capacity changed: '+prefix
assert params['measurement_min_seconds'] == seconds and params['warmup_seconds'] == warmup
assert params['cache_policy'] == policy and params['profile_stages'] == (profile == 'true')
assert row['metrics']['elapsed_wall_s'] >= seconds
count = json.loads(Path(evidence).read_text())['occurrence_count']
assert row['counts']['dataset_passes'] >= 1
assert row['counts']['sample_occurrences'] == count*row['counts']['dataset_passes']
if kind == 'inference':
    assert row['quality']['sample_count'] == count
if protocol != 'pilot':
    detail = load_artifact(row['artifacts']['cache_evidence'])
    assert detail['warmup']['elapsed_wall_s'] >= warmup
    assert row['cache_mode'] == ('mixed_measured' if protocol == '--ic-protocol' else 'cold_verified')
print('Reusing valid measurement: '+prefix)
PY
        continue
      fi
      check_gpu
      echo "Measuring $model/$name repeat $((repeat + 1))/3"
      "${python[@]}" -m fastlens.benchmarks.dct_models.measure_process \
        --output "$directory/$name.repeat$repeat.process.json" \
        --gpu-uuid "$CUDA_VISIBLE_DEVICES" --rss-limit-mib 32768 -- \
        "$PYTHON" -B -m fastlens.benchmarks.baselines.dct_baselines bench \
        --config "$directory/$name.contract/matched.json" --backend "$backend" "${extra[@]}" \
        --evidence "$directory/$name.verified.json" --device cuda:0 --run-kind "$run_kind" --repeats 1 \
        "${bench_options[@]}" "${measurement_options[@]}" --output "$directory/$name.repeat$repeat.jsonl" \
        2>&1 | tee "$directory/$name.repeat$repeat.log"
      check_gpu
    done
  done
done
if "$verify_only"; then
  echo "Passed: $expected-image numerical verification. Results: $output"
else
  "${python[@]}" -m fastlens.benchmarks.baselines.dct_baselines report "$output"/*/*.jsonl --output "$output/summary.json"
  echo "Passed: $expected-image $run_kind ($protocol), ${#models[@]} models x ${#names[@]} systems x 3 repeats. Results: $output/summary.json"
fi
