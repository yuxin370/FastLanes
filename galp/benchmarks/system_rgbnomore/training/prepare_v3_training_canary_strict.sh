#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s nullglob

# Build an exact 900/100 or 9000/1000 ImageNet-train canary.  The selector
# rejects unsupported JPEG sampling before compression, and every binary used
# to produce or read the dataset is frozen under the canary root.

REPO="${REPO:-/home/tangyuxin/gfastlanes/FastLanes}"
PY="${PY:-/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python}"
CLASS_TARS="${CLASS_TARS:-$REPO/galp/data/imagedataset/ILSVRC2012_img_train}"
INDEX_CSV="${INDEX_CSV:-/home/tangyuxin/RGB-no-more/assets/indexbase_train.csv}"
CANARY_BASE="${CANARY_BASE:-$REPO/galp/data/system_rgbnomore/e2e_v3/galp-v3-training-canary-strict}"
CANARY_TAG="${CANARY_TAG:-v3a}"
TOOL="${TOOL:-$REPO/build/galp/tools/jpeg_dct/galp_jpeg_dct_tool}"
TORCH_BINDING_DIR="${TORCH_BINDING_DIR:-$REPO/build/galp/torch}"
SELECTOR="$REPO/galp/benchmarks/system_rgbnomore/training/select_imagenet_canary.py"
PREFLIGHT="$REPO/galp/benchmarks/system_rgbnomore/training/manifest_preflight.py"
MANIFEST_GENERATOR="$REPO/galp/benchmarks/system_rgbnomore/training/generate_imagenet_manifests.py"
THREADS="${THREADS:-12}"
SHARD_WORKERS="${SHARD_WORKERS:-4}"
ROWGROUPS_PER_SHARD="${ROWGROUPS_PER_SHARD:-8192}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: prepare_v3_training_canary_strict.sh 1k|10k

Use a fresh CANARY_TAG for every attempt.  The script never overwrites or
resumes an existing root.  It emits a complete, frozen data-preparation record;
it does not start a GPU training benchmark.
EOF
}

validate_environment() {
    test -x "$PY" || die "Python is not executable: $PY"
    test -x "$TOOL" || die "JPEG-DCT tool is not executable: $TOOL"
    test -d "$TORCH_BINDING_DIR" || die "Torch binding directory is missing: $TORCH_BINDING_DIR"
    test -d "$CLASS_TARS" || die "class-tar directory is missing: $CLASS_TARS"
    test -f "$INDEX_CSV" || die "ImageNet index is missing: $INDEX_CSV"
    test -f "$SELECTOR" || die "canary selector is missing: $SELECTOR"
    test -f "$PREFLIGHT" || die "manifest preflight is missing: $PREFLIGHT"
    test -f "$MANIFEST_GENERATOR" || die "manifest generator is missing: $MANIFEST_GENERATOR"
    [[ "$CANARY_TAG" =~ ^[A-Za-z0-9._-]+$ ]] || die "unsafe CANARY_TAG: $CANARY_TAG"
    [[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || die "THREADS must be positive"
    [[ "$SHARD_WORKERS" =~ ^[1-9][0-9]*$ ]] || die "SHARD_WORKERS must be positive"
    [[ "$ROWGROUPS_PER_SHARD" =~ ^[1-9][0-9]*$ ]] || die "ROWGROUPS_PER_SHARD must be positive"
    local archives=( "$CLASS_TARS"/*.tar )
    test "${#archives[@]}" -eq 1000 || die "expected 1000 class tar files, found ${#archives[@]}"
    local bindings=( "$TORCH_BINDING_DIR"/_galp_direct_dct*.so )
    test "${#bindings[@]}" -eq 1 || die "expected one DirectDct Torch binding, found ${#bindings[@]}"
}

main() {
    test "$#" -eq 1 || { usage >&2; exit 2; }
    local scale="$1"
    local image_count
    local train_count
    local val_count
    case "$scale" in
        -h|--help)
            usage
            return 0
            ;;
        1k)
            image_count=1000
            train_count=900
            val_count=100
            ;;
        10k)
            image_count=10000
            train_count=9000
            val_count=1000
            ;;
        *)
            usage >&2
            die "scale must be 1k or 10k"
            ;;
    esac
    validate_environment

    local root="$CANARY_BASE/train-$scale-$CANARY_TAG"
    local jpeg_root="$root/jpeg/train"
    local dct_root="$root/dct"
    local manifests="$root/training_manifests"
    local frozen="$root/frozen"
    local frozen_tool="$frozen/galp_jpeg_dct_tool"
    local frozen_binding_dir="$frozen/torch"
    test ! -e "$root" || die "refusing to reuse canary root: $root"
    mkdir -p "$root" "$frozen_binding_dir"
    trap 'printf "failed_at=%s\n" "$(date --iso-8601=seconds)" > "$root/FAILED"' ERR

    git -C "$REPO" rev-parse HEAD > "$root/git_commit.txt"
    git -C "$REPO" status --porcelain=v1 > "$root/git_status.txt"
    git -C "$REPO" diff --binary | sha256sum > "$root/git_tracked_diff.sha256"

    cp --reflink=auto --preserve=timestamps "$TOOL" "$frozen_tool"
    local bindings=( "$TORCH_BINDING_DIR"/_galp_direct_dct*.so )
    cp --reflink=auto --preserve=timestamps "${bindings[0]}" "$frozen_binding_dir/"
    cmp -s "$TOOL" "$frozen_tool" || die "tool changed while being frozen"
    cmp -s "${bindings[0]}" "$frozen_binding_dir/$(basename "${bindings[0]}")" ||
        die "Torch binding changed while being frozen"
    sha256sum "$frozen_tool" "$frozen_binding_dir"/_galp_direct_dct*.so > "$root/frozen_binaries.sha256"
    ldd "$frozen_tool" > "$root/tool.ldd.txt"

    "$PY" "$SELECTOR" \
        --class-tar-root "$CLASS_TARS" \
        --index-csv "$INDEX_CSV" \
        --output-dir "$jpeg_root" \
        --selected-index-csv "$root/selected_index.csv" \
        --output-json "$root/selection.json" \
        --image-count "$image_count"

    # Keep layout and order explicit; the strict preflight below independently
    # proves that CLI normalization preserved both properties.
    time "$frozen_tool" \
        --shard \
        --out-dir "$dct_root" \
        --policy ragged \
        --preset random-access \
        --metadata-profile reconstruct \
        --physical-layout image-major-vector-rowgroups \
        --spatial-order tiled-z-32 \
        --shard-images 8192 \
        --rowgroups-per-shard "$ROWGROUPS_PER_SHARD" \
        --threads "$THREADS" \
        --shard-workers "$SHARD_WORKERS" \
        "$jpeg_root" \
        2>&1 | tee "$root/compress.log"

    PYTHONPATH="$REPO/galp/benchmarks/system_rgbnomore" "$PY" "$PREFLIGHT" \
        "$dct_root/manifest.bin" \
        --expected-manifest-version 3 \
        --expected-physical-layout image-major-vector-rowgroups \
        --expected-spatial-order tiled-z32 \
        --expected-image-count "$image_count" \
        > "$root/manifest_preflight.json"

    time "$frozen_tool" \
        --verify-manifest "$dct_root/manifest.bin" \
        "$jpeg_root" \
        2>&1 | tee "$root/verify_manifest.log"
    grep -qx 'exact: true' "$root/verify_manifest.log" || die "coefficient-exact verification failed"

    "$PY" "$MANIFEST_GENERATOR" \
        --jpeg-root "$jpeg_root" \
        --index-csv "$root/selected_index.csv" \
        --galp-manifest "$dct_root/manifest.bin" \
        --output-dir "$manifests" \
        --source-split train \
        --train-count "$train_count" \
        --val-count "$val_count" \
        --seed 11997733 \
        --probe-dimensions \
        --progress-interval 1000

    "$PY" -c '
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
expected = tuple(map(int, sys.argv[2:]))
actual = tuple(
    len(json.loads((root / name).read_text(encoding="utf-8"))["samples"])
    for name in ("train.json", "val.json")
)
if actual != expected:
    raise SystemExit(f"split cardinality mismatch: expected={expected}, actual={actual}")
print(f"split cardinality OK: train={actual[0]} val={actual[1]}")
' "$manifests" "$train_count" "$val_count"

    sha256sum \
        "$root/selection.json" \
        "$root/selected_index.csv" \
        "$root/manifest_preflight.json" \
        "$dct_root/manifest.bin" \
        "$manifests/train.json" \
        "$manifests/val.json" \
        > "$root/artifacts.sha256"
    trap - ERR
    echo "Created strict v3 training canary: $root"
}

main "$@"
