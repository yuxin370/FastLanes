#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s nullglob

# Build reproducible ImageNet-train v3 canaries without expanding the complete
# 138 GiB population.  The source directory contains 1000 per-class tar files;
# this script extracts a balanced 1 or 10 images per class, then writes and
# validates an image-major-vector-rowgroups manifest.

export REPO="${REPO:-/home/tangyuxin/gfastlanes/FastLanes}"
export PY="${PY:-/home/tangyuxin/miniconda3/envs/fastlanes-cuda/bin/python}"

export TRAIN_CLASS_TARS="${TRAIN_CLASS_TARS:-$REPO/galp/data/imagedataset/ILSVRC2012_img_train}"
export CANARY_BASE="${CANARY_BASE:-$REPO/galp/data/system_rgbnomore/e2e_v3/galp-v3-train-canary}"
export INDEX_CSV="${INDEX_CSV:-/home/tangyuxin/RGB-no-more/assets/indexbase_train.csv}"

export TOOL="${TOOL:-$REPO/build/galp/tools/jpeg_dct/galp_jpeg_dct_tool}"
export TORCH_BINDING="${TORCH_BINDING:-$REPO/build/galp/torch}"
export PREPARE_DATASET="${PREPARE_DATASET:-$REPO/galp/benchmarks/system_rgbnomore/dataset/prepare_dataset.py}"
export GENERATE_MANIFESTS="${GENERATE_MANIFESTS:-$REPO/galp/benchmarks/system_rgbnomore/training/generate_imagenet_manifests.py}"

export CANARY_TAG="${CANARY_TAG:-v3a}"
export THREADS="${THREADS:-12}"
export SHARD_WORKERS="${SHARD_WORKERS:-4}"
export SHARD_IMAGES="${SHARD_IMAGES:-8192}"
export ROWGROUPS_PER_SHARD="${ROWGROUPS_PER_SHARD:-8192}"

usage() {
    cat <<'EOF'
Usage:
  run_v3_train_canary.sh [both|1k|10k]

The default is "both".  Outputs are written below:

  $CANARY_BASE/train-1k-$CANARY_TAG
  $CANARY_BASE/train-10k-$CANARY_TAG

Useful environment overrides:

  CANARY_TAG=v3b          Select a fresh output name.
  THREADS=12             JPEG conversion threads.
  SHARD_WORKERS=4        Concurrent shard workers.
  TRAIN_CLASS_TARS=...   Directory containing 1000 n*.tar files.
  CANARY_BASE=...        Output root.

The script never overwrites completed/partial DCT output.  It can resume the
verified pre-compression state produced by the original v3a layout bug.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

validate_environment() {
    test -x "$PY" || die "Python is not executable: $PY"
    test -x "$TOOL" || die "GALP JPEG-DCT tool is not executable: $TOOL"
    test -d "$TRAIN_CLASS_TARS" || die "class-tar directory is missing: $TRAIN_CLASS_TARS"
    test -d "$TORCH_BINDING" || die "Torch binding directory is missing: $TORCH_BINDING"
    test -f "$INDEX_CSV" || die "ImageNet train index is missing: $INDEX_CSV"
    test -f "$PREPARE_DATASET" || die "dataset preparation script is missing: $PREPARE_DATASET"
    test -f "$GENERATE_MANIFESTS" || die "manifest generator is missing: $GENERATE_MANIFESTS"

    [[ "$CANARY_TAG" =~ ^[A-Za-z0-9._-]+$ ]] || die "unsafe CANARY_TAG: $CANARY_TAG"
    [[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || die "THREADS must be positive"
    [[ "$SHARD_WORKERS" =~ ^[1-9][0-9]*$ ]] || die "SHARD_WORKERS must be positive"
    [[ "$SHARD_IMAGES" =~ ^[1-9][0-9]*$ ]] || die "SHARD_IMAGES must be positive"
    [[ "$ROWGROUPS_PER_SHARD" =~ ^[1-9][0-9]*$ ]] || die "ROWGROUPS_PER_SHARD must be positive"

    local class_tars=( "$TRAIN_CLASS_TARS"/*.tar )
    test "${#class_tars[@]}" -eq 1000 ||
        die "expected 1000 class tar files, found ${#class_tars[@]}"

    mkdir -p "$CANARY_BASE"
}

extract_balanced_train_subset() {
    local output_dir="$1"
    local image_count="$2"
    local selection_csv="$3"

    test ! -e "$output_dir" || die "refusing to reuse extraction directory: $output_dir"

    "$PY" - \
        "$TRAIN_CLASS_TARS" \
        "$INDEX_CSV" \
        "$output_dir" \
        "$image_count" \
        "$selection_csv" <<'PY'
import csv
import os
import shutil
import sys
import tarfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

class_tar_root = Path(sys.argv[1])
index_csv = Path(sys.argv[2])
output_root = Path(sys.argv[3])
image_count = int(sys.argv[4])
selection_csv = Path(sys.argv[5])

if image_count <= 0:
    raise ValueError("image_count must be positive")
if output_root.exists():
    raise FileExistsError(output_root)

# ImageNet-1K: distribute the requested population across labels 0..999.
class_count = 1000
base_count, remainder = divmod(image_count, class_count)
quota = {
    label: base_count + (1 if label < remainder else 0)
    for label in range(class_count)
}
selected = {label: [] for label in range(class_count)}
wnid_for_label = {}
remaining = image_count

with index_csv.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    if not {"Filepath", "Label"}.issubset(reader.fieldnames or ()):
        raise ValueError(f"unexpected index columns: {reader.fieldnames}")

    for row in reader:
        rel = PurePosixPath(row["Filepath"].strip())
        parts = rel.parts
        if len(parts) != 3 or parts[0] != "train":
            continue

        label = int(row["Label"])
        if label not in quota:
            raise ValueError(f"label outside ImageNet-1K range: {label}")

        _, wnid, filename = parts
        previous_wnid = wnid_for_label.setdefault(label, wnid)
        if previous_wnid != wnid:
            raise ValueError(
                f"label {label} maps to both {previous_wnid} and {wnid}"
            )

        if len(selected[label]) >= quota[label]:
            continue

        selected[label].append(
            {
                "rel": str(rel),
                "label": label,
                "wnid": wnid,
                "filename": filename,
            }
        )
        remaining -= 1
        if remaining == 0:
            break

missing = {
    label: quota[label] - len(selected[label])
    for label in range(class_count)
    if len(selected[label]) != quota[label]
}
if missing:
    raise RuntimeError(f"index does not provide requested balanced subset: {missing}")

items = [item for label in range(class_count) for item in selected[label]]
if len(items) != image_count:
    raise AssertionError((len(items), image_count))

by_wnid = defaultdict(list)
for item in items:
    by_wnid[item["wnid"]].append(item)

output_root.mkdir(parents=True)
extracted_images = 0

for ordinal, wnid in enumerate(sorted(by_wnid), start=1):
    archive = class_tar_root / f"{wnid}.tar"
    if not archive.is_file():
        raise FileNotFoundError(archive)

    class_output = output_root / wnid
    class_output.mkdir()

    with tarfile.open(archive, mode="r:") as tf:
        members = {
            PurePosixPath(member.name).name: member
            for member in tf.getmembers()
            if member.isfile()
        }

        for item in sorted(by_wnid[wnid], key=lambda value: value["filename"]):
            filename = item["filename"]
            member = members.get(filename)
            if member is None:
                raise FileNotFoundError(f"{archive}: missing {filename}")

            source = tf.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read {archive}:{filename}")

            destination = class_output / filename
            with source, destination.open("xb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            os.utime(destination, (member.mtime, member.mtime))
            extracted_images += 1

    if ordinal % 100 == 0 or ordinal == len(by_wnid):
        print(
            f"extracted classes={ordinal}/{len(by_wnid)} "
            f"images={extracted_images}/{image_count}",
            flush=True,
        )

selection_csv.parent.mkdir(parents=True, exist_ok=True)
with selection_csv.open("x", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["Filepath", "Label"])
    writer.writeheader()
    for item in sorted(items, key=lambda value: value["rel"]):
        writer.writerow({"Filepath": item["rel"], "Label": item["label"]})

print(
    f"balanced extraction complete: images={len(items)}, "
    f"classes={len(by_wnid)}, output={output_root}"
)
PY

    local actual_count
    actual_count="$({
        find "$output_dir" -type f \
            \( -iname '*.JPEG' -o -iname '*.jpg' -o -iname '*.jpeg' \)
    } | wc -l)"
    test "$actual_count" -eq "$image_count" ||
        die "expected $image_count extracted JPEGs, found $actual_count"
}

snapshot_binaries() {
    local root="$1"
    local frozen_dir="$2"
    local frozen_tool="$3"
    local frozen_binding_dir="$4"

    mkdir -p "$frozen_binding_dir"

    cp --reflink=auto --preserve=timestamps "$TOOL" "$frozen_tool"
    cmp -s "$TOOL" "$frozen_tool" || die "tool changed while it was being snapshotted"
    sha256sum "$TOOL" > "$root/source_tool.sha256"
    sha256sum "$frozen_tool" > "$root/tool.sha256"
    ldd "$frozen_tool" > "$root/tool.ldd.txt"

    local binding_sources=( "$TORCH_BINDING"/_galp_direct_dct*.so )
    test "${#binding_sources[@]}" -eq 1 ||
        die "expected exactly one DirectDct binding, found ${#binding_sources[@]}"

    local binding_name
    binding_name="$(basename "${binding_sources[0]}")"
    cp --reflink=auto --preserve=timestamps \
        "${binding_sources[0]}" \
        "$frozen_binding_dir/$binding_name"
    cmp -s "${binding_sources[0]}" "$frozen_binding_dir/$binding_name" ||
        die "Torch binding changed while it was being snapshotted"

    sha256sum "${binding_sources[0]}" > "$root/source_torch_binding.sha256"
    sha256sum "$frozen_binding_dir/$binding_name" > "$root/torch_binding.sha256"
}

manifest_v3_preflight() {
    local manifest="$1"

    "$PY" -c '
import pathlib
import struct
import sys

manifest = pathlib.Path(sys.argv[1])
data = manifest.read_bytes()

assert data[:8] == b"GJDCTSH1", "invalid manifest magic"
version = struct.unpack_from("<I", data, 8)[0]
assert version == 3, f"expected manifest v3, got v{version}"
assert b"image-major-vector-rowgroups" in data, "v3 layout tag missing"

# The base v3 manifest does not persist the CLI spelling of spatial order.
# When the optional Compact-v3 trailer is present, it stores the canonical
# descriptor spelling instead.
if b"GJDCCV31" in data:
    assert b"tiled-z32" in data, "Compact-v3 spatial-order tag missing"

print(
    f"manifest preflight OK: version={version}, "
    f"bytes={len(data)}, path={manifest}"
)
' "$manifest"
}

resume_precompression_canary() {
    local root="$1"
    local jpeg_root="$2"
    local jpeg_view="$3"
    local image_count="$4"
    local dct_out="$5"
    local benchmark_manifests="$6"
    local frozen_tool="$7"

    local migration_tmp="$root/.jpeg-legacy-layout"

    # Resume is intentionally limited to the exact failure boundary before any
    # DCT shard was written.  Partial compression output requires a fresh tag.
    test -d "$root" || return 1
    test ! -e "$dct_out" || return 1
    test ! -e "$benchmark_manifests" || return 1
    test ! -e "$root/label_map.json" || return 1
    test ! -e "$root/selected_index.csv" || return 1
    test ! -e "$root/selection.json" || return 1
    test ! -e "$root/reader_verify.json" || return 1
    test -x "$frozen_tool" || return 1
    test -f "$root/tool.sha256" || return 1
    test -f "$root/torch_binding.sha256" || return 1
    test -f "$root/extraction_selection.csv" || return 1
    grep -qx "image_count=$image_count" "$root/config.env" || return 1
    test "$(wc -l < "$root/extraction_selection.csv")" -eq "$((image_count + 1))" || return 1

    # v3a's first attempt wrote jpeg/<WNID>/... .  Label preflight needs the
    # split marker, so migrate it atomically to jpeg/train/<WNID>/... .
    if test -d "$migration_tmp"; then
        mkdir -p "$jpeg_root"
        test ! -e "$jpeg_view" || return 1
        mv "$migration_tmp" "$jpeg_view"
    elif test -d "$jpeg_root" && ! test -e "$jpeg_view"; then
        local legacy_count
        local legacy_classes
        legacy_count="$({
            find "$jpeg_root" -type f \
                \( -iname '*.JPEG' -o -iname '*.jpg' -o -iname '*.jpeg' \)
        } | wc -l)"
        legacy_classes="$(find "$jpeg_root" -mindepth 1 -maxdepth 1 -type d | wc -l)"
        test "$legacy_count" -eq "$image_count" || return 1
        test "$legacy_classes" -eq 1000 || return 1

        mv "$jpeg_root" "$migration_tmp"
        mkdir "$jpeg_root"
        mv "$migration_tmp" "$jpeg_view"
        echo "Migrated legacy canary layout to: $jpeg_view"
    fi

    test -d "$jpeg_view" || return 1
    local actual_count
    actual_count="$({
        find "$jpeg_view" -type f \
            \( -iname '*.JPEG' -o -iname '*.jpg' -o -iname '*.jpeg' \)
    } | wc -l)"
    test "$actual_count" -eq "$image_count" || return 1

    sha256sum --check "$root/tool.sha256"
    sha256sum --check "$root/torch_binding.sha256"
    echo "Resuming verified pre-compression canary: $root"
}

make_v3_train_canary() {
    local name="$1"
    local image_count="$2"
    local val_count="$3"

    local root="$CANARY_BASE/$name"
    local jpeg_root="$root/jpeg"
    local jpeg_view="$jpeg_root/train"
    local dct_out="$root/dct"
    local benchmark_manifests="$root/training_manifests"
    local extraction_selection="$root/extraction_selection.csv"

    local frozen_dir="$root/frozen"
    local frozen_tool="$frozen_dir/galp_jpeg_dct_tool"
    local frozen_binding_dir="$frozen_dir/torch"

    if test -e "$root"; then
        resume_precompression_canary \
            "$root" \
            "$jpeg_root" \
            "$jpeg_view" \
            "$image_count" \
            "$dct_out" \
            "$benchmark_manifests" \
            "$frozen_tool" ||
            die "existing canary is not safely resumable; use a fresh CANARY_TAG: $root"
    else
        mkdir -p "$root" "$frozen_dir"

        git -C "$REPO" rev-parse HEAD > "$root/git_commit.txt"
        git -C "$REPO" status --porcelain=v1 > "$root/git_status.txt"
        git -C "$REPO" diff --binary | sha256sum > "$root/git_tracked_diff.sha256"

        snapshot_binaries "$root" "$frozen_dir" "$frozen_tool" "$frozen_binding_dir"

        {
            printf 'name=%s\n' "$name"
            printf 'image_count=%s\n' "$image_count"
            printf 'val_count=%s\n' "$val_count"
            printf 'selection=balanced_by_imagenet_label\n'
            printf 'physical_layout=image-major-vector-rowgroups\n'
            printf 'spatial_order=tiled-z-32\n'
            printf 'shard_images=%s\n' "$SHARD_IMAGES"
            printf 'rowgroups_per_shard=%s\n' "$ROWGROUPS_PER_SHARD"
            printf 'threads=%s\n' "$THREADS"
            printf 'shard_workers=%s\n' "$SHARD_WORKERS"
        } > "$root/config.env"

        extract_balanced_train_subset "$jpeg_view" "$image_count" "$extraction_selection"
    fi

    # jpeg_view already contains exactly the selected population, so no second
    # hard-linked ImageFolder view is necessary.
    "$PY" "$PREPARE_DATASET" \
        --split train \
        --input-dir "$jpeg_view" \
        --out-dir "$dct_out" \
        --limit "$image_count" \
        --expected-image-count "$image_count" \
        --index-file "$INDEX_CSV" \
        --selected-index-csv "$root/selected_index.csv" \
        --label-map-json "$root/label_map.json" \
        --write-label-map-only \
        --output-json "$root/selection.json"

    time "$frozen_tool" \
        --shard \
        --out-dir "$dct_out" \
        --policy ragged \
        --preset random-access \
        --metadata-profile reconstruct \
        --physical-layout image-major-vector-rowgroups \
        --spatial-order tiled-z-32 \
        --shard-images "$SHARD_IMAGES" \
        --rowgroups-per-shard "$ROWGROUPS_PER_SHARD" \
        --threads "$THREADS" \
        --shard-workers "$SHARD_WORKERS" \
        "$jpeg_view" \
        2>&1 | tee "$root/compress.log"

    sha256sum --check "$root/tool.sha256"
    sha256sum --check "$root/torch_binding.sha256"
    manifest_v3_preflight "$dct_out/manifest.bin"

    time "$frozen_tool" \
        --verify-manifest "$dct_out/manifest.bin" \
        "$jpeg_view" \
        2>&1 | tee "$root/verify_manifest.log"

    "$PY" "$PREPARE_DATASET" \
        --verify-only \
        --split train \
        --input-dir "$jpeg_view" \
        --out-dir "$dct_out" \
        --manifest "$dct_out/manifest.bin" \
        --expected-image-count "$image_count" \
        --validate-sample-images 16 \
        --torch-binding-dir "$frozen_binding_dir" \
        --output-json "$root/reader_verify.json"

    "$PY" "$GENERATE_MANIFESTS" \
        --jpeg-root "$jpeg_view" \
        --index-csv "$root/selected_index.csv" \
        --galp-manifest "$dct_out/manifest.bin" \
        --output-dir "$benchmark_manifests" \
        --source-split train \
        --train-count 0 \
        --val-count "$val_count" \
        --seed 11997733 \
        --probe-dimensions \
        --progress-interval 1000

    sha256sum \
        "$dct_out/manifest.bin" \
        "$extraction_selection" \
        "$root/selected_index.csv" \
        "$root/label_map.json" \
        "$root/selection.json" \
        "$root/reader_verify.json" \
        "$benchmark_manifests/train.json" \
        "$benchmark_manifests/val.json" \
        > "$root/artifacts.sha256"

    echo
    echo "Created: $root"
    du -sh "$root"
}

main() {
    local target="${1:-both}"
    case "$target" in
        -h|--help)
            usage
            return 0
            ;;
        both|1k|10k)
            ;;
        *)
            usage >&2
            die "unknown target: $target"
            ;;
    esac

    validate_environment

    echo "REPO=$REPO"
    echo "TRAIN_CLASS_TARS=$TRAIN_CLASS_TARS"
    echo "CANARY_BASE=$CANARY_BASE"
    echo "CANARY_TAG=$CANARY_TAG"
    echo "target=$target"

    case "$target" in
        both)
            make_v3_train_canary "train-1k-$CANARY_TAG" 1000 100
            make_v3_train_canary "train-10k-$CANARY_TAG" 10000 1000
            ;;
        1k)
            make_v3_train_canary "train-1k-$CANARY_TAG" 1000 100
            ;;
        10k)
            make_v3_train_canary "train-10k-$CANARY_TAG" 10000 1000
            ;;
    esac
}

main "$@"
