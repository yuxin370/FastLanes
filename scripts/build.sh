#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build="${FASTLENS_BUILD_DIR:-$root/build}"
cmake -S "$root" -B "$build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFASTLENS_BUILD_TESTS=ON \
  -DFASTLENS_BUILD_BENCHMARKS=ON \
  -DFASTLENS_BUILD_TORCH=ON \
  -DPython3_EXECUTABLE="$(command -v python3)" \
  "$@"
cmake --build "$build" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-$(getconf _NPROCESSORS_ONLN)}"
