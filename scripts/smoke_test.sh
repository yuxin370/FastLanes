#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build="${FASTLENS_BUILD_DIR:-$root/build}"
ctest --test-dir "$build" --output-on-failure --no-tests=error \
  -R '^(FastLensPublicApiSmoke|FastLensPublicApiBoundaries|FastLensPackageConsumerSmoke|FastLensDirectDctTorchImport)$' \
  "$@"
