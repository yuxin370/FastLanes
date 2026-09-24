#!/usr/bin/env bash
# Every GALP run step uses the same CUDA toolchain and Python environment.
set -eo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fastlanes-cuda
exec bash --noprofile --norc -eo pipefail "$@"
