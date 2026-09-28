#!/usr/bin/env bash
# Keep system Clang/GCC as host compilers; select Conda CUDA/Python explicitly.
set -eo pipefail
export GALP_CUDA_ENV="$HOME/miniconda3/envs/fastlanes-cuda"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$GALP_CUDA_ENV/bin"
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export CUDACXX="$GALP_CUDA_ENV/bin/nvcc"
export CUDAHOSTCXX=/usr/bin/g++
# Also handle a runner/caller started from an activated Conda environment.
# Its flags and sysroot must not be applied to the system host compilers.
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CONDA_BUILD_SYSROOT
exec /usr/bin/bash --noprofile --norc -eo pipefail "$@"
