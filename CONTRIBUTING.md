# Contributing to FastLanes

For CPU parallel preparation/Wizard race checks on a platform supporting
ThreadSanitizer, use a separate build (TSan cannot be combined with ASan):

```bash
cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=Release -DFLS_BUILD_TESTING=ON \
  -DFLS_ENABLE_TSAN=ON -DFLS_ENABLE_GALP_TESTING_AND_BENCHMARKING=OFF
cmake --build build-tsan --target unit_test --parallel
TMPDIR="$HOME/tmp" ctest --test-dir build-tsan --output-on-failure \
  -R 'ParallelPreparation|ParallelEncoder.AutomaticWizard'
```

This instruments both the CPU library and tests. A TSan startup/runtime failure
is an environment limitation, not a passing race check.

The same `FLS_ENABLE_TSAN=ON` build can check GALP host ownership with the
CUDA fault-injection executable (CUDA toolkit headers/compiler must be discoverable):

```bash
cmake -S . -B build-host-tsan -DCMAKE_BUILD_TYPE=Release -DFLS_BUILD_TESTING=ON \
  -DFLS_ENABLE_TSAN=ON -DFLS_ENABLE_GALP_TESTING_AND_BENCHMARKING=ON \
  -DGALP_BUILD_TESTS=ON -DGALP_BUILD_TORCH=OFF -DGALP_BUILD_BENCHMARKS=OFF \
  -DGALP_BUILD_TOOLS=OFF -DGALP_WITH_JPEG_DCT=OFF
cmake --build build-host-tsan --target galp_cuda_transfer_failure_tests --parallel
TSAN_OPTIONS=halt_on_error=1 build-host-tsan/galp/tests/galp_cuda_transfer_failure_tests
```

This executable compiles the real DevicePool, TransferTracker and DeviceArena,
but supplies thread-safe test CUDA symbols and does **not** link cudart. It checks
host bookkeeping, ownership and injected interleavings, not driver or GPU races.
Do not build/run the real CUDA targets under this global TSan configuration.
Run their ordinary CUDA tests, multi-GPU ownership test and native Torch smoke
in the normal build separately. A sanitizer-incompatible driver is not a reason
to skip the host fault-injection race checks.

We welcome contributions of all kinds:

- bug fixes  
- performance improvements  
- documentation  
- new features  

This document describes the contribution process and testing requirements.

---

## Getting Started

1. **Fork the repository** on GitHub and create a feature branch:
   ```bash
   git checkout -b my-feature
    ```

2. **Make your changes**, following our coding guidelines (see below).
3. **Run the test suite** locally before submitting your PR.
4. **Open a Pull Request (PR)** against the `dev` branch.

---

## Testing

FastLanes includes several layers of testing. All PRs must pass the full test suite in CI.

### Fuzzy Tests

FastLanes uses **fuzzy tests** with an incremental bump seed to check edge cases.
To keep results reproducible, the fuzz seed is stored in the repository.

* Each PR must **increment the seed** so tests run with a new value.
* Bump the seed by running:

  ```bash
  make bump
  ```

---

## Submitting a PR

* Open your PR against the **`dev` branch**.
* Make sure your branch is **rebased on top of `dev`** before submission.
* Verify that **CI passes** (unit tests + fuzzy tests + Python tests if applicable).
* Keep PRs focused — small, logical changes are easier to review and merge.

---

## Communication

* For questions, join the FastLanes Discord.
* Use GitHub Issues for bug reports and feature requests.

---

Thanks for helping improve FastLanes — we appreciate your contributions!
