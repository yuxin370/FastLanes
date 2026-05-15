#!/usr/bin/env python3
"""Check lightweight GALP include-boundary rules.

This script intentionally checks include direction only. It does not move files,
inspect transitive dependencies, or classify algorithms. Production source
membership is taken from the galp_core source list rather than inferred only from
paths, because some current src/ files are developer/tool support debt.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]')

SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".cu",
    ".cuh",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
}

PRIVATE_IMPLEMENTATION_PREFIXES = (
    "alp/",
    "compression/",
    "decompression/",
    "engine/",
    "generator/",
    "memory/",
)

BENCHMARK_GENERATED_PREFIXES = (
    "generated-bindings/",
    "galp_bench/generated/",
    "benchmark/generated-bindings/",
    "benchmark/include/generated-bindings/",
    "galp/benchmark/generated-bindings/",
    "galp/benchmark/include/generated-bindings/",
)

BENCHMARK_HEADER_PREFIXES = (
    "galp_bench/",
    "benchmark/",
    "galp/benchmark/",
)

TEST_HEADER_PREFIXES = (
    "test/",
    "galp/test/",
)

SUPPORT_HEADER_PREFIXES = (
    "galp_support/",
    "support/",
    "galp/support/",
)

BENCHMARK_HEADER_BASENAMES = {
    "data.cuh",
    "verification.cuh",
}

PUBLIC_HEADER_ALLOWLIST: dict[str, set[str]] = {}


def normalize_header(header: str) -> str:
    return header.replace("\\", "/").lstrip("./")


def rel_to_repo(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def is_source_file(path: Path) -> bool:
    return path.suffix in SOURCE_EXTENSIONS


def iter_source_files(galp_root: Path) -> list[Path]:
    scan_roots = [
        galp_root / "include",
        galp_root / "src",
        galp_root / "benchmark",
        galp_root / "test",
        galp_root / "tools",
        galp_root / "examples",
    ]
    files: list[Path] = []
    for root in scan_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and is_source_file(path):
                files.append(path)
    return sorted(files)


def parse_includes(path: Path) -> list[tuple[int, str]]:
    includes: list[tuple[int, str]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as exc:
        raise RuntimeError(f"failed to read {path}: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        match = INCLUDE_RE.match(line)
        if match:
            includes.append((line_number, normalize_header(match.group(1))))
    return includes


def parse_cmake_source_set(cmake_file: Path, variable: str) -> list[str]:
    text = cmake_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_set = False
    entries: list[str] = []
    start_re = re.compile(rf"^\s*set\s*\(\s*{re.escape(variable)}(?:\s+|$)")

    for raw_line in text:
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if not in_set:
            if start_re.match(line):
                in_set = True
                line = start_re.sub("", line).strip()
            else:
                continue
        if ")" in line:
            before_close = line.split(")", 1)[0].strip()
            if before_close:
                entries.extend(before_close.split())
            break
        entries.extend(line.split())

    return [entry for entry in entries if entry]


def galp_core_sources(galp_root: Path) -> set[str]:
    source_entries = parse_cmake_source_set(galp_root / "src" / "CMakeLists.txt", "GALP_CORE_SOURCES")
    return {(galp_root / "src" / entry).resolve().as_posix() for entry in source_entries}


def include_is_benchmark_generated(header: str) -> bool:
    return header.startswith(BENCHMARK_GENERATED_PREFIXES)


def include_is_benchmark(header: str) -> bool:
    return header.startswith(BENCHMARK_HEADER_PREFIXES) or header in BENCHMARK_HEADER_BASENAMES


def include_is_test(header: str) -> bool:
    return header.startswith(TEST_HEADER_PREFIXES)


def include_is_support(header: str) -> bool:
    return header.startswith(SUPPORT_HEADER_PREFIXES)


def include_is_private_implementation(header: str) -> bool:
    return header.startswith(PRIVATE_IMPLEMENTATION_PREFIXES)


def public_header_violation(file_rel: str, header: str) -> str | None:
    if header in PUBLIC_HEADER_ALLOWLIST.get(file_rel, set()):
        return None
    if include_is_benchmark_generated(header):
        return "public headers must not include benchmark-generated bindings"
    if include_is_benchmark(header):
        return "public headers must not include benchmark headers"
    if include_is_test(header):
        return "public headers must not include test headers"
    if include_is_support(header):
        return "public headers must not include support headers"
    if include_is_private_implementation(header):
        return "public headers must not include private implementation headers"
    return None


def production_include_violation(header: str) -> str | None:
    if include_is_benchmark_generated(header):
        return "production code must not include benchmark-generated bindings"
    if include_is_benchmark(header):
        return "production code must not include benchmark headers"
    if include_is_test(header):
        return "production code must not include test headers"
    if include_is_support(header):
        return "production code must not include support headers"
    return None


def is_public_header(path: Path, galp_root: Path) -> bool:
    public_root = galp_root / "include" / "galp"
    return path.resolve().is_relative_to(public_root.resolve()) and path.suffix in {".h", ".hh", ".hpp", ".hxx", ".cuh"}


def is_private_header(path: Path, galp_root: Path) -> bool:
    private_root = galp_root / "src" / "include"
    return path.resolve().is_relative_to(private_root.resolve()) and path.suffix in {".h", ".hh", ".hpp", ".hxx", ".cuh"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root; defaults to the parent of scripts/",
    )
    parser.add_argument(
        "--galp-root",
        type=Path,
        default=None,
        help="GALP subtree root; defaults to <repo-root>/galp",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    galp_root = (args.galp_root or (repo_root / "galp")).resolve()
    if not galp_root.exists():
        print(f"error: GALP root does not exist: {galp_root}", file=sys.stderr)
        return 2

    try:
        core_sources = galp_core_sources(galp_root)
    except OSError as exc:
        print(f"error: failed to read galp_core source list: {exc}", file=sys.stderr)
        return 2

    violations: list[tuple[str, int, str, str]] = []
    for path in iter_source_files(galp_root):
        file_rel = rel_to_repo(path, repo_root)
        is_core_source = path.resolve().as_posix() in core_sources
        is_public = is_public_header(path, galp_root)
        is_private = is_private_header(path, galp_root)

        for line_number, header in parse_includes(path):
            reason: str | None = None
            if is_public:
                reason = public_header_violation(file_rel, header)
            if reason is None and is_core_source:
                reason = production_include_violation(header)
            if reason is None and is_private:
                reason = production_include_violation(header)
            if reason is not None:
                violations.append((file_rel, line_number, header, reason))

    for file_rel, line_number, header, reason in violations:
        print(f"{file_rel}:{line_number}: includes {header} -- {reason}")

    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
