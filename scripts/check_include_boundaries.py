#!/usr/bin/env python3
"""Check lightweight GALP include-boundary rules.

This script intentionally checks only lightweight include and CMake target
boundaries. It does not move files, inspect transitive dependencies, or classify
algorithms. Production source membership is taken from the galp_core source list
rather than inferred only from paths.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]')
CMAKE_CALL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
CMAKE_SCOPES = {"PUBLIC", "PRIVATE", "INTERFACE"}

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
    "codecs/",
    "core/",
    "cuda/",
    "execution/",
    "io/",
    "runtime/",
    "storage/",
    # Legacy private prefixes must not leak into public headers either.
    "alp/",
    "compression/",
    "decompression/",
    "engine/",
    "galp/internal/",
    "memory/",
)

BENCHMARK_GENERATED_PREFIXES = (
    "galp_bench/generated/",
    "generated/bindings/",
    "benchmarks/generated/bindings/",
    "benchmarks/include/galp_bench/generated/",
    "galp/benchmarks/generated/bindings/",
    "galp/benchmarks/include/galp_bench/generated/",
    "benchmark/generated/bindings/",
    "benchmark/include/galp_bench/generated/",
    "galp/benchmark/generated/bindings/",
    "galp/benchmark/include/galp_bench/generated/",
)

BENCHMARK_HEADER_PREFIXES = (
    "galp_bench/",
    "benchmarks/",
    "galp/benchmarks/",
    "benchmark/",
    "galp/benchmark/",
)

TEST_HEADER_PREFIXES = (
    "tests/",
    "galp/tests/",
    "test/",
    "galp/test/",
)

SUPPORT_HEADER_PREFIXES = (
    "galp_extensions/",
    "galp_tools/benchmark_support/",
    "galp_tools/data/",
    "galp_support/",
    "support/",
    "galp/support/",
)

BENCHMARK_HEADER_BASENAMES = {
    "data.cuh",
    "verification.cuh",
}

PUBLIC_HEADER_ALLOWLIST: dict[str, set[str]] = {}
FORBIDDEN_GALP_CORE_LINKS = {
    "galp_benchmark_support",
    "galp_generated_bindings",
    "generated-bindings",
    "galp_tests",
    "galp_alp_support",
}
FORBIDDEN_CORE_SOURCE_ROOTS = (
    "benchmarks",
    "extensions",
    "support",
    "test",
    "tests",
    "tools",
    # Legacy roots kept here so a partial migration still fails loudly.
    "benchmark",
)
FORBIDDEN_EXPOSED_INCLUDE_PARTS = (
    "galp/benchmarks/include",
    "galp/extensions",
    "galp/src",
    "galp/support/include",
    "galp/benchmark/include",
    "galp/src/include",
    "galp/tools/benchmark_support",
    "galp/tools/data",
    "benchmarks/include",
    "extensions",
    "support/include",
    "benchmark/include",
    "src/include",
    "tools/benchmark_support",
    "tools/data",
)


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
        galp_root / "benchmarks",
        galp_root / "extensions",
        galp_root / "tests",
        galp_root / "benchmark",
        galp_root / "support",
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


def parse_cmake_source_set_entries(cmake_file: Path, variable: str) -> list[tuple[int, str]]:
    text = cmake_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_set = False
    entries: list[tuple[int, str]] = []
    start_re = re.compile(rf"^\s*set\s*\(\s*{re.escape(variable)}(?:\s+|$)")

    for line_number, raw_line in enumerate(text, start=1):
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
                entries.extend((line_number, entry) for entry in before_close.split())
            break
        entries.extend((line_number, entry) for entry in line.split())

    return [(line_number, entry) for line_number, entry in entries if entry]


def parse_cmake_source_set(cmake_file: Path, variable: str) -> list[str]:
    return [entry for _, entry in parse_cmake_source_set_entries(cmake_file, variable)]


def galp_core_sources(galp_root: Path) -> set[str]:
    source_entries = parse_cmake_source_set(galp_root / "src" / "CMakeLists.txt", "GALP_CORE_SOURCES")
    return {(galp_root / "src" / entry).resolve().as_posix() for entry in source_entries}


def iter_cmake_files(galp_root: Path) -> list[Path]:
    return sorted(galp_root.rglob("CMakeLists.txt"))


def strip_cmake_comment(line: str) -> str:
    return line.split("#", 1)[0]


def iter_cmake_calls(cmake_file: Path) -> list[tuple[str, int, str]]:
    lines = cmake_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    calls: list[tuple[str, int, str]] = []
    index = 0
    while index < len(lines):
        line = strip_cmake_comment(lines[index])
        match = CMAKE_CALL_RE.match(line)
        if not match:
            index += 1
            continue

        command = match.group(1)
        start_line = index + 1
        block = [line]
        depth = line.count("(") - line.count(")")
        index += 1
        while depth > 0 and index < len(lines):
            line = strip_cmake_comment(lines[index])
            block.append(line)
            depth += line.count("(") - line.count(")")
            index += 1
        calls.append((command, start_line, "\n".join(block)))
    return calls


def cmake_call_args(command: str, text: str) -> list[str]:
    body = re.sub(rf"^\s*{re.escape(command)}\s*\(", "", text, count=1, flags=re.IGNORECASE).strip()
    if body.endswith(")"):
        body = body[:-1]
    return [token.strip('"') for token in re.findall(r'"[^"]*"|[^\s()]+', body)]


def collect_targets_under(galp_root: Path, subdir: str) -> set[str]:
    root = galp_root / subdir
    if not root.exists():
        return set()
    targets: set[str] = set()
    for cmake_file in root.rglob("CMakeLists.txt"):
        for command, _, text in iter_cmake_calls(cmake_file):
            if command.lower() not in {"add_library", "add_executable", "add_custom_target"}:
                continue
            args = cmake_call_args(command, text)
            if args:
                targets.add(args[0])
    return targets


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


def path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def exposed_include_path_violation(token: str) -> str | None:
    normalized = normalize_header(token.strip('"'))
    if any(part in normalized for part in FORBIDDEN_EXPOSED_INCLUDE_PARTS):
        return "install/export/public usage requirements must not expose private, benchmark, or support include dirs"
    return None


def galp_core_link_violation(token: str, forbidden_targets: set[str]) -> str | None:
    if token in CMAKE_SCOPES:
        return None
    if token in forbidden_targets or token in FORBIDDEN_GALP_CORE_LINKS:
        return "galp_core must not link benchmark, test, support, or benchmark-generated targets"

    lowered = token.lower()
    if "gtest" in lowered or "gmock" in lowered:
        return "galp_core must not link test-only dependencies"
    if "benchmark" in lowered:
        return "galp_core must not link benchmark-only dependencies"
    if "nvcomp" in lowered:
        return "galp_core must not link benchmark-only nvCOMP dependencies"
    return None


def check_galp_core_sources(galp_root: Path, repo_root: Path) -> list[tuple[str, int, str, str, str]]:
    cmake_file = galp_root / "src" / "CMakeLists.txt"
    violations: list[tuple[str, int, str, str, str]] = []
    forbidden_roots = [(name, galp_root / name) for name in FORBIDDEN_CORE_SOURCE_ROOTS]

    for line_number, entry in parse_cmake_source_set_entries(cmake_file, "GALP_CORE_SOURCES"):
        entry_path = Path(entry)
        source_path = entry_path if entry_path.is_absolute() else (galp_root / "src" / entry)
        resolved = source_path.resolve()
        for root_name, root in forbidden_roots:
            if path_is_under(resolved, root):
                violations.append(
                    (
                        rel_to_repo(cmake_file, repo_root),
                        line_number,
                        entry,
                        f"galp_core sources must not include files from galp/{root_name}/",
                        "references",
                    )
                )
                break
    return violations


def check_cmake_boundaries(galp_root: Path, repo_root: Path) -> list[tuple[str, int, str, str, str]]:
    violations: list[tuple[str, int, str, str, str]] = []
    forbidden_targets = (
        collect_targets_under(galp_root, "benchmarks")
        | collect_targets_under(galp_root, "extensions")
        | collect_targets_under(galp_root, "tests")
        | collect_targets_under(galp_root, "tools")
        | collect_targets_under(galp_root, "benchmark")
        | collect_targets_under(galp_root, "test")
        | collect_targets_under(galp_root, "support")
    )

    violations.extend(check_galp_core_sources(galp_root, repo_root))

    for cmake_file in iter_cmake_files(galp_root):
        file_rel = rel_to_repo(cmake_file, repo_root)
        for command, line_number, text in iter_cmake_calls(cmake_file):
            command_lower = command.lower()
            args = cmake_call_args(command, text)
            if not args:
                continue

            if command_lower == "target_link_libraries" and args[0] == "galp_core":
                for token in args[1:]:
                    reason = galp_core_link_violation(token, forbidden_targets)
                    if reason is not None:
                        violations.append((file_rel, line_number, token, reason, "references"))

            if command_lower == "target_include_directories" and args[0] == "galp_core":
                scope: str | None = None
                for token in args[1:]:
                    if token in CMAKE_SCOPES:
                        scope = token
                        continue
                    if scope not in {"PUBLIC", "INTERFACE"}:
                        continue
                    reason = exposed_include_path_violation(token)
                    if reason is not None:
                        violations.append((file_rel, line_number, token, reason, "references"))

            if command_lower in {"install", "export"}:
                for token in args:
                    reason = exposed_include_path_violation(token)
                    if reason is not None:
                        violations.append((file_rel, line_number, token, reason, "references"))

    return violations


def is_public_header(path: Path, galp_root: Path) -> bool:
    public_root = galp_root / "include" / "galp"
    return path.resolve().is_relative_to(public_root.resolve()) and path.suffix in {".h", ".hh", ".hpp", ".hxx", ".cuh"}


def is_private_header(path: Path, galp_root: Path) -> bool:
    private_root = galp_root / "src"
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

    violations: list[tuple[str, int, str, str, str]] = []
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
                violations.append((file_rel, line_number, header, reason, "includes"))

    violations.extend(check_cmake_boundaries(galp_root, repo_root))

    for file_rel, line_number, item, reason, verb in violations:
        print(f"{file_rel}:{line_number}: {verb} {item} -- {reason}")

    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
