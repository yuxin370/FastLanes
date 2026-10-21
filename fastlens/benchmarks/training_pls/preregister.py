#!/usr/bin/env python3
"""Write the immutable core PLS convergence/accuracy experiment matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import write_json
from .matrix import core_matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix = core_matrix()
    write_json(args.output.resolve(), matrix)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
