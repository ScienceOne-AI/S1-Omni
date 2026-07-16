#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spec2mol.jdx import load_spectra_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=str(PROJECT_ROOT / "spec2mol" / "dataset" / "test_data" / "000010"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = load_spectra_dir(args.input_dir)
    shapes = [tuple(tensor.shape) for tensor in bundle.context]
    print("present_spectra:", bundle.present_spectra)
    print("zero_filled_spectra:", bundle.zero_filled_spectra)
    print("shapes:", shapes)
    assert shapes == [(1, 1, 701), (1, 1, 3501), (1, 1, 3501)]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
