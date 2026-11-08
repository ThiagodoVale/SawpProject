from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sawp.workflows.cooling import build_event_table


def _series(path: Path, column: str) -> pd.Series:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    return frame[column]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one SAWP-rainfall event table")
    parser.add_argument("sst_csv", type=Path)
    parser.add_argument("rainfall_csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sst-column", default="sst")
    parser.add_argument("--rainfall-column", default="precipitation_mm")
    args = parser.parse_args()
    table = build_event_table(
        _series(args.sst_csv, args.sst_column),
        _series(args.rainfall_csv, args.rainfall_column),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
